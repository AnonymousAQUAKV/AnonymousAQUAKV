#!/usr/bin/env python
# coding: utf-8
import os
import torch
import torch.nn as nn
from typing import Optional, Callable

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache

@torch.no_grad()
def evaluate_perplexity(
        model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device,
        amp_dtype: Optional[torch.dtype] = None, step_size: Optional[int] = None,
        cache_factory: Optional[Callable[[], DynamicCache]] = None
        ) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    if step_size is None:
        step_size = seqlen
    inps = [
            data[:, start : start + seqlen] for start in range(0, data.shape[1], seqlen) if start + seqlen < data.shape[1]
            ]  # ignore last incomplete sequence as in the GPTQ paper
    num_sequences_without_padding = len(inps)


    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity")):
        input_ids = input_ids.to(device)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            if cache_factory is None:
                cache = DynamicCache()
            else:
                cache = cache_factory()
            dtype = amp_dtype or next(model.parameters()).dtype
            lm_logits = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], model.get_output_embeddings().out_features), device=device, dtype=dtype)
            for i in range(0, input_ids.shape[1], step_size):
                out = model(input_ids[:, i: i + step_size], use_cache=True, past_key_values=cache)
                assert out.past_key_values is cache
                lm_logits[:, i: i + step_size, ...] = out.logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()
        else:
            raise RuntimeError

    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()




@torch.no_grad()
def eval_ppl_wikitext_llama3b(cache_factory, model_id="/home/ermakoviv/Llama-3.2-3B/", seqlen=8192, steps_in_seq=8):
    device = "cuda"
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, torch_dtype='auto', low_cpu_mem_usage=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, config=config, padding_side="left")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")['input_ids']
    step_size = seqlen // steps_in_seq
    return evaluate_perplexity(model, testenc, seqlen, device=device, step_size=step_size, cache_factory=cache_factory)
