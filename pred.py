import os

import transformers
from datasets import load_dataset
import torch
import json
import socket
import time

from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from functools import partial
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from custom_cache import get_quanto_prediction_cache, get_test_quanto_prediction_cache, get_rht_cache, \
    get_higgs_predictors

QUANT_BITS = 1

DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "passage_retrieval_en",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "hotpotqa"
]


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "llama-3.2-3B", "llama-3.2-3B-test", "llama-3.2-3B-Instruct"])
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--custom_cache_name', type=str, default=None, choices=["quanto-pred", "test-quanto-pred", "rht","higgs_predictors"])
    parser.add_argument('--out_path', type=str, default="pred")
    parser.add_argument("--quantization_group_size",
                        type=int,
                        default=128,
                        help="Accumulate at least this many tokens before quantizing.")
    parser.add_argument("--hadamard_groupsize", type=int)
    parser.add_argument("--edenn_n", type=int)
    parser.add_argument("--edenn_d", type=int)
    parser.add_argument("--prefix_size",
                        type=int,
                        default=4,
                        help="The number of first tokens that will not be quantized, because of attention sink.")

    parser.add_argument('-e', dest='e', action='store_true')
    parser.add_argument('-a', dest='all', action='store_true', help="Evaluate on slow tasks as well")
    parser.add_argument('--datasets', type=str, default=None)

    parser.add_argument("--predictors_input_path",type=str,default=None)

    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif model_name.lower().endswith("instruct"):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                       tokenize=False, add_generation_prompt=True)
        if prompt.startswith(tokenizer.bos_token):  # BOS will be added again in tokenization
            prompt = prompt[len(tokenizer.bos_token):]

    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, use_quantization=False, custom_quantization=False, cache=None):
    free_port = find_free_port()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(free_port)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids[0].shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            if custom_quantization:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    #cache_position=cache_position,
                    past_key_values=cache(device),
                    use_cache=True
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    cache_implementation="quantized" if use_quantization else None,
                    cache_config={"backend": "quanto", "nbits": QUANT_BITS} if use_quantization else None
                )[0]
        else:
            if custom_quantization:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    past_key_values=cache(device),
                    use_cache=True
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    cache_implementation="quantized" if use_quantization else None,
                    cache_config={"backend": "quanto", "nbits": QUANT_BITS} if use_quantization else None
                )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "llama-3.2" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16  # float16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = model.eval()
    return model, tokenizer


if __name__ == '__main__':
    time_started = time.time()
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    custom_cache_name = args.custom_cache_name
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.all:
        datasets = DATASETS
    else:
        assert args.datasets is not None
        datasets = [ds.strip() for ds in args.datasets.split(",")]
        print(datasets)
        for dataset in datasets:
            if dataset not in DATASETS:
                raise ValueError(f"dataset {dataset} not supported")
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if args.quantize:
        suffix = '_quantize' + str(QUANT_BITS)
    else:
        suffix = ''
    if not os.path.exists(args.out_path + suffix):
        os.makedirs(args.out_path + suffix)
    if not os.path.exists("pred_e" + suffix):
        os.makedirs("pred_e" + suffix)
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name + suffix}"):
                os.makedirs(f"pred_e/{model_name + suffix}")
            out_path = f"pred_e/{model_name + suffix}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"{args.out_path}/{model_name + suffix}"):
                os.makedirs(f"{args.out_path}/{model_name + suffix}")
            out_path = f"{args.out_path}/{model_name + suffix}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        if custom_cache_name is not None:
            custom_quantization = True
            if custom_cache_name == 'quanto-pred':
                cache = get_quanto_prediction_cache
            elif custom_cache_name == 'test-quanto-pred':
                cache = get_test_quanto_prediction_cache
            elif custom_cache_name == 'rht':
                cache = partial(
                    get_rht_cache,
                    quantization_group_size=args.quantization_group_size,
                    hadamard_groupsize=args.hadamard_groupsize,
                    edenn_n=args.edenn_n,
                    edenn_d=args.edenn_d,
                )
            elif  custom_cache_name == 'higgs_predictors':
                if args.predictors_input_path:
                    key_values = torch.load(args.predictors_input_path)
                    key_predictors, value_predictors = key_values["key_predictors"], key_values["value_predictors"]
                else:
                    key_predictors = None
                    value_predictors = None

                cache = partial(get_higgs_predictors,
                                hadamard_groupsize=args.hadamard_groupsize,
                                edenn_n=args.edenn_n,
                                edenn_d=args.edenn_d,
                                quantization_group_size=args.quantization_group_size,
                                prefix_size=args.prefix_size,
                                config=transformers.AutoConfig.from_pretrained(model2path[model_name]),
                                key_predictors=key_predictors,
                                value_predictors=value_predictors,
                                quantizer_type="higgs"
                                )

            # place for new custom cache types
        else:
            custom_quantization = False
            cache = None
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
            max_gen, prompt_format, dataset, device, model_name, model2path, out_path, args.quantize, \
            custom_quantization, cache))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    print("Total time took: ", time.time() - time_started)
