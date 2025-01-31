import torch
import transformers

from prekv.quantizers import HiggsQuantizer
from prekv.cache_utils import TreatPrefixSeparately,PredictorHiggsCache,SingleChunkQuantizedCacheWithPredictors
from functools import partial
from ppl import evaluate_perplexity
from datasets import load_dataset

def make_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model_name",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "--edenn_d",
        type=int,
        help="The grid dimension d for HIGGS.",
    )
    parser.add_argument(
        "--edenn_n",
        type=int,
        help="The grid size n for HIGGS.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=8192,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument("--devices",
                        metavar="N",
                        type=str,
                        nargs="+",
                        default=None,
                        help="List of devices")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
             "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--chunk_size", #<- need to be renamed
        type=int,
        default=32,
        help="Number of tokens processed in one forward pass for simulated sequential generation.",
    )
    parser.add_argument(
        "--quantization_group_size",#<- need to be renamed
        type=int,
        default=128,
        help="Accumulate at least this many tokens before quantizing.",
    )
    parser.add_argument(
        "--hadamard_groupsize",
        type=int,
        default=1024,
        help="Groupsize of Hadamard transform for HIGGS.",
    )
    parser.add_argument(
        "--predictors_input_path",
        type=str,
        default="./key_value_predictors.pt",
        help="Path to saved trained predictors for Key and Values",
    )
    parser.add_argument("--prefix_size",
                        type=int,
                        default=4,
                        help="The number of first tokens that will not be quantized, because of attention sink.")

    return parser

def main():
    parser = make_arg_parser()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    args = parser.parse_args()
    if args.devices is None:
        if torch.cuda.is_available():
            args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            args.devices = [torch.device("cpu")]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert len(args.devices) == 1, "parallelism is still WIP"
    # loading predictors
    key_values_predictors = torch.load(args.predictors_input_path)
    key_predictors, value_predictors = key_values_predictors["key_predictors"], key_values_predictors["value_predictors"]
    [key_predictors[i].to(args.devices[0]) for i in key_predictors]
    [value_predictors[i].to(args.devices[0]) for i in value_predictors]
    #loading model and datasets
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    config = transformers.AutoConfig.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, torch_dtype=args.torch_dtype, low_cpu_mem_usage=True, device_map=args.devices[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, config=config, padding_side="left")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")['input_ids']
    step_size = args.chunk_size

    with torch.no_grad():
        # TODO: make this option in args to eval other quantizers
        quantizer = HiggsQuantizer(args.hadamard_groupsize, args.edenn_d, args.edenn_n)
        cache_factory = lambda: TreatPrefixSeparately(prefix_size=args.prefix_size,
                                                      prefix_cache=transformers.DynamicCache(),
                                                      suffix_cache=PredictorHiggsCache(
                                                          config=model.config, min_buffer_size=args.quantization_group_size,
                                                          save_dequantized_values=True,
                                                          make_quantized_cache=partial(
                                                              SingleChunkQuantizedCacheWithPredictors,
                                                              quantizer=quantizer,
                                                              key_predictors=key_predictors,
                                                              value_predictors=value_predictors
                                                          )
                                                      ))

        ppl_quantized = evaluate_perplexity(model, testenc, args.model_seqlen, device=args.devices[0],
                                            step_size=step_size, cache_factory=cache_factory)

    print(f"PPL on with quantized cache {ppl_quantized}\n")

if __name__ == "__main__":
    main()
