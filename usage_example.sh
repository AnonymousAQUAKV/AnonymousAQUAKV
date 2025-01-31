#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export HF_HOME=your huggingface home directory
export HF_DATASETS_CACHE=your huggingface dataset caches directory

 
export HIGGS_GRIDS_PATH=AnonymousAQUAKV/grids/
# ^-- not actually needed for predictors eval; legacy
 
# higgs
EDENN_D=2
EDENN_N=16

HADAMARD_GROUPSIZE=1024
QUANTIZATION_GROUP_SIZE=128

PREFIX_SIZE=4
CUSTOM_CACHE_NAME=higgs_predictors

# model
MODEL_PATH=unsloth/Llama-3.2-3B-Instruct
MODEL_SEQLEN=8192


LONGBENCH_MODEL_NAME=llama-3.2-3B-Instruct
LONGBENCH_DATASETS=samsum,2wikimqa,trec,hotpotqa,multi_news,triviaqa,qmsum,passage_count,multifieldqa_en,musique,qasper,passage_retrieval_en,narrativeqa,gov_report
LONGBENCH_OUT_PATH=./higgs_predictors_2_16_128

# calibration
CALIBRATION_DATASET=pajama
TOTAL_NSAMPLES=256
VALIDATION_NSAMPLES=32
PREDICTORS_SAVE_PATH=saved_predictors/key_value_predictors_instruct_2_16_128.pt


python train_predictors.py --model_name $MODEL_PATH --model_seqlen $MODEL_SEQLEN --predictors_output_path $PREDICTORS_SAVE_PATH \
 --dataset $CALIBRATION_DATASET --total_nsamples $TOTAL_NSAMPLES --valid_nsamples $VALIDATION_NSAMPLES \
 --edenn_d $EDENN_D --edenn_n $EDENN_N --hadamard_groupsize $HADAMARD_GROUPSIZE --offload_activations
# ^-- note: train_predictors currently ignores $PREFIX_SIZE (attention sinks) and QUANTIZATION_GROUP_SIZE (buffering) during calibration;

python evaluate_perplexity.py --model_name $MODEL_PATH --model_seqlen $MODEL_SEQLEN --predictors_input_path $PREDICTORS_SAVE_PATH \
 --edenn_d $EDENN_D --edenn_n $EDENN_N --hadamard_groupsize $HADAMARD_GROUPSIZE \
 --prefix_size $PREFIX_SIZE --quantization_group_size $QUANTIZATION_GROUP_SIZE --quanto_nbits -custom_cache_name $CUSTOM_CACHE_NAME

# LongBench predictions (to be followed by eval.py)
python pred.py --model $LONGBENCH_MODEL_NAME --predictors_input_path=$PREDICTORS_SAVE_PATH \
 --edenn_d $EDENN_D --edenn_n $EDENN_N --hadamard_groupsize $HADAMARD_GROUPSIZE \
 --prefix_size $PREFIX_SIZE --quantization_group_size $QUANTIZATION_GROUP_SIZE \
 --datasets=$LONGBENCH_DATASETS --out_path=$LONGBENCH_OUT_PATH --custom_cache_name $CUSTOM_CACHE_NAME

python eval.py --model $LONGBENCH_MODEL_NAME --logbench_out_path=$LONGBENCH_OUT_PATH
