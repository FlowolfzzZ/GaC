#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate gac

export HF_ENDPOINT="https://hf-mirror.com"

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="/home/user_name/Models/Qwen3-32B"
WANDB_ARGS="project=lm-eval-harness"

lm_eval --model vllm \
  --model_args pretrained=${MODEL_PATH},tensor_parallel_size=2,max_model_len=8192 \
  --tasks "minerva_math500,aime24,gpqa_diamond_cot_n_shot" \
  --device auto \
  --batch_size 32 \
  --apply_chat_template \
  --log_samples \
  --num_fewshot 4 \
  --output_path output \
  --gen_kwargs temperature=0.7,top_p=0.8,top_k=20,max_gen_toks=4096 \
  --wandb_args ${WANDB_ARGS}