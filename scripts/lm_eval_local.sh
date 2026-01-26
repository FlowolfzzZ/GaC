#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate gac

export HF_ENDPOINT="https://hf-mirror.com"

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/home/user_name/Models/Qwen-32B"
BASE_URL="http://localhost:8000/v1/chat/completions"
WANDB_ARGS="project=lm-eval-harness"

lm_eval --model local-chat-completions \
  --model_args pretrained=${MODEL_PATH},base_url=${BASE_URL},num_concurrent=8,timeout=1800,max_retries=3,tokenized_requests=False,max_tokens=4096 \
  --tasks "minerva_math500,aime24,gpqa_diamond_cot_n_shot" \
  --device auto \
  --apply_chat_template \
  --log_samples \
  --num_fewshot 0 \
  --output_path output \
  --gen_kwargs temperature=0.7,top_p=0.8,top_k=20,max_gen_toks=4096 \
  --wandb_args ${WANDB_ARGS}
