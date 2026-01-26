#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate gac

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python gac_api_server.py \
  --config-path example_configs/3model_ensemble_every_step.yaml \
  --host 0.0.0.0 \
  --port 8000
