#!/bin/bash

# Single-node multi-GPU launch script for RynnWorld-Teleop SFT pretraining
# Usage: bash scripts/rynnworld_teleop_pretrain_single_node.sh

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Set CUDA_HOME for DeepSpeed compatibility (fixes nvcc detection in containers)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NNODES=1
export NUM_PROCESSES=8

# Base model path — override via env if downloaded elsewhere.
export MODEL_PATH="${MODEL_PATH:-pretrained/Wan2.2-TI2V-5B-Diffusers}"

export LAUNCHER="accelerate launch \
    --config_file configs_acc/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "

export PROGRAM="\
finetune.py \
    --model_path $MODEL_PATH \
    --model_name rynnworld_teleop_pretrain \
    --model_type rynnworld_teleop_pretrain \
    --training_type sft \
    --output_dir training/rynnworld-teleop-SFT-pretrain \
    --report_to tensorboard \
    --train_resolution 81x480x832 \
    --train_epochs 1 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --mixed_precision bf16 \
    --num_workers 8 \
    --pin_memory True \
    --nccl_timeout 7200 \
    --checkpointing_steps 100 \
    --checkpointing_limit 20 \
    --resume_from_checkpoint latest \
    --do_validation false \
    --validation_dir /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/sample_data.json \
    --cache_dir data \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --beta2 0.999 \
    --weight_decay 0.01 \
    --epsilon 1e-10 \
    --lr_scheduler cosine \
    --lr_warmup_steps 200 \
    --ema_decay 0.999 \
    --ema_start_step 300 \
    --prompt '' \
"

export CMD="$LAUNCHER $PROGRAM"
eval "$CMD"
