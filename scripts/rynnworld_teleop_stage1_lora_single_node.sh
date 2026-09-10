#!/bin/bash

# Single-node multi-GPU launch script for RynnWorld-Teleop Stage 1 LoRA training
# Usage: bash scripts/rynnworld_teleop_stage1_lora_single_node.sh

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

# ============================================================
# Training strategy: Train control_patch_embedding + LoRA together
# with different learning rates to prevent catastrophic forgetting
# ============================================================
export PROGRAM="\
finetune.py \
    --model_path $MODEL_PATH \
    --model_name rynnworld_teleop \
    --model_type rynnworld_teleop \
    --training_type lora \
    --rank 64 \
    --lora_alpha 64 \
    --target_modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 ffn.net.0.proj ffn.net.2 \
    --output_dir training/rynnworld-teleop-stage1-lora \
    --report_to tensorboard \
    --train_resolution 81x480x832 \
    --train_epochs 1 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --mixed_precision bf16 \
    --num_workers 8 \
    --pin_memory True \
    --nccl_timeout 7200 \
    --checkpointing_steps 100 \
    --checkpointing_limit 100 \
    --do_validation false \
    --validation_dir /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/sample_data.json \
    --cache_dir data \
    --control_type add \
    --learning_rate 2e-5 \
    --control_lr 2e-5 \
    --max_grad_norm 1.0 \
    --beta2 0.999 \
    --weight_decay 0.01 \
    --epsilon 1e-10 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --ema_decay 0.999 \
    --ema_start_step 200 \
    --prompt '' \
    --reg_weight_init 0.01 \
    --reg_weight_decay_steps 2000 \
"

export CMD="$LAUNCHER $PROGRAM"
eval "$CMD"
