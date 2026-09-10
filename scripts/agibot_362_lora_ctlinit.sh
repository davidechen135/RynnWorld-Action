#!/bin/bash
# AgiBot real-robot ego LoRA fine-tune — TASK 362 "Folding shorts", MULTI-episode.
#
# This is the task-362 sibling of scripts/agibot_lora_2gpu.sh (which did task-357,
# path A = single "washing dishes" episode -> 40 clips). The user narrowed scope to
# task 362 only and opted into downloading MORE data: 20 whole training episodes
# (+3 held-out) instead of one. So the data volume is ~10x v2's and the run length
# is scaled up accordingly.
#
# Data: /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel.json — 20 task-362 episodes, each AV1 head_color.mp4
# sliced into 81-frame clips, real end-effector trajectory rendered to a 21-keypoint
# skeleton video and encoded by the official VAE as control (same control path as
# v2 skeleton, docs/agibot_finetune.md). Held-out = 3 whole unseen episodes
# (/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel_heldout/), zero frame overlap with training.
#
# Scene diversity is still 1 (all 220 in-range episodes are shorts-on-bed); the gain
# over v2 is episode/trajectory count + frame volume, NOT scene variety. Documented.
#
# 2x H20 96G. LoRA rank 32 + control_patch_embedding, control_type add.
#
# Run length: clip count is discovered at prep time and written into the json. With
# ~20 episodes averaging ~1600 frames -> ~19 clips/episode -> ~380 clips (vs v2's 40).
# grad_accum 8 / 2 GPU => 16 clips/step => ~24 steps/epoch. 4 epochs => ~95 steps.
# EMA decay 0.99 needs enough updates to matter; start at step 10 so it averages over
# the back ~85 steps (v2's EMA only reached 11% of trained magnitude because it had 12
# updates — with ~85 it actually converges). checkpointing every 12 steps, keep 8.
# lr/control_lr and reg identical to v2 so the only deliberate change vs v2 is DATA
# SCALE — that isolates "does more real-episode data help" as the question under test.
#
# Usage: bash scripts/agibot_362_lora_2gpu.sh
set -e
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MASTER_ADDR=localhost
export MASTER_PORT=29517
export NNODES=1
export NUM_PROCESSES=2
export MODEL_PATH="${MODEL_PATH:-pretrained/Wan2.2-TI2V-5B-Diffusers}"
export DATA_JSON="${DATA_JSON:-data/agibot_362_skel.json}"

LAUNCHER="accelerate launch \
    --config_file configs_acc/2gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES"

PROGRAM="finetune.py \
    --model_path $MODEL_PATH \
    --model_name rynnworld_teleop \
    --model_type rynnworld_teleop \
    --training_type lora \
    --rank 32 \
    --lora_alpha 32 \
    --target_modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 ffn.net.0.proj ffn.net.2 \
    --output_dir training/agibot_362_lora_ctlinit \
    --report_to tensorboard \
    --train_resolution 81x480x832 \
    --train_epochs 3 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision bf16 \
    --num_workers 4 \
    --pin_memory True \
    --nccl_timeout 7200 \
    --checkpointing_steps 24 \
    --checkpointing_limit 8 \
    --do_validation false \
    --validation_dir $DATA_JSON \
    --cache_dir data \
    --control_type add \
    --learning_rate 1e-4 \
    --control_lr 5e-5 \
    --max_grad_norm 1.0 \
    --beta2 0.999 \
    --weight_decay 0.01 \
    --epsilon 1e-10 \
    --lr_scheduler cosine \
    --lr_warmup_steps 5 \
    --ema_decay 0.99 \
    --ema_start_step 10 \
    --prompt '' \
    --control_init_from pretrained/RynnWorld-Teleop \
    --reg_weight_init 0.01 \
    --reg_weight_decay_steps 500"

eval "$LAUNCHER $PROGRAM"
