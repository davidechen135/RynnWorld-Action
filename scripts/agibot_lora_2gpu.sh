#!/bin/bash
# AgiBot real-robot ego LoRA fine-tune (path A: single task-357 "washing dishes"
# episode sliced into ~40 clips). Crosses the "no fine-tune" boundary of the
# zero-shot probes (docs/agibot_crossdomain.md) so the frozen world model actually
# adapts to the real dual-arm robot domain.
#
# 2x H20. Small data (40 clips) -> LoRA rank 32 + control_patch_embedding, short
# run, EMA early, to adapt without catastrophic forgetting.
#
# Hyper-params corrected after reading the v1 curves
# (outputs/agibot_lora_357/training_curves.png):
#   * v1 ran 8 epochs = 24 optimizer steps, but the loss trough was at ~step 21 and
#     it rebounded to 0.282 by step 24 -> over-fitting on 40 clips. Cut to 6 epochs.
#   * v1 set --ema_start_step 40 > the 24 total steps, so EMA.update() NEVER ran and
#     `ema_final` was the *pre-training* shadow: its LoRA B matrices were still the
#     zero-init, i.e. a no-op adapter. Start EMA at 6 so it actually averages.
#   * v1 warmup was 20 of 24 steps and the cosine period was sized for a much longer
#     run (lr was still climbing at the end). Warmup 3, cosine over the real length.
#
# v2 also switches the control signal: `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel.json` is built with
# --control-mode skeleton, i.e. the real end-effector trajectory rendered into a
# hand-pose-style video and encoded by the official VAE. v1's hand-built latent
# matched mu/sigma but averaged only 0.0398 temporal delta over the 40 clips, 25% of
# the real hand-pose control's 0.1587; the rendered one reaches 0.0745 (47%). Both
# track arm speed about equally (r~0.70); the gain is in level, not in responsiveness.
# See scripts/agibot_skeleton_render.py.
#
# Usage: bash scripts/agibot_lora_2gpu.sh
set -e
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MASTER_ADDR=localhost
export MASTER_PORT=29512
export NNODES=1
export NUM_PROCESSES=2
export MODEL_PATH="${MODEL_PATH:-pretrained/Wan2.2-TI2V-5B-Diffusers}"
export DATA_JSON="${DATA_JSON:-data/agibot_357_skel.json}"

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
    --output_dir training/agibot_357_lora_v2_skel \
    --report_to tensorboard \
    --train_resolution 81x480x832 \
    --train_epochs 6 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision bf16 \
    --num_workers 4 \
    --pin_memory True \
    --nccl_timeout 7200 \
    --checkpointing_steps 6 \
    --checkpointing_limit 5 \
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
    --lr_warmup_steps 3 \
    --ema_decay 0.99 \
    --ema_start_step 6 \
    --prompt '' \
    --reg_weight_init 0.01 \
    --reg_weight_decay_steps 500"

eval "$LAUNCHER $PROGRAM"
