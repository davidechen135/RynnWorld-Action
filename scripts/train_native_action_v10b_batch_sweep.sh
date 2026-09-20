#!/bin/bash
# One-factor-at-a-time batch size sweep on top of the v10b native_trajectory config.
# Everything else (LR, steps, schedule, loss weights) stays fixed at the same
# values as training/native_action_v10b_state_256_600 so batch_size is the only
# manipulated variable. Requires the per-sample action-dropout fix in
# rynnworld_teleop_trainer.py (2026-09-16) -- at batch_size>1 the old whole-batch
# dropout would have confounded this sweep.
#
# bs=8 excluded (2026-09-16): two separate smoke tests OOM'd on GPU0 during the
# gradient-checkpointed backward pass. GPU0 has persistent co-tenant processes
# using ~42GB (32.46GB + 9.67GB) plus a third transient process at ~37GB during
# both attempts; bs=4 alone already peaks at 29.4GB reserved, so bs=8 would need
# ~50-55GB, which does not fit alongside the other tenants on this shared A100-80GB
# host. Not a fragmentation issue (PYTORCH_CUDA_ALLOC_CONF wouldn't help) -- it's a
# genuine capacity shortfall from concurrent unrelated jobs.
set -uo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29760}"

LOG_DIR="training/logs"
mkdir -p "$LOG_DIR"

for BS in 2 4; do
  OUT="training/native_action_v10b_state_256_bs${BS}_600"
  LOG="$LOG_DIR/native_action_v10b_bs${BS}_600.log"
  echo "=== batch_size=${BS} -> ${OUT} ==="
  BATCH_SIZE="$BS" \
  OUTPUT_DIR="$OUT" \
  TRAIN_STEPS=600 \
  CHECKPOINTING_STEPS=200 \
  MASTER_PORT=$((MASTER_PORT_BASE + BS)) \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  bash scripts/train_native_action_v10b_state_256.sh > "$LOG" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "=== batch_size=${BS} FAILED (exit $status), see $LOG ==="
  else
    echo "=== batch_size=${BS} DONE ==="
  fi
done
