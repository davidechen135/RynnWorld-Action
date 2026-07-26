#!/usr/bin/env bash
# Drive official inference_user.py over all 8 example cases (SFT).
set -e
CASES=(
  assemble_jenga_001
  basic_fold_009
  basic_pick_place_000
  clean_surface_001
  clip_unclip_papers_006
  color_004
  flip_pages_008
  fold_unfold_paper_basic_008
)
for c in "${CASES[@]}"; do
  out="outputs/repro_sft/$c"
  if [ -f "$out/rollout_seed42.mp4" ] || [ -f "$out/generated_seed42.mp4" ]; then
    echo "== skip $c (exists) =="; continue
  fi
  echo "===== RUN $c ====="
  python -u inference_user.py \
    --image "example/$c/first_frame.png" \
    --control_video "example/$c/control_video.mp4" \
    --text_embedding "example/$c/text_embedding.safetensors" \
    --output "$out" \
    --checkpoint pretrained/RynnWorld-Teleop \
    --mode sft --control_type add --seeds "42" 2>&1 | tail -3
done
echo "ALL DONE"
