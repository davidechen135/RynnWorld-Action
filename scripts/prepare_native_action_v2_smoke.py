#!/usr/bin/env python3
"""Build a balanced raw-33D smoke set from the audited Gate-D latent cache."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


REPO = Path(__file__).resolve().parents[1]
SOURCE_RUN = REPO / "reports/direct_action/gate_d/run_015"
SOURCE_CACHE = Path("/tmp/scratch/gate_d_run001_cache")
OUTPUT = REPO / "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_raw33_v2_smoke"
PER_TASK = 128


def main() -> None:
    selected = json.loads(
        (SOURCE_RUN / "artifacts/selected_5000_scaled.json").read_text()
    )
    stats_doc = json.loads(
        (SOURCE_RUN / "artifacts/action_stats_combined.json").read_text()
    )
    per_task = stats_doc["stats"]["per_task"]
    counts: dict[str, int] = {}
    index = []
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for record in selected:
        task_id = str(record["task_id"])
        if counts.get(task_id, 0) >= PER_TASK:
            continue
        source = SOURCE_CACHE / (
            f"train_ep{int(record['episode']):06d}_f{int(record['start']):06d}.pt"
        )
        if not source.exists():
            continue
        sample = torch.load(source, map_location="cpu", weights_only=False)
        action = sample["robot_trajectory"].squeeze(0).float()
        if action.shape != (33, 33):
            raise ValueError(f"{source}: expected raw action [33,33], got {action.shape}")
        mean = torch.tensor(per_task[task_id]["mean"], dtype=torch.float32)
        std = torch.tensor(per_task[task_id]["std"], dtype=torch.float32)
        action = (action - mean) / std
        ordinal = counts.get(task_id, 0)
        target = OUTPUT / f"task{task_id}_{ordinal:04d}.safetensors"
        save_file(
            {
                "video_latents": sample["video_latent"].squeeze(0).contiguous(),
                "img_latent": sample["img_latent"].squeeze(0).contiguous(),
                "robot_trajectory": action.contiguous(),
            },
            target,
        )
        index.append(
            {
                "task_id": task_id,
                "episode": int(record["episode"]),
                "start_frame": int(record["start"]),
                "condition_mode": "native_trajectory",
                "trajectory_schema": "d_action_core_raw33_zscore_v2",
                "video_latent_path": str(target),
            }
        )
        counts[task_id] = ordinal + 1
        if len(counts) == 2 and all(v >= PER_TASK for v in counts.values()):
            break

    if len(counts) != 2 or any(v != PER_TASK for v in counts.values()):
        raise RuntimeError(f"could not build balanced smoke set: {counts}")
    index_path = OUTPUT / "train.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(json.dumps({"index": str(index_path), "counts": counts, "samples": len(index)}))


if __name__ == "__main__":
    main()
