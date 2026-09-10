#!/usr/bin/env python3
"""Build the balanced V2 smoke set with post-conversion 37D rot6D actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
SOURCE_RUN = REPO / "reports/direct_action/gate_d/run_015"
SOURCE_CACHE = Path("/tmp/scratch/gate_d_run001_cache")
STATS_PATH = REPO / "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_33f_v1/action_stats_core37.json"
OUTPUT = REPO / "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_rot6d37_v2_smoke"
PER_TASK = 128


def quaternion_xyzw_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8, None
    )
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    rotation = np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
            2 * (x * z + y * w), 2 * (x * y + z * w),
            1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ), axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return rotation[..., :, :2].reshape(*quaternion.shape[:-1], 6)


def convert_core_33d_to_37d(core33: np.ndarray) -> np.ndarray:
    # Raw core layout: end position 6, two xyzw quaternions 8,
    # joint position 14, waist position 5.
    position, orientation = core33[..., :6], core33[..., 6:14]
    joint, waist = core33[..., 14:28], core33[..., 28:33]
    return np.concatenate(
        [position,
         quaternion_xyzw_to_rotation6d(orientation[..., :4]),
         quaternion_xyzw_to_rotation6d(orientation[..., 4:]),
         joint, waist], axis=-1,
    )


def normalize_window(raw_window: np.ndarray, task_id: str, stats: dict) -> np.ndarray:
    selected = stats["merged"] if stats["policy"] == "merged" else stats["per_task"][task_id]
    return (raw_window - np.asarray(selected["mean"])) / np.asarray(selected["std"])


def main() -> None:
    selected = json.loads(
        (SOURCE_RUN / "artifacts/selected_5000_scaled.json").read_text()
    )
    stats = json.loads(STATS_PATH.read_text())
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
        raw33 = sample["robot_trajectory"].squeeze(0).float()
        if raw33.shape != (33, 33):
            raise ValueError(f"{source}: expected raw action [33,33], got {raw33.shape}")
        raw37 = convert_core_33d_to_37d(raw33.numpy())
        action = torch.from_numpy(normalize_window(raw37, task_id, stats)).float()

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
                "trajectory_schema": "d_action_core_rot6d37_zscore_v2",
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
