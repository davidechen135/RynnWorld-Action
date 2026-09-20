#!/usr/bin/env python3
"""Build factorized V8 features from the delay-aligned rot6D37 cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.control.native_action_features import build_v8_features, relative_and_velocity


DEFAULT_SOURCE = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_rot6d37_delay_aligned_v3/train.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v8_factorized_v1"
)
DEFAULT_STATS = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_core_33f_v1/action_stats_core37.json"
)


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def task_stats(stats: dict, task_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    selected = stats["merged"] if stats["policy"] == "merged" else stats["per_task"][task_id]
    return torch.tensor(selected["mean"]), torch.tensor(selected["std"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()

    manifest = json.loads(args.source.read_text())
    stats = json.loads(args.stats.read_text())
    if not manifest:
        raise ValueError(f"empty source manifest: {args.source}")

    cached: list[tuple[dict, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]] = []
    accumulators = defaultdict(
        lambda: {
            "relative_sum_sq": torch.zeros(37, dtype=torch.float64),
            "velocity_sum_sq": torch.zeros(37, dtype=torch.float64),
            "count": 0,
        }
    )
    for item in manifest:
        task_id = str(item["task_id"])
        tensors = load_tensors(Path(item["video_latent_path"]))
        target = tensors["robot_trajectory"].float()
        mean, std = task_stats(stats, task_id)
        raw = target * std + mean
        relative, velocity = relative_and_velocity(raw)
        acc = accumulators[task_id]
        acc["relative_sum_sq"] += relative.double().square().sum(dim=0)
        acc["velocity_sum_sq"] += velocity.double().square().sum(dim=0)
        acc["count"] += raw.shape[0]
        cached.append((item, tensors, raw, mean, std))

    scales = {}
    for task_id, acc in accumulators.items():
        count = max(acc["count"], 1)
        scales[task_id] = {
            "relative": (acc["relative_sum_sq"] / count).sqrt().clamp_min(1e-4).float(),
            "velocity": (acc["velocity_sum_sq"] / count).sqrt().clamp_min(1e-4).float(),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    output_manifest = []
    for ordinal, (item, tensors, raw, mean, std) in enumerate(cached):
        task_id = str(item["task_id"])
        relative_scale = scales[task_id]["relative"]
        velocity_scale = scales[task_id]["velocity"]
        tensors["robot_trajectory"] = build_v8_features(
            raw, mean, std, relative_scale, velocity_scale
        ).contiguous()
        # These tensors let evaluation rebuild physically consistent held,
        # reversed, and shifted counterfactuals.
        tensors["robot_trajectory_raw37"] = raw.contiguous()
        tensors["robot_trajectory_mean37"] = mean.contiguous()
        tensors["robot_trajectory_std37"] = std.contiguous()
        tensors["robot_relative_scale37"] = relative_scale.contiguous()
        tensors["robot_velocity_scale37"] = velocity_scale.contiguous()

        target = args.output / f"task{task_id}_{ordinal:04d}.safetensors"
        temporary = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
        save_file(tensors, temporary)
        os.replace(temporary, target)
        output_manifest.append(
            {
                **item,
                "trajectory_schema": "target_relative_velocity_rot6d111_v1",
                "video_latent_path": str(target),
            }
        )

    (args.output / "train.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    serializable_scales = {
        task_id: {name: value.tolist() for name, value in task_scale.items()}
        for task_id, task_scale in scales.items()
    }
    (args.output / "feature_scales.json").write_text(
        json.dumps(serializable_scales, indent=2) + "\n"
    )
    (args.output / "build_summary.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "samples": len(output_manifest),
                "tasks": sorted(scales),
                "feature_dim": 111,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
