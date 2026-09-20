#!/usr/bin/env python3
"""Build a leak-free observed-state + future-action dataset from V8 caches.

The split is grouped by global episode, so no episode can occur in both train
and development manifests.  Only the observed state at the window start is
read; future states are never stored or used as model inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.control.native_action_features import build_v10_features  # noqa: E402
from scripts.prepare_native_action_v10_spatial import state37  # noqa: E402

DEFAULT_SOURCE = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v8_factorized_v1/train.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v10_state_256_v1"
)
FORCED_DEV_EPISODES = {
    "3400": {203, 437},
    "3401": {848, 1044},
}


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def choose_dev_episodes(
    records: list[dict], count_per_task: int, seed: int
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for task_id in sorted({str(item["task_id"]) for item in records}):
        episodes = sorted(
            {int(item["episode"]) for item in records if str(item["task_id"]) == task_id}
        )
        forced = FORCED_DEV_EPISODES.get(task_id, set()) & set(episodes)
        if len(forced) > count_per_task:
            raise ValueError(f"forced dev episodes exceed count for task {task_id}")
        candidates = [episode for episode in episodes if episode not in forced]
        random.Random(seed + int(task_id)).shuffle(candidates)
        result[task_id] = forced | set(candidates[: count_per_task - len(forced)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-episodes-per-task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = json.loads(args.source.read_text())
    dev_episodes = choose_dev_episodes(
        records, args.dev_episodes_per_task, args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)

    field_cache: dict[Path, dict] = {}
    state_cache: dict[Path, np.ndarray] = {}
    output_records: list[dict] = []
    for index, item in enumerate(records):
        parquet = Path(item["source_parquet"])
        if parquet not in state_cache:
            frame = pd.read_parquet(parquet, columns=["observation.state"])
            state_cache[parquet] = np.stack(frame["observation.state"].to_numpy())
            info_path = parquet.parents[2] / "meta/info.json"
            info = json.loads(info_path.read_text())
            field_cache[parquet] = info["features"]["observation.state"][
                "field_descriptions"
            ]

        start = int(item["start_frame"])
        state = state_cache[parquet]
        if not 0 <= start < len(state):
            raise IndexError(f"start {start} outside state sequence {parquet}: {len(state)}")
        observed = torch.from_numpy(
            state37(state[start], field_cache[parquet])
        ).float()

        packed = load_tensors(Path(item["video_latent_path"]))
        raw = packed["robot_trajectory_raw37"].float()
        features = build_v10_features(
            raw,
            observed,
            packed["robot_trajectory_mean37"].float(),
            packed["robot_trajectory_std37"].float(),
            packed["robot_relative_scale37"].float(),
            packed["robot_velocity_scale37"].float(),
        )
        task_id = str(item["task_id"])
        target = args.output / f"task{task_id}_{index:04d}.safetensors"
        output_tensors = {
            "video_latents": packed["video_latents"].contiguous(),
            "img_latent": packed["img_latent"].contiguous(),
            "robot_trajectory": features.contiguous(),
            "robot_trajectory_raw37": raw.contiguous(),
            "robot_observed_state37": observed.contiguous(),
            "robot_trajectory_mean37": packed["robot_trajectory_mean37"].contiguous(),
            "robot_trajectory_std37": packed["robot_trajectory_std37"].contiguous(),
            "robot_relative_scale37": packed["robot_relative_scale37"].contiguous(),
            "robot_velocity_scale37": packed["robot_velocity_scale37"].contiguous(),
        }
        temporary = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
        save_file(output_tensors, temporary)
        os.replace(temporary, target)

        output_records.append(
            {
                **item,
                "trajectory_schema": "state_target_relative_velocity_rot6d148_v1",
                "video_latent_path": str(target),
            }
        )
        if (index + 1) % 32 == 0 or index + 1 == len(records):
            print(f"built {index + 1}/{len(records)}", flush=True)

    train = [
        item for item in output_records
        if int(item["episode"]) not in dev_episodes[str(item["task_id"])]
    ]
    dev = [
        item for item in output_records
        if int(item["episode"]) in dev_episodes[str(item["task_id"])]
    ]
    train_keys = {(str(item["task_id"]), int(item["episode"])) for item in train}
    dev_keys = {(str(item["task_id"]), int(item["episode"])) for item in dev}
    if train_keys & dev_keys:
        raise RuntimeError(f"episode leakage: {sorted(train_keys & dev_keys)}")

    for name, rows in (("all", output_records), ("train", train), ("dev", dev)):
        (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "samples": len(output_records),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "train_episodes": len(train_keys),
        "dev_episodes": len(dev_keys),
        "dev_episode_ids": {
            task: sorted(episodes) for task, episodes in dev_episodes.items()
        },
        "feature_dim": 148,
        "spatial_control": False,
        "future_state_leakage": False,
        "split_group": "task_id+global_episode",
        "seed": args.seed,
    }
    (args.output / "build_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
