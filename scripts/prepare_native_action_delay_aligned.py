#!/usr/bin/env python3
"""Build a delay-aligned rot6D37 cache from the existing smoke set.

Gate A measured command-to-state response delays on real AgiBot episodes:
end position +4 frames, end orientation +2, joints +2, and waist +7.  The
existing cache pairs action[t] with video[t].  This builder keeps each cached
video latent unchanged and replaces its trajectory with the earlier command
that best corresponds to each video frame.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from reports.direct_action.gate_b.run_001.artifacts.action_core_schema import slice_core_33d
from scripts.prepare_native_action_v2_rot6d37_smoke import convert_core_33d_to_37d


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_rot6d37_v2_smoke/train.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_rot6d37_delay_aligned_v3"
)
DEFAULT_SELECTION = REPO / "reports/direct_action/gate_d/run_015/artifacts/selected_5000_scaled.json"
DEFAULT_STATS = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_core_33f_v1/action_stats_core37.json"
)
DEFAULT_EXTRACTED_ROOT = Path("/mnt/workspace/umi-world-model-lab/datasets/agibot_extracted")

# Slices refer to the post-conversion rot6D37 layout.
FIELD_DELAYS = {
    "end_position": (slice(0, 6), 4),
    "end_orientation": (slice(6, 18), 2),
    "joint_position": (slice(18, 32), 2),
    "waist_position": (slice(32, 37), 7),
}


def load_cached(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def normalize(action: np.ndarray, task_id: str, stats: dict) -> np.ndarray:
    selected = stats["merged"] if stats["policy"] == "merged" else stats["per_task"][task_id]
    mean = np.asarray(selected["mean"], dtype=np.float32)
    std = np.asarray(selected["std"], dtype=np.float32)
    return (action - mean) / std


def aligned_window(raw37: np.ndarray, start: int, length: int) -> tuple[np.ndarray, dict[str, int]]:
    target = np.arange(start, start + length)
    aligned = np.empty((length, 37), dtype=np.float32)
    left_pad: dict[str, int] = {}
    for name, (field_slice, delay) in FIELD_DELAYS.items():
        source = target - delay
        left_pad[name] = int((source < 0).sum())
        source = np.clip(source, 0, raw37.shape[0] - 1)
        aligned[:, field_slice] = raw37[source, field_slice]
    return aligned, left_pad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED_ROOT)
    args = parser.parse_args()

    source_index = json.loads(args.source.read_text())
    selected = json.loads(args.selection.read_text())
    stats = json.loads(args.stats.read_text())
    lookup = {
        (str(row["task_id"]), int(row["episode"]), int(row["start"])): row
        for row in selected
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_index = []
    pad_totals = {name: 0 for name in FIELD_DELAYS}
    for ordinal, item in enumerate(source_index):
        key = (str(item["task_id"]), int(item["episode"]), int(item["start_frame"]))
        if key not in lookup:
            raise KeyError(f"source sample is absent from selected-window provenance: {key}")
        provenance = lookup[key]
        parquet = Path(provenance["parquet"])
        if not parquet.exists():
            legacy_root = Path("/mnt/workspace/agibot_extracted")
            parquet = args.extracted_root / parquet.relative_to(legacy_root)
        frame = pd.read_parquet(parquet, columns=["action"])
        raw = np.stack(frame["action"].to_numpy())
        raw37 = convert_core_33d_to_37d(slice_core_33d(raw)).astype(np.float32)

        cached = load_cached(Path(item["video_latent_path"]))
        length = int(cached["robot_trajectory"].shape[0])
        aligned, left_pad = aligned_window(raw37, key[2], length)
        for name, count in left_pad.items():
            pad_totals[name] += count
        cached["robot_trajectory"] = torch.from_numpy(
            normalize(aligned, key[0], stats)
        ).contiguous()

        target = args.output / f"task{key[0]}_{ordinal:04d}.safetensors"
        temporary = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
        save_file(cached, temporary)
        os.replace(temporary, target)
        output_index.append(
            {
                **item,
                "trajectory_schema": "d_action_core_rot6d37_zscore_delay_aligned_v3",
                "video_latent_path": str(target),
                "field_response_delay_frames": {
                    name: delay for name, (_, delay) in FIELD_DELAYS.items()
                },
                "source_parquet": str(parquet),
            }
        )

    (args.output / "train.json").write_text(json.dumps(output_index, indent=2) + "\n")
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "samples": len(output_index),
        "field_response_delay_frames": {
            name: delay for name, (_, delay) in FIELD_DELAYS.items()
        },
        "left_edge_padded_values": pad_totals,
    }
    (args.output / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
