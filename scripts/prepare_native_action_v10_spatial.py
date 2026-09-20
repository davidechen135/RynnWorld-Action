#!/usr/bin/env python3
"""Build a 16-window state+action dataset with leak-free fitted spatial control."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.control.native_action_features import (  # noqa: E402
    build_spatial_action_control,
    build_v10_features,
)
from scripts.agibot_skeleton_render import fit_camera, project  # noqa: E402

DEFAULT_SOURCE = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v8_factorized_v1/train.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v10_state_spatial_16_v1"
)

STATE_FIELDS = {
    "orientation": "state/end/arm_orientation",
    "position": "state/end/arm_position",
    "joint": "state/joint/position",
    "waist": "state/waist/position",
}


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def quaternion_xyzw_to_rot6d_rows(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8, None
    )
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ),
        axis=-1,
    )


def state37(state: np.ndarray, field_descriptions: dict) -> np.ndarray:
    def field(name: str) -> np.ndarray:
        return state[..., field_descriptions[name]["indices"]]

    orientation = field(STATE_FIELDS["orientation"])
    return np.concatenate(
        (
            field(STATE_FIELDS["position"]),
            quaternion_xyzw_to_rot6d_rows(orientation[..., :4]),
            quaternion_xyzw_to_rot6d_rows(orientation[..., 4:]),
            field(STATE_FIELDS["joint"]),
            field(STATE_FIELDS["waist"]),
        ),
        axis=-1,
    ).astype(np.float32)


def resolve_video(parquet: Path, info: dict, episode_index: int) -> Path:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    relative = info["video_path"].format(
        episode_chunk=chunk,
        video_key="observation.images.top_head",
        episode_index=episode_index,
    )
    return parquet.parents[2] / relative


def motion_score(tensors: dict[str, torch.Tensor]) -> float:
    video = tensors["video_latents"].float()
    video_motion = (video[:, 1:] - video[:, :-1]).square().mean().sqrt()
    raw = tensors["robot_trajectory_raw37"].float()
    endpoint_motion = (raw[-1, :6] - raw[0, :6]).square().mean().sqrt()
    return float(video_motion + 2.0 * endpoint_motion)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument("--min-prefix", type=int, default=300)
    parser.add_argument("--max-camera-error", type=float, default=18.0)
    parser.add_argument("--min-camera-r2", type=float, default=0.2)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    candidates: dict[str, list[tuple[float, dict]]] = {}
    for item in source:
        if int(item["start_frame"]) < args.min_prefix:
            continue
        tensors = load_tensors(Path(item["video_latent_path"]))
        candidates.setdefault(str(item["task_id"]), []).append(
            (motion_score(tensors), item)
        )
    for values in candidates.values():
        values.sort(key=lambda pair: pair[0], reverse=True)

    args.output.mkdir(parents=True, exist_ok=True)
    selected: list[dict] = []
    seen_episodes: set[tuple[str, int]] = set()
    for task_id in sorted(candidates):
        accepted = 0
        for score, item in candidates[task_id]:
            episode_key = (task_id, int(item["episode"]))
            if episode_key in seen_episodes:
                continue
            parquet = Path(item["source_parquet"])
            frame = pd.read_parquet(parquet, columns=["observation.state"])
            state = np.stack(frame["observation.state"].to_numpy())
            info = json.loads((parquet.parents[2] / "meta/info.json").read_text())
            local_episode = int(parquet.stem.removeprefix("episode_"))
            # Only a static camera map is fitted, and only frames strictly before
            # the target window are used. Future state/video never enters features.
            video = resolve_video(parquet, info, local_episode)
            if not video.exists():
                continue
            fields = info["features"]["observation.state"]["field_descriptions"]
            start = int(item["start_frame"])
            positions = state[:, fields[STATE_FIELDS["position"]]["indices"]].reshape(-1, 2, 3)
            try:
                coeffs, camera_report = fit_camera(
                    str(video), positions, max_frames=start
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            if max(report["median_err_px"] for report in camera_report) > args.max_camera_error:
                continue
            if min(
                (report["r2_u"] + report["r2_v"]) / 2.0
                for report in camera_report
            ) < args.min_camera_r2:
                continue

            packed = load_tensors(Path(item["video_latent_path"]))
            raw = packed["robot_trajectory_raw37"].float()
            observed = torch.from_numpy(state37(state[start], fields)).float()
            mean = packed["robot_trajectory_mean37"].float()
            std = packed["robot_trajectory_std37"].float()
            relative_scale = packed["robot_relative_scale37"].float()
            velocity_scale = packed["robot_velocity_scale37"].float()
            features = build_v10_features(
                raw, observed, mean, std, relative_scale, velocity_scale
            )

            uv = torch.from_numpy(
                project(coeffs, raw[:, :6].numpy().reshape(-1, 2, 3), out_w=26, out_h=15)
            ).float()
            reference_uv = torch.from_numpy(
                project(coeffs, observed[:6].numpy().reshape(1, 2, 3), out_w=26, out_h=15)[0]
            ).float()
            spatial = build_spatial_action_control(uv, reference_uv)
            in_bounds = (
                (uv[..., 0] >= 0) & (uv[..., 0] < 26)
                & (uv[..., 1] >= 0) & (uv[..., 1] < 15)
            ).float().mean().item()
            if in_bounds < 0.75:
                continue

            ordinal = len(selected)
            target = args.output / f"task{task_id}_{ordinal:02d}.safetensors"
            output_tensors = {
                "video_latents": packed["video_latents"].contiguous(),
                "img_latent": packed["img_latent"].contiguous(),
                "robot_trajectory": features.contiguous(),
                "robot_spatial_control": spatial.contiguous(),
                "robot_trajectory_raw37": raw.contiguous(),
                "robot_observed_state37": observed.contiguous(),
                "robot_trajectory_mean37": mean.contiguous(),
                "robot_trajectory_std37": std.contiguous(),
                "robot_relative_scale37": relative_scale.contiguous(),
                "robot_velocity_scale37": velocity_scale.contiguous(),
                "robot_trajectory_uv": uv.contiguous(),
                "robot_state_uv": reference_uv.contiguous(),
                "camera_coefficients": torch.from_numpy(coeffs).float().contiguous(),
            }
            temporary = target.with_suffix(target.suffix + f".tmp{os.getpid()}")
            save_file(output_tensors, temporary)
            os.replace(temporary, target)

            record = {
                **item,
                "trajectory_schema": "state_target_relative_velocity_rot6d148_spatial6_v1",
                "video_latent_path": str(target),
                "source_video": str(video),
                "motion_score": score,
                "camera_fit_prefix_frames": start,
                "camera_fit": camera_report,
                "trajectory_uv_in_bounds": in_bounds,
            }
            selected.append(record)
            seen_episodes.add(episode_key)
            accepted += 1
            print(
                f"accepted task={task_id} episode={item['episode']} start={start} "
                f"score={score:.4f} in_bounds={in_bounds:.3f} camera={camera_report}",
                flush=True,
            )
            if accepted == args.per_task:
                break
        if accepted != args.per_task:
            raise RuntimeError(f"only accepted {accepted}/{args.per_task} samples for task {task_id}")

    by_task: dict[str, list[dict]] = {}
    for item in selected:
        by_task.setdefault(str(item["task_id"]), []).append(item)
    train, dev = [], []
    for items in by_task.values():
        train.extend(item for i, item in enumerate(items) if i not in (3, 7))
        dev.extend(item for i, item in enumerate(items) if i in (3, 7))
    for name, rows in (("all", selected), ("train", train), ("dev", dev)):
        (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "samples": len(selected),
        "train_samples": len(train),
        "dev_samples": len(dev),
        "tasks": {task: len(items) for task, items in by_task.items()},
        "feature_dim": 148,
        "spatial_shape": [6, 9, 15, 26],
        "future_state_leakage": False,
    }
    (args.output / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
