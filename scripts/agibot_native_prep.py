"""Build timestamp-aligned AgiBot clips with native 24D robot trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import numpy as np
import torch
from safetensors.torch import save_file


SOURCE_FPS = 30.0
TARGET_FPS = 16.0
TARGET_DURATION_NS = 5_000_000_000
TARGET_FRAMES = 81
WINDOW_STRIDE = 75
HEIGHT, WIDTH = 480, 832
TASK_ID = 362
TASK_TEXT_EMBEDDING = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings/agibot_task362.safetensors"
EPISODE_ROOT = Path("/root/autodl-tmp/agibot_362_data")
SELECTION = Path("/root/autodl-tmp/agibot_362_meta/selection_362.json")


def quaternion_xyzw_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.clip(
        np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8, None
    )
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    rotation = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return rotation[..., :, :2].reshape(*quaternion.shape[:-1], 6)


def load_episode(episode_id: str) -> dict:
    h5_path = (
        EPISODE_ROOT
        / "proprio_stats"
        / str(TASK_ID)
        / episode_id
        / "proprio_stats.h5"
    )
    video_path = (
        EPISODE_ROOT
        / "observations"
        / str(TASK_ID)
        / episode_id
        / "videos"
        / "head_color.mp4"
    )
    with h5py.File(h5_path, "r") as handle:
        position = handle["state/end/position"][:].astype(np.float32)
        orientation = handle["state/end/orientation"][:].astype(np.float32)
        rotation6d = quaternion_xyzw_to_rotation6d(orientation)
        gripper = handle["state/effector/position"][:].astype(np.float32)
        head = handle["state/head/position"][:].astype(np.float32)
        waist = handle["state/waist/position"][:].astype(np.float32)
        timestamp = handle["timestamp"][:]
        valid = handle["action/end/index"][:].astype(np.int64)
    trajectory = np.concatenate(
        (
            position.reshape(len(position), 6),
            rotation6d.reshape(len(rotation6d), 12),
            gripper,
            head,
            waist,
        ),
        axis=-1,
    )
    if trajectory.shape[1] != 24 or not np.isfinite(trajectory).all():
        raise ValueError(f"invalid trajectory for episode {episode_id}: {trajectory.shape}")
    return {
        "h5_path": h5_path,
        "video_path": video_path,
        "trajectory": trajectory,
        "timestamp": timestamp,
        "valid": valid,
    }


def contiguous_segments(indices: np.ndarray) -> list[np.ndarray]:
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def windows_for_episode(episode_id: str) -> list[dict]:
    episode = load_episode(episode_id)
    rows = []
    for segment_id, segment in enumerate(contiguous_segments(episode["valid"])):
        if len(segment) < int(TARGET_DURATION_NS / 1e9 * SOURCE_FPS):
            continue
        for start in range(int(segment[0]), int(segment[-1]) + 1, WINDOW_STRIDE):
            target_timestamps = (
                episode["timestamp"][start]
                + np.linspace(0, TARGET_DURATION_NS, TARGET_FRAMES)
            )
            source_indices = np.searchsorted(episode["timestamp"], target_timestamps)
            source_indices = np.clip(source_indices, 1, len(episode["timestamp"]) - 1)
            left = source_indices - 1
            choose_left = (
                np.abs(episode["timestamp"][left] - target_timestamps)
                <= np.abs(episode["timestamp"][source_indices] - target_timestamps)
            )
            source_indices = np.where(choose_left, left, source_indices).astype(np.int64)
            if not np.all(np.isin(source_indices, segment)):
                continue
            timestamps = episode["timestamp"][source_indices]
            max_timestamp_error = np.max(np.abs(timestamps - target_timestamps))
            if max_timestamp_error > 35_000_000:
                continue
            rows.append(
                {
                    "episode_id": episode_id,
                    "segment_id": segment_id,
                    "source_start": int(start),
                    "source_end": int(source_indices[-1]),
                    "source_indices": source_indices.tolist(),
                    "timestamps_ns": timestamps.tolist(),
                    "duration_seconds": float((timestamps[-1] - timestamps[0]) / 1e9),
                    "max_timestamp_error_ms": float(max_timestamp_error / 1e6),
                    "raw_trajectory": episode["trajectory"][source_indices],
                    "video_path": str(episode["video_path"]),
                }
            )
    return rows


def decode_window(video_path: str, source_indices: list[int]) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    start = source_indices[0]
    end = source_indices[-1]
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='between(n,{start},{end})'",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, timeout=180)
    frame_bytes = 640 * 480 * 3
    count = len(result.stdout) // frame_bytes
    expected_count = end - start + 1
    if result.returncode or count != expected_count:
        raise RuntimeError(
            f"decode failed {video_path} [{start},{end}]: "
            f"return={result.returncode} frames={count} stderr={result.stderr[-300:]}"
        )
    frames = np.frombuffer(result.stdout, np.uint8).reshape(count, 480, 640, 3)
    sampled = frames[np.asarray(source_indices) - start]
    return np.stack([cv2.resize(frame, (WIDTH, HEIGHT)) for frame in sampled])


def compute_stats(rows: list[dict]) -> dict[str, np.ndarray]:
    values = np.concatenate([row["raw_trajectory"] for row in rows], axis=0)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    constant = std < 1e-6
    std[constant] = 1.0
    return {"mean": mean, "std": std, "constant_mask": constant}


def write_split(
    split: str,
    rows: list[dict],
    stats: dict[str, np.ndarray],
    output_root: Path,
    encode: bool,
    vae=None,
) -> list[dict]:
    if encode:
        from inference_user import encode_image_to_latent, encode_video_to_latent

    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    index = []
    for number, row in enumerate(rows):
        episode_root = split_root / row["episode_id"]
        episode_root.mkdir(parents=True, exist_ok=True)
        output_path = episode_root / f"clip_{row['source_start']:06d}.safetensors"
        trajectory = (row["raw_trajectory"] - stats["mean"]) / stats["std"]
        tensors = {"robot_trajectory": torch.from_numpy(trajectory).float()}
        if encode:
            frames = decode_window(row["video_path"], row["source_indices"])
            tensors.update(
                {
                    "video_latents": encode_video_to_latent(
                        vae, frames, torch.device("cuda"), torch.bfloat16
                    ).contiguous(),
                    "img_latent": encode_image_to_latent(
                        vae, frames[0], torch.device("cuda"), torch.bfloat16
                    ).contiguous(),
                }
            )
        save_file(tensors, output_path)
        meta = {
            key: value
            for key, value in row.items()
            if key not in {"raw_trajectory"}
        }
        meta.update(
            {
                "trajectory_schema": "ee_pose6d_gripper_head_waist_v1",
                "trajectory_stats": str(output_root / "stats.safetensors"),
                "target_fps": TARGET_FPS,
                "video_latent_path": str(output_path),
                "text_embedding_path": TASK_TEXT_EMBEDDING,
            }
        )
        index.append(meta)
        print(
            f"[{split} {number + 1:03d}/{len(rows):03d}] "
            f"ep={row['episode_id']} f={row['source_start']}-{row['source_end']} "
            f"duration={row['duration_seconds']:.3f}s encode={encode}",
            flush=True,
        )
    (output_root / f"{split}.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_native_v1"))
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    selection = json.loads(SELECTION.read_text())
    train_episodes = [str(value) for value in selection["train_episodes"]]
    dev_episodes = train_episodes[-2:]
    train_episodes = train_episodes[:-2]
    heldout_episodes = [str(value) for value in selection["heldout_episodes"]]
    split_episodes = {
        "train": train_episodes,
        "dev": dev_episodes,
        "heldout": heldout_episodes,
    }
    all_rows = {
        split: [
            row
            for episode_id in episodes
            for row in windows_for_episode(episode_id)
        ]
        for split, episodes in split_episodes.items()
    }
    if args.limit:
        all_rows = {key: value[: args.limit] for key, value in all_rows.items()}
    stats = compute_stats(all_rows["train"])
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: torch.from_numpy(value) for key, value in stats.items()},
        args.output / "stats.safetensors",
    )

    vae = None
    if args.encode:
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(
            "pretrained/Wan2.2-TI2V-5B-Diffusers", subfolder="vae"
        ).to("cuda", torch.bfloat16)
        vae.eval()

    audit = {
        "schema": "ee_pose6d_gripper_head_waist_v1",
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "target_frames": TARGET_FRAMES,
        "target_duration_seconds": TARGET_DURATION_NS / 1e9,
        "window_stride": WINDOW_STRIDE,
        "splits": {
            split: {
                "episodes": episodes,
                "windows": len(all_rows[split]),
                "duration_min": min(row["duration_seconds"] for row in all_rows[split]),
                "duration_max": max(row["duration_seconds"] for row in all_rows[split]),
                "timestamp_error_ms_max": max(
                    row["max_timestamp_error_ms"] for row in all_rows[split]
                ),
            }
            for split, episodes in split_episodes.items()
        },
        "constant_dimensions": np.flatnonzero(stats["constant_mask"]).tolist(),
        "encoded": args.encode,
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    for split, rows in all_rows.items():
        write_split(split, rows, stats, args.output, args.encode, vae)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
