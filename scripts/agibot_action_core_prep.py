"""Gate D run_001, Phase 5: multi-task action-core prep for Pipeline B.

Builds `.safetensors` windows + train/heldout index files consumable by
`core/finetune/datasets/wan_dataset.py`'s `EgoVerseDataset22` in
`condition_mode == "native_trajectory"` with ZERO changes to that dataset
class -- it auto-detects native_trajectory mode from each index entry's
`trajectory_schema` field.

Deliberately NOT a reuse of `scripts/agibot_native_prep.py`: that script
sources trajectories from `state/*` HDF5 fields for task 362. Gate A
validated `action/*` (not `state/*`) as the causal, non-passthrough command
signal, so this script sources the 33D D-action-core from the `action`
parquet column via `action_core_schema.slice_core_33d` (imported, never
re-derived), for tasks 3400 and 3401 across all Gate D Phase 1-3 extracted
shards.

Orientation handling (Gate D Phase 4 architecture note): `action/end/orientation`
is 8D = two 4D quaternions (bimanual). Each 4D half is normalized and converted
to rotation-6D via `quaternion_xyzw_to_rotation6d`, copied verbatim from
`scripts/agibot_native_prep.py:32-51` per the approved plan ("reusing ... as a
pure function, applied per-hand"). This changes the core trajectory dimension
33 -> 37 (8D quaternion replaced by 12D rotation-6D).

Normalization: Gate D Phase 4's `action_stats_combined.json` was computed on
the RAW 33D quaternion core, since rotation-6D didn't exist yet at that point
in the pipeline. Z-scoring raw quaternion components is not the same
operation as z-scoring rotation-6D components, so this script recomputes
per-task/merged stats on the POST-conversion 37D representation (reusing
Phase 4's own `decide_policy`/`per_dim_stats` helpers, not re-deriving the
decision rule) before normalizing. Position/joint/waist dims are unchanged by
the conversion, so their stats are numerically consistent with Phase 4's;
only orientation's 8D->12D reshape necessitates fresh numbers.

Video encoding (--encode) is GPU-gated per the plan's Phase 7 and is included
here only for structural completeness -- it is not invoked by this session's
CPU-only run. Without --encode, each window's `.safetensors` file contains
only `robot_trajectory` (not yet loadable by `EgoVerseDataset22`, which also
needs `video_latents`/`img_latent` in the same file); the manifest and
normalized-trajectory validation this produces is still real, auditable work.

Usage:
    python3 scripts/agibot_action_core_prep.py \
        --tasks 3400,3401 \
        --extracted_root /mnt/workspace/umi-world-model-lab/datasets/agibot_extracted \
        --split reports/direct_action/gate_d/run_001/artifacts/build_combined_split.json \
        --stats reports/direct_action/gate_d/run_001/artifacts/action_stats_combined.json \
        --output /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_v1 \
        --camera_key observation.images.top_head \
        [--encode] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from reports.direct_action.gate_b.run_001.artifacts.action_core_schema import (  # noqa: E402
    field_boundaries,
    slice_core_33d,
)
from reports.direct_action.gate_d.run_001.artifacts.compare_action_distributions import (  # noqa: E402
    decide_policy,
    load_episode_parquet,
    per_dim_stats,
)

FPS = 30
MIN_WINDOW_SEC = 1.0
MAX_WINDOW_SEC = 3.0
TARGET_WINDOW_SEC = 2.0
MIN_FRAMES = round(MIN_WINDOW_SEC * FPS)
MAX_FRAMES = round(MAX_WINDOW_SEC * FPS)
TARGET_FRAMES = round(TARGET_WINDOW_SEC * FPS)
CORE37_FIELD_ORDER = [
    "action/end/position",
    "action/end/orientation/left/rot6d",
    "action/end/orientation/right/rot6d",
    "action/joint/position",
    "action/waist/position",
]


def quaternion_xyzw_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    """Verbatim copy of scripts/agibot_native_prep.py:32-51 (xyzw convention),
    per the approved plan's directive to reuse it as a pure function, applied
    per-hand to each 4D half of the 8D action/end/orientation field."""
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


def convert_core_33d_to_37d(core33: np.ndarray) -> np.ndarray:
    boundaries = field_boundaries()
    pos_s, pos_e = boundaries["action/end/position"]
    ori_s, ori_e = boundaries["action/end/orientation"]
    joint_s, joint_e = boundaries["action/joint/position"]
    waist_s, waist_e = boundaries["action/waist/position"]
    position = core33[..., pos_s:pos_e]
    orientation = core33[..., ori_s:ori_e]
    joint = core33[..., joint_s:joint_e]
    waist = core33[..., waist_s:waist_e]
    left_quat, right_quat = orientation[..., :4], orientation[..., 4:]
    left_rot6d = quaternion_xyzw_to_rotation6d(left_quat)
    right_rot6d = quaternion_xyzw_to_rotation6d(right_quat)
    return np.concatenate([position, left_rot6d, right_rot6d, joint, waist], axis=-1)


def core37_field_boundaries() -> dict:
    dims = [6, 6, 6, 14, 5]
    bounds, cursor = {}, 0
    for name, dim in zip(CORE37_FIELD_ORDER, dims):
        bounds[name] = (cursor, cursor + dim)
        cursor += dim
    assert cursor == 37
    return bounds


def adaptive_segments(n_frames: int) -> list[dict]:
    """Adaptively partition an episode into contiguous, non-overlapping
    windows whose length is ideally in [MIN_WINDOW_SEC, MAX_WINDOW_SEC]
    (1-3s at FPS=30), covering the whole episode -- no per-episode window
    cap. Picks the smallest segment count K such that ceil(n_frames/K) does
    not exceed MAX_FRAMES, then splits n_frames into K near-equal pieces
    (remainder frames distributed one-by-one to the earliest segments so
    every piece differs by at most 1 frame). Episodes shorter than
    MIN_FRAMES (<1s) cannot form a valid window and are skipped."""
    if n_frames < MIN_FRAMES:
        return []
    k = max(1, -(-n_frames // MAX_FRAMES))  # ceil(n_frames / MAX_FRAMES)
    while n_frames // k > MAX_FRAMES:
        k += 1
    while k > 1 and n_frames // k < MIN_FRAMES:
        k -= 1
    base, remainder = divmod(n_frames, k)
    result = []
    cursor = 0
    for i in range(k):
        length = base + (1 if i < remainder else 0)
        result.append({"start": cursor, "length": length})
        cursor += length
    return result


def compute_core37_stats(extracted_root: Path, shards_by_task: dict) -> dict:
    """Recompute Phase 4's decision-rule stats on the POST-conversion 37D
    representation (see module docstring). Reuses decide_policy/per_dim_stats
    from Phase 4's script, not re-derived."""
    task_ids = sorted(shards_by_task)
    per_task_values, per_task_stats = {}, {}
    for task_id in task_ids:
        rows = []
        for shard in shards_by_task[task_id]:
            for ep in shard["train_episodes"]:
                df = load_episode_parquet(extracted_root, task_id, shard["shard_name"], ep)
                action = np.stack(df["action"].to_numpy())
                rows.append(convert_core_33d_to_37d(slice_core_33d(action)))
        values = np.concatenate(rows, axis=0)
        per_task_values[task_id] = values
        per_task_stats[task_id] = per_dim_stats(values)
    comparison = decide_policy(
        per_task_stats[task_ids[0]], per_task_stats[task_ids[1]], boundaries=core37_field_boundaries()
    )
    merged_values = np.concatenate([per_task_values[t] for t in task_ids], axis=0)
    merged_stats = per_dim_stats(merged_values)
    return {
        "policy": comparison["policy"],
        "diverging_dims": comparison["diverging_dims"],
        "per_dim_comparison": comparison["per_dim"],
        "merged": merged_stats,
        "per_task": per_task_stats,
    }


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode_window_frames(video_path: Path, source_indices: list[int], height: int, width: int) -> np.ndarray:
    """imageio/pyav-based decode, matching run_011's validated convention
    (train_5000_corrected.py:123-132) -- decord's bundled ffmpeg cannot decode
    AgiBot's AV1-encoded mp4s (confirmed: `decord.VideoReader(...)` raises
    "cannot find video stream with wanted index" on a real extracted shard
    video, while `imageio.v3.imiter(..., plugin="pyav")` decodes it fine).
    Resizes post-decode via bilinear interpolation (gate_c_visual_v2.py's
    preprocess_video_frames convention) since pyav, unlike decord, has no
    built-in resize-during-decode."""
    import imageio.v3 as iio
    import torch.nn.functional as tf

    start, stop = source_indices[0], source_indices[-1] + 1
    assert list(source_indices) == list(range(start, stop)), "expects contiguous indices"
    frames = []
    for index, frame in enumerate(iio.imiter(str(video_path), plugin="pyav")):
        if index >= stop:
            break
        if index >= start:
            frames.append(frame)
    if len(frames) != len(source_indices):
        raise ValueError(f"{video_path}: decoded {len(frames)} frames, expected {len(source_indices)} at start={start}")
    array = np.stack(frames)  # [F, H, W, C] uint8
    if array.shape[1] != height or array.shape[2] != width:
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
        tensor = tf.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        array = tensor.permute(0, 2, 3, 1).round().clamp(0, 255).byte().numpy()
    return array


def shard_video_path(info: dict, shard_dir: Path, camera_key: str, episode_index: int) -> Path:
    chunks_size = info.get("chunks_size", 1000)
    chunk = episode_index // chunks_size
    rel_path = info["video_path"].format(episode_chunk=chunk, video_key=camera_key, episode_index=episode_index)
    return shard_dir / "data" / rel_path
def build_task_windows(extracted_root: Path, shards_by_task: dict) -> list[dict]:
    windows = []
    for task_id, shards in shards_by_task.items():
        for shard in shards:
            shard_name = shard["shard_name"]
            episodes_by_split = {"train": shard["train_episodes"], "held_out": shard["held_out_episodes"]}
            for split_name, episodes in episodes_by_split.items():
                for ep in episodes:
                    df = load_episode_parquet(extracted_root, task_id, shard_name, ep)
                    action = np.stack(df["action"].to_numpy())
                    core37 = convert_core_33d_to_37d(slice_core_33d(action))
                    for w in adaptive_segments(core37.shape[0]):
                        start, length = w["start"], w["length"]
                        windows.append({
                            "task_id": task_id,
                            "shard_name": shard_name,
                            "episode_index": ep,
                            "split": split_name,
                            "start_frame": start,
                            "num_frames": length,
                            "raw_window": core37[start:start + length],
                        })
    return windows


def normalize_window(raw_window: np.ndarray, task_id: str, stats: dict) -> np.ndarray:
    if stats["policy"] == "merged":
        mean, std = stats["merged"]["mean"], stats["merged"]["std"]
    else:
        mean, std = stats["per_task"][task_id]["mean"], stats["per_task"][task_id]["std"]
    return (raw_window - mean) / std
def write_windows(
    windows: list[dict],
    stats: dict,
    extracted_root: Path,
    output_root: Path,
    camera_key: str,
    text_embedding_path: str | None,
    encode: bool,
    vae=None,
) -> dict:
    if encode:
        from inference_user import encode_image_to_latent, encode_video_to_latent

    info_cache: dict[tuple[str, str], dict] = {}
    index_by_split: dict[str, list[dict]] = {"train": [], "held_out": []}
    task_window_count: dict[str, int] = {}

    for number, w in enumerate(windows):
        task_id, shard_name, ep = w["task_id"], w["shard_name"], w["episode_index"]
        task_window_count[task_id] = task_window_count.get(task_id, 0) + 1

        episode_root = output_root / w["split"] / f"task_{task_id}" / shard_name
        episode_root.mkdir(parents=True, exist_ok=True)
        out_path = episode_root / f"episode_{ep:06d}_start_{w['start_frame']:06d}_len_{w['num_frames']:06d}.safetensors"

        normalized = normalize_window(w["raw_window"], task_id, stats)
        tensors = {"robot_trajectory": torch.from_numpy(normalized).float()}

        if encode:
            cache_key = (task_id, shard_name)
            if cache_key not in info_cache:
                info_cache[cache_key] = json.loads(
                    (extracted_root / f"task_{task_id}" / shard_name / "data" / "meta" / "info.json")
                    .read_text(encoding="utf-8")
                )
            info = info_cache[cache_key]
            shard_dir = extracted_root / f"task_{task_id}" / shard_name
            video_path = shard_video_path(info, shard_dir, camera_key, ep)
            source_indices = list(range(w["start_frame"], w["start_frame"] + w["num_frames"]))
            frames = decode_window_frames(video_path, source_indices, height=480, width=832)
            tensors.update({
                "video_latents": encode_video_to_latent(vae, frames, torch.device("cuda"), torch.bfloat16).contiguous(),
                "img_latent": encode_image_to_latent(vae, frames[0], torch.device("cuda"), torch.bfloat16).contiguous(),
            })

        save_file(tensors, out_path)

        meta = {
            "task_id": task_id,
            "shard_name": shard_name,
            "episode_index": ep,
            "start_frame": w["start_frame"],
            "num_frames": w["num_frames"],
            "trajectory_schema": "d_action_core_rot6d_v1",
            "trajectory_stats_policy": stats["policy"],
            "video_latent_path": str(out_path),
        }
        if text_embedding_path is not None:
            meta["text_embedding_path"] = text_embedding_path
        index_by_split[w["split"]].append(meta)

        if (number + 1) % 200 == 0 or number + 1 == len(windows):
            print(f"[{number + 1}/{len(windows)}] wrote {out_path}", flush=True)

    for split_name, index in index_by_split.items():
        (output_root / f"{'train' if split_name == 'train' else 'heldout'}.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

    return {"task_window_count": task_window_count, "index_by_split": {k: len(v) for k, v in index_by_split.items()}}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="comma-separated task ids, e.g. 3400,3401")
    parser.add_argument("--extracted_root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path, help="Phase 3 build_combined_split.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera_key", default="observation.images.top_head")
    parser.add_argument("--text_embedding_path", default=None)
    parser.add_argument("--encode", action="store_true", help="GPU-gated; decode video + VAE encode (Phase 7)")
    parser.add_argument("--limit", type=int, default=None, help="cap total windows, for smoke-testing")
    args = parser.parse_args()

    task_ids = set(args.tasks.split(","))
    combined_split = json.loads(args.split.read_text(encoding="utf-8"))
    shards_by_task: dict[str, list[dict]] = {}
    for shard in combined_split["shards"]:
        if shard["task_id"] in task_ids:
            shards_by_task.setdefault(shard["task_id"], []).append(shard)
    assert set(shards_by_task) == task_ids, f"expected shards for {task_ids}, found {set(shards_by_task)}"

    print("Computing 37D core stats (post rotation-6D conversion) on train-split episodes...")
    stats = compute_core37_stats(args.extracted_root, shards_by_task)
    print(f"policy={stats['policy']} diverging_dims={len(stats['diverging_dims'])}/37")
    for d in stats["diverging_dims"]:
        print(f"  dim {d['dim']} ({d['field']}): mean_diff_norm={d['mean_diff_norm']:.3f} "
              f"std_ratio={d['std_ratio_3400_over_3401']:.3f}")

    args.output.mkdir(parents=True, exist_ok=True)
    stats_out = {
        "policy": stats["policy"],
        "diverging_dims": stats["diverging_dims"],
        "per_dim_comparison": stats["per_dim_comparison"],
        "merged": {"mean": stats["merged"]["mean"].tolist(), "std": stats["merged"]["std"].tolist(),
                   "n_rows": stats["merged"]["n_rows"]},
        "per_task": {
            t: {"mean": s["mean"].tolist(), "std": s["std"].tolist(), "n_rows": s["n_rows"]}
            for t, s in stats["per_task"].items()
        },
    }
    (args.output / "action_stats_core37.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    print("Building windows across train + held_out episodes...")
    windows = build_task_windows(args.extracted_root, shards_by_task)
    if args.limit is not None:
        windows = windows[: args.limit]
    print(f"{len(windows)} windows total")

    vae = None
    if args.encode:
        raise NotImplementedError(
            "--encode is GPU-gated per the plan's Phase 7 and is not wired up in this "
            "CPU-only session; run without --encode to produce the manifest and "
            "normalized robot_trajectory tensors only."
        )

    result = write_windows(
        windows, stats, args.extracted_root, args.output, args.camera_key,
        args.text_embedding_path, args.encode, vae,
    )
    manifest = {
        "windowing": "adaptive_segments",
        "fps": FPS,
        "min_window_sec": MIN_WINDOW_SEC,
        "max_window_sec": MAX_WINDOW_SEC,
        "target_window_sec": TARGET_WINDOW_SEC,
        "min_frames": MIN_FRAMES,
        "max_frames": MAX_FRAMES,
        "core_dim": 37,
        "stats_policy": stats["policy"],
        "task_window_count": result["task_window_count"],
        "index_counts": result["index_by_split"],
        "encoded": args.encode,
    }
    (args.output / "window_manifest_combined.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: {result['index_by_split']}")
    print(f"Per-task window counts (for WeightedRandomSampler weighting): {result['task_window_count']}")


if __name__ == "__main__":
    main()

