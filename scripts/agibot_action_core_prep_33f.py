"""Gate D run_001, Phase 5b: fixed-33-frame action-core prep for Pipeline B.

Per explicit user instruction, this REPLACES the adaptive 1-3s windowing
scheme (`agibot_action_core_prep.py`'s `adaptive_segments`, still used by the
existing `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_v1` output) with a fixed WINDOW=33-frame,
STRIDE=33 (non-overlapping) scheme for a NEW output tree,
`/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_33f_v1`. The existing variable-length dataset is
left untouched -- this script never writes into `agibot_action_core_v1`.

Everything else is reused, not re-derived, from `agibot_action_core_prep.py`:
- `convert_core_33d_to_37d` / `quaternion_xyzw_to_rotation6d` (rot6D core)
- `compute_core37_stats` (per-task vs merged stats decision on the 37D core)
- `decode_window_frames` / `shard_video_path` (decord-based video decode)
- `normalize_window`, `sha256_file`

Split: identical to Phase 3's `build_combined_split.json`, unchanged
(episode-level train/held_out; no re-derivation).

Video encoding: `--encode` decodes the same `[start, start+33)` frame range
used for the trajectory window (so video and 37D action share one index --
no separate alignment step) and calls the validated Wan VAE encode path.
Resumable: each output `.safetensors` is checked for its expected keys
before (re)computing, and written via a temp-file + atomic rename so a
killed/interrupted run never leaves a half-written file that looks done.

Usage:
    # trajectory-only (CPU):
    python3 scripts/agibot_action_core_prep_33f.py \
        --tasks 3400,3401 \
        --extracted_root /mnt/workspace/umi-world-model-lab/datasets/agibot_extracted \
        --split reports/direct_action/gate_d/run_001/artifacts/build_combined_split.json \
        --output /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_33f_v1

    # + video VAE encode (GPU), smoke test on 12 windows first:
    python3 scripts/agibot_action_core_prep_33f.py ... --encode --limit 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from reports.direct_action.gate_b.run_001.artifacts.action_core_schema import slice_core_33d  # noqa: E402
from reports.direct_action.gate_d.run_001.artifacts.compare_action_distributions import (  # noqa: E402
    load_episode_parquet,
)
from scripts.agibot_action_core_prep import (  # noqa: E402
    compute_core37_stats,
    convert_core_33d_to_37d,
    decode_window_frames,
    normalize_window,
    shard_video_path,
)

WINDOW = 33
STRIDE = 33


def fixed_windows(n_frames: int) -> list[dict]:
    """Non-overlapping WINDOW=33 chunks, STRIDE=33. Trailing frames that
    don't fill a full window are dropped (not padded), per instruction."""
    starts = range(0, n_frames - WINDOW + 1, STRIDE)
    return [{"start": s, "length": WINDOW} for s in starts]


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
                    for w in fixed_windows(core37.shape[0]):
                        start = w["start"]
                        windows.append({
                            "task_id": task_id,
                            "shard_name": shard_name,
                            "episode_index": ep,
                            "split": split_name,
                            "start_frame": start,
                            "num_frames": WINDOW,
                            "raw_window": core37[start:start + WINDOW],
                        })
    return windows


def out_path_for(output_root: Path, w: dict) -> Path:
    episode_root = output_root / w["split"] / f"task_{w['task_id']}" / w["shard_name"]
    return episode_root / f"episode_{w['episode_index']:06d}_start_{w['start_frame']:06d}.safetensors"


def existing_file_has_keys(path: Path, required_keys: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            return required_keys.issubset(set(f.keys()))
    except Exception:
        return False


def atomic_save_file(tensors: dict, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    save_file(tensors, tmp_path)
    os.replace(tmp_path, out_path)


def load_vae(device: torch.device, dtype: torch.dtype):
    """Verbatim path already validated in inference_user.py:236 (canonical,
    already imported elsewhere in this repo for encode_video_to_latent /
    encode_image_to_latent) -- same checkpoint, same .to(device, dtype)."""
    import os as _os

    from diffusers import AutoencoderKLWan

    model_path = _os.environ.get("MODEL_PATH", "pretrained/Wan2.2-TI2V-5B-Diffusers")
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae").to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def write_windows(
    windows: list[dict],
    stats: dict,
    extracted_root: Path,
    output_root: Path,
    camera_key: str,
    text_embedding_path: str | None,
    encode: bool,
    vae,
) -> dict:
    if encode:
        from inference_user import encode_image_to_latent, encode_video_to_latent

    info_cache: dict[tuple[str, str], dict] = {}
    index_by_split: dict[str, list[dict]] = {"train": [], "held_out": []}
    task_window_count: dict[str, int] = {}
    required_keys = {"robot_trajectory", "video_latents", "img_latent"} if encode else {"robot_trajectory"}

    for number, w in enumerate(windows):
        task_id, shard_name, ep = w["task_id"], w["shard_name"], w["episode_index"]
        task_window_count[task_id] = task_window_count.get(task_id, 0) + 1

        out_path = out_path_for(output_root, w)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "task_id": task_id,
            "shard_name": shard_name,
            "episode_index": ep,
            "start_frame": w["start_frame"],
            "num_frames": WINDOW,
            "trajectory_schema": "d_action_core_rot6d_v1",
            "trajectory_stats_policy": stats["policy"],
            "video_latent_path": str(out_path),
        }
        if text_embedding_path is not None:
            meta["text_embedding_path"] = text_embedding_path
        index_by_split[w["split"]].append(meta)

        if existing_file_has_keys(out_path, required_keys):
            if (number + 1) % 200 == 0 or number + 1 == len(windows):
                print(f"[{number + 1}/{len(windows)}] skip (already done) {out_path}", flush=True)
            continue

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
            source_indices = list(range(w["start_frame"], w["start_frame"] + WINDOW))
            frames = decode_window_frames(video_path, source_indices, height=480, width=832)
            tensors.update({
                "video_latents": encode_video_to_latent(vae, frames, torch.device("cuda"), torch.bfloat16).contiguous(),
                "img_latent": encode_image_to_latent(vae, frames[0], torch.device("cuda"), torch.bfloat16).contiguous(),
            })

        atomic_save_file(tensors, out_path)

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
    parser.add_argument("--encode", action="store_true", help="decode video + Wan VAE encode (GPU)")
    parser.add_argument("--limit", type=int, default=None, help="cap total windows, for smoke-testing")
    args = parser.parse_args()

    if args.output.resolve() == (Path('/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop') / "agibot_action_core_v1").resolve():
        raise SystemExit(
            "Refusing to write into /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_v1 (existing adaptive-window "
            "dataset) -- pass a different --output, e.g. /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_33f_v1."
        )

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

    print("Building fixed-33-frame windows across train + held_out episodes...")
    windows = build_task_windows(args.extracted_root, shards_by_task)
    if args.limit is not None:
        windows = windows[: args.limit]
    print(f"{len(windows)} windows total")

    vae = None
    if args.encode:
        print("Loading Wan VAE (cuda, bfloat16)...")
        vae = load_vae(torch.device("cuda"), torch.bfloat16)

    result = write_windows(
        windows, stats, args.extracted_root, args.output, args.camera_key,
        args.text_embedding_path, args.encode, vae,
    )
    manifest = {
        "windowing": "fixed_stride",
        "window": WINDOW,
        "stride": STRIDE,
        "core_dim": 37,
        "stats_policy": stats["policy"],
        "task_window_count": result["task_window_count"],
        "index_counts": result["index_by_split"],
        "encoded": args.encode,
    }
    (args.output / "window_manifest_combined.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: {result['index_by_split']}")
    print(f"Per-task window counts (for balanced sampling): {result['task_window_count']}")


if __name__ == "__main__":
    main()
