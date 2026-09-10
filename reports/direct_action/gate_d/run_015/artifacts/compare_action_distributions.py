"""Gate D run_001, Phase 4: per-task vs merged normalization stats decision.

Computes per-dimension mean/std of the 33D D-action-core (imported from
`action_core_schema.slice_core_33d`, never re-derived) over each task's
train-split episodes (from Phase 3's `build_combined_split.json`), then
applies an explicit, auditable decision rule to choose between:

  - "merged": one global z-score, if ALL 33 dims satisfy BOTH
        |mean_3400 - mean_3401| / pooled_std < MEAN_DIFF_THRESHOLD
        std_3400 / std_3401 (or its reciprocal) in [STD_RATIO_LO, STD_RATIO_HI]
  - "per_task": otherwise, with the specific diverging dims flagged.

This mirrors run_009's `compute_action_stats` methodology (full per-episode
action stream, all train episodes) but generalized across tasks and shards,
and computed on the 33D core rather than run_009's differently-ordered
33D slice (run_009 predates action_core_schema.py and used its own
hand-rolled `extract_33d`; this script uses the canonical Gate B slicer).

Usage:
    python3 compare_action_distributions.py \
        --extracted_root /mnt/data/agibot_extracted \
        --split reports/direct_action/gate_d/run_001/artifacts/build_combined_split.json \
        --out reports/direct_action/gate_d/run_001/artifacts/action_stats_combined.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from reports.direct_action.gate_b.run_001.artifacts.action_core_schema import (  # noqa: E402
    CORE_SLICES,
    field_boundaries,
    slice_core_33d,
)

MEAN_DIFF_THRESHOLD = 0.5
STD_RATIO_LO = 0.5
STD_RATIO_HI = 2.0


def load_episode_parquet(extracted_root: Path, task_id: str, shard_name: str, episode_index: int) -> pd.DataFrame:
    shard_dir = extracted_root / f"task_{task_id}" / shard_name
    info = json.loads((shard_dir / "data" / "meta" / "info.json").read_text(encoding="utf-8"))
    chunks_size = info.get("chunks_size", 1000)
    chunk = episode_index // chunks_size
    rel_path = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    return pd.read_parquet(shard_dir / "data" / rel_path)


def collect_task_core_rows(extracted_root: Path, shard_entries: list[dict]) -> np.ndarray:
    rows = []
    for shard in shard_entries:
        task_id, shard_name = shard["task_id"], shard["shard_name"]
        for ep in shard["train_episodes"]:
            df = load_episode_parquet(extracted_root, task_id, shard_name, ep)
            action = np.stack(df["action"].to_numpy())
            rows.append(slice_core_33d(action))
    return np.concatenate(rows, axis=0)


def per_dim_stats(values: np.ndarray) -> dict:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean, "std": std_safe, "n_rows": int(values.shape[0])}


def decide_policy(stats_3400: dict, stats_3401: dict, boundaries: dict | None = None) -> dict:
    """boundaries defaults to the 33D D-action-core field_boundaries(); callers
    operating on a different per-dim layout (e.g. the 37D rotation-6D core in
    scripts/agibot_action_core_prep.py) pass their own boundaries dict."""
    mean_3400, std_3400 = stats_3400["mean"], stats_3400["std"]
    mean_3401, std_3401 = stats_3401["mean"], stats_3401["std"]
    pooled_std = np.sqrt((std_3400**2 + std_3401**2) / 2.0)
    mean_diff_norm = np.abs(mean_3400 - mean_3401) / pooled_std
    std_ratio = std_3400 / std_3401

    if boundaries is None:
        boundaries = field_boundaries()
    dim_to_field = {}
    for name, (start, end) in boundaries.items():
        for d in range(start, end):
            dim_to_field[d] = name

    per_dim = []
    diverging_dims = []
    for d in range(len(mean_3400)):
        ratio = float(std_ratio[d])
        ratio_ok = STD_RATIO_LO <= ratio <= STD_RATIO_HI or STD_RATIO_LO <= (1.0 / ratio) <= STD_RATIO_HI
        mean_ok = bool(mean_diff_norm[d] < MEAN_DIFF_THRESHOLD)
        ok = mean_ok and ratio_ok
        entry = {
            "dim": d,
            "field": dim_to_field[d],
            "mean_diff_norm": float(mean_diff_norm[d]),
            "std_ratio_3400_over_3401": ratio,
            "ok": ok,
        }
        per_dim.append(entry)
        if not ok:
            diverging_dims.append(entry)

    policy = "merged" if not diverging_dims else "per_task"
    return {
        "policy": policy,
        "mean_diff_threshold": MEAN_DIFF_THRESHOLD,
        "std_ratio_bounds": [STD_RATIO_LO, STD_RATIO_HI],
        "per_dim": per_dim,
        "diverging_dims": diverging_dims,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    combined_split = json.loads(args.split.read_text(encoding="utf-8"))
    shards_by_task: dict[str, list[dict]] = {}
    for shard in combined_split["shards"]:
        shards_by_task.setdefault(shard["task_id"], []).append(shard)

    task_ids = sorted(shards_by_task)
    assert task_ids == ["3400", "3401"], f"expected exactly tasks 3400/3401, got {task_ids}"

    per_task_values = {}
    per_task_stats = {}
    for task_id in task_ids:
        values = collect_task_core_rows(args.extracted_root, shards_by_task[task_id])
        per_task_values[task_id] = values
        per_task_stats[task_id] = per_dim_stats(values)
        print(f"task_{task_id}: {values.shape[0]} rows over {len(shards_by_task[task_id])} shards")

    comparison = decide_policy(per_task_stats["3400"], per_task_stats["3401"])

    merged_values = np.concatenate([per_task_values[t] for t in task_ids], axis=0)
    merged_stats = per_dim_stats(merged_values)

    out = {
        "policy": comparison["policy"],
        "decision_rule": {
            "mean_diff_threshold": comparison["mean_diff_threshold"],
            "std_ratio_bounds": comparison["std_ratio_bounds"],
            "rule": "merged iff ALL 33 dims satisfy |mean_3400-mean_3401|/pooled_std < threshold "
                    "AND std_ratio (either direction) within bounds; else per_task",
        },
        "diverging_dims": comparison["diverging_dims"],
        "per_dim_comparison": comparison["per_dim"],
        "stats": {
            "merged": {
                "mean": merged_stats["mean"].tolist(),
                "std": merged_stats["std"].tolist(),
                "n_rows": merged_stats["n_rows"],
            },
            "per_task": {
                task_id: {
                    "mean": per_task_stats[task_id]["mean"].tolist(),
                    "std": per_task_stats[task_id]["std"].tolist(),
                    "n_rows": per_task_stats[task_id]["n_rows"],
                }
                for task_id in task_ids
            },
        },
        "source": "all train-split episodes (Phase 3 build_combined_split.json), "
                  "full per-episode action stream, action_core_schema.slice_core_33d",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"policy={comparison['policy']} diverging_dims={len(comparison['diverging_dims'])}/33")
    if comparison["diverging_dims"]:
        for d in comparison["diverging_dims"]:
            print(f"  dim {d['dim']} ({d['field']}): mean_diff_norm={d['mean_diff_norm']:.3f} "
                  f"std_ratio={d['std_ratio_3400_over_3401']:.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
