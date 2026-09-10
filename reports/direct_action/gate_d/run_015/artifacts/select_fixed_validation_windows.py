"""Gate D run_001, Phase 6 (second half): fixed per-task validation window.

Generalizes run_011's `select_active_held_out` (train_5000_corrected.py:108-115)
across tasks. run_011 picked a single hardcoded held-out episode (11) and then
chose, within that one episode, the start position maximizing
`||actions[s:s+WINDOW] - actions[s]||` over up to 128 candidate starts
(np.linspace over the episode's valid start range). Episode 11 itself was not
chosen by any rule -- it was just whichever held-out episode run_011's author
picked -- so it is not reproducible across a held-out pool that now spans many
shards per task. This script keeps run_011's activity-score rule but replaces
its fixed WINDOW=33 candidate slicing with `agibot_action_core_prep`'s
adaptive contiguous segments (1-3s windows, no per-episode cap; see that
module for the up-to-date windowing scheme), so the picked validation window
is guaranteed to be one of the actual windows that appears in the held-out
pool rather than an arbitrary WINDOW=33 slice that may not align with any
real window boundary. For each task it scans every (shard, episode, segment)
in the held-out pool and keeps the single one with the globally maximum
activity score. This is deterministic and reproducible, unlike hand-picking
an episode.

Uses the canonical `action_core_schema.slice_core_33d` (not run_011's
hand-rolled `extract_33d`) and `compare_action_distributions.load_episode_parquet`
for shard-relative parquet loading, both imported rather than re-derived.

Usage:
    python3 select_fixed_validation_windows.py \
        --extracted_root /mnt/data/agibot_extracted \
        --split reports/direct_action/gate_d/run_001/artifacts/build_combined_split.json \
        --out reports/direct_action/gate_d/run_001/artifacts/fixed_validation_windows.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from reports.direct_action.gate_b.run_001.artifacts.action_core_schema import slice_core_33d  # noqa: E402
from reports.direct_action.gate_d.run_001.artifacts.compare_action_distributions import (  # noqa: E402
    load_episode_parquet,
)
from scripts.agibot_action_core_prep import adaptive_segments  # noqa: E402


def score_episode(extracted_root: Path, task_id: str, shard_name: str, episode_index: int) -> dict | None:
    df = load_episode_parquet(extracted_root, task_id, shard_name, episode_index)
    actions = slice_core_33d(np.stack(df["action"].to_numpy()))
    segments = adaptive_segments(actions.shape[0])
    if not segments:
        return None
    best = None
    for seg in segments:
        s, length = seg["start"], seg["length"]
        score = float(np.linalg.norm(actions[s:s + length] - actions[s]))
        if best is None or score > best[0]:
            best = (score, s, length)
    score, start, length = best
    return {
        "task_id": task_id,
        "shard_name": shard_name,
        "episode_index": episode_index,
        "start_frame": start,
        "num_frames": length,
        "activity": score,
    }


def select_for_task(extracted_root: Path, shards: list[dict]) -> dict:
    best = None
    for shard in shards:
        for ep in shard["held_out_episodes"]:
            result = score_episode(extracted_root, shard["task_id"], shard["shard_name"], ep)
            if result is None:
                continue
            if best is None or result["activity"] > best["activity"]:
                best = result
    assert best is not None, "no eligible held-out episode found"
    return best


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

    result = {}
    for task_id in sorted(shards_by_task):
        selection = select_for_task(args.extracted_root, shards_by_task[task_id])
        result[task_id] = selection
        print(f"task_{task_id}: shard={selection['shard_name']} episode={selection['episode_index']} "
              f"start={selection['start_frame']} num_frames={selection['num_frames']} "
              f"activity={selection['activity']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
