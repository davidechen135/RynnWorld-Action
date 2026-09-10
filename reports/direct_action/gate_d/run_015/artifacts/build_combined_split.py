"""Gate D run_001, Phase 3: combined multi-task, multi-shard episode split.

Generalizes the original single-shard split rule (documented in
`gate_a/run_002/artifacts/episode_split.json`:
    sha256(shard:episode_index) mod 100 < 15 -> held_out, else train
to a task-qualified key so it is unambiguous across tasks/shards:
    sha256(f"{task_id}:{shard_name}:{episode_index}") mod 100 < 15 -> held_out

Per-episode audit failures (Phase 2, `episode_audit_summary_{task}_{shard}.json`)
are excluded from the split entirely (logged, not silently dropped) -- as of
this run there are none (1376/1376 passed), so this path is exercised but
currently a no-op.

Verification note: `gate_b/run_001/artifacts/split_manifest.json` already
tried 10 different literal key formats for the *original* single-shard rule
against shard task_3400/313498_314085 and could not byte-for-byte reproduce
`episode_split.json`'s held-out set with ANY of them (best overlap 5/19) --
the original rule's exact key format was never disambiguated because no
executable split-generation script survived from Gate A run_001/002/003.
Gate B's resolution was to adopt the original `episode_split.json` as
canonical based on its own internal integrity (disjoint, full coverage),
not on hash-rule recomputation. This script's task-qualified key
(`task_id:shard:episode_index`) was not among Gate B's 10 attempts either,
so it is checked here too, purely for completeness -- a mismatch here is
an EXPECTED continuation of Gate B's finding, not a new bug, and does not
block adoption of this script's own reproducible rule for the 20 shards
that have no historical split to reproduce in the first place.

Usage:
    python3 build_combined_split.py \
        --extracted_root /mnt/data/agibot_extracted \
        --audit_dir reports/direct_action/gate_d/run_001/artifacts \
        --reference_split reports/direct_action/gate_a/run_002/artifacts/episode_split.json \
        --reference_task_id 3400 --reference_shard_name 313498_314085 \
        --out reports/direct_action/gate_d/run_001/artifacts/build_combined_split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HELD_OUT_THRESHOLD = 15


def split_key(task_id: str, shard_name: str, episode_index: int) -> str:
    return f"{task_id}:{shard_name}:{episode_index}"


def assign(task_id: str, shard_name: str, episode_index: int) -> str:
    key = split_key(task_id, shard_name, episode_index)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return "held_out" if int(digest, 16) % 100 < HELD_OUT_THRESHOLD else "train"


def discover_shards(extracted_root: Path) -> list[tuple[str, str, int]]:
    shards = []
    for info_path in sorted(extracted_root.glob("task_*/*/data/meta/info.json")):
        shard_dir = info_path.parent.parent.parent
        task_id = shard_dir.parent.name.removeprefix("task_")
        shard_name = shard_dir.name
        info = json.loads(info_path.read_text(encoding="utf-8"))
        shards.append((task_id, shard_name, info["total_episodes"]))
    return shards


def failing_episodes(audit_dir: Path, task_id: str, shard_name: str) -> set[int]:
    summary_path = audit_dir / f"episode_audit_summary_{task_id}_{shard_name}.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing Phase 2 audit summary for task_{task_id}/{shard_name}: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {rec["episode_index"] for rec in summary["failing_episodes"]}


def build_shard_split(task_id: str, shard_name: str, total_episodes: int, excluded: set[int]) -> dict:
    train, held_out = [], []
    for ep in range(total_episodes):
        if ep in excluded:
            continue
        bucket = assign(task_id, shard_name, ep)
        (held_out if bucket == "held_out" else train).append(ep)
    assert set(train).isdisjoint(held_out)
    return {
        "task_id": task_id,
        "shard_name": shard_name,
        "total_episodes": total_episodes,
        "excluded_episodes": sorted(excluded),
        "train_episodes": train,
        "held_out_episodes": held_out,
        "train_count": len(train),
        "held_out_count": len(held_out),
    }


def verify_against_reference(
    reference_split_path: Path, task_id: str, shard_name: str, total_episodes: int, excluded: set[int]
) -> dict:
    reference = json.loads(reference_split_path.read_text(encoding="utf-8"))
    regenerated = build_shard_split(task_id, shard_name, total_episodes, excluded)
    ref_train = set(reference["train_episodes"])
    ref_held = set(reference["held_out_episodes"])
    new_train = set(regenerated["train_episodes"])
    new_held = set(regenerated["held_out_episodes"])
    exact_match = ref_train == new_train and ref_held == new_held
    return {
        "reference_source": str(reference_split_path),
        "reference_split_method": reference.get("split_method"),
        "new_key_format": "task_id:shard_name:episode_index",
        "exact_match": exact_match,
        "held_out_overlap_with_reference": len(ref_held & new_held),
        "held_out_overlap_denominator": len(ref_held),
        "note": (
            "exact_match False is an EXPECTED continuation of "
            "gate_b/run_001/artifacts/split_manifest.json's finding: none of its 10 "
            "literal key-format attempts reproduced the original rule either. "
            "The original episode_split.json remains canonical for "
            f"task_{task_id}/{shard_name} on the strength of its own integrity checks, "
            "not hash-rule recomputation; this script's own rule is what governs the "
            "20 shards with no historical split to reproduce."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_root", required=True, type=Path)
    parser.add_argument("--audit_dir", required=True, type=Path)
    parser.add_argument("--reference_split", required=True, type=Path)
    parser.add_argument("--reference_task_id", required=True)
    parser.add_argument("--reference_shard_name", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    shards = discover_shards(args.extracted_root)
    if not shards:
        raise RuntimeError(f"no shards discovered under {args.extracted_root}")

    per_shard = []
    all_keys = set()
    for task_id, shard_name, total_episodes in shards:
        excluded = failing_episodes(args.audit_dir, task_id, shard_name)
        shard_split = build_shard_split(task_id, shard_name, total_episodes, excluded)
        per_shard.append(shard_split)
        for ep in range(total_episodes):
            key = split_key(task_id, shard_name, ep)
            assert key not in all_keys, f"duplicate split key across shards: {key}"
            all_keys.add(key)

    reference_check = None
    for task_id, shard_name, total_episodes in shards:
        if task_id == args.reference_task_id and shard_name == args.reference_shard_name:
            excluded = failing_episodes(args.audit_dir, task_id, shard_name)
            reference_check = verify_against_reference(
                args.reference_split, task_id, shard_name, total_episodes, excluded
            )
            break
    if reference_check is None:
        raise RuntimeError(
            f"reference shard task_{args.reference_task_id}/{args.reference_shard_name} "
            "not found among discovered shards -- cannot run required verification step"
        )

    train_total = sum(s["train_count"] for s in per_shard)
    held_out_total = sum(s["held_out_count"] for s in per_shard)
    excluded_total = sum(len(s["excluded_episodes"]) for s in per_shard)

    combined = {
        "split_method": f"sha256(task_id:shard_name:episode_index) mod 100 < {HELD_OUT_THRESHOLD} -> held_out, else train",
        "n_shards": len(per_shard),
        "train_count": train_total,
        "held_out_count": held_out_total,
        "excluded_count": excluded_total,
        "reference_verification": reference_check,
        "shards": per_shard,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print(
        f"Wrote {args.out}: {len(per_shard)} shards, {train_total} train / "
        f"{held_out_total} held_out / {excluded_total} excluded episodes."
    )
    print(f"Reference verification (task_{args.reference_task_id}/{args.reference_shard_name}): "
          f"exact_match={reference_check['exact_match']} "
          f"overlap={reference_check['held_out_overlap_with_reference']}/"
          f"{reference_check['held_out_overlap_denominator']}")
    if not reference_check["exact_match"]:
        print("  (expected -- see reference_check['note'] in the output JSON)")


if __name__ == "__main__":
    main()
