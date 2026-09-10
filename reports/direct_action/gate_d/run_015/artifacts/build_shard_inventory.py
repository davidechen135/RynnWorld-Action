"""Gate D run_001, Phase 0: enumerate task_3400/task_3401 shards without extracting.

Reads only tar headers (via `tarfile`, streaming, no member data pulled to disk) plus
filesystem stat info, so this is safe to run against the raw ~30GB shard files. Any
`.tar.gz.part` file (interrupted download) is recorded separately in
`excluded_part_files` and never appears in `shards` -- it must not be extracted by any
downstream step (see docs/agibot_native_action_plan.md-adjacent Gate D notes on the
2026-07-28 rc=143 interrupted task_3401 downloads).

Usage:
    python3 build_shard_inventory.py \
        --root /mnt/data/datasets/agibot-world/AgiBotWorld2026/ImitationLearning/CommercialSpaces \
        --tasks 3400,3401 \
        --out shard_inventory.json
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def inspect_shard(path: Path) -> dict:
    """Header-only scan: count tar members without extracting/reading file bodies.

    tarfile.open(..., "r|gz") is a *stream* reader -- it walks headers sequentially and
    seeks past data blocks, so this touches far less I/O than `tar -tzf` piped through a
    full listing, though it still must decompress the gzip stream once end-to-end.
    """
    member_count = 0
    has_meta_info = False
    with tarfile.open(path, mode="r|gz") as tar:
        for member in tar:
            member_count += 1
            if member.name.endswith("data/meta/info.json"):
                has_meta_info = True
    return {"member_count": member_count, "has_meta_info": has_meta_info}


def build_task_inventory(task_root: Path, header_scan: bool) -> dict:
    tasks_jsonl_path = task_root / "tasks.jsonl"
    tasks_jsonl = []
    if tasks_jsonl_path.exists():
        tasks_jsonl = [
            json.loads(line)
            for line in tasks_jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    shards = []
    excluded_part_files = []
    for entry in sorted(task_root.iterdir()):
        if entry.is_dir():
            continue
        if entry.name.endswith(".tar.gz.part"):
            excluded_part_files.append(entry.name)
            continue
        if not entry.name.endswith(".tar.gz"):
            continue
        stat = entry.stat()
        record = {
            "name": entry.name,
            "path": str(entry),
            "bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "status": "unextracted",
        }
        if header_scan:
            record.update(inspect_shard(entry))
        shards.append(record)

    return {
        "tasks_jsonl": tasks_jsonl,
        "shards": shards,
        "excluded_part_files": excluded_part_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--tasks", required=True, help="comma-separated task ids, e.g. 3400,3401")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--header-scan",
        action="store_true",
        help="also stream tar headers to count members (slower: touches full gzip stream)",
    )
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    inventory = {"tasks": {}}
    for task_id in task_ids:
        task_root = args.root / f"task_{task_id}"
        if not task_root.exists():
            raise FileNotFoundError(f"task root not found: {task_root}")
        inventory["tasks"][task_id] = build_task_inventory(task_root, args.header_scan)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    for task_id, data in inventory["tasks"].items():
        n_shards = len(data["shards"])
        n_excluded = len(data["excluded_part_files"])
        print(f"task_{task_id}: {n_shards} shards, {n_excluded} excluded .part files")
        for name in data["excluded_part_files"]:
            print(f"  excluded: {name}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
