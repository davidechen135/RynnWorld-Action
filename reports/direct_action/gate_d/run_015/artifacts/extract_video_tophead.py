"""Gate D run_001, Phase 1b: extract ONLY the top_head camera videos.

The Pipe B (33f) prep script consumes exactly one camera --
`--camera_key observation.images.top_head` (see agibot_action_core_prep_33f.py
and its shard_video_path / decode_window_frames call). Phase 1 extracted
meta+parquet only (extract_shard.py defaults include_video=False); the videos
were deferred to the GPU encode phase, which is now (this phase). Running
`extract_shard.py --include_video` would pull all 7 cameras (~7x the bytes of
the single consumed camera). This script pulls ONLY top_head so extraction
stays minimal and fast, while reusing extract_shard.py's schema check and
member-reconstruction logic (same bounded-member tar approach, no full
tar -tzf listing, which times out on ~30GB archives).

Usage:
    python3 extract_video_tophead.py \
        --task_id 3400 --shard_name 313498_314085.tar.gz \
        --shard_path /mnt/data/datasets/.../task_3400/313498_314085.tar.gz \
        --dest_root /mnt/data/agibot_extracted \
        --reference_info reports/direct_action/gate_b/run_001/extracted/info.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from reports.direct_action.gate_d.run_001.artifacts import extract_shard as es

CAMERA_KEY = "observation.images.top_head"


def video_members_only(info: dict) -> list[str]:
    """Member names for just the top_head camera video of every episode."""
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    video_path_tmpl = info["video_path"]
    members = []
    for ep in range(total_episodes):
        chunk = ep // chunks_size
        members.append(
            "data/" + video_path_tmpl.format(episode_chunk=chunk, video_key=CAMERA_KEY, episode_index=ep)
        )
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--shard_name", required=True, help="e.g. 313498_314085.tar.gz")
    parser.add_argument("--shard_path", required=True, type=Path)
    parser.add_argument("--dest_root", required=True, type=Path)
    parser.add_argument("--reference_info", required=True, type=Path)
    args = parser.parse_args()

    shard_stem = args.shard_name.removesuffix(".tar.gz")
    dest = args.dest_root / f"task_{args.task_id}" / shard_stem
    dest.mkdir(parents=True, exist_ok=True)

    reference_info = json.loads(args.reference_info.read_text(encoding="utf-8"))
    info_bytes = es.stream_member(args.shard_path, "data/meta/info.json")
    new_info = json.loads(info_bytes)

    schema_result = es.check_schema(new_info, reference_info)
    if not schema_result["pass"]:
        print(f"SCHEMA MISMATCH: {schema_result['mismatches']}")
        raise SystemExit(2)

    # Require that top_head is actually one of this shard's video keys.
    video_keys = es.video_keys_from_info(new_info)
    if CAMERA_KEY not in video_keys:
        print(f"ERROR: {CAMERA_KEY!r} not in {video_keys}")
        raise SystemExit(2)

    members = video_members_only(new_info)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        memberlist_path = Path(f.name)
        for m in members:
            f.write(m + "\n")

    try:
        proc = subprocess.run(
            ["tar", "-xzf", str(args.shard_path), "-C", str(dest), "-T", str(memberlist_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode not in (0, 2):
            raise RuntimeError(
                f"tar extraction failed rc={proc.returncode} for {args.shard_path}: "
                f"{proc.stderr.decode(errors='replace')[:2000]}"
            )

        missing = [m for m in members if not (dest / m).exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} top_head videos missing after extraction (first 5: {missing[:5]})")
    finally:
        memberlist_path.unlink(missing_ok=True)

    n_video = len(members)
    print(f"OK task_{args.task_id}/{args.shard_name}: extracted {n_video} top_head videos to {dest}")


if __name__ == "__main__":
    main()
