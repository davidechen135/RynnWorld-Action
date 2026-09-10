"""Gate D run_001, Phase 1: parameterized, schema-checked shard extraction.

Extracts one task_3400/task_3401 shard into a durable, task+shard-namespaced
directory under --dest_root, without ever running a full `tar -tzf` listing
(which reliably times out on these ~30GB archives). Instead:

  1. Stream out only `data/meta/info.json` via `tar -xzf <shard> data/meta/info.json -O`.
  2. Parse it for total_episodes, total_chunks, chunks_size, data_path/video_path
     templates, and video feature keys.
  3. Reconstruct the exact list of tar member names for every episode's parquet
     (and, with --include_video, every episode/camera video) and extract them
     via `tar -xzf <shard> -T <memberlist> -C <dest>` (explicit member list,
     not a full listing).

Archive layout note (confirmed empirically in gate_b/run_001/report.md via a
bounded `tar -tzf | head` peek): the tar's own root directory is `data/`, and
info.json's `data_path`/`video_path` templates are themselves already rooted
at `data/...` or `videos/...` relative to the dataset root. So the real tar
member name is `data/` + template, e.g.:
    data/data/chunk-000/episode_000000.parquet   (double "data/", NOT a bug --
                                                    data_path template itself
                                                    starts with "data/")
    data/videos/chunk-000/{camera}/episode_000000.mp4
    data/meta/info.json

By default this extracts meta + parquet only (small, CPU/audit-sufficient).
Video extraction is deferred to Phase 7 (GPU-gated VAE encoding) and only
happens with --include_video.

Usage:
    python3 extract_shard.py \
        --task_id 3401 --shard_name 352507_353983.tar.gz \
        --shard_path /mnt/data/datasets/.../task_3401/352507_353983.tar.gz \
        --dest_root /mnt/data/agibot_extracted \
        --reference_info reports/direct_action/gate_b/run_001/extracted/info.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

SCHEMA_FIELDS_TO_MATCH = ["codebase_version", "robot_type", "fps", "data_path", "video_path"]

# observation.state fields that some shards legitimately log as 0-dim (e.g. no
# base-localization/SLAM active for that episode batch). Neither field is consumed
# by the D-action-core pipeline (which is sliced entirely out of the `action`
# column, not `observation.state` -- see action_core_schema.py) or by Phase 2's
# audit passthrough check (which only needs state/head/position, unaffected by
# these two fields being present/absent). A state_shape mismatch is treated as
# non-fatal only if it is fully explained by one/both of these fields shrinking
# to 0-dim, with every other field's name+dimensions unchanged. Confirmed via
# task_3400/346206_347749.tar.gz on 2026-08-30: 169D reference vs 162D new,
# diff == exactly state/robot/position (3D) + state/robot/orientation (4D) -> 0D.
BENIGN_STATE_FIELD_VARIANTS = {"state/robot/position", "state/robot/orientation"}


def stream_member(shard_path: Path, member: str) -> bytes:
    proc = subprocess.run(
        ["tar", "-xzf", str(shard_path), member, "-O"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to stream member {member!r} from {shard_path}: {proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


def video_keys_from_info(info: dict) -> list[str]:
    feats = info.get("features", {})
    keys = [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    return sorted(keys)


def action_state_shape(info: dict) -> tuple[list, list]:
    feats = info.get("features", {})
    action_shape = feats.get("action", {}).get("shape")
    state_shape = feats.get("observation.state", {}).get("shape")
    return action_shape, state_shape


def state_field_descriptions(info: dict) -> dict:
    return info.get("features", {}).get("observation.state", {}).get("field_descriptions", {}) or {}


def benign_state_shape_diff(new_info: dict, reference_info: dict) -> bool:
    """True iff the only observation.state difference is one/both of
    BENIGN_STATE_FIELD_VARIANTS shrinking to 0-dim (see comment at that
    constant's definition) -- every other field's name and dimensions must be
    byte-identical between shards."""
    new_fields = state_field_descriptions(new_info)
    ref_fields = state_field_descriptions(reference_info)
    if set(new_fields) != set(ref_fields):
        return False
    for name, ref_desc in ref_fields.items():
        new_desc = new_fields[name]
        if new_desc.get("dimensions") == ref_desc.get("dimensions"):
            continue
        if name in BENIGN_STATE_FIELD_VARIANTS and new_desc.get("dimensions") == 0:
            continue
        return False
    return True


def check_schema(new_info: dict, reference_info: dict) -> dict:
    mismatches = {}
    for field in SCHEMA_FIELDS_TO_MATCH:
        if new_info.get(field) != reference_info.get(field):
            mismatches[field] = {"reference": reference_info.get(field), "new": new_info.get(field)}

    ref_action_shape, ref_state_shape = action_state_shape(reference_info)
    new_action_shape, new_state_shape = action_state_shape(new_info)
    if new_action_shape != ref_action_shape:
        mismatches["action_shape"] = {"reference": ref_action_shape, "new": new_action_shape}

    benign_state_note = None
    if new_state_shape != ref_state_shape:
        if benign_state_shape_diff(new_info, reference_info):
            benign_state_note = {
                "reference": ref_state_shape,
                "new": new_state_shape,
                "reason": "state/robot/position and/or state/robot/orientation shrank to 0-dim; "
                          "not part of D-action-core (action-column-only) or Phase 2's passthrough check",
            }
        else:
            mismatches["state_shape"] = {"reference": ref_state_shape, "new": new_state_shape}

    ref_video_keys = video_keys_from_info(reference_info)
    new_video_keys = video_keys_from_info(new_info)
    if new_video_keys != ref_video_keys:
        mismatches["video_keys"] = {"reference": ref_video_keys, "new": new_video_keys}

    result = {"pass": len(mismatches) == 0, "mismatches": mismatches}
    if benign_state_note is not None:
        result["benign_state_shape_diff"] = benign_state_note
    return result


def build_member_list(info: dict, include_video: bool) -> list[str]:
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    data_path_tmpl = info["data_path"]
    video_path_tmpl = info["video_path"]
    video_keys = video_keys_from_info(info) if include_video else []

    members = ["data/meta/info.json"]
    for optional_meta in ("data/meta/episodes.jsonl", "data/meta/tasks.jsonl", "data/meta/stats.json"):
        members.append(optional_meta)

    for ep in range(total_episodes):
        chunk = ep // chunks_size
        members.append("data/" + data_path_tmpl.format(episode_chunk=chunk, episode_index=ep))
        for vk in video_keys:
            members.append(
                "data/" + video_path_tmpl.format(episode_chunk=chunk, video_key=vk, episode_index=ep)
            )
    return members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--shard_name", required=True, help="e.g. 313498_314085.tar.gz")
    parser.add_argument("--shard_path", required=True, type=Path)
    parser.add_argument("--dest_root", required=True, type=Path)
    parser.add_argument("--reference_info", required=True, type=Path)
    parser.add_argument("--include_video", action="store_true", help="also extract mp4s (GPU-phase use, large)")
    parser.add_argument("--verify_only", action="store_true", help="only extract+check meta/info.json, no data files")
    args = parser.parse_args()

    shard_stem = args.shard_name.removesuffix(".tar.gz")
    dest = args.dest_root / f"task_{args.task_id}" / shard_stem
    dest.mkdir(parents=True, exist_ok=True)

    reference_info = json.loads(args.reference_info.read_text(encoding="utf-8"))

    info_bytes = stream_member(args.shard_path, "data/meta/info.json")
    new_info = json.loads(info_bytes)

    schema_result = check_schema(new_info, reference_info)
    schema_out = {
        "task_id": args.task_id,
        "shard_name": args.shard_name,
        "reference": str(args.reference_info),
        **schema_result,
        "total_episodes": new_info.get("total_episodes"),
        "total_frames": new_info.get("total_frames"),
    }
    schema_check_path = dest / "schema_check.json"
    schema_check_path.write_text(json.dumps(schema_out, indent=2), encoding="utf-8")

    if not schema_result["pass"]:
        print(f"SCHEMA MISMATCH for task_{args.task_id}/{args.shard_name}: {schema_result['mismatches']}")
        raise SystemExit(2)

    (dest / "data" / "meta").mkdir(parents=True, exist_ok=True)
    (dest / "data" / "meta" / "info.json").write_bytes(info_bytes)

    if args.verify_only:
        print(f"OK (verify_only) task_{args.task_id}/{args.shard_name}: "
              f"{new_info.get('total_episodes')} episodes, schema matches reference")
        return

    members = build_member_list(new_info, include_video=args.include_video)
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
        # rc==2 from GNU tar commonly means "some members not found" (the three
        # optional meta files above may not exist in every shard) -- only treat
        # this as fatal if any *required* (info.json/parquet/video) member is
        # actually missing on disk afterward.
        if proc.returncode not in (0, 2):
            raise RuntimeError(
                f"tar extraction failed rc={proc.returncode} for {args.shard_path}: "
                f"{proc.stderr.decode(errors='replace')[:2000]}"
            )

        missing_required = []
        for m in members:
            if m.startswith("data/meta/") and m != "data/meta/info.json":
                continue  # optional meta files, already tolerated above
            if not (dest / m).exists():
                missing_required.append(m)
        if missing_required:
            raise RuntimeError(
                f"{len(missing_required)} required members missing after extraction "
                f"(first 5: {missing_required[:5]})"
            )
    finally:
        memberlist_path.unlink(missing_ok=True)

    n_parquet = sum(1 for m in members if m.endswith(".parquet"))
    n_video = sum(1 for m in members if m.endswith(".mp4"))
    print(
        f"OK task_{args.task_id}/{args.shard_name}: extracted {n_parquet} parquet"
        f"{f' + {n_video} video' if n_video else ''} files to {dest}"
    )


if __name__ == "__main__":
    main()
