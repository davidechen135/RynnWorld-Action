"""Gate D run_001, Phase 2: reusable per-episode audit for extracted shards.

Re-derives the Gate A run_002/run_003 per-episode audit (NaN/Inf, exact-equality-
vs-state passthrough check, lag-scan, timestamp sanity) as a standalone, importable
script, since no general-purpose audit script survived Gate A (that code was run
inline and never saved). Rather than re-deriving the two pieces of math that DO
survive, this imports them:

  - `slice_core_33d` from gate_b/run_001/artifacts/action_core_schema.py
  - `verified_scan` from gate_a/run_003/artifacts/sign_audit.py -- its embedded
    self-test (`run_sign_test`) is executed as a guard at the top of `main()`,
    since this is exactly the sign-convention bug Gate A run_003 had to fix once
    already. If the self-test fails, the script aborts before trusting any real
    lag numbers.

Operates on ONE already-extracted shard directory at a time (as produced by
extract_shard.py: `<extracted_root>/data/meta/info.json` +
`<extracted_root>/data/data/chunk-XXX/episode_XXXXXX.parquet`), so it is
parallelizable across all ~20 shards.

Per episode:
  - NaN/Inf on the sliced 33D D-action-core.
  - Exact-equality of each of the 4 core action fields against the matching
    observation.state field at the SAME frame index (passthrough check --
    action/head/position was excluded from the core precisely because it failed
    this check in Gate A; the 4 core fields must NOT fail it).
  - Lag-scan (delays -6..+6) per core field via `verified_scan`, informational
    only (characterizes action-leads-state delay; does not gate pass/fail --
    Gate A already found delays ranging +2..+7 across fields, all valid).
  - Timestamp `dt_mean`/`dt_std` sanity (expect ~1/30s, tight std) and
    frame_index strict monotonicity.

An episode FAILS (excluded from Phase 3's split, logged with reason) iff any of:
  NaN/Inf present, any core field exact-equals its state counterpart at t
  (passthrough), dt_std exceeds threshold, or frame_index is not strictly
  monotonic. Lag-scan results are recorded but never fail an episode.

Usage:
    python3 audit_episodes.py \
        --extracted_root /mnt/data/agibot_extracted/task_3400/314096_314575 \
        --task_id 3400 --shard_name 314096_314575 \
        --out_dir reports/direct_action/gate_d/run_001/artifacts \
        [--episode_range 0-31]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from reports.direct_action.gate_b.run_001.artifacts.action_core_schema import (  # noqa: E402
    CORE_SLICES,
    field_boundaries,
    slice_core_33d,
)
from reports.direct_action.gate_a.run_003.artifacts.sign_audit import (  # noqa: E402
    run_sign_test,
    verified_scan,
)

ACTION_TO_STATE_FIELD = {
    "action/end/position": "state/end/arm_position",
    "action/end/orientation": "state/end/arm_orientation",
    "action/joint/position": "state/joint/position",
    "action/waist/position": "state/waist/position",
}

DT_EXPECTED = 1.0 / 30.0
DT_STD_MAX = 1e-3
LAG_DELAYS = list(range(-6, 7))


def state_field_slice(info: dict, field_name: str) -> slice:
    fd = info["features"]["observation.state"]["field_descriptions"]
    indices = fd[field_name]["indices"]
    assert indices == list(range(indices[0], indices[-1] + 1)), (
        f"{field_name} indices not contiguous: {indices}"
    )
    return slice(indices[0], indices[-1] + 1)


def parse_episode_range(spec: str | None, total_episodes: int) -> range:
    if spec is None:
        return range(total_episodes)
    lo, hi = spec.split("-")
    return range(int(lo), min(int(hi) + 1, total_episodes))


def audit_episode(df, episode_index: int, state_slices: dict) -> dict:
    import pandas as pd  # noqa: F401  (df is a DataFrame; import kept local to mirror caller)

    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    timestamps = df["timestamp"].to_numpy()
    frame_index = df["frame_index"].to_numpy()

    core = slice_core_33d(action)
    nan_count = int(np.isnan(core).sum())
    inf_count = int(np.isinf(core).sum())

    boundaries = field_boundaries()
    exact_equal = {}
    lag_scan = {}
    for field_name, _, _ in CORE_SLICES:
        start, end = boundaries[field_name]
        action_field = core[:, start:end]
        state_field = state[:, state_slices[field_name]]
        exact_equal[field_name] = bool(np.array_equal(action_field, state_field))

        per_dim_best = []
        for d in range(action_field.shape[1]):
            res = verified_scan(action_field[:, d], state_field[:, d], LAG_DELAYS)
            best = min(res, key=res.get)
            per_dim_best.append(best)
        lag_scan[field_name] = {
            "best_delay_per_dim": per_dim_best,
            "median_best_delay": float(np.median(per_dim_best)),
        }

    dt = np.diff(timestamps)
    dt_mean = float(np.mean(dt)) if len(dt) else float("nan")
    dt_std = float(np.std(dt)) if len(dt) else float("nan")
    frame_index_monotonic = bool(np.all(np.diff(frame_index) == 1))

    fail_reasons = []
    if nan_count or inf_count:
        fail_reasons.append(f"nan_inf: nan={nan_count} inf={inf_count}")
    passthrough_fields = [f for f, eq in exact_equal.items() if eq]
    if passthrough_fields:
        fail_reasons.append(f"exact_equal_state_t: {passthrough_fields}")
    if dt_std > DT_STD_MAX:
        fail_reasons.append(f"dt_std {dt_std:.6g} exceeds {DT_STD_MAX:.6g}")
    if not frame_index_monotonic:
        fail_reasons.append("frame_index not strictly monotonic")

    return {
        "episode_index": episode_index,
        "n_frames": int(len(df)),
        "dt_mean": dt_mean,
        "dt_std": dt_std,
        "frame_index_monotonic": frame_index_monotonic,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "exact_equal_state_t": exact_equal,
        "lag_scan": lag_scan,
        "pass": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
    }


def main() -> None:
    sign_test_result = run_sign_test(true_delay=3, delays=range(-6, 7))
    assert sign_test_result["passed"], "sign_audit self-test failed -- aborting, lag numbers untrustworthy"

    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_root", required=True, type=Path)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--shard_name", required=True)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--episode_range", default=None, help="e.g. 0-31; default all episodes")
    args = parser.parse_args()

    import pandas as pd

    info = json.loads((args.extracted_root / "data" / "meta" / "info.json").read_text(encoding="utf-8"))
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    data_path_tmpl = info["data_path"]

    state_slices = {
        field_name: state_field_slice(info, ACTION_TO_STATE_FIELD[field_name])
        for field_name, _, _ in CORE_SLICES
    }

    episode_range = parse_episode_range(args.episode_range, total_episodes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / f"episode_audit_{args.task_id}_{args.shard_name}.jsonl"
    records = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ep in episode_range:
            chunk = ep // chunks_size
            rel_path = data_path_tmpl.format(episode_chunk=chunk, episode_index=ep)
            parquet_path = args.extracted_root / "data" / rel_path
            if not parquet_path.exists():
                record = {
                    "episode_index": ep,
                    "pass": False,
                    "fail_reasons": [f"parquet missing: {parquet_path}"],
                }
            else:
                df = pd.read_parquet(parquet_path)
                record = audit_episode(df, ep, state_slices)
            record["task_id"] = args.task_id
            record["shard_name"] = args.shard_name
            records.append(record)
            f.write(json.dumps(record) + "\n")

    n_pass = sum(1 for r in records if r["pass"])
    n_fail = len(records) - n_pass
    summary = {
        "task_id": args.task_id,
        "shard_name": args.shard_name,
        "sign_audit_self_test": sign_test_result,
        "total_episodes_audited": len(records),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "failing_episodes": [
            {"episode_index": r["episode_index"], "fail_reasons": r["fail_reasons"]}
            for r in records
            if not r["pass"]
        ],
    }
    summary_path = args.out_dir / f"episode_audit_summary_{args.task_id}_{args.shard_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"task_{args.task_id}/{args.shard_name}: {n_pass}/{len(records)} episodes passed audit "
        f"(sign_audit self-test passed). Wrote {jsonl_path} and {summary_path}."
    )
    if n_fail:
        print(f"  FAILING episodes: {[r['episode_index'] for r in records if not r['pass']]}")


if __name__ == "__main__":
    main()
