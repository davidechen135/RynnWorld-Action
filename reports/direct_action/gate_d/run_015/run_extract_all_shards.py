#!/usr/bin/env python3
"""Gate D run_001 Phase 1b driver: extract top_head videos from all 21 shards.

Launches one extract_video_tophead.py subprocess per shard (6 x task_3400 + 15 x task_3401),
all in parallel. Each subprocess streams data/meta/info.json, validates it against the
gate_b reference, reconstructs the top_head member list (~N episodes), and tar-extracts
ONLY those members. Per-shard output is logged to run_001/logs/extract_<task>_<shard>.log.

Idempotent-ish: already-extracted shards (dest dir present) are skipped by the caller,
but extract_video_tophead.py itself always re-runs tar (cheap-ish; it only writes the
missing members). Progress details are in logs/extract_status.json.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path("/mnt/workspace/RynnWorld-Teleop")
RUN = REPO_ROOT / "reports/direct_action/gate_d/run_001"
ART = RUN / "artifacts"
LOG_DIR = RUN / "logs"
DEST_ROOT = Path("/mnt/data/agibot_extracted")
REFERENCE_INFO = Path("/mnt/workspace/RynnWorld-Teleop/reports/direct_action/gate_b/run_001/extracted/info.json")
EXTRACT_SCRIPT = ART / "extract_video_tophead.py"

MAX_PARALLEL = 6  # tar decompress is I/O-bound; keep concurrency modest


def main() -> None:
    inventory = json.loads((ART / "shard_inventory.json").read_text())
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for task_id, task_info in inventory["tasks"].items():
        for shard in task_info["shards"]:
            if shard.get("status") == "extracted":
                continue
            jobs.append((task_id, shard))

    print(f"total jobs to run: {len(jobs)}", flush=True)
    status = {}
    if (LOG_DIR / "extract_status.json").exists():
        status = json.loads((LOG_DIR / "extract_status.json").read_text())

    running = []
    done = set()
    try:
        import queue
        q = queue.Queue()
        for idx, (task_id, shard) in enumerate(jobs):
            q.put((idx, task_id, shard))

        active = {}
        while not q.empty() or active:
            # start new jobs up to concurrency limit
            while len(active) < MAX_PARALLEL and not q.empty():
                idx, task_id, shard = q.get()
                key = f"task_{task_id}/{shard['name']}"
                log_path = LOG_DIR / f"extract_{key.replace('/', '_').replace('.tar.gz', '')}.log"
                logf = open(log_path, "w")
                proc_env = dict(os.environ)
                existing_pypath = proc_env.get("PYTHONPATH", "")
                proc_env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pypath else f"{REPO_ROOT}:{existing_pypath}"
                proc = subprocess.Popen(
                    [
                        "python3", str(EXTRACT_SCRIPT),
                        "--task_id", task_id,
                        "--shard_name", shard["name"],
                        "--shard_path", shard["path"],
                        "--dest_root", str(DEST_ROOT),
                        "--reference_info", str(REFERENCE_INFO),
                    ],
                    stdout=logf, stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                    env=proc_env,
                )
                active[proc.pid] = (proc, logf, key, idx)
                print(f"[start] {key} pid={proc.pid}", flush=True)

            # reap finished
            finished = [pid for pid, (proc, logf, key, idx) in active.items() if proc.poll() is not None]
            for pid in finished:
                proc, logf, key, idx = active.pop(pid)
                rc = proc.returncode
                logf.close()
                status[key] = {"rc": rc, "done": True}
                (LOG_DIR / "extract_status.json").write_text(json.dumps(status, indent=2))
                print(f"[done]  {key} rc={rc} pid={pid}", flush=True)
            import time
            time.sleep(1)
    finally:
        for proc, logf, key, idx in active.values():
            proc.kill()
            logf.close()

    print(f"finished all {len(status)} jobs")


if __name__ == "__main__":
    main()
