#!/usr/bin/env python3
"""Run 5,000 full-range window records from all train episodes.

This wrapper reuses the validated run_005c training/evaluation implementation while
changing only the training-run directory, cache namespace, sample schedule, and
milestones.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
SOURCE = REPO / "reports/direct_action/gate_c/run_005/train_full_range_step3100.py"
RUN = REPO / "reports/direct_action/gate_c/run_006"
CACHE = Path("/tmp/scratch/gate_c_run006_cache")
MANIFEST = REPO / "reports/direct_action/gate_c/run_005/artifacts/window_manifest_full_range.json"
SEED = 42
GLOBAL_START = 3100
STEPS = 5000
MILESTONES = (GLOBAL_START + 1000, GLOBAL_START + 3000, GLOBAL_START + 5000)


def load_source():
    spec = importlib.util.spec_from_file_location("gate_c_run005_impl", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_5000(manifest: dict) -> list[dict]:
    records = [record for record in manifest["records"] if record["split"] == "train"]
    unique = []
    for record in sorted(records, key=lambda item: item["episode"]):
        for window in record["windows"]:
            unique.append(
                {
                    "episode": record["episode"],
                    "start": window["start"],
                    "stratum": window["stratum"],
                    "fraction": window["fraction"],
                    "video": record["video"],
                    "parquet": record["parquet"],
                }
            )
    assert len(unique) == 1456
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(unique)).tolist()
    selected = []
    for index in range(STEPS):
        cycle, position = divmod(index, len(order))
        sample = dict(unique[order[position]])
        sample["cycle"] = cycle
        sample["unique_window_index"] = order[position]
        selected.append(sample)
    assert len(selected) == 5000
    assert len({sample["episode"] for sample in selected}) == 91
    assert len({sample["unique_window_index"] for sample in selected}) == 1456
    return selected


def patch_provenance():
    artifact = RUN / "artifacts/provenance.json"
    if not artifact.exists():
        return
    provenance = json.loads(artifact.read_text())
    provenance.update(
        {
            "run_id": "direct_action/gate_c/run_006",
            "source_run": "direct_action/gate_c/run_005c",
            "source_adapter": str(REPO / "reports/direct_action/gate_c/run_004/checkpoints/adapter_step3100.pt"),
            "source_global_step": GLOBAL_START,
            "selected_record_count": STEPS,
            "unique_training_windows": 1456,
            "train_episode_count": 91,
            "held_out_episode_count": 19,
            "sampling": "one seeded permutation of all 1456 train windows, cycled and truncated at 5000 records",
            "milestones_global": list(MILESTONES),
            "milestones_local": [1000, 3000, 5000],
            "data_manifest": str(MANIFEST),
            "cache": str(CACHE),
        }
    )
    artifact.write_text(json.dumps(provenance, indent=2))


def main():
    module = load_source()
    module.RUN = RUN
    module.CACHE = CACHE
    module.MANIFEST = MANIFEST
    module.GLOBAL_STEP_START = GLOBAL_START
    module.STEPS = STEPS
    module.MILESTONES = MILESTONES
    module.select_500 = select_5000
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "artifacts").mkdir(exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    selected = select_5000(manifest)
    (RUN / "artifacts/selected_5000_full_range.json").write_text(json.dumps(selected, indent=2))
    module.main()
    patch_provenance()


if __name__ == "__main__":
    main()
