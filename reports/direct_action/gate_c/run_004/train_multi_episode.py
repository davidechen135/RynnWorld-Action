#!/usr/bin/env python3
"""Gate C run_004: expand only the data axis across task_3400 episodes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
RUN = REPO / "reports/direct_action/gate_c/run_004"
MANIFEST = RUN / "artifacts/window_manifest.json"
CACHE = Path("/tmp/scratch/gate_c_run004_cache")
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
INITIAL_ADAPTER = REPO / "reports/direct_action/gate_c/run_003/checkpoints/adapter_step1000.pt"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
GLOBAL_STEP_START = 3000
MILESTONES = (3100, 3250, 3500)
STEPS = MILESTONES[-1] - GLOBAL_STEP_START
LR = 1e-4
SEED = 42
HEIGHT, WIDTH = 480, 832
WINDOW = 33
ROLLOUT_STEPS = 20
CFG_SCALE = 1.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_hash(x: torch.Tensor) -> str:
    raw = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def extract_33d(action_40d: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [action_40d[:, 2:8], action_40d[:, 8:16], action_40d[:, 16:30], action_40d[:, 33:38]],
        axis=1,
    )


def select_training_windows(manifest: dict) -> list[dict]:
    by_episode = {}
    for record in manifest["records"]:
        if record["split"] == "train":
            by_episode[record["episode"]] = record
    rng = np.random.default_rng(SEED)
    episodes = sorted(by_episode)
    selected = []
    round_index = 0
    while len(selected) < STEPS:
        for episode in rng.permutation(episodes).tolist():
            starts = by_episode[episode]["window_starts"]
            start = starts[round_index % len(starts)]
            selected.append(
                {
                    "episode": episode,
                    "start": start,
                    "video": by_episode[episode]["video"],
                    "parquet": by_episode[episode]["parquet"],
                }
            )
            if len(selected) == STEPS:
                break
        round_index += 1
    assert len({sample["episode"] for sample in selected}) == len(episodes) == 91
    return selected


def select_reference_window(manifest: dict) -> dict:
    record = next(record for record in manifest["records"] if record["split"] == "held_out")
    return {
        "episode": record["episode"],
        "start": record["window_starts"][0],
        "video": record["video"],
        "parquet": record["parquet"],
    }


def acquire_lock() -> Path:
    lock_path = RUN / "training.lock"
    if lock_path.exists():
        try:
            active_pid = int(lock_path.read_text().strip())
            os.kill(active_pid, 0)
        except (ValueError, ProcessLookupError):
            lock_path.unlink()
        else:
            raise RuntimeError(f"run_004 training is already active as PID {active_pid}")
    lock_path.write_text(str(os.getpid()))
    return lock_path


def decode_prefix(video_path: Path, stop: int) -> np.ndarray:
    frames = []
    for index, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        if index >= stop:
            break
        frames.append(frame)
    if len(frames) != stop:
        raise ValueError(f"{video_path}: decoded {len(frames)} frames, expected {stop}")
    return np.stack(frames)


def prepare_cache(selected: list[dict], cache_namespace: str, helpers, vae, mean, std, device, dtype) -> list[Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for index, sample in enumerate(selected):
        grouped[sample["episode"]].append((index, sample))
    cache_paths = [None] * len(selected)
    for episode in sorted(grouped):
        items = grouped[episode]
        stop = max(sample["start"] + WINDOW for _, sample in items)
        frames = decode_prefix(Path(items[0][1]["video"]), stop)
        df = pd.read_parquet(items[0][1]["parquet"])
        actions = extract_33d(np.stack(df["action"].values))
        for index, sample in items:
            cache_path = CACHE / f"{cache_namespace}_sample_{index:04d}_ep{episode:06d}_f{sample['start']:06d}.pt"
            cache_paths[index] = cache_path
            if cache_path.exists():
                continue
            start = sample["start"]
            clip = torch.from_numpy(frames[start:start + WINDOW])
            video = helpers.preprocess_video_frames(clip, HEIGHT, WIDTH).unsqueeze(0).to(device, dtype=dtype)
            torch.manual_seed(SEED + index)
            latent = helpers.encode_video_frames(vae, video, mean, std)
            trajectory = torch.from_numpy(actions[start:start + WINDOW].copy()).unsqueeze(0)
            payload = {
                "video_latent": latent.cpu(),
                "img_latent": latent[:, :, :1].cpu(),
                "robot_trajectory": trajectory,
                "episode": episode,
                "start_frame": start,
                "video_path": sample["video"],
                "parquet_path": sample["parquet"],
            }
            torch.save(payload, cache_path)
        print(f"[cache] episode={episode:03d} windows={len(items)}", flush=True)
    return cache_paths


def save_checkpoint(encoder, optimizer, global_step: int):
    checkpoint_dir = RUN / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    adapter_path = checkpoint_dir / f"adapter_step{global_step:04d}.pt"
    state_path = checkpoint_dir / f"training_state_step{global_step:04d}.pt"
    torch.save(encoder.state_dict(), adapter_path)
    torch.save(
        {
            "global_step": global_step,
            "adapter_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "initial_adapter_path": str(INITIAL_ADAPTER),
            "initial_adapter_actual_global_step": GLOBAL_STEP_START,
        },
        state_path,
    )
    return adapter_path, state_path


def rollout_metrics(audit, helpers, model, encoder, vae, mean, std, prompt_embedding,
                    reference: dict, global_step: int, device):
    video_latent = reference["video_latent"].to(device)
    img_latent = reference["img_latent"].to(device)
    correct = reference["robot_trajectory"].to(device=device, dtype=torch.float32)
    zero = torch.zeros_like(correct)
    generator = torch.Generator(device=device).manual_seed(314159)
    permutation = torch.randperm(correct.shape[1], generator=generator, device=device)
    conditions = {
        "correct": correct,
        "zero": zero,
        "shuffled": correct[:, permutation],
        "wrong": torch.flip(correct, dims=[1]),
    }
    noise_generator = torch.Generator(device=device).manual_seed(SEED)
    initial_noise = torch.randn(
        video_latent.shape,
        generator=noise_generator,
        device=device,
        dtype=model.patch_embedding.weight.dtype,
    )
    videos = {}
    captures = {}
    embeddings = {}
    audit.ns_decode = helpers.decode_latents_to_video
    encoder.eval()
    with torch.no_grad():
        for name, action in conditions.items():
            captures[name] = {}
            embeddings[name] = encoder(action, video_latent.shape[2] // model.config.patch_size[0]).float().cpu()
            videos[name] = audit.official_rollout(
                model, encoder, vae, img_latent, action, prompt_embedding, mean, std,
                initial_noise, ROLLOUT_STEPS, CFG_SCALE, captures[name],
            )
            audit.save_video(videos[name], RUN / "videos" / f"step{global_step:04d}_{name}.mp4", 30)
    source_first = helpers.decode_latents_to_video(vae, img_latent, mean, std)[0, :, 0].float().cpu()
    source_first = ((source_first.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    result = audit.video_metrics(videos, source_first)
    result.update(
        {
            "global_step": global_step,
            "reference_episode": reference["episode"],
            "reference_start_frame": reference["start_frame"],
            "initial_noise_sha256": tensor_hash(initial_noise),
            "action_sha256": {name: tensor_hash(action) for name, action in conditions.items()},
            "embedding_distances": {
                f"correct__{name}": float(torch.linalg.vector_norm(embeddings["correct"] - embeddings[name]))
                for name in ("zero", "shuffled", "wrong")
            },
            "injection_stats": {
                name: {
                    "norm": captures[name]["adapter_injection_norm"],
                    "mean_abs": captures[name]["adapter_injection_mean_abs"],
                }
                for name in conditions
            },
        }
    )
    (RUN / "artifacts" / f"rollout_metrics_step{global_step:04d}.json").write_text(json.dumps(result, indent=2))
    encoder.train()
    return result


def main():
    lock_path = acquire_lock()
    (RUN / "videos").mkdir(exist_ok=True)
    (RUN / "checkpoints").mkdir(exist_ok=True)
    metrics_path = RUN / "artifacts/training_metrics.json"
    log_jsonl = RUN / "logs/training.jsonl"
    log_jsonl.parent.mkdir(exist_ok=True)

    helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_run004")
    audit = load_module(AUDIT_PATH, "gate_c_corrected_audit_run004")
    manifest = json.loads(MANIFEST.read_text())
    selected = select_training_windows(manifest)
    reference_window = select_reference_window(manifest)
    (RUN / "artifacts/selected_training_windows.json").write_text(json.dumps(selected, indent=2))
    (RUN / "artifacts/reference_window.json").write_text(json.dumps(reference_window, indent=2))

    device = torch.device("cuda")
    dtype = torch.bfloat16
    helpers.apply_monkey_patch(REPO)
    tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
    prompt_embedding = helpers.encode_text(tokenizer, text_encoder, "", device)
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    vae, mean, std = helpers.load_vae(BASE_MODEL, device, dtype)
    cache_paths = prepare_cache(selected, "train", helpers, vae, mean, std, device, dtype)
    reference_path = prepare_cache([reference_window], "held_out_reference", helpers, vae, mean, std, device, dtype)[0]

    model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    encoder = NativeTrajectoryEncoder(input_dim=33).to(device)
    encoder.load_state_dict(torch.load(INITIAL_ADAPTER, map_location="cpu"))
    encoder.train()
    model.native_trajectory_encoder = encoder
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)

    provenance = {
        "run_id": "direct_action/gate_c/run_004",
        "initial_adapter": str(INITIAL_ADAPTER),
        "initial_adapter_filename_step": 1000,
        "initial_adapter_actual_global_step": GLOBAL_STEP_START,
        "train_episode_count": 91,
        "held_out_episode_count": 19,
        "available_train_windows": manifest["train_window_count"],
        "selected_training_windows": len(selected),
        "reference_split": "held_out",
        "reference_episode": reference_window["episode"],
        "reference_start_frame": reference_window["start"],
        "window_size": WINDOW,
        "stride": manifest["stride"],
        "lr": LR,
        "micro_batch_size": 1,
        "milestones": list(MILESTONES),
        "architecture_changed": False,
        "preprocessing_changed": False,
        "action_definition_changed": False,
    }
    (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))

    records = []
    reference = torch.load(reference_path, map_location="cpu")
    timestep_rng = np.random.default_rng(SEED + 1)
    for local_step, cache_path in enumerate(cache_paths, start=1):
        sample = torch.load(cache_path, map_location="cpu")
        video_latent = sample["video_latent"].to(device)
        img_latent = sample["img_latent"].to(device)
        trajectory = sample["robot_trajectory"].to(device=device, dtype=torch.float32)
        timestep = int(timestep_rng.integers(0, 1000))
        loss = helpers.compute_flow_matching_loss(
            model, encoder, video_latent, img_latent, trajectory,
            prompt_embedding, timestep, SEED,
        )
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        global_step = GLOBAL_STEP_START + local_step
        record = {
            "global_step": global_step,
            "local_step": local_step,
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "episode": sample["episode"],
            "start_frame": sample["start_frame"],
            "timestep": timestep,
        }
        records.append(record)
        with log_jsonl.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if local_step % 10 == 0 or global_step in MILESTONES:
            print(
                f"[train] global_step={global_step} loss={float(loss):.6f} "
                f"grad_norm={float(grad_norm):.4f} ep={sample['episode']} start={sample['start_frame']}",
                flush=True,
            )
        if global_step in MILESTONES:
            adapter_path, state_path = save_checkpoint(encoder, optimizer, global_step)
            rollout = rollout_metrics(
                audit, helpers, model, encoder, vae, mean, std,
                prompt_embedding, reference, global_step, device,
            )
            milestone_record = {
                "global_step": global_step,
                "adapter_checkpoint": str(adapter_path),
                "training_state": str(state_path),
                "mean_loss_last10": float(np.mean([r["loss"] for r in records[-10:]])),
                "rollout_metrics": str(RUN / "artifacts" / f"rollout_metrics_step{global_step:04d}.json"),
                "all_first_frames_equal": all(rollout["first_frame_equal_to_correct"].values()),
            }
            provenance.setdefault("completed_milestones", []).append(milestone_record)
            (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
        metrics_path.write_text(json.dumps({"config": provenance, "steps": records}, indent=2))
    (RUN / "training.complete").write_text(str(MILESTONES[-1]))
    lock_path.unlink()
    print(f"[complete] global_step={MILESTONES[-1]}", flush=True)


if __name__ == "__main__":
    main()
