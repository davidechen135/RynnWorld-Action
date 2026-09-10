#!/usr/bin/env python3
"""Gate C run_008: capacity ablation — does a larger native-trajectory adapter fix blur?

Single changed variable: adapter architecture (hidden_dim/layers/heads), controlled
against a same-architecture baseline. BOTH variants train FROM SCRATCH (zero-init
output_projection, random init elsewhere) on the identical full-range window data,
LR, seed, and fixed held-out validation window used in run_005/run_006. Training
from scratch (rather than continuing run_006's step8100 checkpoint) is required
because the larger architecture's weight shapes are incompatible with the existing
checkpoint, and comparing two from-scratch runs keeps the comparison apples-to-apples
instead of confounding "continued pretraining" with "capacity."
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
MANIFEST = REPO / "reports/direct_action/gate_c/run_005/artifacts/window_manifest_full_range.json"
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
CACHE = Path("/tmp/scratch/gate_c_run006_cache")  # reuse run_006's cache: same window/window-length/resolution
WINDOW = 33
HEIGHT, WIDTH = 480, 832
SEED = 42
SHUFFLE_SEED = 314159
LR = 1e-4
STEPS = 2000
MILESTONES = (1000, 2000)
FIXED_VALIDATION_TIMESTEP = 500
ROLLOUT_STEPS = 20
CFG_SCALE = 1.0

VARIANTS = {
    "baseline_scratch": dict(hidden_dim=768, num_layers=2, num_heads=12),   # current architecture, ~16.7M params
    "larger_scratch": dict(hidden_dim=1024, num_layers=3, num_heads=16),    # ~41M params, capacity ablation
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def extract_33d(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[:, 2:8], a[:, 8:16], a[:, 16:30], a[:, 33:38]], axis=1)


def select_windows(manifest: dict, steps: int) -> list[dict]:
    records = {r["episode"]: r for r in manifest["records"] if r["split"] == "train"}
    episodes = sorted(records)
    rng = np.random.default_rng(SEED)
    selected = []
    rounds_needed = -(-steps // len(episodes)) + 1
    for round_index in range(rounds_needed):
        for episode in rng.permutation(episodes).tolist():
            record = records[episode]
            windows = record["windows"]
            digest = hashlib.sha256(f"{SEED}:{episode}".encode()).digest()
            offset = int.from_bytes(digest[:2], "little") % len(windows)
            window = windows[(offset + 5 * round_index) % len(windows)]
            selected.append({
                "episode": episode, "start": window["start"], "stratum": window["stratum"],
                "fraction": window["fraction"], "video": record["video"], "parquet": record["parquet"],
            })
            if len(selected) == steps:
                return selected
    raise AssertionError("failed to select windows")


def select_active_held_out(manifest: dict) -> dict:
    record = next(r for r in manifest["records"] if r["split"] == "held_out" and r["episode"] == 11)
    actions = extract_33d(np.stack(pd.read_parquet(record["parquet"])["action"].values))
    max_start = record["video_frames"] - WINDOW
    candidates = np.unique(np.linspace(0, max_start, min(128, max_start + 1), dtype=int))
    scores = [float(np.linalg.norm(actions[s:s + WINDOW] - actions[s])) for s in candidates]
    start = int(candidates[int(np.argmax(scores))])
    return {"episode": record["episode"], "start": start, "activity": max(scores), "video": record["video"], "parquet": record["parquet"]}


def cache_sample(sample: dict, namespace: str, helpers, vae, mean, std, device, dtype) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{namespace}_ep{sample['episode']:06d}_f{sample['start']:06d}.pt"
    if path.exists():
        return path
    import imageio.v3 as iio
    frames = []
    for index, frame in enumerate(iio.imiter(sample["video"], plugin="pyav")):
        if index >= sample["start"] + WINDOW:
            break
        if index >= sample["start"]:
            frames.append(frame)
    if len(frames) != WINDOW:
        raise ValueError(f"{sample['video']}: decoded {len(frames)} frames at start={sample['start']}")
    frames = np.stack(frames)
    df = pd.read_parquet(sample["parquet"])
    actions = extract_33d(np.stack(df["action"].values))
    video = helpers.preprocess_video_frames(torch.from_numpy(frames), HEIGHT, WIDTH).unsqueeze(0).to(device, dtype=dtype)
    latent = helpers.encode_video_frames(vae, video, mean, std)
    torch.save({
        "video_latent": latent.cpu(), "img_latent": latent[:, :, :1].cpu(),
        "robot_trajectory": torch.from_numpy(actions[sample["start"]:sample["start"] + WINDOW].copy()).unsqueeze(0),
        "raw_video": video.cpu(), "episode": sample["episode"], "start_frame": sample["start"], "stratum": sample.get("stratum"),
    }, path)
    return path


def target_metrics(videos: dict[str, torch.Tensor], target: torch.Tensor) -> dict:
    from skimage.metrics import structural_similarity
    target_np = ((target.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    out = {}
    for name, video in videos.items():
        actual = ((video.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
        target_flow = np.abs(np.diff(np.dot(target_np[..., :3], [0.299, 0.587, 0.114]), axis=0)).mean(axis=(1, 2))
        actual_flow = np.abs(np.diff(np.dot(actual[..., :3], [0.299, 0.587, 0.114]), axis=0)).mean(axis=(1, 2))
        out[name] = {
            "target_pixel_mad": float(np.mean(np.abs(actual.astype(np.float32) - target_np.astype(np.float32)))),
            "target_rmse": float(np.sqrt(np.mean((actual.astype(np.float32) - target_np.astype(np.float32)) ** 2))),
            "target_ssim_mean": float(np.mean([structural_similarity(actual[i], target_np[i], channel_axis=2, data_range=255) for i in range(WINDOW)])),
            "target_flow_magnitude_mae": float(np.mean(np.abs(actual_flow - target_flow))),
        }
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").eval()
        target_lp = torch.from_numpy(target_np).permute(0, 3, 1, 2).float() / 127.5 - 1
        for name, video in videos.items():
            actual = torch.from_numpy(((video.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)).permute(0, 3, 1, 2).float() / 127.5 - 1
            values = []
            for begin in range(0, WINDOW, 4):
                values.extend(lp(actual[begin:begin + 4], target_lp[begin:begin + 4]).view(-1).tolist())
            out[name]["target_lpips_mean"] = float(np.mean(values))
    except Exception as exc:
        out["lpips_error"] = repr(exc)
    return out


def evaluate(audit, helpers, model, encoder, vae, mean, std, prompt, validation, step, device, run_dir):
    latent = validation["video_latent"].to(device)
    image = validation["img_latent"].to(device)
    correct = validation["robot_trajectory"].to(device=device, dtype=torch.float32)
    wrong = torch.roll(correct, shifts=correct.shape[1] // 2, dims=1).clone()
    permutation = torch.randperm(WINDOW, generator=torch.Generator(device=device).manual_seed(SHUFFLE_SEED), device=device)
    conditions = {
        "correct": correct, "zero": torch.zeros_like(correct), "shuffled": correct[:, permutation],
        "reversed": torch.flip(correct, dims=[1]), "wrong": wrong,
    }
    initial_noise = torch.randn(latent.shape, generator=torch.Generator(device=device).manual_seed(SEED), device=device, dtype=model.patch_embedding.weight.dtype)
    audit.ns_decode = helpers.decode_latents_to_video
    captures, embeddings, videos = {}, {}, {}
    encoder.eval()
    with torch.no_grad():
        for name, action in conditions.items():
            captures[name] = {}
            embeddings[name] = encoder(action, latent.shape[2] // model.config.patch_size[0]).float().cpu()
            videos[name] = audit.official_rollout(model, encoder, vae, image, action, prompt, mean, std, initial_noise, ROLLOUT_STEPS, CFG_SCALE, captures[name])
            audit.save_video(videos[name], run_dir / "videos" / f"step{step:04d}_{name}.mp4", 30)
    source_first = ((validation["raw_video"][0, :, 0].float().permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    metrics = audit.video_metrics(videos, source_first)
    metrics.update({
        "global_step": step,
        "embedding_distances": {name: float(torch.linalg.vector_norm(embeddings["correct"] - embeddings[name])) for name in conditions if name != "correct"},
        "injection": {name: {"norm": captures[name]["adapter_injection_norm"]} for name in conditions},
        "target_metrics": target_metrics(videos, validation["raw_video"][0].float()),
    })
    (run_dir / "artifacts" / f"rollout_metrics_step{step:04d}.json").write_text(json.dumps(metrics, indent=2))
    encoder.train()
    return metrics


def run_variant(variant_name: str, encoder_kwargs: dict):
    run_dir = REPO / "reports/direct_action/gate_c/run_008" / variant_name
    lock = run_dir / "training.lock"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    if lock.exists():
        raise RuntimeError(f"{variant_name} lock exists: {lock.read_text().strip()}")
    lock.write_text(str(os.getpid()))
    try:
        manifest = json.loads(MANIFEST.read_text())
        selected = select_windows(manifest, STEPS)
        validation_spec = select_active_held_out(manifest)
        (run_dir / "artifacts/selected_windows.json").write_text(json.dumps(selected, indent=2))
        (run_dir / "artifacts/fixed_validation_window.json").write_text(json.dumps(validation_spec, indent=2))

        helpers = load_module(HELPERS_PATH, f"gate_c_visual_v2_run008_{variant_name}")
        audit = load_module(AUDIT_PATH, f"gate_c_protocol_run008_{variant_name}")
        device, dtype = torch.device("cuda"), torch.bfloat16
        helpers.apply_monkey_patch(REPO)
        tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
        prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
        del tokenizer, text_encoder
        torch.cuda.empty_cache()
        vae, mean, std = helpers.load_vae(BASE_MODEL, device, dtype)
        cache_paths = [cache_sample(sample, "train", helpers, vae, mean, std, device, dtype) for sample in selected]
        validation_path = cache_sample(validation_spec, "fixed_validation", helpers, vae, mean, std, device, dtype)
        validation = torch.load(validation_path, map_location="cpu")

        model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
        from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
        encoder = NativeTrajectoryEncoder(input_dim=33, **encoder_kwargs).to(device)
        param_count = sum(p.numel() for p in encoder.parameters())
        model.native_trajectory_encoder = encoder
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
        timestep_rng = np.random.default_rng(SEED + 3)
        records = []
        provenance = {
            "run_id": f"direct_action/gate_c/run_008/{variant_name}",
            "only_experiment_variable": "adapter_capacity_vs_baseline_scratch",
            "encoder_kwargs": encoder_kwargs, "adapter_param_count": param_count,
            "trained_from_scratch": True, "lr": LR, "gradient_clip_norm": 1.0,
            "steps": STEPS, "seed": SEED, "fixed_validation_timestep": FIXED_VALIDATION_TIMESTEP,
            "validation": validation_spec, "data_manifest": str(MANIFEST),
        }
        (run_dir / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
        for index, path in enumerate(cache_paths, 1):
            sample = torch.load(path, map_location="cpu")
            latent = sample["video_latent"].to(device)
            image = sample["img_latent"].to(device)
            action = sample["robot_trajectory"].to(device=device, dtype=torch.float32)
            timestep = int(timestep_rng.integers(0, 1000))
            loss = helpers.compute_flow_matching_loss(model, encoder, latent, image, action, prompt, timestep, SEED)
            optimizer.zero_grad(); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                fixed_loss = helpers.compute_flow_matching_loss(model, encoder, validation["video_latent"].to(device), validation["img_latent"].to(device), validation["robot_trajectory"].to(device=device, dtype=torch.float32), prompt, FIXED_VALIDATION_TIMESTEP, SEED)
            step = index
            record = {"global_step": step, "loss": float(loss), "fixed_validation_loss": float(fixed_loss), "grad_norm": float(grad_norm), "episode": sample["episode"], "start": sample["start_frame"], "timestep": timestep}
            records.append(record)
            with (run_dir / "logs/training.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if index % 20 == 0 or step in MILESTONES:
                print(f"[{variant_name}] step={step} loss={float(loss):.6f} val={float(fixed_loss):.6f} grad={float(grad_norm):.4f}", flush=True)
            if step in MILESTONES:
                checkpoint = run_dir / "checkpoints" / f"adapter_step{step:04d}.pt"
                torch.save(encoder.state_dict(), checkpoint)
                rollout = evaluate(audit, helpers, model, encoder, vae, mean, std, prompt, validation, step, device, run_dir)
                provenance.setdefault("milestones", []).append({
                    "step": step, "checkpoint": str(checkpoint),
                    "all_first_frames_equal": all(rollout["first_frame_equal_to_correct"].values()),
                    "embedding_distances": rollout["embedding_distances"],
                    "target_metrics": rollout["target_metrics"],
                })
            (run_dir / "artifacts/training_metrics.json").write_text(json.dumps({"provenance": provenance, "steps": records}, indent=2))
        (run_dir / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
        (run_dir / "training.complete").write_text(str(STEPS))
    finally:
        lock.unlink(missing_ok=True)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else None
    if variant is None:
        for name, kwargs in VARIANTS.items():
            run_variant(name, kwargs)
    else:
        run_variant(variant, VARIANTS[variant])


if __name__ == "__main__":
    main()
