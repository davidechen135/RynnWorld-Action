#!/usr/bin/env python3
"""Gate C run_009: single-episode overfit diagnostic with action normalization + raised LR.

User-specified decision tree (root-cause isolation before touching adapter capacity again):
  1. Action normalization audit found gate_c scripts feed RAW 33D action values straight
     from parquet into NativeTrajectoryEncoder — no per-dimension, dataset-level z-score
     standardization, unlike the original project's agibot_native_prep.py (which
     z-scores against train-split mean/std before ever saving robot_trajectory). Measured
     raw scale spread: per-dim std ranges 0.066-0.741 (>11x), several dims have |mean|>1,
     only 74.9% of raw values fall within [-1,1].
  2. This run fixes that gap: compute per-dimension mean/std over ALL 91 train episodes'
     full action streams (matches original pipeline's methodology), z-score every action
     tensor before it reaches the encoder.
  3. Combined with a raised LR (1e-3, 10x run_004-008's 1e-4) and a single-episode dense
     (stride=16, ~50 window) overfit test for 200 epochs, no shuffling of the *window set*
     (only shuffling *order* within an epoch) and no augmentation.

Two changed variables vs run_008's baseline_scratch: action normalization (was: none) and
LR (was: 1e-4). This is intentional per the user's own request — the point of this
diagnostic is to determine whether normalization/LR were masking the model's true
capacity, not to keep single-variable discipline. Architecture is held IDENTICAL to
run_008's baseline_scratch (hidden_dim=768, num_layers=2, num_heads=12, 16.7M params) so
any result here is NOT confounded by capacity.

Decision rule (per user's request):
  - loss does not approach ~0 on this tiny, memorizable set -> normalization/loss-mask
    bug or LR still too small, investigate further before touching capacity again.
  - loss approaches ~0 -> architecture itself is fine; the bottleneck for the full
    91-episode run was under-fitting, likely fixable with more capacity/steps/LR on the
    full dataset, not a structural bug.
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
RUN = REPO / "reports/direct_action/gate_c/run_009"
MANIFEST = REPO / "reports/direct_action/gate_c/run_005/artifacts/window_manifest_full_range.json"
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
CACHE = Path("/tmp/scratch/gate_c_run009_cache")
STATS_CACHE = RUN / "artifacts/action_stats.json"

WINDOW = 33
STRIDE = 16
HEIGHT, WIDTH = 480, 832
SEED = 42
SHUFFLE_SEED = 314159
OVERFIT_EPISODE = 102  # video_frames=837 -> 51 dense stride-16 windows, closest to the requested ~50
LR = 1e-3
EPOCHS = 200
FIXED_VALIDATION_TIMESTEP = 500
ROLLOUT_STEPS = 20
CFG_SCALE = 1.0
ENCODER_KWARGS = dict(hidden_dim=768, num_layers=2, num_heads=12)  # identical to run_008 baseline_scratch


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_33d(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[:, 2:8], a[:, 8:16], a[:, 16:30], a[:, 33:38]], axis=1)


def compute_action_stats(manifest: dict) -> dict:
    if STATS_CACHE.exists():
        return json.loads(STATS_CACHE.read_text())
    records = [r for r in manifest["records"] if r["split"] == "train"]
    all_rows = []
    for record in records:
        df = pd.read_parquet(record["parquet"])
        all_rows.append(extract_33d(np.stack(df["action"].values)))
    values = np.concatenate(all_rows, axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    stats = {
        "mean": mean.tolist(), "std": std.tolist(),
        "source": "all 91 train episodes, full per-episode action stream, matches agibot_native_prep.py methodology",
        "n_rows": int(values.shape[0]),
        "raw_fraction_within_[-1,1]": float(np.mean((values >= -1) & (values <= 1))),
    }
    STATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    STATS_CACHE.write_text(json.dumps(stats, indent=2))
    return stats


def normalize(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (action - mean) / std


def dense_windows(manifest: dict, episode: int) -> dict:
    record = next(r for r in manifest["records"] if r["split"] == "train" and r["episode"] == episode)
    max_start = record["video_frames"] - WINDOW
    starts = list(range(0, max_start + 1, STRIDE))
    return {"episode": episode, "starts": starts, "video": record["video"], "parquet": record["parquet"], "video_frames": record["video_frames"]}


def select_active_held_out(manifest: dict) -> dict:
    record = next(r for r in manifest["records"] if r["split"] == "held_out" and r["episode"] == 11)
    actions = extract_33d(np.stack(pd.read_parquet(record["parquet"])["action"].values))
    max_start = record["video_frames"] - WINDOW
    candidates = np.unique(np.linspace(0, max_start, min(128, max_start + 1), dtype=int))
    scores = [float(np.linalg.norm(actions[s:s + WINDOW] - actions[s])) for s in candidates]
    start = int(candidates[int(np.argmax(scores))])
    return {"episode": record["episode"], "start": start, "activity": max(scores), "video": record["video"], "parquet": record["parquet"]}


def cache_sample(episode: int, start: int, video: str, parquet: str, namespace: str, helpers, vae, mean, std, device, dtype) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{namespace}_ep{episode:06d}_f{start:06d}.pt"
    if path.exists():
        return path
    import imageio.v3 as iio
    frames = []
    for index, frame in enumerate(iio.imiter(video, plugin="pyav")):
        if index >= start + WINDOW:
            break
        if index >= start:
            frames.append(frame)
    if len(frames) != WINDOW:
        raise ValueError(f"{video}: decoded {len(frames)} frames at start={start}")
    frames = np.stack(frames)
    df = pd.read_parquet(parquet)
    actions = extract_33d(np.stack(df["action"].values))
    video_t = helpers.preprocess_video_frames(torch.from_numpy(frames), HEIGHT, WIDTH).unsqueeze(0).to(device, dtype=dtype)
    latent = helpers.encode_video_frames(vae, video_t, mean, std)
    torch.save({
        "video_latent": latent.cpu(), "img_latent": latent[:, :, :1].cpu(),
        "robot_trajectory_raw": torch.from_numpy(actions[start:start + WINDOW].copy()).unsqueeze(0),
        "raw_video": video_t.cpu(), "episode": episode, "start_frame": start,
    }, path)
    return path


def target_metrics(video: torch.Tensor, target: torch.Tensor) -> dict:
    from skimage.metrics import structural_similarity
    target_np = ((target.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    actual = ((video.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    target_flow = np.abs(np.diff(np.dot(target_np[..., :3], [0.299, 0.587, 0.114]), axis=0)).mean(axis=(1, 2))
    actual_flow = np.abs(np.diff(np.dot(actual[..., :3], [0.299, 0.587, 0.114]), axis=0)).mean(axis=(1, 2))
    out = {
        "target_pixel_mad": float(np.mean(np.abs(actual.astype(np.float32) - target_np.astype(np.float32)))),
        "target_rmse": float(np.sqrt(np.mean((actual.astype(np.float32) - target_np.astype(np.float32)) ** 2))),
        "target_ssim_mean": float(np.mean([structural_similarity(actual[i], target_np[i], channel_axis=2, data_range=255) for i in range(actual.shape[0])])),
        "target_flow_magnitude_mae": float(np.mean(np.abs(actual_flow - target_flow))),
    }
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").eval()
        target_lp = torch.from_numpy(target_np).permute(0, 3, 1, 2).float() / 127.5 - 1
        actual_t = torch.from_numpy(actual).permute(0, 3, 1, 2).float() / 127.5 - 1
        values = []
        for begin in range(0, actual.shape[0], 4):
            values.extend(lp(actual_t[begin:begin + 4], target_lp[begin:begin + 4]).view(-1).tolist())
        out["target_lpips_mean"] = float(np.mean(values))
    except Exception as exc:
        out["lpips_error"] = repr(exc)
    return out


def rollout_and_eval(audit, helpers, model, encoder, vae, mean, std, prompt, sample, mean_t, std_t, tag, device, run_dir):
    latent = sample["video_latent"].to(device)
    image = sample["img_latent"].to(device)
    raw_action = sample["robot_trajectory_raw"].to(device=device, dtype=torch.float32)
    action = normalize(raw_action, mean_t, std_t)
    initial_noise = torch.randn(latent.shape, generator=torch.Generator(device=device).manual_seed(SEED), device=device, dtype=model.patch_embedding.weight.dtype)
    capture = {}
    audit.ns_decode = helpers.decode_latents_to_video
    encoder.eval()
    with torch.no_grad():
        video = audit.official_rollout(model, encoder, vae, image, action, prompt, mean, std, initial_noise, ROLLOUT_STEPS, CFG_SCALE, capture)
    audit.save_video(video, run_dir / "videos" / f"{tag}.mp4", 30)
    metrics = target_metrics(video, sample["raw_video"][0].float())
    metrics["adapter_injection_norm"] = capture["adapter_injection_norm"]
    encoder.train()
    return metrics


def main():
    lock = RUN / "training.lock"
    if lock.exists():
        raise RuntimeError(f"lock exists: {lock.read_text().strip()}")
    lock.write_text(str(os.getpid()))
    try:
        manifest = json.loads(MANIFEST.read_text())
        stats = compute_action_stats(manifest)
        print(f"[action_stats] n_rows={stats['n_rows']} raw_fraction_within_[-1,1]={stats['raw_fraction_within_[-1,1]']:.4f}", flush=True)
        episode_spec = dense_windows(manifest, OVERFIT_EPISODE)
        n_windows = len(episode_spec["starts"])
        print(f"[episode {OVERFIT_EPISODE}] video_frames={episode_spec['video_frames']} dense_windows(stride={STRIDE})={n_windows}", flush=True)
        validation_spec = select_active_held_out(manifest)

        helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_run009")
        audit = load_module(AUDIT_PATH, "gate_c_protocol_run009")
        device, dtype = torch.device("cuda"), torch.bfloat16
        helpers.apply_monkey_patch(REPO)
        tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
        prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
        del tokenizer, text_encoder
        torch.cuda.empty_cache()
        vae, vae_mean, vae_std = helpers.load_vae(BASE_MODEL, device, dtype)

        cache_paths = [
            cache_sample(OVERFIT_EPISODE, start, episode_spec["video"], episode_spec["parquet"], "overfit", helpers, vae, vae_mean, vae_std, device, dtype)
            for start in episode_spec["starts"]
        ]
        validation_path = cache_sample(validation_spec["episode"], validation_spec["start"], validation_spec["video"], validation_spec["parquet"], "fixed_validation", helpers, vae, vae_mean, vae_std, device, dtype)
        validation = torch.load(validation_path, map_location="cpu")

        model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
        from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
        encoder = NativeTrajectoryEncoder(input_dim=33, **ENCODER_KWARGS).to(device)
        param_count = sum(p.numel() for p in encoder.parameters())
        model.native_trajectory_encoder = encoder
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)

        mean_t = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
        std_t = torch.tensor(stats["std"], device=device, dtype=torch.float32)

        provenance = {
            "run_id": "direct_action/gate_c/run_009",
            "purpose": "single-episode overfit diagnostic: action normalization + raised LR, capacity held fixed at run_008 baseline",
            "changed_vs_run_008_baseline_scratch": ["action_normalization (per-dim z-score over 91 train episodes, was: none)", "lr (1e-3, was: 1e-4)", "data: single episode dense windows (was: full 91-episode manifest)"],
            "held_fixed_vs_run_008_baseline_scratch": ["architecture (hidden_dim=768,num_layers=2,num_heads=12)", "seed", "fixed_validation_window", "gradient_clip_norm=1.0"],
            "overfit_episode": OVERFIT_EPISODE, "n_windows": n_windows, "stride": STRIDE,
            "epochs": EPOCHS, "total_steps": EPOCHS * n_windows,
            "lr": LR, "gradient_clip_norm": 1.0, "seed": SEED,
            "action_stats": {"n_rows": stats["n_rows"], "raw_fraction_within_[-1,1]": stats["raw_fraction_within_[-1,1]"], "stats_file": str(STATS_CACHE)},
            "adapter_param_count": param_count,
            "fixed_validation_timestep": FIXED_VALIDATION_TIMESTEP, "validation": validation_spec,
        }
        (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))

        milestone_epochs = {1, 5, 10, 25, 50, 100, 150, 200}
        timestep_rng = np.random.default_rng(SEED + 3)
        order_rng = np.random.default_rng(SEED + 7)
        cached = [torch.load(p, map_location="cpu") for p in cache_paths]
        step = 0
        milestones_log = []
        for epoch in range(1, EPOCHS + 1):
            order = order_rng.permutation(n_windows).tolist()
            epoch_losses = []
            for window_index in order:
                sample = cached[window_index]
                latent = sample["video_latent"].to(device)
                image = sample["img_latent"].to(device)
                raw_action = sample["robot_trajectory_raw"].to(device=device, dtype=torch.float32)
                action = normalize(raw_action, mean_t, std_t)
                timestep = int(timestep_rng.integers(0, 1000))
                loss = helpers.compute_flow_matching_loss(model, encoder, latent, image, action, prompt, timestep, SEED)
                optimizer.zero_grad(); loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
                step += 1
                epoch_losses.append(float(loss))
                if step % 200 == 0:
                    with torch.no_grad():
                        fixed_loss = helpers.compute_flow_matching_loss(model, encoder, validation["video_latent"].to(device), validation["img_latent"].to(device), normalize(validation["robot_trajectory_raw"].to(device=device, dtype=torch.float32), mean_t, std_t), prompt, FIXED_VALIDATION_TIMESTEP, SEED)
                    record = {"step": step, "epoch": epoch, "loss": float(loss), "epoch_running_mean": float(np.mean(epoch_losses)), "fixed_validation_loss": float(fixed_loss), "grad_norm": float(grad_norm)}
                    with (RUN / "logs/training.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(f"[run_009] epoch={epoch} step={step} loss={float(loss):.6f} epoch_mean={np.mean(epoch_losses):.6f} val={float(fixed_loss):.6f} grad={float(grad_norm):.4f}", flush=True)
            if epoch in milestone_epochs:
                checkpoint = RUN / "checkpoints" / f"adapter_epoch{epoch:04d}.pt"
                torch.save(encoder.state_dict(), checkpoint)
                epoch_mean_loss = float(np.mean(epoch_losses))
                print(f"[run_009] === milestone epoch={epoch} step={step} epoch_mean_loss={epoch_mean_loss:.6f} ===", flush=True)
                milestones_log.append({"epoch": epoch, "step": step, "epoch_mean_loss": epoch_mean_loss, "checkpoint": str(checkpoint)})
                (RUN / "artifacts/milestones.json").write_text(json.dumps(milestones_log, indent=2))
        # Final rollout: overfit-target window (memorized) + fixed held-out window, for visual/metric confirmation.
        overfit_target = cached[0]
        rollout_metrics = {
            "overfit_train_window": rollout_and_eval(audit, helpers, model, encoder, vae, vae_mean, vae_std, prompt, overfit_target, mean_t, std_t, "final_overfit_window", device, RUN),
            "fixed_held_out": rollout_and_eval(audit, helpers, model, encoder, vae, vae_mean, vae_std, prompt, validation, mean_t, std_t, "final_held_out", device, RUN),
        }
        (RUN / "artifacts/final_rollout_metrics.json").write_text(json.dumps(rollout_metrics, indent=2))
        provenance["milestones"] = milestones_log
        provenance["final_rollout_metrics"] = rollout_metrics
        (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
        (RUN / "training.complete").write_text(str(step))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
