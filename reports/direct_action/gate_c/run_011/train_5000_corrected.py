#!/usr/bin/env python3
"""Gate C run_011: full 91-episode, 5000-step training with both confirmed fixes + lower LR.

run_009/run_010's single-episode (episode 102, 51 dense windows, 200-epoch) overfit diagnostic
established two things:
  1. Action normalization gap (fixed in run_009): gate_c scripts through run_008 fed raw,
     un-normalized 33D actions straight from parquet into NativeTrajectoryEncoder, unlike the
     original project's agibot_native_prep.py, which z-scores per-dimension against train-split
     mean/std before ever saving robot_trajectory. Fixed by reusing run_009's precomputed stats
     (mean/std over all 91 train episodes' full action streams, 172,310 rows).
  2. Timestep train/inference mismatch (fixed in run_010): gate_c_visual_v2.py's shared
     compute_flow_matching_loss (used by every gate_c training script since run_003) passed the
     model a raw, unshifted scalar timestep uniformly across all frames. The real trainer
     (rynnworld_teleop_trainer.py compute_loss) and the inference-time official_rollout both use
     a flow-shifted, per-patch, frame-0-zeroed timestep tensor instead. Fixed by reusing run_010's
     compute_flow_matching_loss_corrected.

With both fixes, run_010's single-episode plateau dropped from ~0.25-0.36 (run_009, buggy
timestep) to ~0.20-0.21 (run_010, fixed), but LR=1e-3 showed validation-loss instability
(spikes to 0.4-0.9) by epoch150-200 on that tiny 51-window set. This run tests whether the two
fixes generalize to the full 91-train-episode distribution: LR lowered to 5e-4 (the other value
originally proposed alongside 1e-3), 5000 training steps drawn from all 1456 unique full-range
windows across all 91 train episodes (run_006's sampling method: one seeded permutation, cycled
to 5000), checkpointed and evaluated every 1000 steps.

Trained from scratch: capacity is held at the run_008 baseline_scratch architecture (16.7M-param
encoder; run_008 already showed a 2.5x larger encoder does not help), but action normalization
and timestep signal differ from every previous multi-episode run (run_004-run_008), so continuing
from an old checkpoint would mix inconsistent training signals with the corrected ones.
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
RUN = REPO / "reports/direct_action/gate_c/run_011"
MANIFEST = REPO / "reports/direct_action/gate_c/run_005/artifacts/window_manifest_full_range.json"
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
CACHE = Path("/tmp/scratch/gate_c_run006_cache")  # reuse: caches RAW actions + latents, unaffected by normalization/timestep fixes
STATS_CACHE = REPO / "reports/direct_action/gate_c/run_009/artifacts/action_stats.json"  # reuse: computed over all 91 train episodes

WINDOW = 33
HEIGHT, WIDTH = 480, 832
SEED = 42
SHUFFLE_SEED = 314159
LR = 5e-4  # lowered from run_009/010's 1e-3 (which showed late-stage validation-loss instability); still 5x the original run_004-008 baseline of 1e-4
STEPS = 5000
MILESTONES = (1000, 2000, 3000, 4000, 5000)
FIXED_VALIDATION_TIMESTEP = 500
ROLLOUT_STEPS = 20
CFG_SCALE = 1.0
ENCODER_KWARGS = dict(hidden_dim=768, num_layers=2, num_heads=12)  # identical to run_008 baseline_scratch / run_009 / run_010


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_33d(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[:, 2:8], a[:, 8:16], a[:, 16:30], a[:, 33:38]], axis=1)


def normalize(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (action - mean) / std


def select_5000(manifest: dict) -> list[dict]:
    records = [record for record in manifest["records"] if record["split"] == "train"]
    unique = []
    for record in sorted(records, key=lambda item: item["episode"]):
        for window in record["windows"]:
            unique.append({
                "episode": record["episode"], "start": window["start"], "stratum": window["stratum"],
                "fraction": window["fraction"], "video": record["video"], "parquet": record["parquet"],
            })
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
    assert len(selected) == STEPS
    assert len({sample["episode"] for sample in selected}) == 91
    assert len({sample["unique_window_index"] for sample in selected}) == 1456
    return selected


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


def compute_flow_matching_loss_corrected(model, encoder, video_latent, img_latent, robot_trajectory,
                                          text_embedding, timestep_idx, seed,
                                          flow_shift: float = 5.0, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Matches rynnworld_teleop_trainer.py compute_loss() timestep construction exactly:
    shifted (not raw) timestep, per-patch tensor, frame-0 zeroed -- same convention
    official_rollout already uses for inference. Verbatim from run_010."""
    device = video_latent.device
    dtype = video_latent.dtype
    torch.manual_seed(seed)
    noise = torch.randn_like(video_latent)
    s = timestep_idx / num_train_timesteps
    sigma_t = torch.tensor(flow_shift * s / (1 + (flow_shift - 1) * s), device=device, dtype=dtype)
    noisy_latents = (1 - sigma_t) * video_latent + sigma_t * noise
    noisy_latents[:, :, 0:1] = img_latent
    target = noise - video_latent

    b, c, n_f, h, w = video_latent.shape
    p_t, p_h, p_w = model.config.patch_size
    first_frame_mask = torch.ones((1, 1, n_f, h, w), device=device, dtype=torch.float32)
    first_frame_mask[:, :, 0] = 0
    shifted_timestep = float(sigma_t) * num_train_timesteps
    timestep_tensor = (first_frame_mask[0, 0, :, ::p_h, ::p_w] * shifted_timestep).flatten().unsqueeze(0).to(dtype)

    pred = model(
        hidden_states=noisy_latents, timestep=timestep_tensor,
        encoder_hidden_states=text_embedding, encoder_hidden_states_image=None,
        robot_trajectory=robot_trajectory, null_condition=False,
    ).sample

    timestep_weight = torch.clamp(1.0 / (sigma_t * (1 - sigma_t) + 1e-5), max=10.0)
    timestep_weight = timestep_weight / timestep_weight.mean()
    loss_per_sample = torch.mean((pred[:, :, 1:] - target[:, :, 1:]) ** 2, dim=(1, 2, 3, 4))
    loss = torch.mean(timestep_weight * loss_per_sample)
    return loss


def target_metrics(videos: dict, target: torch.Tensor) -> dict:
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


def evaluate(audit, helpers, model, encoder, vae, mean, std, prompt, validation, mean_t, std_t, step, device, run_dir):
    latent = validation["video_latent"].to(device)
    image = validation["img_latent"].to(device)
    correct_raw = validation["robot_trajectory"].to(device=device, dtype=torch.float32)
    correct = normalize(correct_raw, mean_t, std_t)
    wrong_raw = torch.roll(correct_raw, shifts=correct_raw.shape[1] // 2, dims=1).clone()
    wrong = normalize(wrong_raw, mean_t, std_t)
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


def main():
    lock = RUN / "training.lock"
    if lock.exists():
        raise RuntimeError(f"lock exists: {lock.read_text().strip()}")
    lock.write_text(str(os.getpid()))
    try:
        manifest = json.loads(MANIFEST.read_text())
        stats = json.loads(STATS_CACHE.read_text())
        print(f"[action_stats] reused from run_009: n_rows={stats['n_rows']} raw_fraction_within_[-1,1]={stats['raw_fraction_within_[-1,1]']:.4f}", flush=True)
        selected = select_5000(manifest)
        validation_spec = select_active_held_out(manifest)
        (RUN / "artifacts/selected_5000_full_range.json").write_text(json.dumps(selected, indent=2))
        (RUN / "artifacts/fixed_validation_window.json").write_text(json.dumps(validation_spec, indent=2))

        helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_run011")
        audit = load_module(AUDIT_PATH, "gate_c_protocol_run011")
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
        encoder = NativeTrajectoryEncoder(input_dim=33, **ENCODER_KWARGS).to(device)
        param_count = sum(p.numel() for p in encoder.parameters())
        model.native_trajectory_encoder = encoder
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
        timestep_rng = np.random.default_rng(SEED + 3)

        mean_t = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
        std_t = torch.tensor(stats["std"], device=device, dtype=torch.float32)

        provenance = {
            "run_id": "direct_action/gate_c/run_011",
            "purpose": "full 91-train-episode, 5000-step training with both run_009/010 fixes (action normalization + corrected timestep signal) and a lowered LR",
            "changed_vs_run_010": ["data: 5000 draws from all 1456 unique full-range windows across 91 train episodes (was: single episode 102, 51 dense windows)", "lr=5e-4 (was: 1e-3)", "steps=5000 total, no epoch repetition (was: 200 epochs over the same 51 windows)"],
            "held_fixed_vs_run_010": ["architecture (hidden_dim=768,num_layers=2,num_heads=12, 16.7M params)", "action_normalization_stats", "corrected_timestep_signal", "seed", "fixed_validation_window", "gradient_clip_norm=1.0"],
            "run_009_010_result_for_comparison": "single-episode (51-window) overfit plateau: run_009 (buggy timestep) ~0.25-0.36 epoch_mean_loss; run_010 (fixed timestep) ~0.20-0.21 epoch_mean_loss, neither approached 0",
            "trained_from_scratch": True, "lr": LR, "gradient_clip_norm": 1.0,
            "steps": STEPS, "seed": SEED, "fixed_validation_timestep": FIXED_VALIDATION_TIMESTEP,
            "validation": validation_spec, "data_manifest": str(MANIFEST),
            "action_stats": {"n_rows": stats["n_rows"], "raw_fraction_within_[-1,1]": stats["raw_fraction_within_[-1,1]"], "stats_file": str(STATS_CACHE)},
            "adapter_param_count": param_count,
        }
        (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))

        for index, path in enumerate(cache_paths, 1):
            sample = torch.load(path, map_location="cpu")
            latent = sample["video_latent"].to(device)
            image = sample["img_latent"].to(device)
            action_raw = sample["robot_trajectory"].to(device=device, dtype=torch.float32)
            action = normalize(action_raw, mean_t, std_t)
            timestep = int(timestep_rng.integers(0, 1000))
            loss = compute_flow_matching_loss_corrected(model, encoder, latent, image, action, prompt, timestep, SEED)
            optimizer.zero_grad(); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                validation_action = normalize(validation["robot_trajectory"].to(device=device, dtype=torch.float32), mean_t, std_t)
                fixed_loss = compute_flow_matching_loss_corrected(model, encoder, validation["video_latent"].to(device), validation["img_latent"].to(device), validation_action, prompt, FIXED_VALIDATION_TIMESTEP, SEED)
            step = index
            record = {"global_step": step, "loss": float(loss), "fixed_validation_loss": float(fixed_loss), "grad_norm": float(grad_norm), "episode": sample["episode"], "start": sample["start_frame"], "timestep": timestep}
            with (RUN / "logs/training.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if index % 50 == 0 or step in MILESTONES:
                print(f"[run_011] step={step} loss={float(loss):.6f} val={float(fixed_loss):.6f} grad={float(grad_norm):.4f}", flush=True)
            if step in MILESTONES:
                checkpoint = RUN / "checkpoints" / f"adapter_step{step:04d}.pt"
                torch.save(encoder.state_dict(), checkpoint)
                rollout = evaluate(audit, helpers, model, encoder, vae, mean, std, prompt, validation, mean_t, std_t, step, device, RUN)
                provenance.setdefault("milestones", []).append({
                    "step": step, "checkpoint": str(checkpoint),
                    "all_first_frames_equal": all(rollout["first_frame_equal_to_correct"].values()),
                    "embedding_distances": rollout["embedding_distances"],
                    "target_metrics": rollout["target_metrics"],
                })
                (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
                print(f"[run_011] === milestone step={step} target_metrics(correct)={rollout['target_metrics']['correct']} ===", flush=True)
        (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
        (RUN / "training.complete").write_text(str(STEPS))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
