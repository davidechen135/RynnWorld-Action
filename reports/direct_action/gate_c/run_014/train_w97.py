#!/usr/bin/env python3
"""Gate C run_014: window length ablation -- WINDOW=97 (~3.2s @30fps) vs run_011's WINDOW=33.

Follow-up to run_012/013, which showed the model's generated motion is nearly identical
across correct/zero/shuffled/reversed/wrong action conditioning at BOTH WINDOW=33 and
WINDOW=97 (inference-time only, using run_011's WINDOW=33-trained checkpoint). The user then
asked whether *training* itself should use a longer window (~5s), and specifically whether to
resample with small overlapping strides (e.g. start=1, start=30). Analysis of the 91 train
episodes' actual lengths (min=699, median=1454, max=4755 frames) showed that forcing a fixed
16-windows/episode target at WINDOW=149 (~5s) would require ~42% median / 75% max adjacent
overlap -- and that an ad hoc small fixed stride (like start=1, start=30) would reproduce
exactly the front-loaded-coverage bug run_005 already fixed. WINDOW=97 is a gentler version of
the same idea (already validated end-to-end at inference in run_013): reusing run_005's exact
stratified_starts() method (STRIDE=16 eligible grid, up to 16 windows/episode picked evenly by
index) at WINDOW=97 yields the SAME 1456 total train windows as WINDOW=33, with only 18%
median / 67% max worst-case adjacent overlap (35/91 episodes have zero overlap) -- so no ad hoc
resampling was needed, just rebuilding the manifest at the new window length.

Trained from scratch (fresh encoder init), matching run_011's own precedent for introducing a
new independent variable: mixing a WINDOW=33-trained checkpoint's gradients with WINDOW=97
batches would confound the window-length ablation. Everything else held fixed vs run_011:
same architecture (16.7M-param encoder), same LR=5e-4, same action-normalization stats
(run_009), same corrected timestep signal (run_010) -- both already frame-count-agnostic by
construction (compute_flow_matching_loss_corrected uses n_f = video_latent.shape[2] dynamically;
NativeTrajectoryEncoder.forward() interpolates to whatever output_frames the latent implies).

STEPS reduced from run_011's 5000 to 2500: WINDOW=97 is ~2.94x longer than WINDOW=33, and
run_012 vs run_013's wall-clock (single 5-condition rollout eval) went from ~2min to ~6.5min
(~3x), so training compute cost is expected to scale similarly per step. 2500 steps still
cycles through all 1456 unique windows at least once (matching run_011's cycling method).

Fixed validation window changed from run_011's activity-heuristic pick (episode 11, start=1012,
motion score ~13.2) to the high-motion window identified and already used in run_012/013
(episode 40, start=2477, motion score 25.8) -- chosen for the same reason run_012/013 used it:
a low-motion validation window makes it hard to visually/quantitatively tell whether the model
is following action content at all.
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
RUN = REPO / "reports/direct_action/gate_c/run_014"
MANIFEST = RUN / "artifacts/window_manifest_full_range_w97.json"
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
CACHE = Path("/tmp/scratch/gate_c_run014_cache")  # new cache: WINDOW=97 tensors differ in shape from run_006/011's WINDOW=33 cache
STATS_CACHE = REPO / "reports/direct_action/gate_c/run_009/artifacts/action_stats.json"  # reused: stats are per-dimension, window-length independent

WINDOW = 97
HEIGHT, WIDTH = 480, 832
SEED = 42
SHUFFLE_SEED = 314159
LR = 5e-4  # unchanged from run_011: isolate window-length as the only new variable
STEPS = 2500
MILESTONES = (500, 1000, 1500, 2000, 2500)
FIXED_VALIDATION_TIMESTEP = 500
ROLLOUT_STEPS = 20
CFG_SCALE = 1.0
ENCODER_KWARGS = dict(hidden_dim=768, num_layers=2, num_heads=12)  # unchanged from run_008/009/010/011

VALIDATION_EPISODE = 40
VALIDATION_START = 2477


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


def select_windows(manifest: dict, steps: int) -> list[dict]:
    records = [record for record in manifest["records"] if record["split"] == "train"]
    unique = []
    for record in sorted(records, key=lambda item: item["episode"]):
        for window in record["windows"]:
            unique.append({
                "episode": record["episode"], "start": window["start"], "stratum": window["stratum"],
                "fraction": window["fraction"], "video": record["video"], "parquet": record["parquet"],
            })
    assert len(unique) == 1456, f"expected 1456 unique WINDOW={WINDOW} train windows, got {len(unique)}"
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(unique)).tolist()
    selected = []
    for index in range(steps):
        cycle, position = divmod(index, len(order))
        sample = dict(unique[order[position]])
        sample["cycle"] = cycle
        sample["unique_window_index"] = order[position]
        selected.append(sample)
    assert len(selected) == steps
    assert len({sample["episode"] for sample in selected}) == 91
    return selected


def select_validation(manifest: dict) -> dict:
    record = next(r for r in manifest["records"] if r["split"] == "held_out" and r["episode"] == VALIDATION_EPISODE)
    assert VALIDATION_START + WINDOW <= record["video_frames"], "validation window out of bounds"
    return {"episode": VALIDATION_EPISODE, "start": VALIDATION_START, "video": record["video"], "parquet": record["parquet"]}


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
    """Verbatim from run_010/011 -- already frame-count-agnostic (n_f derived from video_latent.shape)."""
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
            "source_flow_magnitude_mean": float(np.mean(target_flow)),
            "generated_flow_magnitude_mean": float(np.mean(actual_flow)),
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
        selected = select_windows(manifest, STEPS)
        validation_spec = select_validation(manifest)
        (RUN / "artifacts/selected_windows_w97.json").write_text(json.dumps(selected, indent=2))
        (RUN / "artifacts/fixed_validation_window.json").write_text(json.dumps(validation_spec, indent=2))

        helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_run014")
        audit = load_module(AUDIT_PATH, "gate_c_protocol_run014")
        device, dtype = torch.device("cuda"), torch.bfloat16
        helpers.apply_monkey_patch(REPO)
        tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
        prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
        del tokenizer, text_encoder
        torch.cuda.empty_cache()
        vae, mean, std = helpers.load_vae(BASE_MODEL, device, dtype)
        cache_paths = [cache_sample(sample, "train_w97", helpers, vae, mean, std, device, dtype) for sample in selected]
        validation_path = cache_sample(validation_spec, "fixed_validation_w97", helpers, vae, mean, std, device, dtype)
        validation = torch.load(validation_path, map_location="cpu")

        model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
        # WINDOW=97 first attempt OOM'd (CUDA OOM, ~79.15GiB allocated, 268MiB requested) during the
        # first training-step forward/backward -- 97 raw frames -> 25 latent frames (vs 9 at window=33)
        # means substantially larger activation memory to retain for backprop through the frozen 5B
        # backbone. enable_gradient_checkpointing() trades recompute for memory (standard diffusers API,
        # gated on `torch.is_grad_enabled() and self.gradient_checkpointing` in WanTransformer3DModel.forward,
        # confirmed NOT gated on self.training, so this is safe despite model.eval() below/above).
        model.enable_gradient_checkpointing()
        from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
        encoder = NativeTrajectoryEncoder(input_dim=33, **ENCODER_KWARGS).to(device)
        param_count = sum(p.numel() for p in encoder.parameters())
        model.native_trajectory_encoder = encoder
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
        timestep_rng = np.random.default_rng(SEED + 3)

        mean_t = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
        std_t = torch.tensor(stats["std"], device=device, dtype=torch.float32)

        provenance = {
            "run_id": "direct_action/gate_c/run_014",
            "purpose": "window-length ablation: WINDOW=97 (~3.2s) vs run_011's WINDOW=33 (~1.1s), all else held fixed",
            "changed_vs_run_011": [
                "window=97 raw frames (was 33)",
                "new stratified manifest at window=97 via run_005's exact stratified_starts method (STRIDE=16, 16/episode) -- still 1456 unique train windows, median 18%/max 67% adjacent overlap (was: 0% overlap at window=33)",
                "steps=2500 (was 5000; reduced for expected ~3x per-step compute cost at 97 vs 33 frames)",
                "fixed validation window = episode 40, start=2477 (motion score 25.8; was: episode 11, start=1012, motion score ~13.2, picked by an activity heuristic over episode 11 only)",
                "transformer.enable_gradient_checkpointing() added (was: not needed at window=33) -- first attempt at window=97 without it OOM'd (79.15/79.25GiB allocated, ~87MiB free) during the first training step's forward/backward through the frozen 5B backbone; cache-building (all 1457 samples) had already completed successfully before that crash, so this run reuses the existing cache directory unchanged",
            ],
            "held_fixed_vs_run_011": ["architecture (hidden_dim=768,num_layers=2,num_heads=12, 16.7M params)", "lr=5e-4", "action_normalization_stats (run_009)", "corrected_timestep_signal (run_010)", "seed", "gradient_clip_norm=1.0"],
            "run_011_result_for_comparison": "best checkpoint step3000: target SSIM=0.813/LPIPS=0.138 on episode11/start1012 (low motion); but run_012/013 (post-hoc rollout audits at window=33 and window=97 using this checkpoint) found correct/zero/shuffled/reversed/wrong nearly indistinguishable (SSIM/LPIPS within ~0.001-0.02) on the high-motion episode40/start2477 window at both window lengths, and generated motion only ~28-38% of real motion magnitude -- action conditioning appears to not meaningfully affect output",
            "motivating_question": "does training on longer windows change whether the model's output actually responds to action content, or does the same near-total action-insensitivity persist regardless of window length?",
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
                print(f"[run_014] step={step} loss={float(loss):.6f} val={float(fixed_loss):.6f} grad={float(grad_norm):.4f}", flush=True)
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
                print(f"[run_014] === milestone step={step} target_metrics(correct)={rollout['target_metrics']['correct']} ===", flush=True)
        (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
        (RUN / "training.complete").write_text(str(STEPS))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
