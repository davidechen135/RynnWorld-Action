#!/usr/bin/env python3
"""Gate C run_012: high-motion held-out rollout audit for run_011's checkpoints.

All prior five-condition evaluations (run_004 through run_011) used the same single
fixed held-out window (episode 11, start=1012), chosen by an activity-score heuristic
over episode 11 only. That window's motion is modest. This script scores ALL held-out
episodes' windows by total action variation and picks a genuinely high-motion window
(episode 40, start=2477, motion score 25.8 -- roughly 2x episode 11/1012's ~13.2) so the
rollout shows visibly larger movement, using the same official_rollout protocol (shared
first frame per checkpoint, only the 33D action tensor differs across conditions).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
RUN = REPO / "reports/direct_action/gate_c/run_012"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
STATS_CACHE = REPO / "reports/direct_action/gate_c/run_009/artifacts/action_stats.json"
ADAPTER = REPO / "reports/direct_action/gate_c/run_011/checkpoints/adapter_step3000.pt"
CACHE = Path("/tmp/scratch/gate_c_run012_cache")

WINDOW = 33
HEIGHT, WIDTH = 480, 832
SEED = 42
SHUFFLE_SEED = 314159
ROLLOUT_STEPS = 20
CFG_SCALE = 1.0
ENCODER_KWARGS = dict(hidden_dim=768, num_layers=2, num_heads=12)

HIGH_MOTION_EPISODE = 40
HIGH_MOTION_START = 2477
HIGH_MOTION_SCORE = 25.8


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


def find_record(manifest: dict, episode: int) -> dict:
    return next(r for r in manifest["records"] if r["split"] == "held_out" and r["episode"] == episode)


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
        "robot_trajectory": torch.from_numpy(actions[start:start + WINDOW].copy()).unsqueeze(0),
        "raw_video": video_t.cpu(), "episode": episode, "start_frame": start,
    }, path)
    return path


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


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "videos").mkdir(exist_ok=True)
    (RUN / "artifacts").mkdir(exist_ok=True)

    manifest = json.loads((REPO / "reports/direct_action/gate_c/run_005/artifacts/window_manifest_full_range.json").read_text())
    stats = json.loads(STATS_CACHE.read_text())
    record = find_record(manifest, HIGH_MOTION_EPISODE)

    helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_run012")
    audit = load_module(AUDIT_PATH, "gate_c_protocol_run012")
    device, dtype = torch.device("cuda"), torch.bfloat16
    helpers.apply_monkey_patch(REPO)
    tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
    prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    vae, mean, std = helpers.load_vae(BASE_MODEL, device, dtype)

    sample_path = cache_sample(HIGH_MOTION_EPISODE, HIGH_MOTION_START, record["video"], record["parquet"], "highmotion", helpers, vae, mean, std, device, dtype)
    sample = torch.load(sample_path, map_location="cpu")

    model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    encoder = NativeTrajectoryEncoder(input_dim=33, **ENCODER_KWARGS).to(device)
    encoder.load_state_dict(torch.load(ADAPTER, map_location="cpu"))
    encoder.eval()
    model.native_trajectory_encoder = encoder

    mean_t = torch.tensor(stats["mean"], device=device, dtype=torch.float32)
    std_t = torch.tensor(stats["std"], device=device, dtype=torch.float32)

    latent = sample["video_latent"].to(device)
    image = sample["img_latent"].to(device)
    correct_raw = sample["robot_trajectory"].to(device=device, dtype=torch.float32)
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
    with torch.no_grad():
        for name, action in conditions.items():
            captures[name] = {}
            embeddings[name] = encoder(action, latent.shape[2] // model.config.patch_size[0]).float().cpu()
            videos[name] = audit.official_rollout(model, encoder, vae, image, action, prompt, mean, std, initial_noise, ROLLOUT_STEPS, CFG_SCALE, captures[name])
            audit.save_video(videos[name], RUN / "videos" / f"{name}.mp4", 30)

    source_first = ((sample["raw_video"][0, :, 0].float().permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    metrics = audit.video_metrics(videos, source_first)
    metrics.update({
        "checkpoint": str(ADAPTER),
        "episode": HIGH_MOTION_EPISODE, "start_frame": HIGH_MOTION_START, "motion_score": HIGH_MOTION_SCORE,
        "reference_episode11_start1012_motion_score": 13.236601829528809,
        "embedding_distances": {name: float(torch.linalg.vector_norm(embeddings["correct"] - embeddings[name])) for name in conditions if name != "correct"},
        "injection": {name: {"norm": captures[name]["adapter_injection_norm"]} for name in conditions},
        "target_metrics": target_metrics(videos, sample["raw_video"][0].float()),
    })
    (RUN / "artifacts/rollout_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"target_metrics_correct": metrics["target_metrics"]["correct"], "embedding_distances": metrics["embedding_distances"]}, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
