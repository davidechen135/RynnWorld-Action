#!/usr/bin/env python3
"""Gate D run_016: 20,000-step training on 4x run_015's window volume, framework verbatim from run_011.

run_015 validated the 5000-seeded-window recipe on the full Gate D manifest (48,800 unique
full-range windows across 1,162 train episodes), applying run_011's two-fix recipe unchanged.
This run scales the same seed=42 permutation to the first 20,000 windows (41% of the 48,800
train pool, touching all 1,162 train episodes at least once) for 20,000 training steps. The
first 5,000 elements of the permutation are identical to run_015's selection, so their cached
latents are reused verbatim (copied from run_015's cache directory) rather than re-encoded.

Two changes vs run_015, both data-volume-specific, neither touching the algorithm/framework:
  - STEPS = 20000 (was 5000); MILESTONES every 5000 steps (was every 1000).
  - CACHE moved to /mnt/workspace (network mount) since 20,002 encoded windows (~1.6TB) exceed
    the rootfs's free space; run_015's cache lived on rootfs /tmp/scratch.

Held identical to run_015/run_011: architecture (hidden_dim=768, num_layers=2, num_heads=12,
16.7M params), LR=5e-4, seed, gradient clip 1.0, corrected timestep, per-task action
normalization (stats.per_task["3400"]/["3401"]), fixed validation (same two activity-peak
windows), and the rollout protocol.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
RUN = REPO / "reports/direct_action/gate_d/run_016"
MANIFEST = RUN / "artifacts/window_manifest_full_dataset.json"
FIXED_VALIDATION = RUN / "artifacts/fixed_validation_windows.json"
STATS_CACHE = RUN / "artifacts/action_stats_combined.json"  # policy=per_task
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
CACHE = Path("/mnt/workspace/scratch/gate_d_run016_cache")  # network mount: 20,002 windows ~1.6TB, rootfs too small
DEST_ROOT = Path("/mnt/workspace/agibot_extracted")  # top_head extraction root (== parquet root)

WINDOW = 33
HEIGHT, WIDTH = 480, 832
SEED = 42
SHUFFLE_SEED = 314159
LR = 5e-4  # unchanged from run_011/run_015
STEPS = 20000
MILESTONES = (5000, 10000, 15000, 20000)
FIXED_VALIDATION_TIMESTEP = 500
ROLLOUT_STEPS = 20
CFG_SCALE = 1.0
ENCODER_KWARGS = dict(hidden_dim=768, num_layers=2, num_heads=12)  # identical to run_008/009/010/011/015

EXPECT_UNIQUE = 48800
EXPECT_TRAIN_EPISODES = 1162
EXPECT_STEPS_EPISODES = 1162  # distinct episodes among the first 20,000 seeded windows (computed offline)


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


def build_video_path(record: dict) -> str:
    return str(DEST_ROOT / f"task_{record['task_id']}" / record["shard"] / record["video_relpath"])


def select_scaled(manifest: dict) -> list[dict]:
    """Same seed=42 permutation as run_015's select_5000, extended to STEPS=20000 windows.
    The first 5000 elements of this list are identical to run_015's selection (same pool,
    same permutation, same prefix) so their cached latents can be reused without re-encoding."""
    records = [record for record in manifest["records"] if record["split"] == "train"]
    unique = []
    for record in sorted(records, key=lambda item: item["episode"]):
        video = build_video_path(record)
        for window in record["windows"]:
            unique.append({
                "episode": record["episode"], "task_id": record["task_id"], "start": window["start"],
                "stratum": window["stratum"], "fraction": window["fraction"],
                "video": video, "parquet": record["parquet"],
            })
    assert len(unique) == EXPECT_UNIQUE
    assert len({u["episode"] for u in unique}) == EXPECT_TRAIN_EPISODES
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
    assert len({sample["episode"] for sample in selected}) == EXPECT_STEPS_EPISODES
    assert len({sample["unique_window_index"] for sample in selected}) == STEPS  # 20000 < 48800, no wrap-around
    return selected


def resolve_global_episode(manifest: dict, task_id: str, shard: str, real_episode: int) -> int:
    for key, info in manifest["global_episode_lookup"].items():
        if info["task_id"] == task_id and info["shard"] == shard and info["real_episode"] == real_episode:
            return int(key)
    raise KeyError(f"no global episode for task={task_id} shard={shard} real_episode={real_episode}")


def select_active_held_out(manifest: dict) -> list[dict]:
    specs = json.loads(FIXED_VALIDATION.read_text())
    out = []
    for task_id, spec in specs.items():
        episode = resolve_global_episode(manifest, spec["task_id"], spec["shard_name"], spec["episode_index"])
        record = next(r for r in manifest["records"] if r["episode"] == episode)
        assert record["split"] == "held_out", f"expected held_out, got {record['split']}"
        out.append({
            "episode": episode, "task_id": record["task_id"], "start": spec["start_frame"],
            "activity": spec["activity"], "video": build_video_path(record), "parquet": record["parquet"],
        })
    return out


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
    official_rollout already uses for inference. Verbatim from run_010/run_011/run_015."""
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
            audit.save_video(videos[name], run_dir / "videos" / f"step{step:04d}_task{validation['task_id']}_{name}.mp4", 30)
    source_first = ((validation["raw_video"][0, :, 0].float().permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    metrics = audit.video_metrics(videos, source_first)
    metrics.update({
        "global_step": step,
        "embedding_distances": {name: float(torch.linalg.vector_norm(embeddings["correct"] - embeddings[name])) for name in conditions if name != "correct"},
        "injection": {name: {"norm": captures[name]["adapter_injection_norm"]} for name in conditions},
        "target_metrics": target_metrics(videos, validation["raw_video"][0].float()),
    })
    (run_dir / "artifacts" / f"rollout_metrics_step{step:04d}_task{validation['task_id']}.json").write_text(json.dumps(metrics, indent=2))
    encoder.train()
    return metrics


def main():
    lock = RUN / "training.lock"
    if lock.exists():
        raise RuntimeError(f"lock exists: {lock.read_text().strip()}")
    lock.write_text(str(os.getpid()))
    (RUN / "checkpoints").mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    (RUN / "videos").mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(MANIFEST.read_text())
        stats = json.loads(STATS_CACHE.read_text())
        assert stats["policy"] == "per_task", f"expected per_task policy, got {stats['policy']}"
        per_task = stats["stats"]["per_task"]
        print(f"[action_stats] policy={stats['policy']} tasks={ {t: d['n_rows'] for t, d in per_task.items()} }", flush=True)

        selected = select_scaled(manifest)
        validation_specs = select_active_held_out(manifest)
        (RUN / "artifacts/selected_20000_scaled.json").write_text(json.dumps(selected, indent=2))
        (RUN / "artifacts/fixed_validation_window.json").write_text(json.dumps(validation_specs, indent=2))

        helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_gated")
        audit = load_module(AUDIT_PATH, "gate_c_protocol_gated")
        device, dtype = torch.device("cuda"), torch.bfloat16
        helpers.apply_monkey_patch(REPO)
        tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
        prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
        del tokenizer, text_encoder
        torch.cuda.empty_cache()
        vae, mean, std = helpers.load_vae(BASE_MODEL, device, dtype)
        cache_paths = [cache_sample(sample, "train", helpers, vae, mean, std, device, dtype) for sample in selected]
        validation_paths = [cache_sample(spec, "fixed_validation", helpers, vae, mean, std, device, dtype) for spec in validation_specs]
        validations = []
        for spec, path in zip(validation_specs, validation_paths):
            sample = torch.load(path, map_location="cpu")
            sample["task_id"] = spec["task_id"]
            validations.append(sample)

        model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
        from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
        encoder = NativeTrajectoryEncoder(input_dim=33, **ENCODER_KWARGS).to(device)
        param_count = sum(p.numel() for p in encoder.parameters())
        model.native_trajectory_encoder = encoder
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
        timestep_rng = np.random.default_rng(SEED + 3)

        per_task_mean = {t: torch.tensor(d["mean"], device=device, dtype=torch.float32) for t, d in per_task.items()}
        per_task_std = {t: torch.tensor(d["std"], device=device, dtype=torch.float32) for t, d in per_task.items()}
        task_label = {t: label for t, label in zip(sorted(per_task.keys()), ["A", "B", "C", "D"])}

        provenance = {
            "run_id": "direct_action/gate_d/run_016",
            "purpose": "20,000-step training on 4x run_015's window volume (41% of the 48,800-window full Gate D train pool, all 1,162 train episodes touched), same seed=42 permutation and recipe as run_015",
            "changed_vs_run_015": ["data: 20,000 seeded draws from the same 48,800-window pool via the same seed=42 permutation (was: 5,000 draws); first 5,000 elements identical, cache reused", "milestones every 5000 steps (was: every 1000 steps)", "cache directory moved to network mount /mnt/workspace (was: rootfs /tmp/scratch) -- 20,002 cached windows exceed rootfs free space"],
            "held_fixed_vs_run_015": ["architecture (hidden_dim=768,num_layers=2,num_heads=12, 16.7M params)", "lr=5e-4", "corrected_timestep_signal", "seed=42", "gradient_clip_norm=1.0", "rollout protocol", "per-task action normalization", "fixed validation windows (same two activity-peak windows, one per task)"],
            "run_015_result_for_comparison": "see reports/direct_action/gate_d/run_015/artifacts/provenance.json",
            "run_011_result_for_comparison": "see reports/direct_action/gate_c/run_011/artifacts/provenance.json",
            "trained_from_scratch": True, "lr": LR, "gradient_clip_norm": 1.0,
            "steps": STEPS, "seed": SEED, "fixed_validation_timestep": FIXED_VALIDATION_TIMESTEP,
            "validation": validation_specs, "data_manifest": str(MANIFEST),
            "action_stats": {"policy": stats["policy"], "per_task": {t: {"n_rows": d["n_rows"]} for t, d in per_task.items()}, "stats_file": str(STATS_CACHE)},
            "adapter_param_count": param_count,
        }
        (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))

        for index, path in enumerate(cache_paths, 1):
            sample = torch.load(path, map_location="cpu")
            latent = sample["video_latent"].to(device)
            image = sample["img_latent"].to(device)
            action_raw = sample["robot_trajectory"].to(device=device, dtype=torch.float32)
            task_id = selected[index - 1]["task_id"]
            action = normalize(action_raw, per_task_mean[task_id], per_task_std[task_id])
            timestep = int(timestep_rng.integers(0, 1000))
            loss = compute_flow_matching_loss_corrected(model, encoder, latent, image, action, prompt, timestep, SEED)
            optimizer.zero_grad(); loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                fixed_losses = {}
                for validation, spec in zip(validations, validation_specs):
                    vtask = spec["task_id"]
                    validation_action = normalize(validation["robot_trajectory"].to(device=device, dtype=torch.float32), per_task_mean[vtask], per_task_std[vtask])
                    fixed_losses[vtask] = float(compute_flow_matching_loss_corrected(
                        model, encoder, validation["video_latent"].to(device), validation["img_latent"].to(device),
                        validation_action, prompt, FIXED_VALIDATION_TIMESTEP, SEED))
            step = index
            record = {"global_step": step, "loss": float(loss), "fixed_validation_loss": fixed_losses,
                      "grad_norm": float(grad_norm), "episode": sample["episode"], "task_id": task_id,
                      "start": sample["start_frame"], "timestep": timestep}
            with (RUN / "logs/training.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if index % 250 == 0 or step in MILESTONES:
                val_str = ",".join(f"{task_label[t]}={v:.4f}" for t, v in fixed_losses.items())
                print(f"[gate_d] step={step} loss={float(loss):.6f} val={val_str} grad={float(grad_norm):.4f}", flush=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            if step in MILESTONES:
                checkpoint = RUN / "checkpoints" / f"adapter_step{step:04d}.pt"
                torch.save(encoder.state_dict(), checkpoint)
                milestone_rollouts = []
                for validation, spec in zip(validations, validation_specs):
                    vtask = spec["task_id"]
                    rollout = evaluate(audit, helpers, model, encoder, vae, mean, std, prompt, validation,
                                       per_task_mean[vtask], per_task_std[vtask], step, device, RUN)
                    milestone_rollouts.append({
                        "task_id": vtask, "episode": validation["episode"],
                        "all_first_frames_equal": all(rollout["first_frame_equal_to_correct"].values()),
                        "embedding_distances": rollout["embedding_distances"],
                        "target_metrics": rollout["target_metrics"],
                    })
                provenance.setdefault("milestones", []).append({
                    "step": step, "checkpoint": str(checkpoint), "rollouts": milestone_rollouts,
                })
                (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
                print(f"[gate_d] === milestone step={step} rolled {len(validations)} val episodes ===", flush=True)
        (RUN / "artifacts/provenance.json").write_text(json.dumps(provenance, indent=2))
        (RUN / "training.complete").write_text(str(STEPS))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
