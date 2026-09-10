#!/usr/bin/env python3
"""Gate C fixed-protocol four-way rollout audit.

All controls share the same source frame, prompt, initial noise, scheduler,
CFG, steps, resolution, and duration. Only the 33D action tensor changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
SCRIPT = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
SOURCE_VIDEO = Path("/tmp/scratch/gate_c_video/data/videos/chunk-000/observation.images.top_head/episode_000008.mp4")
SOURCE_PARQUET = Path("/tmp/scratch/gate_b_task3400/data/data/chunk-000/episode_000008.parquet")
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CKPT = REPO / "pretrained/RynnWorld-Teleop-Causal"
CHECKPOINT = REPO / "reports/direct_action/gate_c/run_003/checkpoints/adapter_step1000.pt"
OUT = REPO / "reports/direct_action/gate_c/run_003_fixed_protocol_v4"
SEED = 42
SHUFFLE_SEED = 314159
STEPS = 20
CFG_SCALE = 1.0
HEIGHT, WIDTH, MAX_FRAMES = 480, 832, 33


def sha256_tensor(x: torch.Tensor) -> str:
    raw = x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def stats(x: torch.Tensor) -> dict:
    y = x.detach().float()
    return {"shape": list(y.shape), "mean": float(y.mean()), "std": float(y.std()), "norm": float(y.norm()), "min": float(y.min()), "max": float(y.max())}


def load_namespace():
    import argparse as ap
    old = ap.ArgumentParser.parse_args
    ap.ArgumentParser.parse_args = lambda self, args=None, namespace=None: ap.Namespace()
    ns = {"__file__": str(SCRIPT), "__name__": "gate_c_visual_v2_audit"}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), ns)
    ap.ArgumentParser.parse_args = old
    return ns


def official_rollout(model, encoder, vae, img_latent, action, prompt_embedding,
                     latents_mean, latents_std, initial_noise, steps, cfg_scale,
                     capture: dict) -> torch.Tensor:
    from diffusers import UniPCMultistepScheduler

    device = img_latent.device
    dtype = model.patch_embedding.weight.dtype
    b, c, _, h, w = img_latent.shape
    latents = initial_noise.clone()
    n_f = latents.shape[2]
    p_t, p_h, p_w = model.config.patch_size
    first_mask = torch.ones((b, 1, n_f, h, w), device=device, dtype=dtype)
    first_mask[:, :, 0] = 0
    condition = img_latent

    scheduler = UniPCMultistepScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    scheduler.set_timesteps(steps, device=device)
    capture["scheduler_class"] = scheduler.__class__.__name__
    capture["scheduler_timesteps"] = [int(x) for x in scheduler.timesteps.detach().cpu().tolist()]

    # Record the exact adapter injection tensor for this condition.
    with torch.no_grad():
        post_patch_frames = n_f // p_t
        native = encoder(action, post_patch_frames).to(dtype)
        capture["native_features"] = native.detach().float().cpu()
        capture["adapter_input"] = action.detach().float().cpu()

    def run_model(model_input, t):
        # Official Wan TI2V convention: frame 0 has timestep zero.
        t_frame = (first_mask[0, 0, :, ::p_h, ::p_w] * t).flatten().unsqueeze(0)
        return model(
            hidden_states=model_input,
            timestep=t_frame,
            encoder_hidden_states=prompt_embedding,
            encoder_hidden_states_image=None,
            robot_trajectory=action,
            null_condition=False,
        ).sample

    with torch.no_grad():
        for t in scheduler.timesteps:
            model_input = (1 - first_mask) * condition + first_mask * latents
            pred = run_model(model_input, t)
            if cfg_scale != 1.0:
                # Kept explicit so every control has the same CFG policy.
                uncond = model(
                    hidden_states=model_input,
                    timestep=(first_mask[0, 0, :, ::p_h, ::p_w] * t).flatten().unsqueeze(0),
                    encoder_hidden_states=prompt_embedding,
                    encoder_hidden_states_image=None,
                    robot_trajectory=action,
                    null_condition=False,
                ).sample
                pred = uncond + cfg_scale * (pred - uncond)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            latents = (1 - first_mask) * condition + first_mask * latents

    capture["first_frame_latent_hash"] = sha256_tensor(latents[:, :, :1])
    capture["first_frame_mask_zero_count"] = int((first_mask[:, :, 0] == 0).sum())
    capture["adapter_injection_norm"] = float(capture["native_features"].norm())
    capture["adapter_injection_mean_abs"] = float(capture["native_features"].abs().mean())
    video = ns_decode(vae, latents, latents_mean, latents_std)
    return video[0].detach().float().cpu()


# Set by main after namespace loading; keeps official helper use explicit.
ns_decode = None


def read_video(path: Path):
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")
    return frames


def save_video(frames: torch.Tensor, path: Path, fps: int):
    import imageio
    x = ((frames.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    imageio.mimsave(str(path), x, fps=fps)


def video_metrics(videos: dict[str, torch.Tensor], first_source: np.ndarray):
    from skimage.metrics import structural_similarity
    names = list(videos)
    arrays = {k: ((v.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8) for k, v in videos.items()}
    first_hashes = {k: hashlib.sha256(a[0].tobytes()).hexdigest() for k, a in arrays.items()}
    first_equal = {k: bool(np.array_equal(a[0], arrays[names[0]][0])) for k, a in arrays.items()}
    out = {"first_frame_sha256": first_hashes, "first_frame_equal_to_correct": first_equal, "pairwise": {}, "per_video": {}}
    for name, a in arrays.items():
        gray = np.dot(a[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        flow = []
        for i in range(1, len(gray)):
            flow.append(float(np.mean(np.abs(gray[i] - gray[i - 1]))))
        ssim_source = float(structural_similarity(first_source, a[0], channel_axis=2, data_range=255))
        out["per_video"][name] = {"shape": list(a.shape), "frame_count": int(len(a)), "mean_abs_frame_delta": float(np.mean(flow)), "std_abs_frame_delta": float(np.std(flow)), "optical_flow_proxy_mean": float(np.mean(flow)), "optical_flow_proxy_std": float(np.std(flow)), "ssim_to_source_first_frame": ssim_source}
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            a, b = arrays[a_name].astype(np.float32), arrays[b_name].astype(np.float32)
            out["pairwise"][f"{a_name}__{b_name}"] = {"mean_abs_pixel_distance": float(np.mean(np.abs(a - b))), "rmse": float(np.sqrt(np.mean((a - b) ** 2))), "ssim_mean": float(np.mean([structural_similarity(a[j], b[j], channel_axis=2, data_range=255) for j in range(len(a))]))}
    try:
        import lpips
        lp = lpips.LPIPS(net="alex").eval()
        for i, a_name in enumerate(names):
            for b_name in names[i + 1:]:
                xa = torch.from_numpy(arrays[a_name]).permute(0, 3, 1, 2).float() / 127.5 - 1
                xb = torch.from_numpy(arrays[b_name]).permute(0, 3, 1, 2).float() / 127.5 - 1
                vals = []
                for start in range(0, len(xa), 4):
                    vals.extend(lp(xa[start:start + 4], xb[start:start + 4]).view(-1).tolist())
                out["pairwise"][f"{a_name}__{b_name}"]["lpips_mean"] = float(np.mean(vals))
    except Exception as exc:
        out["lpips_error"] = repr(exc)
    return out


def main():
    global ns_decode
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "videos").mkdir()
    (OUT / "artifacts").mkdir()
    ns = load_namespace()
    ns_decode = ns["decode_latents_to_video"]
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    ns["apply_monkey_patch"](REPO)
    tokenizer, text_encoder = ns["load_text_encoder"](BASE_MODEL, device, dtype)
    prompt = ""
    prompt_embedding = ns["encode_text"](tokenizer, text_encoder, prompt, device)
    vae, latents_mean, latents_std = ns["load_vae"](BASE_MODEL, device, dtype)
    frames, metadata = ns["load_and_decode_video"](SOURCE_VIDEO, max_frames=MAX_FRAMES)
    pre = ns["preprocess_video_frames"](frames, HEIGHT, WIDTH).unsqueeze(0).to(device, dtype=dtype)
    video_latent = ns["encode_video_frames"](vae, pre, latents_mean, latents_std)
    img_latent = video_latent[:, :, :1].clone()
    _, actions_np = ns["load_episode_data"](SOURCE_PARQUET, metadata)
    correct = torch.from_numpy(actions_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    zero = torch.zeros_like(correct)
    g = torch.Generator(device=device).manual_seed(SHUFFLE_SEED)
    perm = torch.randperm(correct.shape[1], generator=g, device=device)
    shuffled = correct[:, perm]
    wrong = torch.flip(correct, dims=[1])
    conditions = {"correct": correct, "zero": zero, "shuffled": shuffled, "wrong": wrong}

    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    encoder = NativeTrajectoryEncoder(input_dim=33).to(device)
    state = torch.load(CHECKPOINT, map_location="cpu")
    encoder.load_state_dict(state)
    encoder.eval()
    model = ns["load_model"](RYNN_CKPT, BASE_MODEL, sys.stdout)
    model.native_trajectory_encoder = encoder
    adapter_attached = model.native_trajectory_encoder is encoder
    initial_noise = torch.randn((1, video_latent.shape[1], video_latent.shape[2], video_latent.shape[3], video_latent.shape[4]), device=device, dtype=model.patch_embedding.weight.dtype)

    provenance = {"checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "checkpoint_filename_global_step": 1000, "checkpoint_actual_global_step": 3000, "adapter_attached_identity": adapter_attached, "adapter_param_count": sum(p.numel() for p in encoder.parameters()), "source_video": str(SOURCE_VIDEO), "source_parquet": str(SOURCE_PARQUET), "prompt": prompt, "seed": SEED, "shuffle_seed": SHUFFLE_SEED, "cfg_scale": CFG_SCALE, "inference_steps": STEPS, "resolution": [HEIGHT, WIDTH], "duration_frames": MAX_FRAMES, "fps": int(metadata["fps"]), "video_latent_shape": list(video_latent.shape), "img_latent_shape": list(img_latent.shape), "initial_noise_sha256": sha256_tensor(initial_noise), "initial_noise_shape": list(initial_noise.shape), "text_embedding_shape": list(prompt_embedding.shape), "text_embedding_sha256": sha256_tensor(prompt_embedding)}
    (OUT / "artifacts" / "provenance.json").write_text(json.dumps(provenance, indent=2))

    action_info, embeddings, captures, videos = {}, {}, {}, {}
    with torch.no_grad():
        for name, action in conditions.items():
            action_info[name] = {"sha256": sha256_tensor(action), **stats(action)}
            emb = encoder(action, video_latent.shape[2] // model.config.patch_size[0])
            embeddings[name] = emb.detach().float().cpu()
            captures[name] = {}
            videos[name] = official_rollout(model, encoder, vae, img_latent, action, prompt_embedding, latents_mean, latents_std, initial_noise, STEPS, CFG_SCALE, captures[name])
            save_video(videos[name], OUT / "videos" / f"step3000_{name}.mp4", int(metadata["fps"]))
    distances = {a: {b: float(torch.linalg.vector_norm(embeddings[a] - embeddings[b])) for b in embeddings if b != a} for a in embeddings}
    injection = {name: {"norm": captures[name]["adapter_injection_norm"], "mean_abs": captures[name]["adapter_injection_mean_abs"], "delta_vs_zero_norm": float(torch.linalg.vector_norm(embeddings[name] - embeddings["zero"]))} for name in captures}
    source_first = ((pre[0, :, 0].float().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    metrics = video_metrics(videos, source_first)
    metrics.update({"action_tensors": action_info, "trajectory_embedding_distances": distances, "per_injection_layer_activation_deltas": {"post_patch_single_injection": injection}, "captures": {k: {x: v for x, v in c.items() if x not in ("native_features", "adapter_input")} for k, c in captures.items()}, "pass": bool(all(metrics["first_frame_equal_to_correct"].values()) and metrics["per_video"]["correct"]["mean_abs_frame_delta"] > 0 and all(v > 0 for v in distances["correct"].values())), "pass_note": "Automated pass is structural/motion-only; visual review remains required."})
    (OUT / "artifacts" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "artifacts" / "config.json").write_text(json.dumps({"protocol": "official_clean_first_frame_mask", "conditions": list(conditions), "only_variable": "33D_action_tensor", "scheduler": "UniPCMultistepScheduler", "cfg_scale": CFG_SCALE, "steps": STEPS, "seed": SEED, "initial_noise_shared": True}, indent=2))
    print(json.dumps({"output": str(OUT), "pass": metrics["pass"], "checkpoint_actual_global_step": 3000, "videos": {k: str(OUT / "videos" / f"step3000_{k}.mp4") for k in videos}}, indent=2))


if __name__ == "__main__":
    main()
