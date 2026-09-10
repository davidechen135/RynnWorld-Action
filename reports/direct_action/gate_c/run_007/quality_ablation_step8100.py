#!/usr/bin/env python3
"""Quality ablation at step8100: in-distribution vs held-out, real CFG, inference steps.

The shared corrected_rollout_protocol.py `official_rollout` implements a dead
CFG branch: its "uncond" call passes the exact same robot_trajectory and
null_condition=False as the conditional call, so pred is unchanged for any
cfg_scale != 1.0. This script implements a corrected uncond branch
(null_condition=True, which makes wan_forward skip native-trajectory
injection) so CFG can be tested for real, without modifying the frozen
run_003 artifact.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
RUN = REPO / "reports/direct_action/gate_c/run_007"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
ADAPTER = REPO / "reports/direct_action/gate_c/run_006/checkpoints/adapter_step8100.pt"
HELD_OUT_CACHE = Path("/tmp/scratch/gate_c_run006_cache/fixed_validation_ep000011_f001012.pt")
TRAIN_CACHE = Path("/tmp/scratch/gate_c_run006_cache/train_ep000083_f001088.pt")
SEED = 42
WINDOW = 33


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def target_metrics_single(video: torch.Tensor, target: torch.Tensor) -> dict:
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


def rollout_with_cfg(model, encoder, img_latent, action, prompt_embedding, initial_noise,
                      steps: int, cfg_scale: float, capture: dict) -> torch.Tensor:
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

    with torch.no_grad():
        post_patch_frames = n_f // p_t
        native = encoder(action, post_patch_frames).to(dtype)
        capture["native_features"] = native.detach().float().cpu()

    def cond_forward(model_input, t_frame):
        return model(
            hidden_states=model_input, timestep=t_frame,
            encoder_hidden_states=prompt_embedding, encoder_hidden_states_image=None,
            robot_trajectory=action, null_condition=False,
        ).sample

    def uncond_forward(model_input, t_frame):
        # Real unconditional branch: null_condition=True makes wan_forward
        # skip native-trajectory injection entirely (see wan_forward's
        # `elif not has_control or null_condition` path).
        return model(
            hidden_states=model_input, timestep=t_frame,
            encoder_hidden_states=prompt_embedding, encoder_hidden_states_image=None,
            robot_trajectory=action, null_condition=True,
        ).sample

    with torch.no_grad():
        for t in scheduler.timesteps:
            model_input = (1 - first_mask) * condition + first_mask * latents
            t_frame = (first_mask[0, 0, :, ::p_h, ::p_w] * t).flatten().unsqueeze(0)
            pred = cond_forward(model_input, t_frame)
            if cfg_scale != 1.0:
                uncond = uncond_forward(model_input, t_frame)
                pred = uncond + cfg_scale * (pred - uncond)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            latents = (1 - first_mask) * condition + first_mask * latents

    capture["adapter_injection_norm"] = float(capture["native_features"].norm())
    video = decode_fn(vae_ref[0], latents, mean_ref[0], std_ref[0])
    return video[0].detach().float().cpu()


decode_fn = None
vae_ref = [None]
mean_ref = [None]
std_ref = [None]


def main():
    global decode_fn
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "videos").mkdir(exist_ok=True)
    (RUN / "artifacts").mkdir(exist_ok=True)

    helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_run007")
    device, dtype = torch.device("cuda"), torch.bfloat16
    helpers.apply_monkey_patch(REPO)
    tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
    prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    vae, mean, std = helpers.load_vae(BASE_MODEL, device, dtype)
    vae_ref[0], mean_ref[0], std_ref[0] = vae, mean, std
    decode_fn = helpers.decode_latents_to_video

    model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    encoder = NativeTrajectoryEncoder(input_dim=33).to(device)
    encoder.load_state_dict(torch.load(ADAPTER, map_location="cpu"))
    encoder.eval()
    model.native_trajectory_encoder = encoder

    held_out = torch.load(HELD_OUT_CACHE, map_location="cpu")
    train_sample = torch.load(TRAIN_CACHE, map_location="cpu")

    configs = [
        {"name": "heldout_cfg1_steps20", "sample": "held_out", "cfg": 1.0, "steps": 20},
        {"name": "heldout_cfg3_steps20", "sample": "held_out", "cfg": 3.0, "steps": 20},
        {"name": "heldout_cfg5_steps20", "sample": "held_out", "cfg": 5.0, "steps": 20},
        {"name": "heldout_cfg1_steps50", "sample": "held_out", "cfg": 1.0, "steps": 50},
        {"name": "train_cfg1_steps20", "sample": "train", "cfg": 1.0, "steps": 20},
        {"name": "train_cfg1_steps50", "sample": "train", "cfg": 1.0, "steps": 50},
    ]

    samples = {"held_out": held_out, "train": train_sample}
    results = {}
    for cfg in configs:
        sample = samples[cfg["sample"]]
        img_latent = sample["img_latent"].to(device)
        action = sample["robot_trajectory"].to(device=device, dtype=torch.float32)
        latent_shape = sample["video_latent"].shape
        initial_noise = torch.randn(
            latent_shape, generator=torch.Generator(device=device).manual_seed(SEED),
            device=device, dtype=model.patch_embedding.weight.dtype,
        )
        capture = {}
        video = rollout_with_cfg(model, encoder, img_latent, action, prompt, initial_noise,
                                  cfg["steps"], cfg["cfg"], capture)
        video_path = RUN / "videos" / f"{cfg['name']}.mp4"
        x = ((video.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
        import imageio
        imageio.mimsave(str(video_path), x, fps=30)
        metrics = target_metrics_single(video, sample["raw_video"][0].float())
        metrics["adapter_injection_norm"] = capture["adapter_injection_norm"]
        results[cfg["name"]] = {**cfg, "metrics": metrics, "video": str(video_path)}
        print(json.dumps({cfg["name"]: metrics}, indent=2), flush=True)

    (RUN / "artifacts/quality_ablation_results.json").write_text(json.dumps({
        "adapter": str(ADAPTER),
        "held_out_source": str(HELD_OUT_CACHE),
        "train_source": str(TRAIN_CACHE),
        "note": "uncond branch uses null_condition=True (real CFG), unlike corrected_rollout_protocol.py's dead CFG branch",
        "results": results,
    }, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
