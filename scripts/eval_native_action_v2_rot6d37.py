#!/usr/bin/env python3
"""Fixed-noise held-out action audit for V2 rot6D37 checkpoints."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, set_peft_model_state_dict

REPO = Path(__file__).resolve().parents[1]
HELPERS = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
BASE = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN = REPO / "pretrained/RynnWorld-Teleop-Causal"
IS_V3 = os.environ.get("EVAL_V3") == "1"
IS_V4 = os.environ.get("EVAL_V4") == "1"
IS_V5 = os.environ.get("EVAL_V5") == "1"
IS_V6 = os.environ.get("EVAL_V6") == "1"
IS_V7 = os.environ.get("EVAL_V7") == "1"
IS_V8 = os.environ.get("EVAL_V8") == "1"
IS_V9 = os.environ.get("EVAL_V9") == "1"
IS_V10 = os.environ.get("EVAL_V10") == "1"
IS_V11 = os.environ.get("EVAL_V11") == "1"
USES_OBSERVED_STATE = IS_V10 or IS_V11
IS_FACTORIZED = IS_V8 or IS_V9 or USES_OBSERVED_STATE
IS_V3_PLUS = IS_V3 or IS_V4 or IS_V5 or IS_V6 or IS_V7 or IS_FACTORIZED
TRAIN = Path(os.environ.get("EVAL_TRAIN", REPO / ("training/native_action_v3_rot6d37_500" if IS_V3_PLUS else "training/native_action_v2_rot6d37_500")))
OUT = Path(os.environ.get("EVAL_OUT", REPO / ("reports/native_action_v3_rot6d37_500_eval" if IS_V3_PLUS else "reports/native_action_v2_rot6d37_500_eval")))
STATS = REPO / "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_33f_v1/action_stats_core37.json"
VALIDATION = Path("/tmp/scratch/gate_d_run001_cache/fixed_validation_ep001348_f000534.pt")
TRAIN_SAMPLE = os.environ.get("EVAL_SAMPLE_SAFETENSORS")
CHECKPOINTS = tuple(int(x) for x in os.environ.get("EVAL_CHECKPOINTS", "250,300,350").split(","))
BASE_CONDITIONS = ("correct", "held", "zero", "reversed", "shifted")
V10_ABLATIONS = ("correct_no_spatial", "held_correct_spatial")
CONDITIONS = BASE_CONDITIONS + V10_ABLATIONS if USES_OBSERVED_STATE else BASE_CONDITIONS
if os.environ.get("EVAL_CONDITIONS"):
    CONDITIONS = tuple(os.environ["EVAL_CONDITIONS"].split(","))
STEPS = int(os.environ.get("EVAL_STEPS", "20"))
SEED = int(os.environ.get("EVAL_SEED", "42"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def convert37(core33: torch.Tensor) -> torch.Tensor:
    def rot6(q):
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x, y, z, w = q.unbind(-1)
        return torch.stack(
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*y+z*w),
             1-2*(x*x+z*z), 2*(x*z-y*w), 2*(y*z+x*w)], dim=-1
        )
    return torch.cat([core33[..., :6], rot6(core33[..., 6:10]),
                      rot6(core33[..., 10:14]), core33[..., 14:]], dim=-1)


def load_lora(model, checkpoint: Path):
    from diffusers import WanImageToVideoPipeline
    state = WanImageToVideoPipeline.lora_state_dict(str(checkpoint / "high_noise_lora"))
    # Pipeline serialization namespaces transformer LoRA keys, while the
    # standalone WanTransformer3DModel PEFT adapter does not.
    state = {k.removeprefix("transformer."): v for k, v in state.items()}
    result = set_peft_model_state_dict(model, state, adapter_name="high_noise")
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected LoRA keys: {result.unexpected_keys[:5]}")


@torch.no_grad()
def rollout(model, scheduler, image, action, prompt, noise, spatial_control=None):
    dtype, device = model.patch_embedding.weight.dtype, image.device
    latents = noise.clone()
    b, _, frames, h, w = latents.shape
    _, ph, pw = model.config.patch_size
    mask = torch.ones((b, 1, frames, h, w), device=device, dtype=dtype)
    mask[:, :, 0] = 0
    scheduler.set_timesteps(STEPS, device=device)
    for t in scheduler.timesteps:
        model_input = (1-mask)*image + mask*latents
        timestep = (mask[0, 0, :, ::ph, ::pw] * t).flatten().unsqueeze(0)
        pred = model(hidden_states=model_input, timestep=timestep,
                     encoder_hidden_states=prompt,
                     encoder_hidden_states_image=None,
                     robot_trajectory=action,
                     robot_spatial_control=spatial_control,
                     null_condition=False).sample
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
        latents = (1-mask)*image + mask*latents
    return latents


def array(video):
    return np.rint(
        (video.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5
    ).clip(0, 255).astype(np.uint8)


def gt_motion_region(ground_truth: np.ndarray, quantile: float = 0.8) -> np.ndarray:
    motion = np.abs(np.diff(ground_truth.astype(np.float32), axis=0)).mean(axis=(0, 3))
    return motion >= np.quantile(motion, quantile)


def reconstruction_metrics(
    ground_truth: np.ndarray, generated: np.ndarray, motion_region: np.ndarray
) -> dict[str, float]:
    gt = ground_truth.astype(np.float32)
    pred = generated.astype(np.float32)
    error = np.abs(gt - pred)
    gt_motion = np.diff(gt, axis=0)
    pred_motion = np.diff(pred, axis=0)
    gt_roi_motion = gt_motion[:, motion_region, :].reshape(-1)
    pred_roi_motion = pred_motion[:, motion_region, :].reshape(-1)
    temporal_cosine = float(
        np.dot(gt_roi_motion, pred_roi_motion)
        / (np.linalg.norm(gt_roi_motion) * np.linalg.norm(pred_roi_motion) + 1e-12)
    )
    return {
        "gt_pixel_mad": float(error.mean()),
        "gt_rmse": float(np.square(gt - pred).mean() ** 0.5),
        "motion_region_pixel_mad": float(error[:, motion_region, :].mean()),
        "background_pixel_mad": float(error[:, ~motion_region, :].mean()),
        "motion_region_temporal_mad": float(
            np.abs(gt_motion[:, motion_region, :] - pred_motion[:, motion_region, :]).mean()
        ),
        "motion_region_temporal_cosine": temporal_cosine,
    }


def main():
    from diffusers import UniPCMultistepScheduler
    import imageio

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    helpers = load_module(HELPERS, "rot6d37_eval_helpers")
    helpers.apply_monkey_patch(REPO)
    device, dtype = torch.device("cuda"), torch.bfloat16

    tokenizer, text_encoder = helpers.load_text_encoder(BASE, device, dtype)
    prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    vae, latent_mean, latent_std = helpers.load_vae(BASE, device, dtype)
    if os.environ.get("PURE_WAN_BASELINE") == "1":
        from diffusers import WanTransformer3DModel
        model = WanTransformer3DModel.from_pretrained(BASE, subfolder="transformer")
        model = model.to(device=device, dtype=dtype).requires_grad_(False).eval()
    else:
        model = helpers.load_model(RYNN, BASE, sys.stdout)

    if TRAIN_SAMPLE:
        from safetensors.torch import load_file
        packed = load_file(TRAIN_SAMPLE)
        sample = {
            "video_latent": packed["video_latents"].unsqueeze(0),
            "img_latent": packed["img_latent"].unsqueeze(0),
        }
        spatial_actions = {name: None for name in CONDITIONS}
        if IS_FACTORIZED:
            from core.control.native_action_features import (
                build_spatial_action_control,
                build_v8_features,
                build_v10_features,
            )

            raw = packed["robot_trajectory_raw37"].unsqueeze(0).float().to(device)
            mean = packed["robot_trajectory_mean37"].float().to(device)
            std = packed["robot_trajectory_std37"].float().to(device)
            relative_scale = packed["robot_relative_scale37"].float().to(device)
            velocity_scale = packed["robot_velocity_scale37"].float().to(device)

            def make_v8(source):
                return build_v8_features(
                    source, mean, std, relative_scale, velocity_scale
                )

            if USES_OBSERVED_STATE:
                state = packed["robot_observed_state37"].unsqueeze(0).float().to(device)

                def make_features(source):
                    return build_v10_features(
                        source, state, mean, std, relative_scale, velocity_scale
                    )

                held_raw = state[:, None].expand_as(raw)
                raw_conditions = {
                    "correct": raw,
                    "held": held_raw,
                    "reversed": raw.flip(1),
                    "shifted": raw.roll(raw.shape[1] // 2, 1),
                }
                correct = make_features(raw)
                actions = {
                    name: make_features(value) for name, value in raw_conditions.items()
                }
                actions["zero"] = torch.zeros_like(correct)
                actions["correct_no_spatial"] = actions["correct"]
                actions["held_correct_spatial"] = actions["held"]
                if "robot_trajectory_uv" in packed and "robot_state_uv" in packed:
                    uv = packed["robot_trajectory_uv"].unsqueeze(0).float().to(device)
                    state_uv = packed["robot_state_uv"].unsqueeze(0).float().to(device)
                    held_uv = state_uv[:, None].expand_as(uv)
                    uv_conditions = {
                        "correct": uv,
                        "held": held_uv,
                        "reversed": uv.flip(1),
                        "shifted": uv.roll(uv.shape[1] // 2, 1),
                    }
                    spatial_actions = {
                        name: build_spatial_action_control(value, state_uv)
                        for name, value in uv_conditions.items()
                    }
                    spatial_actions["zero"] = torch.zeros_like(
                        spatial_actions["correct"]
                    )
                    spatial_actions["correct_no_spatial"] = torch.zeros_like(
                        spatial_actions["correct"]
                    )
                    spatial_actions["held_correct_spatial"] = spatial_actions["correct"]
            else:
                correct = make_v8(raw)
                held_raw = raw[:, :1].expand_as(raw)
                actions = {
                    "correct": correct,
                    "held": make_v8(held_raw),
                    "zero": torch.zeros_like(correct),
                    "reversed": make_v8(raw.flip(1)),
                    "shifted": make_v8(raw.roll(raw.shape[1] // 2, 1)),
                }
        else:
            # Smoke-set safetensors store task-wise normalized rot6D37.
            correct = packed["robot_trajectory"].unsqueeze(0).float().to(device)
    else:
        if IS_FACTORIZED:
            raise ValueError("V8 evaluation requires EVAL_SAMPLE_SAFETENSORS")
        sample = torch.load(VALIDATION, map_location="cpu", weights_only=False)
        raw37 = convert37(sample["robot_trajectory"].float())
        stats = json.loads(STATS.read_text())["per_task"]["3401"]
        mean = torch.tensor(stats["mean"]); std = torch.tensor(stats["std"])
        correct = ((raw37-mean)/std).to(device)
    if not IS_FACTORIZED:
        spatial_actions = {name: None for name in CONDITIONS}
        actions = {
            "correct": correct,
            "held": correct[:, :1].expand_as(correct),
            "zero": torch.zeros_like(correct),
            "reversed": correct.flip(1),
            "shifted": correct.roll(correct.shape[1] // 2, 1),
        }
    image = sample["img_latent"].to(device=device, dtype=dtype)
    shape = sample["video_latent"].shape
    noise = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(SEED),
                        device=device, dtype=dtype)
    scheduler = UniPCMultistepScheduler.from_pretrained(BASE, subfolder="scheduler")

    ground_truth_video = helpers.decode_latents_to_video(
        vae, sample["video_latent"].to(device=device, dtype=dtype),
        latent_mean, latent_std,
    )[0].float().cpu()
    ground_truth_array = array(ground_truth_video)
    motion_region = gt_motion_region(ground_truth_array)
    imageio.mimsave(OUT/"videos"/"ground_truth.mp4", ground_truth_array, fps=30)

    if os.environ.get("BASELINE_ONLY") == "1":
        from core.control.native_trajectory_encoder import NativeTrajectoryConditionerV2, NativeTrajectoryConditionerV3, NativeTrajectoryEncoder
        if os.environ.get("V1_FROZEN_BASELINE") == "1":
            encoder = NativeTrajectoryConditionerV3(37)
            old = NativeTrajectoryEncoder(33)
            old.load_state_dict(torch.load(
                REPO/"reports/direct_action/gate_d/run_015/checkpoints/adapter_step1000.pt",
                map_location="cpu", weights_only=True,
            ))
            old.eval()
            with torch.no_grad():
                encoder.base_residual.copy_(old(torch.zeros(1, 1, 33), 1))
            model.native_trajectory_encoder = encoder.to(device=device, dtype=dtype).eval()
        else:
            model.native_trajectory_encoder = NativeTrajectoryConditionerV2(37).to(device=device, dtype=dtype).eval()
        latent = rollout(model, scheduler, image, torch.zeros_like(correct), prompt, noise)
        video = helpers.decode_latents_to_video(vae, latent, latent_mean, latent_std)[0].float().cpu()
        name = ("v1_frozen_baseline.mp4" if os.environ.get("V1_FROZEN_BASELINE") == "1"
                else "pure_wan_base.mp4" if os.environ.get("PURE_WAN_BASELINE") == "1"
                else "rynn_no_control.mp4")
        imageio.mimsave(OUT/"videos"/name, array(video), fps=30)
        print(json.dumps({"baseline": str(OUT/"videos"/name)}))
        return

    if not IS_V3_PLUS:
        model.add_adapter(LoraConfig(r=8, lora_alpha=8, target_modules=[
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"
        ]), adapter_name="high_noise")
        model.requires_grad_(False).eval()

    all_metrics = {}
    for step in CHECKPOINTS:
        checkpoint = TRAIN / f"checkpoint-{step}"
        if not IS_V3_PLUS:
            load_lora(model, checkpoint)
        from core.control.native_trajectory_encoder import (
            NativeTrajectoryConditionerV2,
            NativeTrajectoryConditionerV3,
            NativeTrajectoryConditionerV4,
            NativeTrajectoryConditionerV5,
            NativeTrajectoryConditionerV6,
            NativeTrajectoryConditionerV7,
            NativeTrajectoryConditionerV8,
            NativeTrajectoryConditionerV9,
            NativeTrajectoryConditionerV10,
            NativeTrajectoryConditionerV11,
        )
        encoder_cls = (
            NativeTrajectoryConditionerV11 if IS_V11
            else NativeTrajectoryConditionerV10 if IS_V10
            else NativeTrajectoryConditionerV9 if IS_V9
            else NativeTrajectoryConditionerV8 if IS_V8
            else NativeTrajectoryConditionerV7 if IS_V7
            else NativeTrajectoryConditionerV6 if IS_V6
            else NativeTrajectoryConditionerV5 if IS_V5
            else NativeTrajectoryConditionerV4 if IS_V4
            else NativeTrajectoryConditionerV3 if IS_V3
            else NativeTrajectoryConditionerV2
        )
        encoder = encoder_cls(
            input_dim=148 if USES_OBSERVED_STATE else 111 if IS_FACTORIZED else 37
        ).to(device=device, dtype=dtype)
        encoder.load_state_dict(torch.load(checkpoint/"native_trajectory_encoder.bin",
                                           map_location="cpu", weights_only=True))
        encoder.eval(); model.native_trajectory_encoder = encoder
        videos = {}
        for name in CONDITIONS:
            latent = rollout(
                model, scheduler, image, actions[name], prompt, noise,
                spatial_control=spatial_actions[name],
            )
            video = helpers.decode_latents_to_video(vae, latent, latent_mean, latent_std)[0].float().cpu()
            videos[name] = array(video)
            imageio.mimsave(OUT/"videos"/f"step{step}_{name}.mp4", videos[name], fps=30)
        correct_video = videos["correct"].astype(np.float32)
        pairs = {}
        for name in CONDITIONS:
            other = videos[name].astype(np.float32)
            item = reconstruction_metrics(ground_truth_array, other, motion_region)
            if name != "correct":
                item.update({
                    "pixel_mad": float(np.abs(correct_video-other).mean()),
                    "rmse": float(np.square(correct_video-other).mean()**0.5),
                })
            pairs[name] = item
        all_metrics[str(step)] = pairs
        (OUT/"metrics.json").write_text(json.dumps(all_metrics, indent=2))
        print(json.dumps({"step": step, "vs_correct": pairs}), flush=True)

    config = {
        "checkpoints": CHECKPOINTS,
        "conditions": CONDITIONS,
        "seed": SEED,
        "steps": STEPS,
        "shared_initial_noise": True,
    }
    if TRAIN_SAMPLE:
        config["sample_safetensors"] = str(Path(TRAIN_SAMPLE).resolve())
    else:
        config.update({
            "heldout_task": "3401",
            "heldout_episode": 1348,
            "start": 534,
        })
    (OUT/"config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
