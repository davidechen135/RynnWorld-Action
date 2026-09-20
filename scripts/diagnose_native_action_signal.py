#!/usr/bin/env python3
"""Audit temporal alignment and feature scaling of the native action path.

This script is intentionally CPU-only.  It reads cached training samples and a
native trajectory adapter, then reports:

* lagged correlation between action-change energy and video-latent motion;
* the response to 0x/0.5x/1x/2x action magnitude before and after LayerNorm;
* the RMS of the residual injected into DiT relative to video patch tokens;
* the corresponding AdaLN modulation RMS.

It does not instantiate the 5B DiT, so it is suitable for a quick pre-training
diagnostic on a busy machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from core.control import NativeTrajectoryConditionerV6, NativeTrajectoryConditionerV7


DEFAULT_MANIFEST = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_rot6d37_v2_smoke/train.json"
)
DEFAULT_CHECKPOINT = Path(
    "training/native_action_v6_causal_rot6d37_300/checkpoint-300/"
    "native_trajectory_encoder.bin"
)
DEFAULT_TRANSFORMER = Path("pretrained/Wan2.2-TI2V-5B-Diffusers/transformer")


def rms(value: torch.Tensor) -> float:
    return value.float().square().mean().sqrt().item()


def correlation(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.float().flatten()
    y = y.float().flatten()
    if x.numel() < 3 or x.numel() != y.numel():
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom.item() < 1e-12:
        return None
    return (x * y).sum().div(denom).item()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def causal_action_input(action: torch.Tensor, output_frames: int) -> torch.Tensor:
    """Mirror NativeTrajectoryConditionerV6._encode up to the input MLP."""
    delta = torch.zeros_like(action)
    delta[:, 1:] = action[:, 1:] - action[:, :-1]
    local_input = torch.cat([action, delta], dim=-1)
    if local_input.shape[1] == 1 + 4 * (output_frames - 1):
        first = local_input[:, :1]
        rest = local_input[:, 1:].reshape(
            local_input.shape[0], output_frames - 1, 4, local_input.shape[-1]
        ).mean(dim=2)
        return torch.cat([first, rest], dim=1)
    if local_input.shape[1] != output_frames:
        return F.adaptive_avg_pool1d(
            local_input.transpose(1, 2), output_frames
        ).transpose(1, 2)
    return local_input


def load_sample(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        action = handle.get_tensor("robot_trajectory").float().unsqueeze(0)
        video = handle.get_tensor("video_latents").float().unsqueeze(0)
    return action, video


def load_patch_embedding(transformer_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_file = transformer_dir / index["weight_map"]["patch_embedding.weight"]
    with safe_open(weight_file, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("patch_embedding.weight").float()
        bias = handle.get_tensor("patch_embedding.bias").float()
    return weight, bias


def estimate_patch_rms(
    video: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    max_patches: int,
) -> float:
    """Estimate token RMS from evenly sampled non-overlapping video patches."""
    kernel = tuple(weight.shape[2:])
    patches = (
        video.unfold(2, kernel[0], kernel[0])
        .unfold(3, kernel[1], kernel[1])
        .unfold(4, kernel[2], kernel[2])
        .permute(0, 2, 3, 4, 1, 5, 6, 7)
        .reshape(-1, weight[0].numel())
    )
    if patches.shape[0] > max_patches:
        indices = torch.linspace(0, patches.shape[0] - 1, max_patches).long()
        patches = patches[indices]
    return rms(F.linear(patches, weight.flatten(1), bias))


def lagged_alignment(
    action_motion: list[torch.Tensor], video_motion: list[torch.Tensor], max_lag: int
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for lag in range(-max_lag, max_lag + 1):
        action_parts: list[torch.Tensor] = []
        video_parts: list[torch.Tensor] = []
        for action_energy, video_energy in zip(action_motion, video_motion):
            if lag < 0:
                a = action_energy[-lag:]
                v = video_energy[:lag]
            elif lag > 0:
                a = action_energy[:-lag]
                v = video_energy[lag:]
            else:
                a = action_energy
                v = video_energy
            if a.numel() >= 3:
                action_parts.append(a)
                video_parts.append(v)
        corr = correlation(torch.cat(action_parts), torch.cat(video_parts))
        result[str(lag)] = corr
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conditioner-version", choices=("v6", "v7"), default="v6")
    parser.add_argument("--transformer-dir", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--feature-samples", type=int, default=8)
    parser.add_argument("--patch-samples", type=int, default=64)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())[: args.max_samples]
    if not manifest:
        raise ValueError(f"empty manifest: {args.manifest}")

    conditioner_cls = {
        "v6": NativeTrajectoryConditionerV6,
        "v7": NativeTrajectoryConditionerV7,
    }[args.conditioner_version]
    model = conditioner_cls(input_dim=37).float().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    patch_weight, patch_bias = load_patch_embedding(args.transformer_dir)

    scales = (0.0, 0.5, 1.0, 2.0)
    scale_accumulator: dict[float, dict[str, list[float]]] = {
        scale: {
            "pre_norm_delta_rms": [],
            "post_norm_delta_rms": [],
            "residual_delta_rms": [],
            "adaln_delta_rms": [],
            "residual_to_video_patch_rms": [],
        }
        for scale in scales
    }
    action_motion: list[torch.Tensor] = []
    video_motion: list[torch.Tensor] = []
    sample_summaries: list[dict[str, Any]] = []

    with torch.inference_mode():
        for sample_index, item in enumerate(manifest):
            path = Path(item["video_latent_path"])
            action, video = load_sample(path)
            output_frames = video.shape[2]

            action_input = causal_action_input(action, output_frames)
            action_level = action_input[..., :37]
            action_velocity = action_input[..., 37:]
            action_energy = action_velocity[:, 1:].square().mean(dim=-1).sqrt().squeeze(0)
            latent_delta = video[:, :, 1:] - video[:, :, :-1]
            latent_energy = latent_delta.square().mean(dim=(1, 3, 4)).sqrt().squeeze(0)
            action_motion.append(action_energy)
            video_motion.append(latent_energy)

            video_patch_rms = None
            if sample_index < args.feature_samples:
                video_patch_rms = estimate_patch_rms(
                    video, patch_weight, patch_bias, args.patch_samples
                )
                null_input = causal_action_input(torch.zeros_like(action), output_frames)
                null_pre = model.input_projection(null_input)
                null_post = model.output_norm(null_pre)
                for scale in scales:
                    scaled_input = causal_action_input(action * scale, output_frames)
                    pre = model.input_projection(scaled_input)
                    post = model.output_norm(pre)
                    centered = model.encode_centered(action * scale, output_frames)
                    residual = model.input_residual_projection(centered)
                    modulation = model.adaln_projection(centered)

                    metrics = scale_accumulator[scale]
                    metrics["pre_norm_delta_rms"].append(rms(pre - null_pre))
                    metrics["post_norm_delta_rms"].append(rms(post - null_post))
                    metrics["residual_delta_rms"].append(rms(residual))
                    metrics["adaln_delta_rms"].append(rms(modulation))
                    metrics["residual_to_video_patch_rms"].append(
                        rms(residual) / max(video_patch_rms, 1e-12)
                    )

            sample_summaries.append(
                {
                    "task_id": str(item.get("task_id")),
                    "episode": item.get("episode"),
                    "start_frame": item.get("start_frame"),
                    "path": str(path),
                    "action_level_rms": rms(action_level),
                    "action_velocity_rms": rms(action_velocity[:, 1:]),
                    "video_latent_motion_rms": rms(latent_delta),
                    "video_patch_rms": video_patch_rms,
                }
            )

    scale_response = {
        str(scale): {name: mean(values) for name, values in metrics.items()}
        for scale, metrics in scale_accumulator.items()
    }
    reference = scale_response["1.0"]
    for scale in scales:
        row = scale_response[str(scale)]
        for name in (
            "pre_norm_delta_rms",
            "post_norm_delta_rms",
            "residual_delta_rms",
            "adaln_delta_rms",
        ):
            denom = reference[name]
            row[f"{name}_gain_vs_1x"] = (
                row[name] / denom if denom is not None and denom > 0 else None
            )

    lag_correlation = lagged_alignment(action_motion, video_motion, args.max_lag)
    finite_lags = {int(k): v for k, v in lag_correlation.items() if v is not None}
    best_lag = max(finite_lags, key=lambda key: finite_lags[key]) if finite_lags else None
    top_samples = sorted(
        sample_summaries, key=lambda row: row["video_latent_motion_rms"], reverse=True
    )[:10]
    top_action_samples = sorted(
        sample_summaries, key=lambda row: row["action_velocity_rms"], reverse=True
    )[:10]

    report = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "conditioner_version": args.conditioner_version,
        "num_samples": len(sample_summaries),
        "num_feature_samples": min(len(sample_summaries), args.feature_samples),
        "definitions": {
            "lag": "positive means video motion follows action change by lag latent tokens",
            "gain_vs_1x": "ideal linear amplitude response is approximately the input scale",
            "residual_to_video_patch_rms": "action residual RMS divided by clean video patch-token RMS",
        },
        "dataset_summary": {
            "action_level_rms_mean": mean([r["action_level_rms"] for r in sample_summaries]),
            "action_velocity_rms_mean": mean([r["action_velocity_rms"] for r in sample_summaries]),
            "video_latent_motion_rms_mean": mean(
                [r["video_latent_motion_rms"] for r in sample_summaries]
            ),
        },
        "lagged_action_video_motion_correlation": lag_correlation,
        "best_lag": best_lag,
        "scale_response": scale_response,
        "highest_video_motion_samples": top_samples,
        "highest_action_motion_samples": top_action_samples,
    }

    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
