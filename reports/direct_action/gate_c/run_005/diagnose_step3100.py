#!/usr/bin/env python3
"""Diagnose action/value/order sensitivity at the run_004 step3100 checkpoint."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import imageio.v3 as iio

REPO = Path("/mnt/workspace/RynnWorld-Teleop")
RUN = REPO / "reports/direct_action/gate_c/run_005"
MANIFEST = REPO / "reports/direct_action/gate_c/run_004/artifacts/window_manifest.json"
CACHE = Path("/tmp/scratch/gate_c_run004_cache")
CHECKPOINT = REPO / "reports/direct_action/gate_c/run_004/checkpoints/adapter_step3100.pt"
HELPERS_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
AUDIT_PATH = REPO / "reports/direct_action/gate_c/run_003/artifacts/corrected_rollout_protocol.py"
BASE_MODEL = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN_CHECKPOINT = REPO / "pretrained/RynnWorld-Teleop-Causal"
SEED = 42
SHUFFLE_SEED = 314159
STEPS = 20
CFG_SCALE = 1.0
HEIGHT, WIDTH = 480, 832
WINDOW = 33


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(x: torch.Tensor) -> str:
    y = x.detach().cpu().contiguous()
    return hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()


def summary(x: torch.Tensor) -> dict:
    y = x.detach().float().cpu()
    return {
        "shape": list(y.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "sha256": sha(x),
        "norm": float(y.norm()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "storage_ptr": int(x.untyped_storage().data_ptr()),
    }


def extract_33d(a):
    return np.concatenate([a[:, 2:8], a[:, 8:16], a[:, 16:30], a[:, 33:38]], axis=1)


def select_active_window(actions: np.ndarray, max_start: int) -> tuple[int, float]:
    candidates = np.unique(np.linspace(0, max_start, num=min(128, max_start + 1), dtype=int))
    scores = [float(np.linalg.norm(actions[start:start + WINDOW] - actions[start])) for start in candidates]
    index = int(np.argmax(scores))
    return int(candidates[index]), scores[index]


def load_latent_for_window(record: dict, start: int, helpers, vae, mean, std, device, dtype):
    frames = []
    for index, frame in enumerate(iio.imiter(record["video"], plugin="pyav")):
        if index >= start + WINDOW:
            break
        if index >= start:
            frames.append(frame)
    if len(frames) != WINDOW:
        raise ValueError(f"decoded {len(frames)} instead of {WINDOW} frames for episode {record['episode']}")
    video = helpers.preprocess_video_frames(torch.from_numpy(np.stack(frames)), HEIGHT, WIDTH).unsqueeze(0).to(device, dtype=dtype)
    latent = helpers.encode_video_frames(vae, video, mean, std)
    return latent, latent[:, :, :1].clone()


def main():
    helpers = load_module(HELPERS_PATH, "gate_c_visual_v2_run005_diag")
    audit = load_module(AUDIT_PATH, "gate_c_protocol_run005_diag")
    manifest = json.loads(MANIFEST.read_text())
    held = [r for r in manifest["records"] if r["split"] == "held_out"]
    reference_record = next(r for r in held if r["episode"] == 11)
    wrong_record = next(r for r in held if r["episode"] != 11)
    reference_actions = extract_33d(np.stack(pd.read_parquet(reference_record["parquet"])["action"].values))
    wrong_actions = extract_33d(np.stack(pd.read_parquet(wrong_record["parquet"])["action"].values))
    reference_start, reference_activity = select_active_window(reference_actions, reference_record["video_frames"] - WINDOW)
    wrong_start, wrong_activity = select_active_window(wrong_actions, wrong_record["video_frames"] - WINDOW)
    correct = torch.from_numpy(reference_actions[reference_start:reference_start + WINDOW].copy()).unsqueeze(0).float()
    wrong = torch.from_numpy(wrong_actions[wrong_start:wrong_start + WINDOW].copy()).unsqueeze(0).float()
    generator = torch.Generator().manual_seed(SHUFFLE_SEED)
    permutation = torch.randperm(WINDOW, generator=generator)
    conditions = {
        "correct": correct.clone(),
        "zero": torch.zeros_like(correct),
        "shuffled": correct[:, permutation].clone(),
        "reversed": torch.flip(correct, dims=[1]).clone(),
        "wrong": wrong,
    }
    device = torch.device("cuda")
    dtype = torch.bfloat16
    helpers.apply_monkey_patch(REPO)
    tokenizer, text_encoder = helpers.load_text_encoder(BASE_MODEL, device, dtype)
    prompt_embedding = helpers.encode_text(tokenizer, text_encoder, "", device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    vae, mean, std = helpers.load_vae(BASE_MODEL, device, dtype)
    reference_latent, img_latent = load_latent_for_window(reference_record, reference_start, helpers, vae, mean, std, device, dtype)
    wrong_latent, _ = load_latent_for_window(wrong_record, wrong_start, helpers, vae, mean, std, device, dtype)
    model = helpers.load_model(RYNN_CHECKPOINT, BASE_MODEL, sys.stdout)
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    encoder = NativeTrajectoryEncoder(input_dim=33).to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    encoder.eval()
    model.native_trajectory_encoder = encoder
    reference_latent = reference_latent.to(device)
    img_latent = img_latent.to(device)
    output_frames = reference_latent.shape[2] // model.config.patch_size[0]
    initial_noise = torch.randn(reference_latent.shape, generator=torch.Generator(device=device).manual_seed(SEED), device=device, dtype=model.patch_embedding.weight.dtype)

    captures = {name: {"encoder_calls": [], "model_actions": [], "patch_outputs": [], "rollout": {}} for name in conditions}
    original_encoder_forward = encoder.forward
    original_model_forward = model.forward
    original_patch_forward = model.patch_embedding.forward
    active = {"name": None}

    def encoder_forward(trajectory, frames):
        name = active["name"]
        if name is not None and len(captures[name]["encoder_calls"]) == 0:
            captures[name]["encoder_calls"].append({"input": summary(trajectory), "output_frames": frames})
        output = original_encoder_forward(trajectory, frames)
        if name is not None and len(captures[name]["encoder_calls"]) == 1:
            captures[name]["encoder_calls"][0]["output"] = summary(output)
        return output

    def model_forward(*args, **kwargs):
        name = active["name"]
        action = kwargs.get("robot_trajectory")
        if name is not None and action is not None and len(captures[name]["model_actions"]) == 0:
            captures[name]["model_actions"].append(summary(action))
        return original_model_forward(*args, **kwargs)

    def patch_forward(*args, **kwargs):
        output = original_patch_forward(*args, **kwargs)
        name = active["name"]
        if name is not None and len(captures[name]["patch_outputs"]) == 0:
            captures[name]["patch_outputs"].append(summary(output))
        return output

    encoder.forward = encoder_forward
    model.forward = model_forward
    model.patch_embedding.forward = patch_forward
    videos = {}
    audit.ns_decode = helpers.decode_latents_to_video
    with torch.no_grad():
        for name, action_cpu in conditions.items():
            active["name"] = name
            action = action_cpu.to(device)
            videos[name] = audit.official_rollout(
                model, encoder, vae, img_latent, action, prompt_embedding, mean, std,
                initial_noise, STEPS, CFG_SCALE, captures[name]["rollout"],
            )
            audit.save_video(videos[name], RUN / "videos" / f"step3100_{name}.mp4", 30)
    active["name"] = None
    source_first = helpers.decode_latents_to_video(vae, img_latent, mean, std)[0, :, 0].float().cpu()
    source_first = ((source_first.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    metrics = audit.video_metrics(videos, source_first)
    serializable_captures = {}
    for name, capture in captures.items():
        rollout_capture = capture["rollout"]
        serializable_captures[name] = {
            "encoder_calls": capture["encoder_calls"],
            "model_actions": capture["model_actions"],
            "patch_outputs": capture["patch_outputs"],
            "rollout": {
                key: (summary(value) if isinstance(value, torch.Tensor) else value)
                for key, value in rollout_capture.items()
                if key not in ("native_features", "adapter_input")
            },
            "rollout_native_features": summary(rollout_capture["native_features"]),
            "rollout_adapter_input": summary(rollout_capture["adapter_input"]),
        }
    condition_summary = {name: summary(action) for name, action in conditions.items()}
    embedding_distances = {}
    for name in conditions:
        if name != "correct":
            a = captures["correct"]["encoder_calls"][0]["output"]
            b = captures[name]["encoder_calls"][0]["output"]
            embedding_distances[name] = {"sha_equal": a["sha256"] == b["sha256"], "norm_distance": float(torch.linalg.vector_norm(encoder(conditions["correct"].to(device), output_frames) - encoder(conditions[name].to(device), output_frames)).cpu())}
    result = {
        "checkpoint": str(CHECKPOINT),
        "reference_episode": reference_record["episode"],
        "wrong_episode": wrong_record["episode"],
        "reference_start": reference_start,
        "wrong_start": wrong_start,
        "reference_activity": reference_activity,
        "wrong_activity": wrong_activity,
        "conditions": condition_summary,
        "condition_storage_ptr_unique": len({x["storage_ptr"] for x in condition_summary.values()}) == len(condition_summary),
        "captures": serializable_captures,
        "embedding_distances": embedding_distances,
        "metrics": metrics,
        "protocol": {"steps": STEPS, "seed": SEED, "shuffle_seed": SHUFFLE_SEED, "cfg_scale": CFG_SCALE, "shared_initial_noise": True},
    }
    (RUN / "artifacts/diagnostic_step3100.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(RUN / "artifacts/diagnostic_step3100.json"), "embedding_distances": embedding_distances, "video_pairwise": metrics["pairwise"]}, indent=2))


if __name__ == "__main__":
    main()
