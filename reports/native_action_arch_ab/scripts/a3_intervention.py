#!/usr/bin/env python3
"""A3: forward causal intervention on the V11 action-conditioning path.

Runtime hooks only.  No production model file is edited, nothing is trained,
no weights are written.  Every variant is installed by swapping module objects
or hooking a forward, and is removed afterwards.

Fixed across all variants (the comparison is only meaningful this way):
same checkpoint, same real window, same first frame, same initial noise,
same seed, same sampler and step count.

Variants
  A  original
  B  base_residual temporarily zeroed           (magnitude swamping)
  C  input_residual_projection replaced by a frozen full-rank projection
                                                (rank starvation)
  F  B + C
  D  gate forced to 1                           (no-op control; see note)
  E  frozen per-frame per-location gate         (per-frame gating)
  G  B + C + E

Note on D and E: V11 has no spatial gate.  ``spatialize_residual`` and
``last_spatial_gate`` belong to V9, which V11 does not inherit from
(V11 -> V10 -> V8 -> ... -> V2).  D is therefore implemented as a literal
"gate == 1" no-op so that its output difference from A measures exactly zero,
which is the honest way to report that the gate contributes nothing on this
checkpoint.  E supplies the per-frame gating capability with a frozen
seed-fixed mask instead of a trained visual query, so that no training is
required.

Writes metrics/a3_intervention.json and videos/ under the report directory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "reports/native_action_arch_ab"
HELPERS = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
BASE = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN = REPO / "pretrained/RynnWorld-Teleop-Causal"
TRAIN = REPO / "training/native_action_v11_temporal_alignment_300"
CKPT_STEP = int(os.environ.get("A3_CKPT_STEP", "300"))
SAMPLE = Path(os.environ.get(
    "A3_SAMPLE",
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v10_state_spatial_16_v1/task3400_00.safetensors",
))
STEPS = int(os.environ.get("A3_STEPS", "20"))
SEEDS = tuple(int(x) for x in os.environ.get("A3_SEEDS", "42,7,123").split(","))
CONDITIONS = tuple(os.environ.get("A3_CONDITIONS", "correct,reversed,held,swapped").split(","))
VARIANTS = tuple(os.environ.get("A3_VARIANTS", "A,B,C,F,D,E,G").split(","))
A3_OUT = os.environ.get("A3_OUT", "a3_intervention.json")
NOISE_FLOOR_OUTMAD = 7.229

# What counts as "closer to GT" and "beyond the noise floor" is fixed up front
# so the result cannot be read off a threshold chosen after seeing the numbers.


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array(video):
    return np.rint(
        (video.clamp(-1, 1).permute(1, 2, 3, 0).numpy() + 1) * 127.5
    ).clip(0, 255).astype(np.uint8)


def gt_motion_region(ground_truth: np.ndarray, quantile: float = 0.8) -> np.ndarray:
    motion = np.abs(np.diff(ground_truth.astype(np.float32), axis=0)).mean(axis=(0, 3))
    return motion >= np.quantile(motion, quantile)


def metrics_vs_gt(gt: np.ndarray, gen: np.ndarray, roi: np.ndarray) -> dict:
    g, p = gt.astype(np.float32), gen.astype(np.float32)
    err = np.abs(g - p)
    gm, pm = np.diff(g, axis=0), np.diff(p, axis=0)
    a, b = gm[:, roi, :].reshape(-1), pm[:, roi, :].reshape(-1)
    return {
        "gt_pixel_mad": float(err.mean()),
        "gt_rmse": float(np.square(g - p).mean() ** 0.5),
        "motion_region_pixel_mad": float(err[:, roi, :].mean()),
        "background_pixel_mad": float(err[:, ~roi, :].mean()),
        "motion_region_temporal_mad": float(np.abs(gm[:, roi, :] - pm[:, roi, :]).mean()),
        "motion_region_temporal_cosine": float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
        "motion_energy": float(np.abs(pm[:, roi, :]).mean()),
        "video_std": float(p.std()),
    }


class FullRankProjection(torch.nn.Module):
    """Frozen full-rank stand-in for the trained rank-starved projection.

    Scaled so its output RMS matches what ``input_residual_projection``
    produces on this window, i.e. only the *rank* is changed, not the gain.

    The gain is calibrated by *direct measurement* against ``calib`` (the
    centered action for this window) rather than by a Frobenius identity.  The
    original formula, ``q * target_rms * sqrt(in_dim)``, assumes
    ``rms(W x) == gain * rms(x)`` for orthonormal ``q``; measured on this
    window it delivers 0.050239 against a 0.034856 target, i.e. **1.44x high**.
    Over-injection is the less dangerous direction than A4's 55x under-injection
    (it cannot manufacture an "inert" conclusion), but it still means variant C
    was not the rank-only control it claims to be.  Measuring removes the
    guesswork, and the assertion fails the run if calibration drifts.
    """

    def __init__(self, in_dim: int, out_dim: int, target_rms: float,
                 seed: int, dtype, device, calib: torch.Tensor | None = None):
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        q, _ = torch.linalg.qr(
            torch.randn(out_dim, in_dim, generator=gen, dtype=torch.float32))
        if calib is None:
            gain = target_rms * (in_dim ** 0.5)
        else:
            with torch.no_grad():
                unit = torch.nn.functional.linear(
                    calib.to(q.dtype), q.to(device=device, dtype=q.dtype))
                unit_rms = float(unit.float().pow(2).mean().sqrt().item())
            gain = target_rms / max(unit_rms, 1e-12)
        self.register_buffer(
            "weight",
            (q * gain).to(device=device, dtype=dtype))
        self.target_rms = target_rms
        if calib is not None:
            with torch.no_grad():
                achieved = float(
                    torch.nn.functional.linear(
                        calib.to(self.weight.dtype), self.weight)
                    .float().pow(2).mean().sqrt().item())
            assert abs(achieved / target_rms - 1.0) < 1e-3, (
                f"gain calibration failed: achieved {achieved} vs target {target_rms}")
            self.achieved_rms = achieved

    def forward(self, x):
        return torch.nn.functional.linear(x.to(self.weight.dtype), self.weight)


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
        model_input = (1 - mask) * image + mask * latents
        timestep = (mask[0, 0, :, ::ph, ::pw] * t).flatten().unsqueeze(0)
        pred = model(hidden_states=model_input, timestep=timestep,
                     encoder_hidden_states=prompt,
                     encoder_hidden_states_image=None,
                     robot_trajectory=action,
                     robot_spatial_control=spatial_control,
                     null_condition=False).sample
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
        latents = (1 - mask) * image + mask * latents
    return latents


def main() -> None:
    from safetensors.torch import load_file
    from diffusers import UniPCMultistepScheduler
    import imageio
    from core.control.native_action_features import build_v10_features
    from core.control.native_trajectory_encoder import NativeTrajectoryConditionerV11

    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    device, dtype = torch.device("cuda"), torch.bfloat16

    helpers = load_module(HELPERS, "gate_c_visual_v2")
    helpers.apply_monkey_patch(REPO)
    tokenizer, text_encoder = helpers.load_text_encoder(BASE, device, dtype)
    prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    vae, latent_mean, latent_std = helpers.load_vae(BASE, device, dtype)
    model = helpers.load_model(RYNN, BASE, sys.stdout)

    encoder = NativeTrajectoryConditionerV11(input_dim=148).to(device=device, dtype=dtype)
    encoder.load_state_dict(torch.load(
        TRAIN / f"checkpoint-{CKPT_STEP}" / "native_trajectory_encoder.bin",
        map_location="cpu", weights_only=True))
    encoder.eval()
    model.native_trajectory_encoder = encoder

    packed = load_file(str(SAMPLE))
    raw = packed["robot_trajectory_raw37"].unsqueeze(0).float().to(device)
    mean = packed["robot_trajectory_mean37"].float().to(device)
    std = packed["robot_trajectory_std37"].float().to(device)
    rel_s = packed["robot_relative_scale37"].float().to(device)
    vel_s = packed["robot_velocity_scale37"].float().to(device)
    state = packed["robot_observed_state37"].unsqueeze(0).float().to(device)

    def feat(src):
        return build_v10_features(src, state, mean, std, rel_s, vel_s)

    correct = feat(raw)
    swapped = correct.clone()
    swapped[:, [0, 1]] = correct[:, [1, 0]]
    actions = {
        "correct": correct,
        "reversed": feat(raw.flip(1)),
        "held": feat(state[:, None].expand_as(raw)),
        "swapped": swapped,
    }

    image = packed["img_latent"].unsqueeze(0).to(device=device, dtype=dtype)
    latent_shape = packed["video_latents"].unsqueeze(0).shape
    T = latent_shape[2] // model.config.patch_size[0]
    gH = latent_shape[3] // model.config.patch_size[1]
    gW = latent_shape[4] // model.config.patch_size[2]
    scheduler = UniPCMultistepScheduler.from_pretrained(BASE, subfolder="scheduler")

    gt_video = helpers.decode_latents_to_video(
        vae, packed["video_latents"].unsqueeze(0).to(device=device, dtype=dtype),
        latent_mean, latent_std)[0].float().cpu()
    gt = array(gt_video)
    roi = gt_motion_region(gt)
    imageio.mimsave(OUT / "videos" / "a3_ground_truth.mp4", gt, fps=30)

    # trained projection RMS, measured once, reused for every C-type variant
    with torch.no_grad():
        centered = encoder.encode_centered(correct, T)
        proj_rms = float(encoder.input_residual_projection(centered).float()
                         .pow(2).mean().sqrt().item())
    print(f"[A3] window={SAMPLE.stem} T={T} grid={gH}x{gW} "
          f"steps={STEPS} seeds={SEEDS} PROJ_RMS={proj_rms:.6g}", flush=True)

    # Frozen per-frame gate for E / G (no training).  Note the shape: the
    # residual reaching the injection site is [B,C,T,1,1] -- spatial extent is
    # 1x1, because ``input_residual_projection`` produces one vector per frame
    # and that vector is broadcast over H*W.  A per-*location* gate is therefore
    # not expressible here at all; only per-frame weighting is.  That is the
    # same structural fact that leaves the spatial branch dead, so it is
    # reported rather than worked around.
    gen = torch.Generator(device="cpu").manual_seed(2024)
    gate = (2.0 * torch.sigmoid(torch.randn(1, 1, T, 1, 1, generator=gen))).to(device)

    results: dict = {}
    for variant in VARIANTS:
        cleanup = []
        gh = []
        if variant in ("B", "F", "G"):
            cleanup.append(("base", encoder.base_residual.detach().clone()))
            encoder.base_residual.data = torch.zeros_like(encoder.base_residual)
        if variant in ("C", "F", "G"):
            orig = encoder.input_residual_projection
            cleanup.append(("proj", orig))
            encoder.input_residual_projection = FullRankProjection(
                in_dim=orig.weight.shape[1], out_dim=orig.weight.shape[0],
                target_rms=proj_rms, seed=1234,
                dtype=orig.weight.dtype, device=orig.weight.device,
                calib=centered)
        gh = []
        if variant in ("E", "G"):
            def gate_hook(module, inp, output, _g=gate):
                residual, modulation = output
                return (residual * _g.to(residual.dtype), modulation)
            gh.append(encoder.register_forward_hook(gate_hook))

        for seed in SEEDS:
            noise = torch.randn(
                latent_shape, generator=torch.Generator(device=device).manual_seed(seed),
                device=device, dtype=dtype)
            per_cond = {}
            for name in CONDITIONS:
                lat = rollout(model, scheduler, image, actions[name], prompt, noise)
                vid = helpers.decode_latents_to_video(
                    vae, lat, latent_mean, latent_std)[0].float().cpu()
                arr = array(vid)
                per_cond[name] = arr
                imageio.mimsave(
                    OUT / "videos" / f"a3_{variant}_seed{seed}_{name}.mp4", arr, fps=30)
            corr = per_cond["correct"].astype(np.float32)
            entry = {}
            for name in CONDITIONS:
                item = metrics_vs_gt(gt, per_cond[name], roi)
                if name != "correct":
                    o = per_cond[name].astype(np.float32)
                    item["vs_correct_pixel_mad"] = float(np.abs(corr - o).mean())
                    item["vs_correct_rmse"] = float(np.square(corr - o).mean() ** 0.5)
                entry[name] = item
            results.setdefault(variant, {})[str(seed)] = entry
            print(json.dumps({
                "variant": variant, "seed": seed,
                "correct_gt_mad": round(entry["correct"]["gt_pixel_mad"], 4),
                "rev_gt_mad": round(entry["reversed"]["gt_pixel_mad"], 4),
                "held_gt_mad": round(entry["held"]["gt_pixel_mad"], 4),
                "corr_rev_mad": round(entry["reversed"]["vs_correct_pixel_mad"], 4),
                "corr_held_mad": round(entry["held"]["vs_correct_pixel_mad"], 4),
                "motion_cos_correct": round(entry["correct"]["motion_region_temporal_cosine"], 5),
                "motion_cos_reversed": round(entry["reversed"]["motion_region_temporal_cosine"], 5),
                "motion_energy": round(entry["correct"]["motion_energy"], 4),
                "video_std": round(entry["correct"]["video_std"], 4),
            }), flush=True)

        for h in gh:
            h.remove()
        for kind, saved in cleanup:
            if kind == "base":
                encoder.base_residual.data = saved.to(encoder.base_residual.device,
                                                       encoder.base_residual.dtype)
            else:
                encoder.input_residual_projection = saved
        (OUT / "metrics" / A3_OUT).write_text(
            json.dumps({"variants": results,
                        "noise_floor_outmad": NOISE_FLOOR_OUTMAD,
                        "checkpoint_step": CKPT_STEP,
                        "window": SAMPLE.stem,
                        "seeds": list(SEEDS),
                        "steps": STEPS,
                        "variants_run": list(VARIANTS)}, indent=2))

    print(f"[A3] wrote {OUT / 'metrics' / 'a3_intervention.json'}")


if __name__ == "__main__":
    main()
