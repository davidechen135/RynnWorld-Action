#!/usr/bin/env python3
"""A4: minimal Phase-B feasibility simulation.  Runtime hooks only.

No training, no saved weights, no production code edits.  The question is not
"does this beat the trained model" but "does the proposed injection scheme even
move the output, and is it more separable than the current single-point add".

Proposed scheme (variant P), contrasted against the current one (variant A):

  A  current:  hidden = patch_embedding + (action + base)   at the input, once
  P  proposed: hidden = patch_embedding + base              at the input
               hidden += scale * gate * action_tokens       at blocks 0/10/20/30

with, in P:
  * ``base_residual`` untouched and kept in its own channel, so the ~20x
    offset never shares a channel with the action term;
  * the action term routed through its own full-rank projection that is not
    shared with the base path;
  * strict time alignment: token n receives the action vector of its own
    latent frame, ``t(n) = n // (gH*gW)``, with no averaging over time;
  * a small independent per-token gate (mean < 1), so the effect is a
    modulation rather than a brute-force amplification.

Measured, for both variants, over the same fixed seeds as A3:
  1. does block-level injection stably change the output
  2. is correct/reversed more separable than the single-point additive
  3. is that achieved without raising the injected magnitude
  4. does the generated picture diverge (or collapse)

Writes metrics/a4_phase_b_sim.json and videos/ under the report directory.
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
CKPT_STEP = int(os.environ.get("A4_CKPT_STEP", "300"))
SAMPLE = Path(os.environ.get(
    "A4_SAMPLE",
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v10_state_spatial_16_v1/task3400_00.safetensors",
))
STEPS = int(os.environ.get("A4_STEPS", "20"))
SEEDS = tuple(int(x) for x in os.environ.get("A4_SEEDS", "42,7,123").split(","))
CONDITIONS = ("correct", "reversed", "held", "swapped")
VARIANTS = tuple(os.environ.get("A4_VARIANTS", "A,P").split(","))
INJECT_BLOCKS = tuple(int(x) for x in os.environ.get("A4_BLOCKS", "0,10,20,30").split(","))
# Target RMS of the injected action token, as a fraction of the base_residual
# RMS.  Chosen so the total injected energy stays below the current channel's.
ACTION_BASE_RATIO = float(os.environ.get("A4_RATIO", "1.0"))
A4_OUT = os.environ.get("A4_OUT", "a4_phase_b_sim.json")
NOISE_FLOOR_OUTMAD = 7.229


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


@torch.no_grad()
def rollout(model, scheduler, image, action, prompt, noise):
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
                     robot_spatial_control=None,
                     null_condition=False).sample
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
        latents = (1 - mask) * image + mask * latents
    return latents


def build_action_tokens(encoder, action: torch.Tensor, weight: torch.Tensor,
                        gate: torch.Tensor, T: int, gH: int, gW: int,
                        scale: float) -> torch.Tensor:
    """[B,T,148] raw features -> [B, N, 3072] strictly t-aligned gated tokens.

    The raw feature vector is 148-dim; the projection expects the 768-dim
    *encoded* representation, so ``encode_centered`` comes first (same step the
    real encoder performs before ``input_residual_projection``).

    Token index is t-major (``index = t*(gH*gW) + h*gW + w``), so repeating
    each frame's vector across its own gH*gW slots keeps t to t with no
    averaging across time.
    """
    centered = encoder.encode_centered(action, T)
    proj = torch.nn.functional.linear(centered.to(weight.dtype), weight)  # [B,T,3072]
    tokens = proj.repeat_interleave(gH * gW, dim=1)                       # [B,N,3072]
    return tokens * gate.to(tokens.dtype) * scale


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
    shape = packed["video_latents"].unsqueeze(0).shape
    T = shape[2] // model.config.patch_size[0]
    gH = shape[3] // model.config.patch_size[1]
    gW = shape[4] // model.config.patch_size[2]
    N = T * gH * gW
    scheduler = UniPCMultistepScheduler.from_pretrained(BASE, subfolder="scheduler")

    gt_video = helpers.decode_latents_to_video(
        vae, packed["video_latents"].unsqueeze(0).to(device=device, dtype=dtype),
        latent_mean, latent_std)[0].float().cpu()
    gt = array(gt_video)
    roi = gt_motion_region(gt)
    imageio.mimsave(OUT / "videos" / "a4_ground_truth.mp4", gt, fps=30)

    # Independent full-rank projection for the block injection, NOT shared with
    # the base/residual path.  Shape matches the trained projection [3072,768];
    # rank is 768 (full column rank) instead of the trained one's effective 1.
    # The gain is set so the injected token RMS equals what the trained
    # projection actually produces on this window -- only the rank changes,
    # which keeps the comparison about *where and how* the action enters
    # rather than about adding energy.
    out_dim, hidden_in = encoder.input_residual_projection.weight.shape
    with torch.no_grad():
        trained_rms = float(
            encoder.input_residual_projection(
                encoder.encode_centered(correct, T)).float().pow(2).mean().sqrt().item())
        in_rms = float(encoder.encode_centered(correct, T).float().pow(2).mean().sqrt().item())
    gen = torch.Generator(device="cpu").manual_seed(4242)
    q, _ = torch.linalg.qr(torch.randn(out_dim, hidden_in, generator=gen, dtype=torch.float32))
    base_rms = float(encoder.base_residual.float().pow(2).mean().sqrt().item())
    target_rms = ACTION_BASE_RATIO * trained_rms
    centered_action = encoder.encode_centered(correct, T)
    # Calibrate the gain by direct measurement rather than by a Frobenius
    # identity.  Two earlier attempts to derive it analytically (``target /
    # (in_rms * sqrt(hidden_in))`` and ``target * sqrt(hidden_in/out_dim) /
    # in_rms``) both missed, the first by 55x and the second by 4x, because the
    # relation between gain and output RMS depends on q's normalisation in a way
    # that is easy to get wrong and invisible in the results.  Measuring the
    # uncalibrated response removes the guesswork.  This calibrates *dosage*
    # only -- it is a magnitude control, set before any outcome is seen, and is
    # not tuned against the metric.
    with torch.no_grad():
        raw_out = torch.nn.functional.linear(
            centered_action.to(q.dtype), q.to(device=device, dtype=q.dtype))
        unit_rms = float(raw_out.float().pow(2).mean().sqrt().item())
    gain = target_rms / max(unit_rms, 1e-12)
    weight = (q * gain).to(device=device, dtype=dtype)
    with torch.no_grad():
        achieved_rms = float(
            torch.nn.functional.linear(
                centered_action.to(weight.dtype), weight)
            .float().pow(2).mean().sqrt().item())
    assert abs(achieved_rms / target_rms - 1.0) < 1e-3, (
        f"gain calibration failed: achieved {achieved_rms} vs target {target_rms}")

    # small independent per-token gate, deterministic, mean well below 1
    ggate = torch.Generator(device="cpu").manual_seed(777)
    gate = (0.5 * torch.sigmoid(torch.randn(1, N, 1, generator=ggate))).to(device)
    # The gate has mean ~0.254, so injecting at gate_scale=1 delivers roughly a
    # quarter of the nominal magnitude.  Scaling it up to compensate is what
    # makes a comparison against the ungated variant a comparison of *rank and
    # placement* rather than of dosage -- otherwise a weaker result is
    # uninterpretable.  A4_GATE_SCALE is set by hand per run, never tuned
    # against the outcome.
    gate_scale = float(os.environ.get("A4_GATE_SCALE", "1.0"))

    print(f"[A4] window={SAMPLE.stem} T={T} grid={gH}x{gW} N={N} "
          f"blocks={INJECT_BLOCKS} ratio={ACTION_BASE_RATIO} "
          f"base_rms={base_rms:.5f} target_action_rms={target_rms:.5f} "
          f"achieved_action_rms={achieved_rms:.6f} "
          f"gate_mean={float(gate.mean()):.5f}", flush=True)

    base_orig = encoder.base_residual.detach().clone()
    results: dict = {}

    for variant in VARIANTS:
        handles = []
        # --- variant wiring -------------------------------------------------
        if variant == "P":
            # 1. action removed from the patch-embedding input; base kept
            def encoder_hook(module, inp, output):
                _residual, modulation = output
                b = _residual.shape[0]
                return (base_orig.to(_residual.dtype).expand(
                    b, -1, T, -1, -1), modulation)
            handles.append(encoder.register_forward_hook(encoder_hook))

            # 2. strict t-to-t block injection at the chosen depths.  The
            #    trainer calls blocks with all-positional args, so the pre-hook
            #    returns a replacement args tuple.
            def make_block_hook(tokens):
                def hook(module, args):
                    hs = args[0]
                    return (hs + tokens.to(hs.dtype),) + tuple(args[1:])
                return hook

        for seed in SEEDS:
            noise = torch.randn(
                shape, generator=torch.Generator(device=device).manual_seed(seed),
                device=device, dtype=dtype)
            per_cond = {}
            for name in CONDITIONS:
                if variant == "P":
                    for h in handles[1:]:
                        h.remove()
                    handles = handles[:1]
                    tokens = build_action_tokens(
                        encoder, actions[name], weight, gate, T, gH, gW, gate_scale)
                    for bi in INJECT_BLOCKS:
                        if bi < len(model.blocks):
                            handles.append(model.blocks[bi].register_forward_pre_hook(
                                make_block_hook(tokens)))
                lat = rollout(model, scheduler, image, actions[name], prompt, noise)
                vid = helpers.decode_latents_to_video(
                    vae, lat, latent_mean, latent_std)[0].float().cpu()
                arr = array(vid)
                per_cond[name] = arr
                imageio.mimsave(
                    OUT / "videos" / f"a4_{variant}_seed{seed}_{name}.mp4", arr, fps=30)
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
                "corr_rev_mad": round(entry["reversed"]["vs_correct_pixel_mad"], 5),
                "corr_held_mad": round(entry["held"]["vs_correct_pixel_mad"], 5),
                "motion_cos_correct": round(entry["correct"]["motion_region_temporal_cosine"], 5),
                "motion_cos_reversed": round(entry["reversed"]["motion_region_temporal_cosine"], 5),
                "video_std": round(entry["correct"]["video_std"], 4),
            }), flush=True)

        for h in handles:
            h.remove()

    encoder.base_residual.data = base_orig
    (OUT / "metrics" / A4_OUT).write_text(json.dumps({
        "variants": results,
        "variant_notes": {
            "A": "current: single additive (action+base) at patch-embedding input",
            "P": "proposed: base-only at input, strict t-to-t action injected at "
                 f"blocks {list(INJECT_BLOCKS)} via an independent full-rank "
                 "projection with a small independent per-token gate",
        },
        "inject_blocks": list(INJECT_BLOCKS),
        "action_base_ratio": ACTION_BASE_RATIO,
        "base_residual_rms": base_rms,
        "target_action_rms": target_rms,
        "projection_achieved_rms": achieved_rms,
        "gate_mean": float(gate.mean()),
        "gate_scale": gate_scale,
        "effective_action_rms": achieved_rms * float(gate.mean()) * gate_scale,
        "noise_floor_outmad": NOISE_FLOOR_OUTMAD,
        "checkpoint_step": CKPT_STEP,
        "window": SAMPLE.stem,
        "seeds": list(SEEDS),
        "steps": STEPS,
    }, indent=2))
    print(f"[A4] wrote {OUT / 'metrics' / A4_OUT}")


if __name__ == "__main__":
    main()
