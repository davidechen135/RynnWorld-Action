#!/usr/bin/env python3
"""Step 2: inference-time A/B/C/D ablation of the native-action injection.

One factor per arm, nothing else changes:

  A  as-is                     current checkpoint, both base and projection intact
  B  base-off                  only `base_residual` zeroed
  C  proj-swap                 only `input_residual_projection` replaced by a
                               random matrix matched to the original's output RMS
  D  base-off + proj-swap      both, at the same time

Why the arms are shaped this way
--------------------------------
With injection `b + f(action)`, the correct-vs-reversed difference is
`f(correct) - f(reversed)` -- the constant `b` cancels exactly. So a large
static base does NOT by itself prove collapse. It can still matter through the
downstream normalization/nonlinearity it shifts the input of. That is an
empirical question, which is why B exists.

Symmetrically, the earlier full-rank projection probe raised the *encoding*
difference 5.2x. That shows the trained projection compresses distinguishable
information; it does NOT show the generated video becomes more correct, and it
does NOT prove a swapped initialization must re-collapse. That is why C/D exist,
and why they are RMS-matched: an unmatched swap would confound "full rank" with
"larger magnitude". Several fixed seeds are run so a single lucky matrix cannot
be read as the effect.

RMS matching
------------
The replacement is `g = c * (W A)` where `A` is a random `hidden -> hidden`
orthogonal matrix and `c` is chosen so that `rms(g(x)) == rms(W(x))` on the
actual batch of `centered` features the encoder produces for this sample. That
makes C/D a pure change of *which* directions carry the signal, at constant
injected magnitude.

Reading the result
------------------
This is a diagnostic, not a fix. Inference-time replacement of a trained
representation is a distribution shift: the DiT was trained against `W`, so even
a "better" `g` can look worse. A null result does not falsify the projection
hypothesis -- it means this test cannot see it, and the answer has to come from
Step 3's joint training instead.

Reported per arm and per condition: action response (output MAD vs its own
arm's `correct`), motion-vs-GT consistency (temporal cosine), flicker
(interframe MAD at fixed stride), image quality (laplacian variance, contrast).

Usage
-----
  python scripts/ablate_native_action_injection.py --list
  python scripts/ablate_native_action_injection.py --arms A,B --conditions correct,reversed
  python scripts/ablate_native_action_injection.py            # full sweep
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HELPERS = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
BASE = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN = REPO / "pretrained/RynnWorld-Teleop-Causal"
TRAIN = REPO / "training/native_action_v10b_state_256_600"
CKPT = 400
DATA = Path("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1")

# The same four high-arm-motion, episode-disjoint dev windows used by the gate in
# reports/native_action_arm_dynamic_dev/summary.json.
SAMPLES = (
    ("task3400_episode443_start768", "task3400_0242.safetensors"),
    ("task3400_episode437_start1328", "task3400_0224.safetensors"),
    ("task3401_episode1055_start1488", "task3401_0192.safetensors"),
    ("task3401_episode1158_start2384", "task3401_0008.safetensors"),
)
DEFAULT_CONDITIONS = ("correct", "reversed", "held", "shifted", "zero")
DEFAULT_PROJ_SEEDS = (0, 1, 2)
ARMS = ("A", "B", "C", "D")

# `zero` is not a free choice: in the eval harness it is `torch.zeros_like(correct)`,
# which makes `centered == 0`, so the residual is exactly `base_residual` and the
# modulation is exactly `base_modulation`. That makes `zero` the base-only rollout
# the gate already reports -- and it is why `zero` is byte-identical across
# checkpoints. Verified as a deliberate design, not a loading bug: the trainer
# freezes both params (rynnworld_teleop_trainer.py:869-871) and a prior baseline is
# copied in (:697). `base_residual` rms is 0.8433 at every checkpoint.
#
# `robot_spatial_control` is left as None, matching reports/native_action_arm_dynamic_dev.
# That is safe here: V10's `spatial_stem[-1].weight` is zero-initialised and still has
# rms 0.0 in checkpoint-400, so the spatial branch contributes exactly zero and the
# spatial-control factor cannot confound this ablation.


# --------------------------------------------------------------------------- #
# Helpers shared with the gate
# --------------------------------------------------------------------------- #
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


def arm_metrics(gt: np.ndarray, pred: np.ndarray, motion_region: np.ndarray) -> dict:
    g = gt.astype(np.float32)
    p = pred.astype(np.float32)
    gt_motion = np.diff(g, axis=0)
    pred_motion = np.diff(p, axis=0)
    a = gt_motion[:, motion_region, :].reshape(-1)
    b = pred_motion[:, motion_region, :].reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # Flicker: frame-to-frame change at a fixed stride, on the whole frame. High
    # values mean temporal instability, independent of whether the motion is
    # correct. Gray frames are excluded from the quality stats below.
    stride = 4
    flicker = float(np.abs(p[stride:] - p[:-stride]).mean())

    gray = p.mean(axis=3)
    lap = (
        gray[:, 2:, 2:] + gray[:, :-2, 2:] + gray[:, 2:, :-2] + gray[:, :-2, :-2]
        - 4 * gray[:, 1:-1, 1:-1]
    )
    return {
        "gt_pixel_mad": float(np.abs(g - p).mean()),
        "motion_region_pixel_mad": float(np.abs(g - p)[:, motion_region, :].mean()),
        "motion_region_temporal_cosine": cos,
        "interframe_delta": float(np.abs(pred_motion).mean()),
        "gt_interframe_delta": float(np.abs(gt_motion).mean()),
        "motion_energy_ratio": float(
            np.abs(pred_motion).mean() / (np.abs(gt_motion).mean() + 1e-12)
        ),
        "flicker_stride4": flicker,
        "laplacian_variance": float(lap.var()),
        "frame_std": float(gray.std()),
    }


# --------------------------------------------------------------------------- #
# The ablation itself
# --------------------------------------------------------------------------- #
def matched_random_projection(weight: torch.Tensor, features: torch.Tensor, seed: int):
    """A random map with the same shape and the same output RMS as `weight`.

    Deliberately a plain Gaussian rather than an orthogonalised map: `weight` is
    [out_dim, in_dim] with out_dim > in_dim, so its rows cannot be orthonormal, and
    even a reduced QR would collapse to a square block of the wrong shape. Matching
    the singular-value spectrum is also not what is wanted here -- preserving the
    original's spectrum would preserve the rank collapse under test. So the only
    things held fixed are shape and output magnitude; everything else is randomised.

    The caller records both matrices' energy spectra, so the difference in rank
    structure is reported rather than hidden.
    """
    out_dim, in_dim = weight.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn(out_dim, in_dim, generator=generator, dtype=torch.float32).to(weight.device)
    with torch.no_grad():
        x = features.float().reshape(-1, in_dim)
        ref = weight.float() @ x.T
        cand = g @ x.T
        scale = (ref.square().mean().sqrt() / (cand.square().mean().sqrt() + 1e-12)).item()
    return g * scale, scale


def energy_spectrum(weight: torch.Tensor, top: int = 4) -> dict:
    sv = torch.linalg.svdvals(weight.float())
    energy = sv.square()
    total = energy.sum()
    return {
        "rank_at_99pct": int((energy.cumsum(0) / total < 0.99).sum().item()) + 1,
        "top_energy_fracs": [float(v) for v in (energy[:top] / total)],
        "stable_rank": float(total / (energy[0] + 1e-12)),
    }


def make_encoder():
    from core.control.native_trajectory_encoder import NativeTrajectoryConditionerV10

    enc = NativeTrajectoryConditionerV10(input_dim=148)
    enc.load_state_dict(
        torch.load(TRAIN / f"checkpoint-{CKPT}/native_trajectory_encoder.bin",
                   map_location="cpu", weights_only=False),
        strict=False,
    )
    return enc.eval()


def apply_arm(enc, arm: str, proj_seed: int, centered_probe: torch.Tensor):
    """Mutate `enc` in place for this arm. Returns a record of what was changed."""
    rec = {"arm": arm, "proj_seed": proj_seed}
    if arm in ("B", "D"):
        with torch.no_grad():
            rec["base_residual_rms_before"] = float(
                enc.base_residual.float().square().mean().sqrt()
            )
            enc.base_residual.zero_()
            rec["base_modulation_rms_before"] = float(
                enc.base_modulation.float().square().mean().sqrt()
            )
            enc.base_modulation.zero_()
    if arm in ("C", "D"):
        w = enc.input_residual_projection.weight.data
        rec["proj_original_spectrum"] = energy_spectrum(w)
        g, scale = matched_random_projection(w, centered_probe, proj_seed)
        rec["proj_swap_scale"] = scale
        rec["proj_swapped_spectrum"] = energy_spectrum(g)
        with torch.no_grad():
            w.copy_(g.to(w.dtype))
    return rec


@torch.no_grad()
def rollout(model, scheduler, image, action, prompt, noise, steps):
    dtype, device = model.patch_embedding.weight.dtype, image.device
    latents = noise.clone()
    b, _, frames, h, w = latents.shape
    _, ph, pw = model.config.patch_size
    mask = torch.ones((b, 1, frames, h, w), device=device, dtype=dtype)
    mask[:, :, 0] = 0
    scheduler.set_timesteps(steps, device=device)
    for t in scheduler.timesteps:
        model_input = (1 - mask) * image + mask * latents
        timestep = (mask[0, 0, :, ::ph, ::pw] * t).flatten().unsqueeze(0)
        pred = model(
            hidden_states=model_input, timestep=timestep,
            encoder_hidden_states=prompt, encoder_hidden_states_image=None,
            robot_trajectory=action, robot_spatial_control=None,
            null_condition=False,
        ).sample
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
        latents = (1 - mask) * image + mask * latents
    return latents


def build_actions(packed, device):
    from core.control.native_action_features import build_v10_features

    raw = packed["robot_trajectory_raw37"].unsqueeze(0).float().to(device)
    state = packed["robot_observed_state37"].unsqueeze(0).float().to(device)
    mean = packed["robot_trajectory_mean37"].float().to(device)
    std = packed["robot_trajectory_std37"].float().to(device)
    rs = packed["robot_relative_scale37"].float().to(device)
    vs = packed["robot_velocity_scale37"].float().to(device)

    def make(src):
        return build_v10_features(src, state, mean, std, rs, vs)

    held = state[:, None].expand_as(raw)
    correct = make(raw)
    return {
        "correct": correct,
        "reversed": make(raw.flip(1)),
        "held": make(held),
        "shifted": make(raw.roll(raw.shape[1] // 2, 1)),
        # Zero the built 148-d feature, matching the eval harness exactly
        # (`actions["zero"] = torch.zeros_like(correct)`). Zeroing the raw 37-d
        # trajectory instead would NOT be the same thing: the builder derives
        # `relative = target - observed_state`, so a zero raw trajectory still
        # carries a non-zero relative term.
        "zero": torch.zeros_like(correct),
    }


def main(args):
    import imageio
    from safetensors.torch import load_file
    from diffusers import UniPCMultistepScheduler

    out = Path(args.output)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    partial_path = out / "partial.json"
    results = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    if args.reset:
        results = {}

    helpers = load_module(HELPERS, "rot6d37_eval_helpers")
    helpers.apply_monkey_patch(REPO)
    device, dtype = torch.device("cuda"), torch.bfloat16

    tokenizer, text_encoder = helpers.load_text_encoder(BASE, device, dtype)
    prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    vae, latent_mean, latent_std = helpers.load_vae(BASE, device, dtype)
    model = helpers.load_model(RYNN, BASE, sys.stdout)
    scheduler = UniPCMultistepScheduler.from_pretrained(BASE, subfolder="scheduler")

    samples = SAMPLES
    if args.samples:
        wanted = set(args.samples.split(","))
        samples = tuple(s for s in SAMPLES if s[0] in wanted)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    proj_seeds = tuple(int(s) for s in args.proj_seeds.split(","))

    for sample_name, filename in samples:
        packed = load_file(str(DATA / filename))
        image = packed["img_latent"].unsqueeze(0).to(device=device, dtype=dtype)
        shape = packed["video_latents"].unsqueeze(0).shape
        noise = torch.randn(
            shape, generator=torch.Generator(device=device).manual_seed(args.seed),
            device=device, dtype=dtype,
        )
        actions = build_actions(packed, device)

        gt_video = helpers.decode_latents_to_video(
            vae, packed["video_latents"].unsqueeze(0).to(device=device, dtype=dtype),
            latent_mean, latent_std,
        )[0].float().cpu()
        gt_array = array(gt_video)
        region = gt_motion_region(gt_array)

        for arm in arms:
            seeds = proj_seeds if arm in ("C", "D") else (0,)
            for pseed in seeds:
                key = f"{sample_name}|{arm}|p{pseed}"
                if key in results and not args.force:
                    print(f"[skip] {key}", flush=True)
                    continue

                enc = make_encoder().to(device=device, dtype=dtype).eval()
                # Probe features the encoder actually produces, so RMS matching is
                # against the real `centered` distribution rather than a guess.
                with torch.no_grad():
                    probe = enc.encode_centered(actions["correct"], 9).float()
                rec = apply_arm(enc, arm, pseed, probe)
                model.native_trajectory_encoder = enc

                # Measure the injected residual's temporal structure per condition.
                dose = {}
                with torch.no_grad():
                    for name in conditions:
                        r, m = enc(actions[name], 9)
                        r = r.float().squeeze(0).squeeze(-1).squeeze(-1)
                        mu = r.mean(dim=1, keepdim=True)
                        tv = r - mu
                        dose[name] = {
                            "residual_total_rms": float(r.square().mean().sqrt()),
                            "residual_tv_rms": float(tv.square().mean().sqrt()),
                            "residual_tv_frac": float(
                                tv.square().mean().sqrt()
                                / (r.square().mean().sqrt() + 1e-12)
                            ),
                            "modulation_rms": float(m.float().square().mean().sqrt()),
                        }

                videos = {}
                for name in conditions:
                    latent = rollout(model, scheduler, image, actions[name], prompt,
                                     noise, args.steps)
                    video = helpers.decode_latents_to_video(
                        vae, latent, latent_mean, latent_std
                    )[0].float().cpu()
                    videos[name] = array(video)
                    imageio.mimsave(
                        out / "videos" / f"{sample_name}_{arm}_p{pseed}_{name}.mp4",
                        videos[name], fps=30,
                    )

                entry = {"arm": arm, "proj_seed": pseed, "injection": dose,
                         "arm_record": rec, "conditions": {}}
                correct = videos["correct"].astype(np.float32)
                for name in conditions:
                    m = arm_metrics(gt_array, videos[name], region)
                    if name != "correct":
                        d = np.abs(correct - videos[name].astype(np.float32))
                        m["output_mad_vs_arm_correct"] = float(d.mean())
                    entry["conditions"][name] = m

                results[key] = entry
                partial_path.write_text(json.dumps(results, indent=2))
                c = entry["conditions"]
                print(
                    f"[{key}] correct gt_mad={c['correct']['gt_pixel_mad']:.2f} "
                    f"cos={c['correct']['motion_region_temporal_cosine']:+.4f} | "
                    f"rev outMAD={c.get('reversed', {}).get('output_mad_vs_arm_correct', float('nan')):.2f} "
                    f"tv_frac={dose['correct']['residual_tv_frac']:.6f}",
                    flush=True,
                )

    (out / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(TRAIN / f"checkpoint-{CKPT}"),
                "samples": [s[0] for s in samples],
                "arms": list(arms),
                "proj_seeds": list(proj_seeds),
                "conditions": list(conditions),
                "steps": args.steps,
                "seed": args.seed,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"[out] {out/'summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    p.add_argument("--proj-seeds", default=",".join(str(s) for s in DEFAULT_PROJ_SEEDS))
    p.add_argument("--samples", default="", help="comma-separated sample names; default all four")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="reports/action_conditioning_diagnosis/ablation")
    p.add_argument("--reset", action="store_true", help="discard existing partial.json")
    p.add_argument("--force", action="store_true", help="redo keys already present")
    p.add_argument("--list", action="store_true", help="print the sample table and exit")
    args = p.parse_args()
    if args.list:
        for n, f in SAMPLES:
            print(f"  {n:34s} {f}")
        raise SystemExit(0)
    main(args)
