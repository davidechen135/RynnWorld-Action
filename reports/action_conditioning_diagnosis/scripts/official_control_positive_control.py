#!/usr/bin/env python3
"""Step 1: measure the control capability the OFFICIAL RynnWorld-Teleop model
actually has, using its own demo latents as the positive control.

Why this runs before anything else
----------------------------------
Every verdict in reports/native_action_v10_v11_execution/ comes from the gate in
scripts/eval_native_action_v2_rot6d37.py. Before reading that gate's verdict
("action conditioning collapsed") as a property of the task, we need the gate's
own number for a model that is known to work: the official SFT checkpoint driven
by the official hand-pose control latent, anchored on the official GT future.

The official demo latent files carry all three tensors:
  video_latents          [48,21,30,52]  the true future
  control_video_latents  [48,21,30,52]  the true skeleton control
  img_latent             [48, 1,30,52]  the true first frame
                                        (byte-identical to video_latents[:, :1])
So the same gt_pixel_mad / motion-region-cosine metrics used on the native action
gates apply verbatim, and here "correct" unambiguously means the right control for
the right future -- which is a property the AgiBot gates never had.

Two questions, in order
-----------------------
1. Correctness: on data the model is good at, does the correct skeleton reproduce
   the demo's own motion? If it does not, the eval's interpretation is wrong and
   must be repaired before any training budget is spent.
2. Sensitivity and correctness under a CHANGED skeleton: does swapping the control
   produce a reasonable response, and does the correct control still win on GT?

On `reversed`
-------------
Reversing a skeleton sequence makes the control START at the arm's END pose while
the first frame is hard-written as the START pose on every denoising step
(core/inference/rynnworld_teleop.py:452,496). That state pairing does not exist in
the training distribution, so `reversed` measures SENSITIVITY -- is the output a
function of the control at all -- and must NOT be read as a correctness test.
`swapped` (another demo's control latent, this demo's first frame and GT) and
`scaled` are the perturbations that keep the control a plausible skeleton.

Every condition reports its control-latent distance to `correct`, so sensitivity
is read as a dose-response rather than as a bare MAD.

Usage:
  python scripts/official_control_positive_control.py
  python scripts/official_control_positive_control.py --demo blowdry_hair_0_0_rgb \
      --text-embedding <path>/blowdry_hair.safetensors
  python scripts/official_control_positive_control.py --swap-from blowdry_hair_0_0_rgb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LATENT_DIR = Path("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents")
TEXT_DIR = Path("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/text_embeddings")
WIDTH, HEIGHT = 832, 480


# --------------------------------------------------------------------------- #
# Control-latent perturbations. Each returns the control tensor the DiT sees.
# --------------------------------------------------------------------------- #
def perturb(name: str, control: torch.Tensor, swap: torch.Tensor | None) -> torch.Tensor:
    if name == "correct":
        return control
    if name == "zero":
        return torch.zeros_like(control)
    if name == "static":  # first frame repeated: plausible skeleton, no motion
        return control[:, :1].expand_as(control).contiguous()
    if name == "reversed":  # sensitivity probe only -- see module docstring
        return control.flip(1).contiguous()
    if name == "shifted":
        return control.roll(control.shape[1] // 2, 1)
    if name == "permuted":
        g = torch.Generator().manual_seed(1234)
        return control[:, torch.randperm(control.shape[1], generator=g)].contiguous()
    if name == "negated":
        return -control
    if name == "smooth":  # 5-tap box filter along latent time (dim 1 of [C,T,H,W])
        t = control.shape[1]
        pad = torch.nn.functional.pad(control, (0, 0, 0, 0, 2, 2), mode="replicate")
        out = sum(pad[:, i : i + t] for i in range(5)) / 5.0
        return out.contiguous()
    if name.startswith("scaled_"):
        return control * float(name.removeprefix("scaled_"))
    if name == "sped_up":  # nearest-neighbour 2x along latent time
        idx = torch.arange(control.shape[1]) // 2
        return control[:, idx].contiguous()
    if name == "swapped":
        if swap is None:
            raise ValueError("`swapped` needs --swap-from")
        if swap.shape != control.shape:
            raise ValueError(f"swap shape {tuple(swap.shape)} != control {tuple(control.shape)}")
        return swap
    raise ValueError(f"unknown condition {name!r}")


# --------------------------------------------------------------------------- #
# Metrics -- copied verbatim from scripts/eval_native_action_v2_rot6d37.py so the
# official numbers are directly comparable to the native-action gate numbers.
# --------------------------------------------------------------------------- #
def to_uint8(frames) -> np.ndarray:
    """Pipeline `output_type='np'` -> uint8 [T,H,W,3]."""
    arr = frames.numpy() if torch.is_tensor(frames) else np.asarray(frames)
    if arr.ndim == 5:  # [1,T,C,H,W] or [1,T,H,W,C]
        arr = arr[0]
    if arr.shape[0] == 3 and arr.ndim == 4:  # [C,T,H,W] float
        arr = np.rint((np.clip(arr, -1, 1).transpose(1, 2, 3, 0) + 1) * 127.5)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.001:
            arr = arr * 255.0
        arr = np.rint(arr)
    return arr.clip(0, 255).astype(np.uint8)


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
        # Motion energy independent of pixel alignment: does the model move at all,
        # and by how much relative to GT? Guards against a low-MAD blur.
        "interframe_delta": float(np.abs(pred_motion).mean()),
        "gt_interframe_delta": float(np.abs(gt_motion).mean()),
        "motion_energy_ratio": float(
            np.abs(pred_motion).mean() / (np.abs(gt_motion).mean() + 1e-12)
        ),
    }


DEFAULT_CONDITIONS = (
    "correct", "swapped", "static", "scaled_0.5", "scaled_1.5",
    "reversed", "shifted", "smooth", "sped_up", "negated", "permuted", "zero",
)


def main(args):
    import inference_user as IU
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file
    from core.inference.rynnworld_teleop import safe_export_to_video

    device, dtype = torch.device("cuda"), torch.bfloat16
    out = Path(args.output)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    demo_path = Path(args.demo)
    if not demo_path.exists():
        demo_path = LATENT_DIR / f"{args.demo}.safetensors"
    assert demo_path.exists(), f"demo latent not found: {demo_path}"

    packed = load_file(str(demo_path))
    control = packed["control_video_latents"].to(device=device, dtype=dtype)
    gt_latent = packed["video_latents"].to(device=device, dtype=dtype)
    img_latent = packed["img_latent"].to(device=device, dtype=dtype)
    print(f"[demo] {demo_path.name}  control={tuple(control.shape)}")

    swap = None
    if args.swap_from:
        swap_path = LATENT_DIR / f"{args.swap_from}.safetensors"
        assert swap_path.exists(), f"swap source not found: {swap_path}"
        swap = load_file(str(swap_path))["control_video_latents"].to(device=device, dtype=dtype)
        print(f"[swap] {swap_path.name}  control={tuple(swap.shape)}")

    # Optional custom first frame; default keeps the official img_latent so the
    # only variable is the control signal.
    if args.image:
        vae = AutoencoderKLWan.from_pretrained(IU.MODEL_PATH, subfolder="vae").to(device=device, dtype=dtype)
        image_np = IU.read_image(args.image, HEIGHT, WIDTH)
        img_latent = IU.encode_image_to_latent(vae, image_np, device, dtype)
        del vae
        torch.cuda.empty_cache()
        print(f"[first frame] {args.image} (custom)")

    prompt_embeds = None
    if args.text_embedding:
        te = load_file(args.text_embedding)
        key = next(k for k in te if "text_embedding" in k or "prompt" in k)
        prompt_embeds = te[key].to(device=device, dtype=dtype).unsqueeze(0)
        print(f"[text] {Path(args.text_embedding).name}[{key}] {tuple(prompt_embeds.shape)}")

    inp = out / "input_latent.safetensors"
    from safetensors.torch import save_file
    save_file(
        {"video_latents": gt_latent, "control_video_latents": control, "img_latent": img_latent},
        str(inp),
    )

    print(f"[model] {args.checkpoint} mode={args.mode} control_type={args.control_type}")
    pipe = IU.load_pipeline(args.checkpoint, args.control_type, dtype,
                            mode=args.mode, use_ema=not args.no_ema)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if "swapped" in conditions and swap is None:
        conditions = [c for c in conditions if c != "swapped"]

    results, controls, videos = {}, {}, {}
    state = {"gt_uint8": None, "motion_region": None}

    def generate(ctl: torch.Tensor, seed: int) -> dict:
        save_file(
            {"video_latents": gt_latent, "control_video_latents": ctl, "img_latent": img_latent},
            str(inp),
        )
        gen = torch.Generator(device=device).manual_seed(seed)
        res = pipe(prompt="" if prompt_embeds is None else None,
                   negative_prompt="", guidance_scale=args.guidance_scale,
                   video_latent_path=str(inp), control_type=args.control_type,
                   prompt_embeds=prompt_embeds, generator=gen,
                   num_inference_steps=args.steps)
        video, gt, ctl_vid, _ = res
        if state["gt_uint8"] is None:
            state["gt_uint8"] = to_uint8(gt.frames[0])
            state["motion_region"] = gt_motion_region(state["gt_uint8"])
            safe_export_to_video(gt.frames[0], str(out / "videos" / "ground_truth.mp4"), fps=16)
            np.save(out / "motion_region.npy", state["motion_region"])
        return {"uint8": to_uint8(video.frames[0]), "raw": video.frames[0],
                "ctl_raw": ctl_vid.frames[0]}

    for name in conditions:
        ctl = perturb(name, control, swap)
        # How far did this perturbation actually move the control the DiT sees?
        ctl_delta = float((ctl.float() - control.float()).abs().mean())
        ctl_dt = float((ctl[:, 1:] - ctl[:, :-1]).abs().mean())
        controls[name] = {"control_delta_vs_correct": ctl_delta, "control_temporal_dt": ctl_dt}

        g = generate(ctl, args.seed)
        videos[name] = g["uint8"]
        m = reconstruction_metrics(state["gt_uint8"], g["uint8"], state["motion_region"])
        results[name] = m
        safe_export_to_video(g["raw"], str(out / "videos" / f"{name}.mp4"), fps=16)
        if name == "correct":
            safe_export_to_video(g["ctl_raw"], str(out / "videos" / "_control_decoded.mp4"), fps=16)
        print(f"  [{name:11s}] ctl_delta={ctl_delta:.4f} gt_mad={m['gt_pixel_mad']:6.2f} "
              f"temporal_cos={m['motion_region_temporal_cosine']:+.4f} "
              f"motion_energy={m['motion_energy_ratio']:.3f}", flush=True)

    # Noise floor: the SAME control re-sampled at other seeds. Every output MAD
    # below is only meaningful relative to this -- a perturbation that moves the
    # output no more than resampling noise has not been demonstrated to be read.
    noise_floor = None
    nf_seeds = [int(s) for s in str(args.noise_floor_seeds).split(",") if s.strip()]
    if nf_seeds and "correct" in videos:
        print(f"\n=== noise floor: `correct` re-sampled at seeds {nf_seeds} ===")
        ds = []
        for s in nf_seeds:
            h = generate(control, s)
            d = float(np.abs(videos["correct"].astype(np.float32)
                             - h["uint8"].astype(np.float32)).mean())
            ds.append(d)
            safe_export_to_video(h["raw"], str(out / "videos" / f"noise_seed{s}.mp4"), fps=16)
            print(f"  [correct@{s}] vs correct@{args.seed}: outMAD={d:.3f}", flush=True)
        noise_floor = float(np.mean(ds))
        print(f"  noise floor = {noise_floor:.3f} +/- {float(np.std(ds)):.3f} (n={len(ds)})")

    # Output-space distances to `correct`: the sensitivity half of the protocol.
    for name in conditions:
        if name == "correct":
            continue
        d = np.abs(videos["correct"].astype(np.float32) - videos[name].astype(np.float32))
        results[name]["pixel_mad_vs_correct"] = float(d.mean())
        results[name]["rmse_vs_correct"] = float(np.square(d).mean() ** 0.5)
        if noise_floor:
            results[name]["sensitivity_over_noise"] = float(d.mean() / noise_floor)

    # Gate-style summary, expressed with the same two quantities the native gate
    # demanded: sensitivity (margin over reversed) and correctness (GT MAD wins).
    corr_mad = results["correct"]["gt_pixel_mad"]
    wins = {n: int(corr_mad <= results[n]["gt_pixel_mad"])
            for n in conditions if n != "correct"}
    summary = {
        "demo": demo_path.name,
        "checkpoint": args.checkpoint,
        "control_type": args.control_type,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "shared_initial_noise": True,
        "conditions": list(conditions),
        "metrics": results,
        "control_perturbation": controls,
        "noise_floor_outmad": noise_floor,
        "correct_gt_pixel_mad": corr_mad,
        "correct_gt_pixel_mad_wins": wins,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== sensitivity (output MAD vs correct) vs dose (control latent MAD) ===")
    print(f"{'condition':12s} {'ctlΔ':>8s} {'out MAD':>9s} {'GT MAD':>8s} {'t-cos':>8s} "
          f"{'x noise':>8s} {'win':>4s}")
    for name in conditions:
        if name == "correct":
            continue
        r = results[name]
        ratio = f"{r['sensitivity_over_noise']:7.2f}x" if noise_floor else f"{'-':>8s}"
        print(f"{name:12s} {controls[name]['control_delta_vs_correct']:8.4f} "
              f"{r['pixel_mad_vs_correct']:9.3f} {r['gt_pixel_mad']:8.2f} "
              f"{r['motion_region_temporal_cosine']:8.4f} {ratio} {wins[name]:4d}")
    print(f"\n[correct] gt_mad={corr_mad:.2f} "
          f"temporal_cos={results['correct']['motion_region_temporal_cosine']:+.4f} "
          f"motion_energy_ratio={results['correct']['motion_energy_ratio']:.3f}")
    if noise_floor:
        print(f"[noise floor] re-sampling the SAME control moves the output by {noise_floor:.2f} MAD; "
              f"any condition at or below 1.0x was not demonstrated to be read.")
    print(f"[out] {out/'summary.json'}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", default="basic_fold_1_6_rgb",
                   help="official demo name (resolved under video_latents/) or path")
    p.add_argument("--swap-from", default="",
                   help="demo whose control latent stands in for `swapped`")
    p.add_argument("--image", default="", help="optional custom first frame")
    p.add_argument("--text-embedding", default="", help="official text embedding .safetensors")
    p.add_argument("--checkpoint", default="pretrained/RynnWorld-Teleop")
    p.add_argument("--mode", default="sft", choices=["sft", "lora"])
    p.add_argument("--control-type", default="add", choices=["add", "add-plus", "concat"])
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-floor-seeds", default="7,123",
                   help="comma-separated seeds that re-sample `correct` to establish "
                        "the sampling noise floor; empty string disables")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--output", default="reports/official_control_positive_control")
    main(p.parse_args())
