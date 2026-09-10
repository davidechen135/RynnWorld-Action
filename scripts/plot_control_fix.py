"""Show the control-signal fix: what the model is actually fed, before and after.

The v1 fine-tune synthesized `control_video_latents` by hand (gaussian blobs + a
sinusoidal carrier, z-scored onto the real control's mu/sigma). It matched the
first-order statistics perfectly and still taught the model nothing, because the
Conv3d control head reads motion out of spatio-temporal gradients and the hand-built
latent moved only a quarter as much frame to frame as the real control does.

Two panels, because two different things need showing:
  1. control_pixels.png  -- what the rendered skeleton control looks like (v2 only;
     v1 never had a pixel form, that IS the point)
  2. control_delta.png   -- per-clip control latent temporal delta, the number that
     actually predicts whether the control head sees anything. Real official control
     is the reference line.

Usage: python scripts/plot_control_fix.py
"""
import argparse
import glob
import os

import numpy as np

LIGHT = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
             grid="#e8e8e6", ref="#52514e",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a")
DARK = dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
            grid="#333331", ref="#c3c2b7",
            s1="#3987e5", s2="#d95926", s3="#199e70")

REAL_TEMPORAL = 0.1587      # measured on the official hand-pose control latent
REAL_SPATIAL = 0.1027


def clip_deltas(pattern):
    """Per-clip control-latent temporal delta, ordered by clip start frame."""
    from safetensors.torch import load_file
    out = []
    for p in sorted(glob.glob(pattern)):
        t = load_file(p)["control_video_latents"].float()
        start = int(os.path.basename(p).split("_")[1].split(".")[0])
        out.append((start, float(t.diff(dim=1).abs().mean())))
    return np.array(out)


def pixels_figure(control_mp4, tk, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as iio

    v = iio.mimread(control_mp4, memtest=False)
    a = np.stack([np.asarray(f[..., :3], np.uint8) for f in v])
    picks = [0, 20, 40, 60, 80]

    fig, axes = plt.subplots(1, len(picks), figsize=(15, 3.5),
                             gridspec_kw=dict(wspace=0.015, left=0.012, right=0.988,
                                              top=0.60, bottom=0.03))
    fig.patch.set_facecolor(tk["surface"])
    fig.text(0.012, 0.975, "The fixed control signal — real end-effector poses rendered as a skeleton video",
             color=tk["primary"], fontsize=15, fontweight="600", ha="left", va="top")
    fig.text(0.012, 0.885,
             "blue = left arm, red = right arm, 21 keypoints each, in the official hand-pose control's visual language.\n"
             "The open fan is an open gripper; the closed fist is a closed one. This goes through the official VAE, so the\n"
             "control head receives exactly the kind of latent it was trained on — instead of a hand-synthesized one.",
             color=tk["secondary"], fontsize=9.5, ha="left", va="top", linespacing=1.5)

    for ax, fi in zip(axes, picks):
        ax.imshow(a[min(fi, len(a) - 1)])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(tk["grid"])
        ax.set_title(f"frame {fi}", color=tk["secondary"], fontsize=9.5, pad=5)
    fig.savefig(out, dpi=110, facecolor=tk["surface"])
    print(f"-> {out}")


def delta_figure(old, new, tk, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.4, 5.6),
                           gridspec_kw=dict(left=0.078, right=0.665, top=0.70, bottom=0.11))
    fig.patch.set_facecolor(tk["surface"])
    ax.set_facecolor(tk["surface"])
    fig.text(0.078, 0.965, "Does the control latent actually move?",
             color=tk["primary"], fontsize=15, fontweight="600", ha="left", va="top")
    fig.text(0.078, 0.895,
             "Temporal Δ of control_video_latents, per training clip. The Conv3d control head reads motion\n"
             "from spatio-temporal gradients, so this — not μ/σ — is what decides whether it sees a signal.\n"
             "Both variants match the real control's μ/σ exactly; only one of them reaches its magnitude.",
             color=tk["secondary"], fontsize=9.5, ha="left", va="top", linespacing=1.5)

    ax.grid(True, color=tk["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(tk["grid"])
    ax.tick_params(colors=tk["secondary"], labelsize=9, length=0)

    ax.axhline(REAL_TEMPORAL, color=tk["ref"], linewidth=2, linestyle=(0, (5, 3)), zorder=3)
    ax.annotate(f"REAL official control  ·  {REAL_TEMPORAL:.3f}",
                (3250, REAL_TEMPORAL), xycoords=("data", "data"), va="center", ha="left",
                color=tk["primary"], fontsize=9.5, fontweight="600", annotation_clip=False)

    for (name, arr, color) in (("v1 action encoder", old, tk["s2"]),
                               ("v2 skeleton render", new, tk["s3"])):
        ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2, solid_capstyle="round",
                marker="o", markersize=5, markeredgecolor=tk["surface"],
                markeredgewidth=1.6, zorder=4)
        pct = arr[:, 1].mean() / REAL_TEMPORAL * 100
        ax.annotate(f"{name}  ·  {arr[:, 1].mean():.3f} avg  ({pct:.0f}% of real)",
                    (3250, arr[-1, 1]), xycoords=("data", "data"), va="center", ha="left",
                    color=tk["primary"], fontsize=9.5, fontweight="600", annotation_clip=False)

    ax.set_xlabel("clip start frame in the episode", color=tk["secondary"], fontsize=10)
    ax.set_ylabel("control latent temporal Δ", color=tk["secondary"], fontsize=10)
    ax.set_ylim(0, REAL_TEMPORAL * 1.25)
    ax.set_xlim(-80, 3230)
    fig.savefig(out, dpi=150, facecolor=tk["surface"])
    print(f"-> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--old", default="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357/clip_*.safetensors")
    p.add_argument("--new", default="/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel/clip_*.safetensors")
    p.add_argument("--control-mp4", default="outputs/agibot_skeleton/clip2000_control.mp4")
    p.add_argument("--output", default="outputs/agibot_control_fix")
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)

    old, new = clip_deltas(a.old), clip_deltas(a.new)
    print(f"v1 mean {old[:, 1].mean():.4f}  v2 mean {new[:, 1].mean():.4f}  real {REAL_TEMPORAL}")
    for tk, sfx in ((LIGHT, ""), (DARK, "_dark")):
        pixels_figure(a.control_mp4, tk, os.path.join(a.output, f"control_pixels{sfx}.png"))
        delta_figure(old, new, tk, os.path.join(a.output, f"control_delta{sfx}.png"))
