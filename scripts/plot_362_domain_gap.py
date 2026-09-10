"""Why the rollout collapses on AgiBot data but not on the official sample clips.

This is the finding that survived every other hypothesis. Fixing the zero-initialized
control path (--control_init_from) did NOT stop the collapse, and the latent statistics
of my clips match the official ones almost exactly (video std 0.80 vs 0.78-0.89, control
std 1.63 vs 1.68-1.70), so neither the harness defect nor a scale mismatch explains it.

What does differ is how much MOTION the tensors carry. Measured as mean |x_t - x_{t-1}|
on the stored latents:

  official clips     control 0.132-0.152   video 0.38-0.41
  my AgiBot clips    control 0.074-0.092   video 0.28

The control signal carries roughly half the official motion, and the AgiBot head-camera
video is itself ~70% as dynamic. The SFT model was trained where control is strong
relative to the video; given a weaker control signal it does not stay anchored and the
rollout drifts, then collapses.

The control experiment that pins this down is the second panel: the SAME model and code,
run on an official clip, does NOT collapse (detail ratio 0.917) while every AgiBot clip
does (0.605-0.66). That rules out inference config and the checkpoint.

Usage: python scripts/plot_362_domain_gap.py [--dark]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file

LIGHT = dict(bg="#fcfcfb", ink="#1a1a19", ink2="#5c5c58", grid="#e4e4e0",
             off="#3f8f5b", mine="#d95f02", gt="#5c5c58")
DARK = dict(bg="#1a1a19", ink="#f2f2ef", ink2="#a8a8a2", grid="#33332f",
            off="#3f9e68", mine="#e07020", gt="#a8a8a2")

CLIPS = [
    ("official\njigsaw", "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents/assemble_disassemble_jigsaw_puzzle_0_0_rgb.safetensors", "off"),
    ("official\nfold", "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents/basic_fold_1_6_rgb.safetensors", "off"),
    ("official\njenga", "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/video_latents/assemble_jenga_0_1_rgb.safetensors", "off"),
    ("mine 362\ntrain", "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel/650989/clip_000000.safetensors", "mine"),
    ("mine 362\nheld-out", "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel_heldout/650190/clip_000729.safetensors", "mine"),
    ("mine 357\n(v2)", "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_357_skel/clip_000000.safetensors", "mine"),
]

# detail ratio measured by the held-out eval, averaged over all 3 held-out episodes
# (per-clip values in outputs/agibot_362_sweep/sweep_metrics_v{1,3}_*.json)
COLLAPSE = [
    ("official fold\n(zero-shot)", 0.917, "off"),
    ("AgiBot mean\n(zero-shot)", 0.651, "mine"),
    ("AgiBot mean\n(LoRA v1)", 0.663, "mine"),
    ("AgiBot mean\n(LoRA warm-start)", 0.635, "mine"),
]


def tdelta(path, key):
    t = load_file(path)[key].float()
    return (t[:, 1:] - t[:, :-1]).abs().mean().item()


def main(a):
    C = DARK if a.dark else LIGHT

    rows = []
    for label, path, kind in CLIPS:
        if not os.path.exists(path):
            continue
        rows.append((label, tdelta(path, "control_video_latents"),
                     tdelta(path, "video_latents"), kind))

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    fig.patch.set_facecolor(C["bg"])
    fig.subplots_adjust(wspace=0.24)
    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.grid(True, axis="y", color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C["grid"])
        ax.tick_params(colors=C["ink2"], labelsize=9)

    # ---- panel 1: motion carried by control vs video ----
    ax = axes[0]
    x = np.arange(len(rows))
    w = 0.38
    ctl = [r[1] for r in rows]
    vid = [r[2] for r in rows]
    cols = [C[r[3]] for r in rows]
    ax.bar(x - w / 2, ctl, w, color=cols, zorder=3)
    ax.bar(x + w / 2, vid, w, color=cols, alpha=0.45, zorder=3)
    for i in range(len(rows)):
        ax.annotate(f"{ctl[i]:.3f}", xy=(i - w / 2, ctl[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=C["ink"], fontsize=8)
        ax.annotate(f"{vid[i]:.3f}", xy=(i + w / 2, vid[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=C["ink2"], fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5, color=C["ink2"])
    ax.set_ylabel("temporal delta of stored latents", color=C["ink"], fontsize=10)
    ax.set_title("My control signal carries ~half the official motion\n"
                 "(solid = control, faded = video)",
                 color=C["ink"], fontsize=10.5, fontweight="bold", loc="left", pad=10)
    ax.annotate("official band", xy=(1, max(ctl[:3])), xytext=(0, 26),
                textcoords="offset points", ha="center",
                color=C["off"], fontsize=9, fontweight="bold")
    ax.annotate("mine", xy=(4, max(ctl[3:])), xytext=(0, 26), textcoords="offset points",
                ha="center", color=C["mine"], fontsize=9, fontweight="bold")
    ax.set_ylim(top=max(vid) * 1.28)

    # ---- panel 2: does it collapse? ----
    ax = axes[1]
    x2 = np.arange(len(COLLAPSE))
    vals = [c[1] for c in COLLAPSE]
    ax.bar(x2, vals, 0.55, color=[C[c[2]] for c in COLLAPSE], zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=C["ink"], fontsize=9, fontweight="bold")
    ax.axhline(1.0, color=C["gt"], lw=1.5, ls="--", zorder=4)
    ax.annotate("GT holds structure (mean 1.105)", xy=(len(COLLAPSE) - 0.5, 1.0), xytext=(0, 6),
                textcoords="offset points", ha="right", color=C["ink"],
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.set_xticks(x2)
    ax.set_xticklabels([c[0] for c in COLLAPSE], fontsize=8.5, color=C["ink2"])
    ax.set_ylabel("detail ratio (last third / first third)", color=C["ink"], fontsize=10)
    ax.set_title("Same model, same code: official clip holds,\nevery AgiBot arm collapses alike",
                 color=C["ink"], fontsize=10.5, fontweight="bold", loc="left", pad=10)

    fig.suptitle("Root cause: a control-strength domain gap, not the harness defect and not data volume",
                 color=C["ink"], fontsize=12.5, fontweight="bold", x=0.005, ha="left", y=1.03)
    fig.text(0.005, -0.07,
             "Latent scale matches the official clips (video std 0.80 vs 0.78-0.89, control std 1.63 vs "
             "1.68-1.70), so this is not a normalization bug.\nWarm-starting the control path from SFT "
             "restored its magnitude (0.0369 vs SFT 0.0364) but did NOT stop the collapse (0.663 -> 0.635 "
             "mean, slightly worse),\nwhich rules out the zero-init defect as the cause. Zero-shot, LoRA v1 "
             "and LoRA warm-start all sit within +/-0.03 of each other and ~0.45\nbelow GT. What remains is "
             "signal strength: AgiBot's fitted-camera skeleton control carries ~half the motion the SFT "
             "model relies on.",
             color=C["ink2"], fontsize=8.7, ha="left", va="top")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[saved] {a.out}")
    for r in rows:
        print(f"  {r[0]:22s} ctl_dt={r[1]:.4f}  vid_dt={r[2]:.4f}  ratio={r[1]/r[2]:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dark", action="store_true")
    p.add_argument("--out", default="outputs/agibot_362_plots/domain_gap_362.png")
    a = p.parse_args()
    if a.dark and a.out == "outputs/agibot_362_plots/domain_gap_362.png":
        a.out = "outputs/agibot_362_plots/domain_gap_362_dark.png"
    main(a)
