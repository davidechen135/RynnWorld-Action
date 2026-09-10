"""The root cause, as one figure: the control path is thrown away at training start.

`control_type=add` builds control_patch_embedding as a ZERO-initialized Conv3d
(core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py, the `else` branch of the
is_concat check) and hardcodes control_scale=0.1. It never reads the trained
control_patch_embedding.bin / control_scale.bin that ship with the SFT checkpoint. So a
short LoRA run does not adapt the pretrained control path -- it relearns it from zero.

Measured after 100 steps: weight absmax 0.001167 vs the SFT checkpoint's 0.036377, i.e.
31x smaller. Control barely reaches the denoiser, the rollout has almost no guidance,
and it melts from frame ~2 (outputs/agibot_362_sweep/good_eps_trace.png).

Log scale is used on the magnitude panel because the values span 1.5 orders of
magnitude; the axis is labeled as log and the ratio is annotated directly so the
compression cannot mislead.

Usage: python scripts/plot_362_control_diag.py [--dark]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

LIGHT = dict(bg="#fcfcfb", ink="#1a1a19", ink2="#5c5c58", grid="#e4e4e0",
             sft="#3f8f5b", run="#d95f02")
DARK = dict(bg="#1a1a19", ink="#f2f2ef", ink2="#a8a8a2", grid="#33332f",
            sft="#3f9e68", run="#e07020")

SOURCES = [
    ("SFT released\n(pretrained)", "pretrained/RynnWorld-Teleop", "sft"),
    ("LoRA v1 ckpt-24", "training/agibot_362_lora/checkpoint-24", "run"),
    ("LoRA v1 ckpt-100", "training/agibot_362_lora/checkpoint-100", "run"),
]


def read(root):
    cpe = torch.load(os.path.join(root, "control_patch_embedding.bin"),
                     map_location="cpu", weights_only=False)
    absmax = max(t.abs().max().item() for t in cpe.values())
    cs_path = os.path.join(root, "control_scale.bin")
    cs = None
    if os.path.exists(cs_path):
        v = torch.load(cs_path, map_location="cpu", weights_only=False)
        cs = v.item() if hasattr(v, "item") else float(v)
    return absmax, cs


def main(a):
    C = DARK if a.dark else LIGHT
    rows = []
    for label, root, kind in SOURCES:
        if not os.path.exists(root):
            continue
        am, cs = read(root)
        rows.append((label, am, cs, kind))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    fig.patch.set_facecolor(C["bg"])
    fig.subplots_adjust(wspace=0.28)
    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.grid(True, axis="y", color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C["grid"])
        ax.tick_params(colors=C["ink2"], labelsize=9)

    x = np.arange(len(rows))
    labels = [r[0] for r in rows]
    colors = [C[r[3]] for r in rows]

    # ---- panel 1: control_patch_embedding magnitude (log) ----
    ax = axes[0]
    vals = [r[1] for r in rows]
    ax.bar(x, vals, 0.55, color=colors, zorder=3)
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.5f}", xy=(i, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=C["ink"], fontsize=8.5, fontweight="bold")
    sft = vals[0]
    for i, v in enumerate(vals[1:], start=1):
        # place the ratio INSIDE the plot above the bar (offsetting downward collided
        # with the x tick labels), keeping it clear of the value label above it
        ax.annotate(f"{sft / v:.0f}× smaller", xy=(i, v), xytext=(0, 20),
                    textcoords="offset points", ha="center", color=C["ink2"], fontsize=8.5)
    ax.set_ylim(top=sft * 2.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=C["ink2"])
    ax.set_ylabel("control_patch_embedding |w| max  (log)", color=C["ink"], fontsize=10)
    ax.set_title("Control path relearned from zero,\nand never catches up",
                 color=C["ink"], fontsize=10.5, fontweight="bold", loc="left", pad=10)

    # ---- panel 2: control_scale ----
    ax = axes[1]
    cs = [(r[0], r[2], r[3]) for r in rows if r[2] is not None]
    x2 = np.arange(len(cs))
    ax.bar(x2, [c[1] for c in cs], 0.55, color=[C[c[2]] for c in cs], zorder=3)
    for i, c in enumerate(cs):
        ax.annotate(f"{c[1]:.4f}", xy=(i, c[1]), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=C["ink"], fontsize=8.5, fontweight="bold")
    ax.axhline(cs[0][1], color=C["sft"], lw=1.4, ls="--", zorder=4)
    ax.annotate("SFT value 0.1094", xy=(len(cs) - 0.5, cs[0][1]), xytext=(0, 5),
                textcoords="offset points", ha="right", color=C["ink"],
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels([c[0] for c in cs], fontsize=9, color=C["ink2"])
    ax.set_ylabel("control_scale", color=C["ink"], fontsize=10)
    ax.set_title("control_scale hardcoded to 0.1,\nnot inherited",
                 color=C["ink"], fontsize=10.5, fontweight="bold", loc="left", pad=10)

    fig.suptitle("Root cause: control_type=add zero-initializes the control path each run",
                 color=C["ink"], fontsize=13, fontweight="bold", x=0.005, ha="left", y=1.02)
    fig.text(0.005, -0.06,
             "rynnworld_teleop_trainer.py builds control_patch_embedding as a zero-init Conv3d and sets "
             "control_scale=0.1, ignoring the\ntrained control_patch_embedding.bin / control_scale.bin in the "
             "SFT checkpoint. A ~100-step LoRA run cannot relearn it,\nso the control signal barely reaches the "
             "denoiser and the rollout collapses. Fix under test: --control_init_from <sft_dir>.",
             color=C["ink2"], fontsize=8.8, ha="left", va="top")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[saved] {a.out}")
    for r in rows:
        print(f"  {r[0]:28s} cpe_absmax={r[1]:.6f}  control_scale={r[2]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dark", action="store_true")
    p.add_argument("--out", default="outputs/agibot_362_plots/control_diag_362.png")
    a = p.parse_args()
    if a.dark and a.out == "outputs/agibot_362_plots/control_diag_362.png":
        a.out = "outputs/agibot_362_plots/control_diag_362_dark.png"
    main(a)
