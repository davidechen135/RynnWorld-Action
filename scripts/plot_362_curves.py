"""Training curves for the task-362 LoRA run: loss / grad_norm / lr from tensorboard.

Three stacked panels sharing the x axis (step) -- NOT a dual-axis chart. loss,
grad_norm and lr have unrelated units, so overlaying them on twin y-axes would
manufacture crossings that mean nothing. Stacked panels keep one scale per axis.

loss is logged per micro-batch (792 points over 100 optimizer steps), so the raw
series is a noise cloud; we draw it faint and overlay a per-optimizer-step mean plus
a rolling mean, which is what "is it converging" actually asks. grad_norm and lr are
already per-step.

Palette validated with the dataviz skill's validate_palette.js (light + dark, all
checks pass); every series is direct-labeled so identity never rests on hue alone.

Usage: python scripts/plot_362_curves.py [--dark]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGDIR = "training/agibot_362_lora/logs/finetrainer-cogvideo"

LIGHT = dict(bg="#fcfcfb", ink="#1a1a19", ink2="#5c5c58", grid="#e4e4e0",
             series=("#1f78b4", "#d95f02", "#3f8f5b"))
DARK = dict(bg="#1a1a19", ink="#f2f2ef", ink2="#a8a8a2", grid="#33332f",
            series=("#3f8ec4", "#e07020", "#3f9e68"))


def load_scalars(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir)
    ea.Reload()
    out = {}
    for tag in ("loss", "lr", "grad_norm"):
        s = ea.Scalars(tag)
        out[tag] = (np.array([x.step for x in s], float),
                    np.array([x.value for x in s], float))
    return out


def per_step_mean(steps, vals):
    """Collapse the per-micro-batch loss cloud to one mean per optimizer step."""
    us = np.unique(steps)
    return us, np.array([vals[steps == s].mean() for s in us])


def main(a):
    C = DARK if a.dark else LIGHT
    d = load_scalars(a.logdir)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.5, 1, 1], hspace=0.16))
    fig.patch.set_facecolor(C["bg"])

    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.grid(True, color=C["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C["grid"])
        ax.tick_params(colors=C["ink2"], labelsize=9)

    # ---- panel 1: loss ----
    ax = axes[0]
    st, lv = d["loss"]
    ax.plot(st, lv, color=C["series"][0], lw=0.8, alpha=0.22, zorder=2)
    ms, mv = per_step_mean(st, lv)
    ax.plot(ms, mv, color=C["series"][0], lw=2.0, zorder=3)
    k = 9
    if len(mv) >= k:
        roll = np.convolve(mv, np.ones(k) / k, mode="valid")
        rx = ms[k - 1:]
        ax.plot(rx, roll, color=C["ink"], lw=2.0, ls="--", zorder=4)
        ax.annotate(f"rolling mean ({k} steps)", xy=(rx[-1], roll[-1]),
                    xytext=(-6, 14), textcoords="offset points", ha="right",
                    color=C["ink"], fontsize=9, fontweight="bold")
    ax.annotate("per-step mean loss", xy=(ms[len(ms) // 3], mv[len(ms) // 3]),
                xytext=(8, 18), textcoords="offset points",
                color=C["series"][0], fontsize=9, fontweight="bold")
    ax.annotate("per-micro-batch (792 pts)", xy=(st[len(st) // 2], lv.max() * 0.92),
                color=C["ink2"], fontsize=8.5)
    ax.set_ylabel("loss", color=C["ink"], fontsize=10)
    ax.set_title("AgiBot task-362 LoRA — 395 clips / 20 episodes, 100 steps, 2×H20",
                 color=C["ink"], fontsize=12.5, fontweight="bold", loc="left", pad=12)

    # ---- panel 2: grad_norm ----
    ax = axes[1]
    gs, gv = d["grad_norm"]
    ax.plot(gs, gv, color=C["series"][1], lw=2.0, zorder=3)
    ax.set_ylabel("grad_norm", color=C["ink"], fontsize=10)
    ax.annotate("grad_norm", xy=(gs[len(gs) // 2], gv[len(gs) // 2]),
                xytext=(8, 16), textcoords="offset points",
                color=C["series"][1], fontsize=9, fontweight="bold")
    ax.axhline(1.0, color=C["ink2"], lw=1.0, ls=":", zorder=2)
    ax.annotate("clip max_grad_norm 1.0", xy=(gs[-1], 1.0), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=C["ink2"], fontsize=8.5)

    # ---- panel 3: lr ----
    ax = axes[2]
    ls_, lvv = d["lr"]
    ax.plot(ls_, lvv, color=C["series"][2], lw=2.0, zorder=3)
    ax.set_ylabel("learning rate", color=C["ink"], fontsize=10)
    ax.set_xlabel("optimizer step", color=C["ink"], fontsize=10)
    ax.annotate("cosine + 5-step warmup", xy=(ls_[len(ls_) // 3], lvv.max() * 0.8),
                xytext=(10, 0), textcoords="offset points",
                color=C["series"][2], fontsize=9, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_color(C["ink2"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[saved] {a.out}")
    print(f"  loss  first-step mean {mv[0]:.4f}  min {mv.min():.4f} @step {int(ms[mv.argmin()])}"
          f"  last {mv[-1]:.4f}")
    print(f"  grad_norm  max {gv.max():.3f}  last {gv[-1]:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default=LOGDIR)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--out", default="outputs/agibot_362_plots/curves_362.png")
    a = p.parse_args()
    if a.dark and a.out == "outputs/agibot_362_plots/curves_362.png":
        a.out = "outputs/agibot_362_plots/curves_362_dark.png"
    main(a)
