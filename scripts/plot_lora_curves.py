"""Plot the AgiBot-357 LoRA training curves from the TensorBoard event file.

Three measures on very different scales (loss ~0.1-0.9, grad_norm ~0.02-3.1,
lr ~0-1e-4) -> small multiples sharing one x-axis, never a dual y-axis.
Loss is logged per micro-batch (160 pts over 24 optimizer steps); raw points are
context at low opacity and the running mean is the emphasis line.

Usage: python scripts/plot_lora_curves.py
"""
import argparse
import os

import numpy as np

# --- design tokens (validated categorical slots 1-3; see dataviz/references/palette.md)
LIGHT = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
             muted="#8a8983", grid="#e8e8e6",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a")
DARK = dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
            muted="#8a8983", grid="#333331",
            s1="#3987e5", s2="#d95926", s3="#199e70")


def read_scalars(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        ev = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev], float),
                    np.array([e.value for e in ev], float))
    return out


def fractional_steps(steps):
    """Micro-batches share an optimizer step; spread them evenly inside it so the
    x-axis reads as continuous training progress instead of 8 points on one tick."""
    x = steps.astype(float).copy()
    for s in np.unique(steps):
        m = steps == s
        n = int(m.sum())
        if n > 1:
            x[m] = s + np.linspace(0, 1, n, endpoint=False)
    return x


def running_mean(y, w=8):
    pad = np.concatenate([np.full(w - 1, y[0]), y])
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def plot(scalars, tk, out_path, title_suffix="", subtitle="", warmup=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.6), sharex=True,
                             gridspec_kw=dict(hspace=0.34, left=0.11, right=0.97,
                                              top=0.875, bottom=0.075))
    fig.patch.set_facecolor(tk["surface"])

    fig.text(0.11, 0.960, f"AgiBot-357 LoRA fine-tune{title_suffix}",
             color=tk["primary"], fontsize=15, fontweight="600", ha="left")
    fig.text(0.11, 0.933, subtitle,
             color=tk["secondary"], fontsize=9.5, ha="left")

    for ax in axes:
        ax.set_facecolor(tk["surface"])
        ax.grid(True, color=tk["grid"], linewidth=1, linestyle="-", alpha=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(tk["grid"])
        ax.tick_params(colors=tk["secondary"], labelsize=9, length=0)

    # --- panel 1: loss (raw micro-batch as context + running mean as the story)
    ls, lv = scalars["loss"]
    lx = fractional_steps(ls)
    ax = axes[0]
    ax.scatter(lx, lv, s=9, color=tk["s1"], alpha=0.22, linewidths=0, zorder=2)
    sm = running_mean(lv, 8)
    ax.plot(lx, sm, color=tk["s1"], linewidth=2, solid_capstyle="round", zorder=3)
    ax.set_ylabel("loss", color=tk["secondary"], fontsize=10)
    ax.set_title("Flow-matching MSE loss  ·  dots = per-micro-batch, line = 8-batch running mean",
                 color=tk["secondary"], fontsize=9.5, loc="left", pad=6)
    for xx, yy, txt, dy in ((lx[0], sm[0], f"{sm[0]:.3f}", 10), (lx[-1], sm[-1], f"{sm[-1]:.3f}", 12)):
        ax.scatter([xx], [yy], s=42, color=tk["s1"], zorder=5,
                   edgecolors=tk["surface"], linewidths=2)
        ax.annotate(txt, (xx, yy), textcoords="offset points", xytext=(0, dy),
                    ha="center", color=tk["primary"], fontsize=9.5, fontweight="600")
    # the trough matters: loss bottoms out mid-run then RISES again -- quoting only
    # step1 -> step10 would hide the late uptick (over-fitting on 40 clips)
    lo = int(np.argmin(sm))
    ax.scatter([lx[lo]], [sm[lo]], s=42, color=tk["s1"], zorder=5,
               edgecolors=tk["surface"], linewidths=2)
    ax.annotate(f"min {sm[lo]:.3f}", (lx[lo], sm[lo]), textcoords="offset points",
                xytext=(0, -16), ha="center", color=tk["primary"],
                fontsize=9.5, fontweight="600")
    ax.set_ylim(0, max(lv) * 1.12)

    # --- panel 2: grad_norm (one point per optimizer step)
    gs, gv = scalars["grad_norm"]
    ax = axes[1]
    ax.plot(gs, gv, color=tk["s2"], linewidth=2, solid_capstyle="round",
            marker="o", markersize=4.5, markeredgecolor=tk["surface"],
            markeredgewidth=1.6, zorder=3)
    ax.set_ylabel("grad norm", color=tk["secondary"], fontsize=10)
    ax.set_title("Gradient norm  ·  one point per optimizer step",
                 color=tk["secondary"], fontsize=9.5, loc="left", pad=6)
    pk = int(np.argmax(gv))
    ax.annotate(f"peak {gv[pk]:.2f} @ step {int(gs[pk])}", (gs[pk], gv[pk]),
                textcoords="offset points", xytext=(8, -2), ha="left",
                color=tk["primary"], fontsize=9.5, fontweight="600")
    ax.annotate(f"{gv[-1]:.4f}", (gs[-1], gv[-1]), textcoords="offset points",
                xytext=(-4, 12), ha="right", color=tk["primary"],
                fontsize=9.5, fontweight="600")
    ax.set_ylim(0, max(gv) * 1.18)

    # --- panel 3: lr (aqua carries a contrast WARN -> direct labels are mandatory)
    rs, rv = scalars["lr"]
    rx = fractional_steps(rs)
    ax = axes[2]
    ax.plot(rx, rv * 1e4, color=tk["s3"], linewidth=2, solid_capstyle="round", zorder=3)
    ax.set_ylabel("lr  (x1e-4)", color=tk["secondary"], fontsize=10)
    ax.set_title(f"Learning rate  ·  {warmup}-step warmup then cosine decay",
                 color=tk["secondary"], fontsize=9.5, loc="left", pad=6)
    pk = int(np.argmax(rv))
    ax.scatter([rx[pk]], [rv[pk] * 1e4], s=42, color=tk["s3"], zorder=5,
               edgecolors=tk["surface"], linewidths=2)
    ax.annotate(f"peak {rv[pk]:.2e}", (rx[pk], rv[pk] * 1e4), textcoords="offset points",
                xytext=(6, 6), ha="left", color=tk["primary"], fontsize=9.5, fontweight="600")
    ax.set_ylim(0, max(rv) * 1e4 * 1.25)
    ax.set_xlabel("optimizer step", color=tk["secondary"], fontsize=10)
    ax.set_xlim(-0.6, max(gs) + 1)

    fig.savefig(out_path, dpi=150, facecolor=tk["surface"])
    print(f"-> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="training/agibot_357_lora_v2_skel/logs/finetrainer-cogvideo")
    p.add_argument("--output", default="outputs/agibot_lora_357_v2")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--subtitle",
                   default="rank 32 · 2xH20 · 40 clips x 6 epochs / grad_accum 8 = 18 optimizer steps"
                           "  ·  skeleton control")
    p.add_argument("--suffix", default=" v2")
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)
    sc = read_scalars(a.logdir)
    plot(sc, LIGHT, os.path.join(a.output, "training_curves.png"), a.suffix, a.subtitle, a.warmup)
    plot(sc, DARK, os.path.join(a.output, "training_curves_dark.png"),
         a.suffix + " — dark", a.subtitle, a.warmup)
