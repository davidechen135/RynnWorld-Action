"""Visualize the held-out AgiBot-357 comparison: real GT vs zero-shot vs the two
fine-tunes (v1, hand-built action control; v2, rendered skeleton control).

Two figures:
  1. heldout_grid.png     -- frame grid, one row per variant, GT on top
  2. heldout_motion.png   -- per-frame motion energy over time, GT as the reference

Motion energy = mean |frame_t - frame_{t-1}| over pixels. The GT line is the target:
a variant that sits far above it is generating motion the real robot never made, one
far below it is frozen.

Usage: python scripts/plot_heldout_compare.py
"""
import argparse
import os

import numpy as np

LIGHT = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
             grid="#e8e8e6", gt="#52514e",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a")
DARK = dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
            grid="#333331", gt="#c3c2b7",
            s1="#3987e5", s2="#d95926", s3="#199e70")


def load(path):
    import imageio.v2 as iio
    v = iio.mimread(path, memtest=False)
    return np.stack([np.asarray(f[..., :3], np.float32) for f in v])


def motion(a):
    return np.abs(np.diff(a, axis=0)).mean(axis=(1, 2, 3))


def grid_figure(vids, tk, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [0, 20, 40, 60, 80]
    n = len(vids)
    fig, axes = plt.subplots(n, len(picks), figsize=(15, 2.2 * n + 1.0),
                             gridspec_kw=dict(hspace=0.05, wspace=0.015, left=0.155,
                                              right=0.985, top=0.865, bottom=0.02))
    fig.patch.set_facecolor(tk["surface"])
    fig.text(0.155, 0.968, "Held-out clip (frames 3969–4049) — never seen in training",
             color=tk["primary"], fontsize=15, fontweight="600", ha="left", va="top")
    fig.text(0.155, 0.923,
             "same first frame · same action control · same seed 42 · only the weights differ",
             color=tk["secondary"], fontsize=9.5, ha="left", va="top")

    for r, (label, color, a) in enumerate(vids):
        for c, fi in enumerate(picks):
            ax = axes[r, c]
            ax.imshow(a[min(fi, len(a) - 1)].astype(np.uint8))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"frame {fi}", color=tk["secondary"], fontsize=9.5, pad=5)
        # Row identity lives left of the images: a color key bar, then the label.
        # Horizontal text (not rotated) so two-line names stay readable.
        ax0 = axes[r, 0]
        head, sub = label.split("\n")
        ax0.text(-0.045, 0.60, head, transform=ax0.transAxes, va="center", ha="right",
                 color=tk["primary"], fontsize=10.5, fontweight="600")
        ax0.text(-0.045, 0.40, sub, transform=ax0.transAxes, va="center", ha="right",
                 color=tk["secondary"], fontsize=9)
        ax0.plot([-0.028, -0.028], [0.32, 0.68], transform=ax0.transAxes,
                 color=color, linewidth=3.5, solid_capstyle="round", clip_on=False)

    fig.savefig(out, dpi=110, facecolor=tk["surface"])
    print(f"-> {out}")


def motion_figure(vids, tk, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 5.4),
                           gridspec_kw=dict(left=0.085, right=0.78, top=0.775, bottom=0.11))
    fig.patch.set_facecolor(tk["surface"])
    ax.set_facecolor(tk["surface"])
    fig.text(0.085, 0.945, "Motion energy per frame — held-out clip",
             color=tk["primary"], fontsize=15, fontweight="600", ha="left", va="top")
    fig.text(0.085, 0.880,
             "mean |Δ| between consecutive frames. The real recording is the target;\n"
             "a curve above it is inventing motion the robot never made.",
             color=tk["secondary"], fontsize=9.5, ha="left", va="top", linespacing=1.4)

    ax.grid(True, color=tk["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(tk["grid"])
    ax.tick_params(colors=tk["secondary"], labelsize=9, length=0)

    # End labels are placed by rank, not by the curve's last value: GT and zero-shot
    # converge at frame 80 and would otherwise print on top of each other.
    curves = []
    for label, color, a in vids:
        m = motion(a)
        x = np.arange(1, len(m) + 1)
        lw = 2.4 if label.startswith("REAL") else 2
        ax.plot(x, m, color=color, linewidth=lw, solid_capstyle="round",
                zorder=4 if label.startswith("REAL") else 3)
        ax.scatter([x[-1]], [m[-1]], s=46, color=color, zorder=5,
                   edgecolors=tk["surface"], linewidths=2)
        curves.append((m[-1], m.mean(), label.splitlines()[0], color))

    ymax = max(max(motion(a)) for _, _, a in vids)
    span = ymax * 1.06
    gap = span * 0.075                      # minimum vertical room between labels
    curves.sort(key=lambda c: -c[0])
    placed = []
    for yend, avg, name, color in curves:
        y = yend
        if placed and placed[-1] - y < gap:
            y = placed[-1] - gap
        placed.append(y)
        ax.annotate(f"{name}  ·  {avg:.2f} avg", (84, y), xycoords=("data", "data"),
                    va="center", ha="left", color=tk["primary"],
                    fontsize=9.5, fontweight="600", annotation_clip=False)
        ax.plot([81.5, 83.2], [yend, y], color=color, linewidth=1.2,
                solid_capstyle="round", clip_on=False, zorder=4)

    ax.set_xlabel("frame", color=tk["secondary"], fontsize=10)
    ax.set_ylabel("motion energy  (mean |Δ| per pixel)", color=tk["secondary"], fontsize=10)
    ax.set_xlim(0, 81.5)
    ax.set_ylim(0, span)
    fig.savefig(out, dpi=150, facecolor=tk["surface"])
    print(f"-> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="outputs/agibot_heldout_v2skel/heldout_gt_real.mp4")
    p.add_argument("--zeroshot", default="outputs/agibot_heldout_v2skel/heldout_zeroshot_seed42.mp4")
    p.add_argument("--v1", default="outputs/agibot_heldout_ckpt24/heldout_finetuned_seed42.mp4")
    p.add_argument("--v2", default="outputs/agibot_heldout_v2skel/heldout_finetuned_seed42.mp4")
    p.add_argument("--output", default="outputs/agibot_heldout_v2skel")
    a = p.parse_args()

    for tk, sfx in ((LIGHT, ""), (DARK, "_dark")):
        vids = [
            ("REAL GT\nrecording", tk["gt"], load(a.gt)),
            ("ZERO-SHOT\nfrozen SFT", tk["s1"], load(a.zeroshot)),
            ("FT v1\naction control", tk["s2"], load(a.v1)),
            ("FT v2\nskeleton control", tk["s3"], load(a.v2)),
        ]
        grid_figure(vids, tk, os.path.join(a.output, f"heldout_grid{sfx}.png"))
        motion_figure(vids, tk, os.path.join(a.output, f"heldout_motion{sfx}.png"))
