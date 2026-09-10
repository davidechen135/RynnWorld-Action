"""Held-out comparison for task-362: motion energy AND structure retention, side by side.

Reading motion energy alone on this run gives the WRONG answer, so this figure is
deliberately two panels:

  left  — motion energy as % of GT. Fine-tuning moves 2 of 3 clips from 69/76% to
          96/105%, which looks like a clean win.
  right — detail_ratio (spatial detail, last third / first third). GT sits at ~1.02;
          every generated run sits at 0.62-0.74, i.e. structure decays.

Frame inspection (outputs/agibot_362_sweep/good_eps_trace.png) shows why: both models
render frame 0 correctly then melt from frame ~2, and the melt itself produces
frame-to-frame change of roughly GT's magnitude. The "96%" is coincidence, not folding
motion. Panel 2 is what actually discriminates, so it is not optional context -- it is
the finding, and the caption says so.

100% reference lines are drawn on both panels because "match GT" is the target for
both measures; bars are the three held-out clips.

Palette validated via the dataviz skill validator (light + dark, all checks pass).

Usage: python scripts/plot_362_heldout.py [--dark]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LIGHT = dict(bg="#fcfcfb", ink="#1a1a19", ink2="#5c5c58", grid="#e4e4e0",
             zs="#1f78b4", ft="#d95f02", gt="#5c5c58")
DARK = dict(bg="#1a1a19", ink="#f2f2ef", ink2="#a8a8a2", grid="#33332f",
            zs="#3f8ec4", ft="#e07020", gt="#a8a8a2")


def main(a):
    C = DARK if a.dark else LIGHT
    d = json.load(open(a.sweep))
    rows = d["clips"]
    labels = [f"{r['episode']}\n{r['clip'].replace('clip_', 'f')}" for r in rows]
    x = np.arange(len(rows))
    w = 0.36

    pct_zs, pct_ft, rat_gt, rat_zs, rat_ft = [], [], [], [], []
    for r in rows:
        g = r["runs"]["gt"]
        pct_zs.append(100 * r["runs"]["zeroshot"]["interframe_delta_mean"] / g["interframe_delta_mean"])
        pct_ft.append(100 * r["runs"]["finetuned"]["interframe_delta_mean"] / g["interframe_delta_mean"])
        rat_gt.append(g["detail_ratio_last_over_first"])
        rat_zs.append(r["runs"]["zeroshot"]["detail_ratio_last_over_first"])
        rat_ft.append(r["runs"]["finetuned"]["detail_ratio_last_over_first"])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    fig.patch.set_facecolor(C["bg"])
    for ax in axes:
        ax.set_facecolor(C["bg"])
        ax.grid(True, axis="y", color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C["grid"])
        ax.tick_params(colors=C["ink2"], labelsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color=C["ink2"])

    # ---- panel 1: motion energy as %GT ----
    ax = axes[0]
    b1 = ax.bar(x - w / 2, pct_zs, w, color=C["zs"], label="zero-shot", zorder=3)
    b2 = ax.bar(x + w / 2, pct_ft, w, color=C["ft"], label="fine-tuned (ckpt-100)", zorder=3)
    ax.axhline(100, color=C["gt"], lw=1.6, ls="--", zorder=4)
    ax.annotate("GT = 100%", xy=(len(rows) - 0.5, 100), xytext=(0, 6),
                textcoords="offset points", ha="right", color=C["ink"],
                fontsize=9, fontweight="bold")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}%", xy=(r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        color=C["ink"], fontsize=8.5)
    ax.set_ylabel("motion energy, % of GT", color=C["ink"], fontsize=10)
    ax.set_title("Motion energy — looks like a win", color=C["ink"],
                 fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=C["ink"], loc="upper left")

    # ---- panel 2: structure retention ----
    ax = axes[1]
    ax.bar(x - w / 2, rat_zs, w, color=C["zs"], zorder=3)
    ax.bar(x + w / 2, rat_ft, w, color=C["ft"], zorder=3)
    ax.plot(x, rat_gt, "o", ms=9, color=C["gt"], zorder=5)
    for i, v in enumerate(rat_gt):
        ax.annotate(f"GT {v:.2f}", xy=(i, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=C["ink"], fontsize=8.5, fontweight="bold")
    for i in range(len(rows)):
        ax.annotate(f"{rat_zs[i]:.2f}", xy=(i - w / 2, rat_zs[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=C["ink"], fontsize=8.5)
        ax.annotate(f"{rat_ft[i]:.2f}", xy=(i + w / 2, rat_ft[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=C["ink"], fontsize=8.5)
    ax.axhline(1.0, color=C["gt"], lw=1.6, ls="--", zorder=4)
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("detail ratio (last third / first third)", color=C["ink"], fontsize=10)
    ax.set_title("Structure retention — the real story: all collapse",
                 color=C["ink"], fontsize=11.5, fontweight="bold", loc="left", pad=10)

    fig.suptitle("AgiBot task-362 held-out: 3 unseen episodes, zero-shot vs LoRA ckpt-100",
                 color=C["ink"], fontsize=13, fontweight="bold", x=0.005, ha="left", y=1.005)
    fig.text(0.005, -0.055,
             "Both models render frame 0 from the conditioning latent, then melt from frame ~2 "
             "(see good_eps_trace.png).\nThat melt generates frame-to-frame change of roughly GT's "
             "magnitude, so \"96%\" on the left is coincidence, not folding motion.\n"
             "GT holds detail ratio ~1.02; every generated run sits at 0.62-0.74. Motion energy alone "
             "cannot detect collapse -- read both panels.",
             color=C["ink2"], fontsize=8.8, ha="left", va="top")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[saved] {a.out}")
    for i, r in enumerate(rows):
        print(f"  {r['episode']}  zs {pct_zs[i]:5.0f}%GT ratio {rat_zs[i]:.3f} | "
              f"ft {pct_ft[i]:5.0f}%GT ratio {rat_ft[i]:.3f} | GT ratio {rat_gt[i]:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="outputs/agibot_362_sweep/sweep_metrics.json")
    p.add_argument("--dark", action="store_true")
    p.add_argument("--out", default="outputs/agibot_362_plots/heldout_362.png")
    a = p.parse_args()
    if a.dark and a.out == "outputs/agibot_362_plots/heldout_362.png":
        a.out = "outputs/agibot_362_plots/heldout_362_dark.png"
    main(a)
