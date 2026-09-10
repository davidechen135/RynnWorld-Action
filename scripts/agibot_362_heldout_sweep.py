"""Multi-clip held-out sweep: run the zero-shot vs fine-tuned comparison on one clip
from EACH held-out episode, so the verdict does not rest on a single clip.

The single-clip run (outputs/agibot_362_heldout/) showed both models collapsing after
frame ~2 with motion energy far ABOVE GT (262% / 315%) while spatial detail sat at
~40% of GT -- the signature of flicker/collapse, not real folding motion. A single
clip cannot distinguish "this clip is hard" from "the model collapses on unseen
episodes", so this sweep repeats it across all three held-out episodes.

Each clip is evaluated by scripts/agibot_heldout_eval.py (which exports the pipeline's
own GT decode as the anchor), then the per-clip metrics are merged into one table.

Usage:
  python scripts/agibot_362_heldout_sweep.py [--per-episode 1]
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(REPO, "scripts", "agibot_heldout_eval.py")
HELD_JSON = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel_heldout.json"
TRAIN_JSON = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel.json"


def pick_clips(per_episode):
    """One mid-episode clip per held-out episode (avoid clip_000000: its first frame is
    the episode's first frame, the easiest possible conditioning)."""
    eps = defaultdict(list)
    for e in json.load(open(HELD_JSON)):
        eps[e["video_latent_path"].split("/")[2]].append(e["video_latent_path"])
    picked = []
    for ep in sorted(eps):
        clips = sorted(eps[ep])
        mid = len(clips) // 2
        picked.extend(clips[mid:mid + per_episode])
    return picked


def main(a):
    clips = pick_clips(a.per_episode)
    print(f"[sweep] {len(clips)} clips across held-out episodes:")
    for c in clips:
        print("   ", c)

    rows = []
    for c in clips:
        ep = c.split("/")[2]
        name = os.path.splitext(os.path.basename(c))[0]
        out = f"outputs/agibot_362_sweep/{ep}_{name}"
        cmd = [sys.executable, EVAL, "--clip", c, "--train-json", TRAIN_JSON,
               "--task-text", "Folding shorts",
               "--lora-checkpoint", a.lora_checkpoint,
               "--output", out]
        print(f"\n=== {ep}/{name} ===", flush=True)
        subprocess.run(cmd, check=True, cwd=REPO)
        m = json.load(open(os.path.join(out, "heldout_metrics.json")))
        rows.append({"episode": ep, "clip": name, "runs": m["runs"]})

    # merged table
    summary = {"lora_checkpoint": a.lora_checkpoint, "clips": rows}
    os.makedirs("outputs/agibot_362_sweep", exist_ok=True)
    sp = "outputs/agibot_362_sweep/sweep_metrics.json"
    json.dump(summary, open(sp, "w"), indent=2)

    print(f"\n{'episode':10s}{'clip':16s}{'GT':>8s}{'zeroshot':>10s}{'%GT':>7s}"
          f"{'finetuned':>11s}{'%GT':>7s}{'zs_ratio':>10s}{'ft_ratio':>10s}")
    for r in rows:
        g = r["runs"]["gt"]["interframe_delta_mean"]
        z = r["runs"]["zeroshot"]
        f = r["runs"]["finetuned"]
        print(f"{r['episode']:10s}{r['clip']:16s}{g:8.4f}"
              f"{z['interframe_delta_mean']:10.4f}{100*z['interframe_delta_mean']/g:6.0f}%"
              f"{f['interframe_delta_mean']:11.4f}{100*f['interframe_delta_mean']/g:6.0f}%"
              f"{z['detail_ratio_last_over_first']:10.3f}{f['detail_ratio_last_over_first']:10.3f}")
    print(f"\n[done] {sp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-episode", type=int, default=1)
    p.add_argument("--lora-checkpoint", default="training/agibot_362_lora/checkpoint-100")
    main(p.parse_args())
