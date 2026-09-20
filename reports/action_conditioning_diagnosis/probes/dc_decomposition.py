#!/usr/bin/env python3
"""Why `reversed` is indistinguishable: measure the per-clip DC of each condition.

The residual is (1, 3072, 9, 1, 1). Its time-mean over the 9 frames is a single
3072-vector -- the per-clip "DC" that every frame receives. The earlier probe found
that ~99.9% of the residual's amplitude is this DC. If that is true, then any
perturbation that preserves the action sequence's time-mean should leave the DC
(and therefore the rollout) almost unchanged.

`reversed` = `raw.flip(1)` is exactly such a perturbation: flipping a sequence
preserves its multiset of values, hence its mean. `held` and `zero` are not.
`shifted` = `raw.roll(T//2)` preserves the raw mean but not the derived `velocity`
term (the roll introduces one large wrap-around difference), so it sits between.

This script measures it directly instead of arguing from the shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

TRAIN = REPO / "training/native_action_v10b_state_256_600"
CKPT = 400
DATA = Path("/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1")

SAMPLES = (
    ("task3400_episode443_start768", "task3400_0242.safetensors"),
    ("task3400_episode437_start1328", "task3400_0224.safetensors"),
    ("task3401_episode1055_start1488", "task3401_0192.safetensors"),
    ("task3401_episode1158_start2384", "task3401_0008.safetensors"),
)
CONDS = ("correct", "reversed", "held", "shifted", "zero")


def build(enc, packed):
    from core.control.native_action_features import build_v10_features

    raw = packed["robot_trajectory_raw37"].unsqueeze(0).float()
    state = packed["robot_observed_state37"].unsqueeze(0).float()
    mean = packed["robot_trajectory_mean37"].float()
    std = packed["robot_trajectory_std37"].float()
    rs = packed["robot_relative_scale37"].float()
    vs = packed["robot_velocity_scale37"].float()
    mk = lambda s: build_v10_features(s, state, mean, std, rs, vs)
    return {
        "correct": mk(raw),
        "reversed": mk(raw.flip(1)),
        "held": mk(state[:, None].expand_as(raw)),
        "shifted": mk(raw.roll(raw.shape[1] // 2, 1)),
        "zero": torch.zeros_like(mk(raw)),
    }


def main():
    from core.control.native_trajectory_encoder import NativeTrajectoryConditionerV10

    enc = NativeTrajectoryConditionerV10(input_dim=148)
    enc.load_state_dict(
        torch.load(TRAIN / f"checkpoint-{CKPT}/native_trajectory_encoder.bin",
                   map_location="cpu", weights_only=False),
        strict=False,
    )
    enc.eval()

    print("Per-condition residual decomposition, checkpoint-400, V10b")
    print("  DC    = r.mean over the 9 time frames, one 3072-vector per clip")
    print("  tv    = r - DC   (the only part that can encode temporal order)")
    print("  |DC| / |r| and |tv| / |r| are the RMS shares.\n")

    agg = {c: {"dc_share": [], "tv_share": [], "cos_dc_vs_correct": []} for c in CONDS}

    for name, fn in SAMPLES:
        packed = load_file(str(DATA / fn))
        acts = build(enc, packed)
        with torch.no_grad():
            out = {}
            for c in CONDS:
                r, _ = enc(acts[c], 9)
                r = r.float().squeeze(0).squeeze(-1).squeeze(-1)   # [3072, 9]
                dc = r.mean(dim=1)
                out[c] = (r, dc)
            print(f"  {name}")
            print(f"    {'cond':10s} {'rms(r)':>9s} {'rms(DC)':>9s} {'rms(tv)':>9s} "
                  f"{'DC share':>9s} {'tv share':>9s} {'cos(DC,cand)':>13s}")
            ref = out["correct"][1]
            for c in CONDS:
                r, dc = out[c]
                rr = r.square().mean().sqrt()
                rd = dc.square().mean().sqrt()
                rt = (r - dc[:, None]).square().mean().sqrt()
                cos = float(torch.dot(dc, ref) / (dc.norm() * ref.norm() + 1e-12))
                print(f"    {c:10s} {rr:9.5f} {rd:9.5f} {rt:9.5f} {rd/rr:9.3%} "
                      f"{rt/rr:9.3%} {cos:+13.6f}")
                agg[c]["dc_share"].append(float(rd / rr))
                agg[c]["tv_share"].append(float(rt / rr))
                agg[c]["cos_dc_vs_correct"].append(cos)
            print()

    print("  MEANS ACROSS 4 SAMPLES")
    print(f"    {'cond':10s} {'DC share':>10s} {'tv share':>10s} {'cos(DC,correct DC)':>20s}")
    for c in CONDS:
        a = agg[c]
        n = len(a["dc_share"])
        print(f"    {c:10s} {sum(a['dc_share'])/n:10.3%} {sum(a['tv_share'])/n:10.3%} "
              f"{sum(a['cos_dc_vs_correct'])/n:+20.6f}")


if __name__ == "__main__":
    main()
