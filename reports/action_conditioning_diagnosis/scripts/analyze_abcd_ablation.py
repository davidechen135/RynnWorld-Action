#!/usr/bin/env python3
"""Summarise reports/action_conditioning_diagnosis/ablation/summary.json.

Answers the Step-2 question with one factor changed per arm:

  A  as-is            reference -- what the shipped checkpoint does now
  B  base-off         is the 20x static `base_residual` the thing that swamps
                      the action? (it cancels in a difference, but it can still
                      shift the input of a downstream normalization)
  C  proj-swap        does replacing the rank-collapsed `input_residual_projection`
                      with an RMS-matched random map restore action response?
  D  base-off+proj    both, to see whether the two are independent or interact

The numbers that matter, in order:

  tv_frac      how much of the injected residual varies over time at all. This is
               the mechanism variable -- if it stays ~0.1%, nothing downstream can
               recover temporal structure no matter what the arms do.
  outMAD       output MAD of a corrupted condition against the SAME arm's
               `correct` rollout. Must be read against the noise floor (re-sampling
               the identical input), never as an absolute.
  gtMAD / cos  reconstruction vs GT and motion-direction agreement. Reported so a
               "more sensitive but less correct" arm is visible as such.
  energy       motion energy ratio; guards against a low-MAD blur being mistaken
               for control.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ABL = REPO / "reports/action_conditioning_diagnosis/ablation/summary.json"
# Same-checkpoint, same-condition resample floor for the offical model. The native
# model's own floor is taken from the gate report when present.
OFFICIAL_FLOOR = 7.229

CONDITIONS = ("correct", "reversed", "held", "shifted", "zero")
ARMS = ("A", "B", "C", "D")


def load_native_floor() -> float | None:
    """The gate's own 3-seed resample floor for the native model, if recorded."""
    for cand in (
        REPO / "reports/native_action_arm_dynamic_dev/summary.json",
        REPO / "reports/native_action_v10b_state_256/summary.json",
    ):
        if not cand.exists():
            continue
        try:
            blob = json.loads(cand.read_text())
        except json.JSONDecodeError:
            continue
        for key in ("noise_floor_outmad", "noise_floor", "resample_noise_floor"):
            if isinstance(blob.get(key), (int, float)):
                return float(blob[key])
        # Otherwise, look for per-seed rollouts of the same condition.
        for v in blob.values():
            if isinstance(v, dict) and "noise_floor" in v:
                return float(v["noise_floor"])
    return None


def main() -> int:
    if not ABL.exists():
        print(f"missing {ABL}", file=sys.stderr)
        return 1
    blob = json.loads(ABL.read_text())
    results = blob["results"]
    samples = blob["samples"]
    native_floor = load_native_floor()

    print("=" * 108)
    print("STEP 2 -- A/B/C/D injection ablation (one factor per arm)")
    print(f"  checkpoint {blob['checkpoint']}")
    print(f"  samples={len(samples)}  arms={blob['arms']}  proj_seeds={blob['proj_seeds']}  "
          f"steps={blob['steps']}  diffusion seed={blob['seed']} (single seed -- no within-run floor)")
    print(f"  noise floor: native={native_floor if native_floor else 'not recorded'}  "
          f"official(comparable order)={OFFICIAL_FLOOR}")
    print("=" * 108)

    # ---------------- mechanism: tv_frac per arm ---------------------------- #
    print("\n[1] MECHANISM -- temporal fraction of the injected residual (condition=correct)")
    print(f"    {'arm':4s} {'proj_seed':>9s} {'total_rms':>10s} {'tv_rms':>10s} {'tv_frac':>10s}   {'rank@99%':>8s} {'top1_energy':>11s}")
    for arm in ARMS:
        for key, entry in results.items():
            if entry["arm"] != arm:
                continue
            frac = entry["injection"]["correct"]["residual_tv_frac"]
            tot = entry["injection"]["correct"]["residual_total_rms"]
            tvr = entry["injection"]["correct"]["residual_tv_rms"]
            spec = entry["arm_record"].get("proj_swapped_spectrum") or \
                   entry["arm_record"].get("proj_original_spectrum") or {}
            rank = spec.get("rank_at_99pct", "-")
            top1 = spec.get("top_energy_fracs", [float("nan")])[0]
            print(f"    {arm:4s} {entry['proj_seed']:>9d} {tot:10.4f} {tvr:10.6f} "
                  f"{frac:10.4%}   {rank:>8} {top1:11.4f}")

    # ---------------- per-arm sensitivity ----------------------------------- #
    print("\n[2] ACTION RESPONSE -- output MAD against this arm's OWN `correct` rollout")
    print("    (each cell is sample-mean; >1x the noise floor means the condition was read)")
    header = f"    {'arm':4s} " + " ".join(f"{c:>19s}" for c in CONDITIONS if c != "correct")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for arm in ARMS:
        cells = []
        for cond in CONDITIONS:
            if cond == "correct":
                continue
            vals = [
                e["conditions"][cond]["output_mad_vs_arm_correct"]
                for e in results.values()
                if e["arm"] == arm and cond in e["conditions"]
                and "output_mad_vs_arm_correct" in e["conditions"][cond]
            ]
            cells.append(f"{sum(vals)/len(vals):8.2f}({sum(vals)/len(vals)/OFFICIAL_FLOOR:4.2f}x)" if vals else f"{'-':>19s}")
        print(f"    {arm:4s} " + " ".join(f"{c:>19s}" for c in cells))

    # ---------------- correctness per arm ----------------------------------- #
    print("\n[3] CORRECTNESS vs GT -- is the arm still reconstructing, or just reacting?")
    print(f"    {'arm':4s} {'gt_mad':>8s} {'motion_cos':>11s} {'energy':>8s} {'flicker':>8s} {'lap_var':>9s} {'frame_std':>10s}")
    for arm in ARMS:
        def mean(metric, cond="correct"):
            vals = [e["conditions"][cond][metric] for e in results.values()
                    if e["arm"] == arm and cond in e["conditions"]]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"    {arm:4s} {mean('gt_pixel_mad'):8.2f} "
              f"{mean('motion_region_temporal_cosine'):+11.4f} "
              f"{mean('motion_energy_ratio'):8.3f} {mean('flicker_stride4'):8.2f} "
              f"{mean('laplacian_variance'):9.1f} {mean('frame_std'):10.2f}")

    # ---------------- the arm effect, per sample ---------------------------- #
    print("\n[4] ARM EFFECT PER SAMPLE -- tv_frac(correct) and reversed outMAD")
    print(f"    {'sample':34s} " + " ".join(f"{a:>22s}" for a in ARMS))
    for s in samples:
        cells = []
        for arm in ARMS:
            e = next((v for k, v in results.items()
                      if k.startswith(s + "|" + arm + "|")), None)
            if e is None:
                cells.append(f"{'--':>22s}")
                continue
            frac = e["injection"]["correct"]["residual_tv_frac"]
            om = e["conditions"].get("reversed", {}).get("output_mad_vs_arm_correct", float("nan"))
            cells.append(f"{frac*100:7.3f}% {om:7.2f}MAD".rjust(22))
        print(f"    {s:34s} " + " ".join(cells))

    # ---------------- verdict ----------------------------------------------- #
    print("\n[5] VERDICT")
    def arm_mean(metric, cond="correct"):
        vals = [e["conditions"][cond][metric] for e in results.values()
                if e["arm"] == arm and cond in e["conditions"]]
        return sum(vals) / len(vals) if vals else float("nan")

    def arm_frac(a):
        vals = [e["injection"]["correct"]["residual_tv_frac"]
                for e in results.values() if e["arm"] == a]
        return sum(vals) / len(vals) if vals else float("nan")

    for arm in ARMS:
        rev = arm_mean("output_mad_vs_arm_correct", "reversed")
        held = arm_mean("output_mad_vs_arm_correct", "held")
        print(f"    {arm}: tv_frac={arm_frac(arm):8.4%}  rev={rev:6.2f}({rev/OFFICIAL_FLOOR:4.2f}x)  "
              f"held={held:6.2f}({held/OFFICIAL_FLOOR:4.2f}x)  "
              f"gt_mad={arm_mean('gt_pixel_mad'):6.2f}  cos={arm_mean('motion_region_temporal_cosine'):+.4f}  "
              f"energy={arm_mean('motion_energy_ratio'):.3f}")
    print("""
    Reading it: if tv_frac stays ~0.1% in every arm, the injection has almost no
    temporal content to begin with and no inference-time projection swap can create
    it -- the fix has to be structural (route time through an addressable axis) and
    then trained. If tv_frac rises sharply in C/D but rev outMAD does NOT clear the
    floor, the projection is not the binding constraint either.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
