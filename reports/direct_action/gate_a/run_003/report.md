# Gate A run_003 — sign-convention audit of the lag/delay scan

Run: `reports/direct_action/gate_a/run_003/`
Date: 2026-08-20
**Verdict: Gate A-core PASS CONFIRMED, with a corrected and independently re-derived delay
measurement.** `run_001` and `run_002` are preserved unmodified. No training source files
were touched. Gate B has not been started.

## The problem the supervisor flagged

run_001/run_002 reported `MAE(action[t], state[t+lag])` minimized at **negative** `lag`
(e.g. -2, -4) and interpreted this as "action leads state." The supervisor correctly pointed
out that if the metric is genuinely `MAE(action[t], state[t+lag])`, a command that leads the
response by N frames should minimize at **positive** lag, not negative — a negative-lag
minimum under that literal definition means `action[t]` best matches `state[t-2]`, i.e.
action reproduces a *past* state, which would mean action lags state (or is a smoothed/
delayed echo of it), the opposite of a command.

## What was actually happening (root cause)

The scan code used in run_001/run_002 was:

```python
for lag in lags:
    if lag < 0:
        x, y = action[:lag], state[-lag:]
    elif lag > 0:
        x, y = action[lag:], state[:-lag]
    else:
        x, y = action, state
    mae = mean(abs(x - y))
```

Tracing the index pairs directly: at `lag=-2`, this pairs `action[0]` with `state[2]`,
`action[1]` with `state[3]`, etc. — i.e. it actually computes `MAE(action[t], state[t+2])`
when `lag=-2`, **not** `MAE(action[t], state[t-2])` as the report prose claimed. So the
code and the prose disagreed on what "negative lag" meant. This is confirmed by direct
index tracing in this run (see conversation trace; reproduced in `artifacts/sign_audit.py`'s
docstring).

Because of this prose/code mismatch, the qualitative conclusion in run_002 ("action leads
state") happened to be *directionally* right, but the numeric delay values (-2, -4, -2, -4)
were not trustworthy as stated and the sign convention was never independently verified
before being used to support a PASS. That is exactly the kind of error the supervisor's
synthetic-sequence-first requirement is designed to catch, so a synthetic test was built and
required to pass before trusting any real-data number in this run.

## Sign test: deterministic synthetic sequence, action-leads-state by +3 frames

`artifacts/sign_audit.py`, function `build_synthetic(true_delay=3)`: constructs
`action_full` as a smooth deterministic signal (sine + small fixed-seed noise), then sets
`state[t] = action[t - true_delay]` for all valid `t` — i.e. state is defined to echo a past
action, which is by construction "action leads state by `true_delay` frames." This is
verified algebraically before use (`assert np.allclose(state_full[t_check], action_full[t_check - true_delay])`).

Convention under test: **`response_delay_frames D`, found by minimizing
`MAE(action[t], state[t+D])` over `t`; `D > 0` means state responds `D` frames *after*
action (action leads); `D < 0` means action matches an earlier state (action lags).**

```
$ python3 reports/direct_action/gate_a/run_003/artifacts/sign_audit.py
{
  "true_delay": 3,
  "recovered_delay": 3,
  "passed": true
}
SIGN TEST PASSED: response_delay_frames convention verified.
```

Recovered delay (+3) matches injected delay (+3) exactly. The **same synthetic case, run
through the original run_001/run_002 code**, recovers `best_lag = -3` — confirming the
original code's sign was inverted relative to this now-verified convention, and explaining
the earlier -2/-4 values.

## Corrected re-scan on real data: all 110 episodes, sign-verified convention

Using the verified `verified_scan()` function (not the original code, and not a relabeling
of old numbers — this is a from-scratch re-computation from the raw parquet files), applied
per-episode to all 110 episodes (widened beyond the original 32) with delay window
`[-12, +12]` frames (widened from an initial `[-6,+6]` after `action/waist/position`'s
optimum landed on that window's edge):

| field | median best delay | mean best delay | % episodes with D>0 | margin vs D=0 |
|---|---|---|---|---|
| `action/joint/position` | +2 | 2.00 | 100.0% | 86.4% |
| `action/end/position` | +4 | 4.42 | 100.0% | 37.7% |
| `action/end/orientation` | +2 | 2.31 | 100.0% | 40.5% |
| `action/waist/position` | +7 | 6.84 | 96.4% | 83.5% |

Full per-field delay distributions, pooled mean-MAE-by-delay curves, and margin figures:
`artifacts/core_delay_scan_corrected_110ep_final.json`. Visualized in
`figures/core_delay_scan_corrected_110ep.png`.

All 4 fields: strictly positive delay in ≥96% of individual episodes, and pooled MAE at the
best delay is 38-86% lower than at delay=0. This is the same qualitative direction as
run_002's claim ("action leads state"), but now: (a) independently re-derived from raw data
rather than reusing the old scan's numbers, (b) sign-verified against a synthetic ground
truth before being trusted, (c) run over all 110 episodes rather than 32, (d) reports an
honest per-episode distribution and a margin-vs-delay-0 metric rather than a single pooled
number, and (e) uses a wide-enough window that no field's optimum is clipped at the search
boundary.

## Same-timestamp columns vs. upstream provenance — explicitly distinguished

Per supervisor instruction, this timing analysis is **supporting evidence, not proof of
provenance**. It establishes that, empirically, `action/*` values at time `t` are more
similar to `state` values `2-7` frames in the future than to `state[t]` itself — consistent
with `action` being a control-loop *setpoint/target* that the physical system tracks with
latency, which is consistent with (but does not, on its own, formally prove) `action` being
issued from an upstream planner/controller rather than being a copy of sensor feedback. The
`action/head/position` exact-equality finding (0 episodes out of 110 show this for the 4
core fields, vs. exact match in all sampled episodes for `head/position`) remains the
stronger, more direct piece of evidence for the *core* fields — same-timestamp exact
equality is a same-timestamp, not a lag-based, check, and is unaffected by this sign bug.
No upstream code/schema documentation was newly found in this run establishing formal
command provenance beyond what run_001/run_002 already reported (official AgiBot-World
glossary: "Action refers to the instructions sent to the hardware abstraction layer, where
controller would respond to these instructions" — a schema-level claim, not a per-field
proof).

## Gate A-core verdict (updated)

**PASS, confirmed with corrected evidence.** The corrected delay scan shows action
genuinely leads state (positive `response_delay_frames`, verified sign convention, 96-100%
episode consistency, real margin over delay=0) for all 4 D-action-core fields
(`joint/position`, `end/position`, `end/orientation`, `waist/position`). This is consistent
with — not a reversal of — run_002's qualitative conclusion, but the quantitative delay
values from run_002 (-2,-4,-2,-4) should be considered **superseded** by this run's
corrected values (+2,+4,+2,+4 median, with `waist/position` actually +7 median once the
window was widened, not +4 as the old sign-inverted number implied).

`D-action-effector` remains quarantined, unaffected by this correction (its own lag scan in
run_002 was flat/inconclusive under either sign convention, so this bug does not change its
status).

## Gate A checklist status (updated)

- [x] Sign convention for any lag/delay metric is explicitly defined and verified against a
      synthetic ground-truth case before being used to support a PASS.
- [x] Corrected delay scan re-run on all 110 episodes (real data), not just relabeled.
- [x] Per-field delay distributions and confidence margins (vs. delay=0) reported.
- [x] Same-timestamp exact-equality checks (head/position passthrough,
      joint/end/waist non-passthrough) explicitly distinguished from lag-based timing
      evidence, per supervisor instruction — timing supports but does not prove provenance.
- [x] No training source files modified during this check.
- [x] Gate B not started.

## Artifacts

- `manifest.json` — machine-readable run manifest, exact commands, bug description.
- `artifacts/sign_audit.py` — runnable, saved synthetic sign test (not just inline/ephemeral).
- `artifacts/core_delay_scan_corrected_110ep_final.json` — full per-field delay distributions,
  mean-MAE-by-delay curves, margins vs. delay=0, all 110 episodes.
- `figures/core_delay_scan_corrected_110ep.png` — per-field delay-scan plots with the
  correct-sign convention labeled directly on the axes.

## Changed files

None. No training source files were modified. `core/finetune/datasets/wan_dataset.py`,
`core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py`, and all other pre-existing dirty
files remain exactly as they were before this check.

## Next action

Gate A-core PASS stands, now on corrected evidence. Per supervisor instruction, proceed to
**Gate B**: zero-training/zero-init smoke inference for C0/no-condition, D2-oracle,
D2-causal (minimal stub), and D-action-core. Do not begin any training. Continue to treat
`D-action-effector` as quarantined/excluded pending new provenance.
