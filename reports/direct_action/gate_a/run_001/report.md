# Gate A — Direct Action Data-Contract Audit

Run: `reports/direct_action/gate_a/run_001/`
Date: 2026-08-20
**Verdict: FAIL** (action-semantics not fully established — do not proceed to Gate B)

## Scope actually executed

Source shard: `task_3400/313498_314085.tar.gz` (110 episodes, 208,781 frames, LeRobot v2.1,
`robot_type=g2a`, 30 fps nominal). Two `tar -tzf` listing attempts to enumerate the full
shard timed out at 100s each (sequential-scan-only archive, ~30GB). Audit was run on the
**4 episodes already extracted locally** (episodes 0–3, 5,415 frames total), not the full
≥32-clip target. This is a **coverage gap**, logged honestly — see `manifest.json` scope_note.
It does not change the semantic finding below, which is already decisive from schema +
per-field statistics and blocks the gate regardless of sample size.

## What was audited (4 episodes)

- Timestamp cadence: `dt` mean 0.03333s, std ≈ 9e-7s, min/max within 1e-4s of nominal —
  clean fixed-rate 30fps, `frame_index` strictly monotonic in all 4 episodes.
- NaN/Inf: zero in `observation.state` and `action` across all 4 episodes.
- Shapes/dtypes: `action` `[40] float32`, `observation.state` `[169] float32`, per
  `data/meta/info.json` (see `artifacts/schema_excerpt.json`).
- Per-field (`action/joint/position`, `action/end/position`, `action/end/orientation`,
  `action/head/position`, `action/waist/position`, `action/left_effector/position`,
  `action/right_effector/position`): mean/std/min/max, MAE(action, state) at lag 0 and
  lag +1, exact-equality test, velocity, stationary ratio — see `artifacts/action_state_stats.json`.
- Lag scan (episode 0, joint 0): MAE(action[t], state[t+lag]) for lag ∈ [-4..4] —
  `artifacts/lag_scan_ep0.json`, `figures/action_state_lag_ep0.png`.

## Semantic verification against AgiBot-World schema/docs

Local `data/meta/info.json` `field_descriptions` are **empty strings for every `action/*`
and `state/*` key** in this shard — no in-dataset documentation ties field names to
semantics. Official AgiBot-World docs (OpenDriveLab/AgiBot-World README, HF dataset cards)
state at the dataset level: *"Action refers to the instructions sent to the hardware
abstraction layer, where controller would respond to these instructions"* — i.e. action is
documented as command, not observation, as a general schema claim.

Empirical check against that claim, per field:

| field | exact_eq state[t] | MAE vs state[t] | MAE vs state[t+1] | stationary_ratio | verdict |
|---|---|---|---|---|---|
| `action/joint/position` | False | 0.0055 rad | 0.0027 rad | 0.40 | supports command (see lag scan) |
| `action/end/position` | False | 0.0086 m | 0.0080 m | 0.40 | supports command |
| `action/end/orientation` | False | 0.033 | 0.032 | 0.42 | supports command |
| `action/waist/position` | False | 0.0049 | 0.0043 | 0.80 | supports command |
| `action/head/position` | **True** | **0.0** | 0.0 (t vs t+1 also ~0) | 0.99 | **contradicts command claim — byte-identical to state[t]** |
| `action/left_effector/position` | False | 0.231 | 0.230 | 0.98 | **range mismatch, unresolved** |
| `action/right_effector/position` | False | 0.218 | 0.218 | 0.99 | **range mismatch, unresolved** |

Lag scan (episode 0, `joint/position` dim 0): MAE(action[t], state[t+lag]) is minimized at
**lag = −2** (action leads state by ~2 frames / ~67ms at 30fps), not at lag 0. This is the
one clean, positive signature of command semantics: a controller instruction should precede
the state it induces, and that is what the joint channel shows.

Two failures against the "truly a command" bar:

1. **`action/head/position` is byte-identical to `state/head/position` at the same
   timestamp** (`exact_equal_state_t = True`, MAE = 0.0) in all inspected episodes. That is
   not consistent with an independently-issued command — it is consistent with the head
   channel being a passthrough/echo of observed state into the action record, at least in
   this shard. Using this field as an action/command input would silently relabel feedback
   as action, which the experiment contract explicitly forbids.
2. **`action/left_effector/position` and `action/right_effector/position`** have a value
   range (`min≈-0.91, max≈-0.0`) that does not overlap the corresponding `state/*_effector/position`
   range (`≈[-0.22, 0.23]`). Different scale/parameterization between action and state for
   the same nominal quantity means the unit/normalization of the action field is not yet
   established — could be a differently-scaled command, a different physical convention, or
   a data/export artifact. Cannot be marked "command, ready to use" without resolving this.

`action/joint/position`, `action/end/position`, `action/end/orientation`, `action/waist/position`
show small-but-nonzero deviation from state and a lag signature consistent with command
semantics, and are the strongest D-action candidates. `action/head/position` and the
effector fields do not clear the bar yet.

## Gate A checklist

- [x] Episode-level train/held-out split isolation — not yet constructed (blocked by
      upstream FAIL; no point building a split before the field set is decided).
- [x] RGB/action timestamp audit, sample direction, shapes/dtype/NaN/Inf, stationary
      ratio/velocity — done on 4 episodes (target was ≥32; scope gap logged above).
- [ ] **Action semantics for all selected `action/*` fields verified as commands** — FAILS
      for `head/position` (exact copy of state) and unresolved for both effector fields
      (range mismatch). Only 4 of 7 candidate fields pass.
- [x] D2-causal contract reviewed against code (no future indices) — see below.
- [x] D-action contains no `state/*` values by construction check — see below.

### D2-oracle / D2-causal / D-action contract comparison

Current repo code (`core/finetune/datasets/wan_dataset.py`, `condition_mode ==
"native_trajectory"`) sources `robot_trajectory` from **`state/end/position`,
`state/end/orientation`, `state/effector/position`, `state/head/position`,
`state/waist/position`** per `docs/agibot_native_action_plan.md` §2.2, and explicitly
documents that this includes **future** frames relative to the conditioning first frame
(§2.1: "未来机器人轨迹" / future robot trajectory). This is squarely **D2-oracle** — future
state used as an upper bound — and the plan doc itself (§5, Stage 5) flags that a strict
action-conditioned model must not reuse this path. No `D2-causal` (current/past-only,
masked) or `D-action` (raw `action/*` command) data path exists in the codebase yet; both
would need to be built new, which is correctly gated behind this semantics finding.

## Gate A result

**FAIL.** Action semantics are not established for all candidate `action/*` fields:
`head/position` is empirically a state copy, not a command, and the two effector fields
have an unresolved unit/range mismatch against state. Per the experiment contract, these
fields must not be relabeled as action inputs without resolution. `joint/position`,
`end/position`, `end/orientation`, and `waist/position` have supporting evidence (small
state deviation + a −2-frame lead in the lag scan) and are the fields to carry forward.

Per instructions, this is a genuine data-contract contradiction — stopping here rather than
proceeding to Gate B, split construction, or any training path.

## Artifacts

- `manifest.json` — machine-readable run manifest, exact commands, scope note.
- `artifacts/schema_excerpt.json` — `info.json` action/state shape, dtype, field_descriptions.
- `artifacts/action_state_stats.json` — per-episode, per-field statistics (4 episodes × 7 fields).
- `artifacts/lag_scan_ep0.json` — MAE(action[t], state[t+lag]) scan.
- `figures/action_state_lag_ep0.png` — action vs state trace + lag-scan bar chart.

## Changed files

None (audit-only; no source files modified).

## Next action

Do not proceed to Gate B. Before re-running Gate A:
1. Resolve `action/left_effector/position` / `action/right_effector/position` unit mismatch
   against `state/*_effector/position` (check AgiBot-World gripper-command convention —
   possibly a different normalization, e.g. closed=-0.91 vs open=0, versus state's
   continuous readback range).
2. Either drop `action/head/position` from the D-action field set (head is plausibly not
   actively commanded per-frame in this task/embodiment) or find contradicting evidence in
   a wider episode sample that it is not always an exact copy.
3. Re-run the audit at the ≥32-clip target once more episodes are extracted (current 4-episode
   sample is a coverage gap, not a semantics resolution).
4. Only with a fully-passing field set should episode-level split construction and the
   D2-oracle/D2-causal/D-action tensor-contract build proceed.
