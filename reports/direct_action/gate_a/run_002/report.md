# Gate A run_002 — D-action-core (narrowed contract)

Run: `reports/direct_action/gate_a/run_002/`
Date: 2026-08-20
**Verdict: PASS** (scope: D-action-core, 4 fields, 33D. D-action-effector remains quarantined, tracked separately, not part of this verdict.)

`run_001` (FAIL, full 7-field/40D contract) is preserved unmodified at
`reports/direct_action/gate_a/run_001/`. This run does not overwrite it — it narrows scope
per supervisor decision and corrects a reporting error discovered in run_001's effector
analysis (below).

## Correction to run_001

run_001's table stated `action/left_effector/position` (`[-0.91, 0.0]`) and
`state/left_effector/position` (mislabeled as `[-0.185, 0.199]`, which was actually
mean±std, not min/max) had non-overlapping ranges. **This was a transcription error in my
own report, not a real finding.** Re-checked directly: episode 0's `state/left_effector/position`
true min/max is `[-0.402, -0.0]`, and across the full 110-episode shard both
`action/*_effector/position` and `state/*_effector/position` share the same outer bound
`[-0.91, -0.0]`. Range non-overlap, as originally claimed, does not hold. Per the
supervisor's instruction, range non-overlap alone would not have been treated as
disqualifying anyway (command and feedback may use different units) — but in this case there
never was a non-overlap. See `artifacts/effector_quarantine_note.md` for the full
correction and the (separate, still-open) reasons the effector fields remain quarantined.

## D-action-core: scope and result

Contract: `action/joint/position` (14D, rad), `action/end/position` (6D, m),
`action/end/orientation` (8D, quaternion), `action/waist/position` (5D). Total 33D.
Full spec: `artifacts/d_action_core_contract.json`.

Excluded from core: `action/head/position` (same-timestamp exact copy of
`state/head/position` in every one of the 4 originally-sampled episodes — a genuine state
passthrough, confirmed in run_001, not re-litigated here). `action/left_effector/position`,
`action/right_effector/position` — quarantined, see below.

### Widened audit: all 110 episodes (target was ≥32)

Extraction: rather than a full `tar -tzf` listing (times out at 100s on this ~30GB archive,
confirmed twice), the exact list of 110 parquet member paths was constructed from
`info.json`'s `data_path` template (`data/chunk-{chunk:03d}/episode_{index:06d}.parquet`)
and `total_episodes=110`, then extracted by explicit member list — `tar` early-exits once
all named members are found, avoiding the full sequential scan. All 110 episodes / 208,781
frames extracted and audited.

- **NaN/Inf**: 0 across all 4 core fields, all 110 episodes, 208,781 frames.
  (`artifacts/core_global_stats.json`, `artifacts/core_audit_all_episodes.json`)
- **Exact-equality vs state[t]**: 0/110 episodes exact-equal for any of the 4 core fields —
  rules out state passthrough for the core set (unlike `head/position`).
- **MAE(action[t], state[t]) per field, across all 110 episodes**:

  | field | mean MAE | min MAE | max MAE | mean stationary ratio |
  |---|---|---|---|---|
  | `action/joint/position` | 0.00398 | 0.00214 | 0.00601 | 0.381 |
  | `action/end/position` | 0.00427 | 0.00189 | 0.00859 | 0.385 |
  | `action/end/orientation` | 0.00742 | 0.00201 | 0.03287 | 0.455 |
  | `action/waist/position` | 0.00486 | 0.00004 | 0.01135 | 0.740 |

  Small, tight, nonzero deviation from state across the entire shard — consistent with a
  command that closely tracks but is not identical to the resulting state.

- **Lag scan (32 episodes, all 4 core fields)**: `artifacts/core_lag_scan_32ep.json`,
  `figures/core_lag_scan_32ep.png`. All 4 fields show MAE(action[t], state[t+lag])
  minimized at negative lag (action leads state by 2-4 frames / 67-133ms at 30fps) and
  monotonically increasing for positive lag. This is the clean, uniform command-precedes-
  response signature the contract requires, now confirmed across 32 episodes (not just the
  1-episode sample in run_001) and across all 4 fields (not just `joint/position`).

- **Timestamp/sampling**: `dt` mean 0.03333s, std <1e-6s, frame_index strictly monotonic,
  confirmed across all 110 episodes (`artifacts/core_audit_all_episodes.json`).

### Episode split

`artifacts/episode_split.json` — deterministic `sha256(shard:episode_index) mod 100 < 15`
split, 91 train / 19 held-out, disjointness verified by assertion. No source-frame overlap
possible since the split operates at episode granularity and each frame belongs to exactly
one episode.

### No future-state leakage

D-action-core reads only `action/*` fields at the conditioning window's own row index —
no `state/*` fields and no future-index reads are part of this contract by construction.
Verified: the contract spec (`artifacts/d_action_core_contract.json`) lists only
`action/joint/position`, `action/end/position`, `action/end/orientation`,
`action/waist/position` as inputs; state fields appear only as the audit's comparison
target, never as a proposed model input. No D2-causal implementation exists yet in the
codebase to audit for a causal mask (only the pre-existing `native_trajectory`/D2-oracle
path exists, in `core/finetune/datasets/wan_dataset.py` and
`docs/agibot_native_action_plan.md` — confirmed to intentionally include future state, and
explicitly out of scope for D-action).

### Not yet finalized (deferred to prep-script implementation, not blocking Gate A)

- Normalization: global (not train-only) mean/std reported here for audit visibility;
  the actual training-time stats must be refit on `episode_split.json`'s `train_episodes`
  only, per `docs/agibot_native_action_plan.md` §2.3 (no per-episode normalization).
- Quaternion normalization + rotation-6D conversion for `end/orientation` (per plan doc §2.2).
- Windowing/padding/interpolation policy for building fixed-length 81-frame samples.

These are prep-implementation details, not data-contract defects — logged so they aren't
silently skipped at build time.

## D-action-effector: quarantine (unchanged verdict, corrected reasoning)

Still excluded from core. See `artifacts/effector_quarantine_note.md`. Summary: no
authoritative unit/sign-convention documentation was found in this repo, the official
AgiBot-World glossary, or accessible converter code; per-episode MAE(action, state) is
inconsistent across episodes (0.005-0.272, an order of magnitude spread the core fields
don't show); and the lag scan is flat rather than showing a clean command-leads-state
minimum. None of this is disqualifying on its own, but none of it clears the bar either.
Marked optional/unverified, excluded from D-action-core v1, not transformed to force a match
with state.

## Gate A checklist (D-action-core scope)

- [x] Episode-level train/held-out split, zero overlap — `artifacts/episode_split.json`.
- [x] Timestamp/sample-direction/shape/dtype/NaN/Inf/stationary-ratio/velocity audit,
      widened to all 110 episodes (target was ≥32).
- [x] Action semantics verified as commands for all 4 D-action-core fields: zero exact-state
      matches (110/110 episodes), tight nonzero MAE, and a uniform command-leads-state lag
      signature (32/32 episodes).
- [x] No future-state leakage: D-action-core contract reads only same-timestep `action/*`.
- [x] D2-causal reviewed: not yet implemented; existing code path is D2-oracle only
      (`native_trajectory`, intentionally future-inclusive per its own docs) — correctly out
      of scope for D-action.
- [x] Tensor contract (dims, units, dtype, coordinate frames, normalization plan) recorded
      in `artifacts/d_action_core_contract.json`.

## Gate A result

**PASS for D-action-core** (4 fields, 33D, 91/19 episode split, 110-episode audit, zero
NaN/Inf, zero exact-state matches, consistent command-leads-state lag signature).
**D-action-effector remains unresolved/quarantined**, excluded from this PASS, tracked
separately for future resolution.

Per supervisor instruction: proceeding only to **Gate B** (zero-training/zero-init smoke
inference and sensitivity checks for D-action-core, C0/D2-oracle/D2-causal as applicable).
**Not proceeding to any training.**

## Artifacts

- `manifest.json` — machine-readable run manifest, exact commands, scope.
- `artifacts/d_action_core_contract.json` — full tensor contract (dims, units, dtype,
  coordinate frames, normalization/padding plan, split, lag-scan summary).
- `artifacts/effector_quarantine_note.md` — correction + quarantine rationale.
- `artifacts/core_audit_all_episodes.json` — per-episode, per-core-field audit, all 110 episodes.
- `artifacts/core_global_stats.json` — full-shard (208,781-frame) per-field mean/std/min/max/NaN/Inf.
- `artifacts/core_lag_scan_32ep.json` — lag scan, 4 fields × 32 episodes.
- `artifacts/episode_split.json` — train/held-out episode split (91/19), disjoint.
- `artifacts/schema_excerpt.json` — info.json shape/dtype/field_descriptions excerpt (carried from run_001).
- `figures/core_lag_scan_32ep.png` — per-field lag-scan plots.
- `figures/core_per_episode_110ep.png` — per-episode MAE and stationary-ratio, all 110 episodes.

## Changed files

None (audit-only; no source files modified).

## Next action

Proceed to **Gate B**: zero-training/zero-init smoke inference for C0/no-condition,
D2-oracle, D2-causal (if a minimal causal-window stub is built for comparison), and
D-action-core. Verify model loading, adapter zero-init behavior, deterministic inputs,
identical base output where mathematically expected; save videos + metrics; no success
claims from loss alone. Do not build the D2-causal or D-action-core data-prep/training code
paths beyond what Gate B's smoke test needs. Do not touch D-action-effector further unless
new provenance is found.
