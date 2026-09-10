# D-action-effector — quarantine note

**Status: quarantined, excluded from D-action-core v1. Not proven, not disproven.**

## What run_001 got wrong

run_001's report table claimed `action/left_effector/position` (range `[-0.91, 0.0]`) and
`state/left_effector/position` (reported as `[-0.185, 0.199]`, i.e. mean±std, mislabeled as
min/max) had non-overlapping ranges. That specific numeric claim was a **reporting error**:
the actual min/max of `state/left_effector/position` in episode 0 is `[-0.402, -0.0]`, and
across the full 110-episode shard both `action/*_effector/position` and
`state/*_effector/position` share the identical outer bound `[-0.91, -0.0]`
(`core_global_stats` — see full-shard `episodes_stats.jsonl` cross-check below). So the
"unit mismatch" as originally stated is **not established** — range non-overlap was a
mis-transcription, not a real observation.

## What is still unresolved (why it stays quarantined, not promoted to core)

1. **No authoritative source defines the sign/zero convention.** `data/meta/info.json`
   `field_descriptions` are empty strings for both fields (same as every other field in this
   shard). The official AgiBot-World glossary (OpenDriveLab/AgiBot-World README, fetched via
   web search) defines `effector` and `position` only generically ("end effector, e.g.
   dexterous hands or grippers"; "spatial position, encoder position, angle, etc.") with no
   documented open/closed sign convention.
2. **Per-episode MAE(action, state) at lag 0 is inconsistent across episodes** (checked on
   episodes 0-3): 0.231, 0.272, 0.005, 0.136 for left effector — an order of magnitude
   spread that core fields do not show (core fields: tight range 0.002-0.011 MAE across all
   110 episodes). This suggests the effector action/state relationship is not stationary in
   the same way core fields are, and needs more targeted investigation (e.g. does the
   command track state exactly except during active grasp transitions?).
3. **Lag scan is flat, not a clean minimum.** Unlike the 4 core fields (clear dip at negative
   lag), episode 0's left-effector lag scan is nearly flat across lag ∈ [-4, 4]
   (0.2301-0.2338), showing no clear command-leads-state signature. This alone doesn't
   disprove command semantics (a held/discretized gripper command changes rarely, which
   would flatten a lag scan even for a true command), but it does not provide the same
   positive evidence the core fields have.

## Per supervisor instruction

Range non-overlap alone is not treated as a contradiction (command and feedback may
legitimately use different units) — and in this case, the original non-overlap claim was
itself incorrect. But absent authoritative provenance or a clean lag signature, this field
does not yet clear the "verified as a command" bar either. It remains **optional/unverified**
and is excluded from D-action-core v1. It may be revisited with:
- targeted analysis of gripper open/close transition frames vs. commanded transitions, or
- discovery of an authoritative AgiBot-World gripper-command convention document/code.

No transform was applied to force alignment with state, per instruction.
