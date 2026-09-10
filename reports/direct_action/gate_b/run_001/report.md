# Gate B run_001 — zero-training structural/inference smoke test

Run: `reports/direct_action/gate_b/run_001/`
Date: 2026-08-20
**Verdict: Gate B PASS (narrowed scope: C0 vs C3 only; C1 and C2 marked UNAVAILABLE by
design, not stubbed).** No weights were optimized. No training source files were modified.
All pre-existing dirty repo state was preserved.

## Scope and controls, as defined for this run

- **C0 = no control.** `null_condition=True`, no `robot_trajectory` passed to
  `WanTransformer3DModel.forward`. Baseline.
- **C1 = D2-oracle (future state), diagnostic leakage upper bound — UNAVAILABLE.**
  No trained oracle checkpoint or genuine future-state conditioning implementation exists
  in this repo. The only trajectory encoder available (`NativeTrajectoryEncoder`) ships with
  a zero-initialized `output_projection`, so its output is byte-identical to C0 regardless of
  what data is fed to it — it cannot demonstrate a real "leakage upper bound," which requires
  trained weights that have actually learned to exploit future information. Rather than fake
  this with a stub, C1 is explicitly marked unavailable, per instruction ("never use a fake
  stub as a valid control").
- **C2 = D2-causal (current/history-only) — UNAVAILABLE.** No genuine causal/history-only
  conditioning pathway is implemented anywhere in the repo. `NativeTrajectoryEncoder` consumes
  a full trajectory window, not a causal-masked one, and no other module implements this.
  Same reasoning as C1: marked unavailable rather than stubbed.
- **C3 = D-action-core.** Real, timestamp-aligned raw 33D D-action-core columns
  (`action/end/position`[6D] + `action/end/orientation`[8D] + `action/joint/position`[14D] +
  `action/waist/position`[5D], concatenated in **raw index order**, not joint-first) from
  genuine task_3400 parquet files, fed through a freshly constructed
  `NativeTrajectoryEncoder(input_dim=33)`.

With C1/C2 unavailable, Gate B's executable inference scope is **C0 vs C3 only**, evaluated
as a plumbing/sensitivity test:
1. At zero-init (`control_scale=0` analogue — the native-trajectory pathway has no separate
   scalar `control_scale`; its zero contribution comes entirely from the adapter's zeroed
   `output_projection`), C3 must reproduce C0 exactly.
2. After perturbing `output_projection` away from zero (a controlled, non-training,
   post-hoc weight edit), C3 must diverge from C0, confirming the pathway is genuinely wired
   into the forward computation and sensitive to its input — **not** a semantic-success claim,
   since the adapter weights are otherwise untrained/random-init.

## Data provenance guardrail

Per explicit correction mid-task: **`data/agibot_362_native_smoke` and its episode IDs
(649616/650872/650989) were NOT used anywhere in this run** — not as input, not as split
evidence, not as metrics, not as PASS evidence. That dataset uses an entirely different
24D `ee_pose6d_gripper_head_waist_v1` schema (task_362) and is unrelated to the audited
Gate A contract. All Gate B data is drawn exclusively from task_3400
(`task_3400/313498_314085.tar.gz`, 110 episodes, 208,781 frames), matching the shard Gate A
run_003 audited. Every evaluated episode (8, 83) was asserted to be a member of the canonical
110-episode split manifest and NOT a held-out episode; no legacy IDs ever appear in the
manifest, loader, or metrics.

## 1. Split manifest

No executable split-generation script was ever saved in Gate A run_001/002/003 — only prose
(`sha256(shard:episode_index) mod 100 < 15 → held_out`) plus the resulting
`episode_split.json`. `artifacts/build_split_manifest.py` attempted to reproduce the split
from ~10 plausible literal readings of the hash-key format; **none reproduced the reference
exactly** (best overlap ~5/19 held-out episodes — consistent with an unrecoverable exact key
format, not an off-by-one bug). This is reported honestly as
`hash_reproduction_status: UNVERIFIED_BY_RECOMPUTATION`, not silently forced to match and not
treated as a blocker, per instruction. `reports/direct_action/gate_a/run_002/artifacts/episode_split.json`
is adopted as the **canonical** Gate B split manifest, and its own internal integrity was
independently re-verified from the JSON: disjoint (train ∩ held_out = ∅), covers all 110
episodes (0–109), counts consistent (91 train / 19 held_out).

```
$ python3 reports/direct_action/gate_b/run_001/artifacts/build_split_manifest.py
GATE B SPLIT MANIFEST: PASS (canonical=reference episode_split.json, hash_reproduction_status=UNVERIFIED_BY_RECOMPUTATION)
```

Output: `artifacts/split_manifest.json`, log: `logs/split_manifest.log`.

## 2. D-action-core 33D channel-order lock

`artifacts/action_core_schema.py` is the single source of truth for how the 33D D-action-core
vector is sliced from the raw 40D `action` column, **in raw index order**:

| output position | field | raw indices | dim |
|---|---|---|---|
| 0:6   | `action/end/position`    | 2:8   | 6  |
| 6:14  | `action/end/orientation` | 8:16  | 8  |
| 14:28 | `action/joint/position`  | 16:30 | 14 |
| 28:33 | `action/waist/position`  | 33:38 | 5  |

Excluded: `action/head/position` (30:33, state passthrough), quarantined effectors
(`action/left_effector/position` 0:1, `action/right_effector/position` 1:2), and
`action/robot/velocity` (38:40, not part of D-action-core). **This is NOT a joint-first
layout** — no reorder was applied or tested; the tensor preserves raw column order exactly.
Self-test (synthetic 40D marker vector, asserts each field lands at its expected output slice
and confirms zero leakage from head/effector/velocity):

```
$ python3 reports/direct_action/gate_b/run_001/artifacts/action_core_schema.py
SELF-TEST PASSED: 33D slice order verified, no head/effector/velocity leakage.
```

Log: `logs/action_core_schema_selftest.log`. All Gate B loader and model code imports
`CORE_SLICES`/`slice_core_33d` from this module — no ad hoc re-derivation.

## 3. Data extraction (scratch, not duplicated into repo)

110 parquet files extracted from `task_3400/313498_314085.tar.gz` to
`/tmp/scratch/gate_b_task3400/data/data/chunk-000/` (outside the repo, per guardrail — the
report tree retains only manifests, hashes, bounded samples, commands, logs, and metrics, not
raw parquet). Archive member path required an extra `data/` prefix
(`data/data/chunk-{:03d}/episode_{:06d}.parquet`) not reflected literally in `meta/info.json`'s
`data_path` template — discovered via bounded `tar -tzf | head` inspection. All 110 members
extracted successfully, 0 errors. `extracted/info.json` retains only the small metadata file
(episode/frame counts, fps, schema pointers), not the parquet payload itself.

## 4. Loader validation

`artifacts/gate_b_smoke.py`, `validate_loader()`. Ran against two deterministically-selected
(seed=42) **train-split** episodes (8, 83) from the real task_3400 data:

```
[LOADER] validating task_3400 parquet loader ...
  ep=008: 2902 frames, action OK, state OK, core OK, dt=0.03333±1.38e-06
  ep=083: 1213 frames, action OK, state OK, core OK, dt=0.03333±8.26e-07
  Determinism: ep_a=8 reads identically across two loads. PASS
[LOADER] PASS
```

Checks performed and passed for both episodes:
- **Schema**: `action` (40D, float32), `observation.state` (169D, float32), `timestamp`,
  `frame_index`, `episode_index` columns present and correctly typed.
- **Shape**: `action_all.shape == (T, 40)`, `state_all.shape == (T, 169)`,
  D-action-core slice `core.shape == (T, 33)`.
- **Order**: D-action-core slice matches the raw-index-order lock in
  `action_core_schema.py` (verified by that module's independent self-test, §2).
- **Timestamps**: strictly increasing, `dt ≈ 1/30s` (fps=30) with std < 1.4e-6.
- **`frame_index`**: contiguous `0..T-1` per episode (no gaps, no accidental
  cross-episode concatenation).
- **Finite values**: no NaN/Inf in `action`, `state`, or the D-action-core slice.
- **Split isolation**: both evaluated episodes confirmed present in `train_episodes` and
  absent from `held_out_episodes`; `episode_index` column inside each parquet matches the
  expected episode number (no accidental mislabeling).
- **No future-state reads**: loader reads `action[t]`/`state[t]` from the same row only; no
  windowing or lookahead is performed at the loader level (the only future-adjacent operation
  in the whole Gate A/B pipeline is Gate A's already-audited/sign-corrected lag *scan*, which
  is diagnostic-only and not used by this loader).
- **Determinism**: same-seed episode selection reproduced identically across two independent
  runs; re-reading the same parquet twice produced bit-identical arrays
  (`np.array_equal` on the full `action` column).

Full metrics: `artifacts/gate_b_metrics.json` → `.loader`.

## 5. Environment

Disposable venv (`/tmp/scratch/gate_b_env`, `--system-site-packages`) reusing the working
CUDA-enabled system torch without modification. Minimal top-level packages installed via
pip's normal resolver, constrained to the pre-existing torch build
(`torch_constraint.txt`: `torch==2.7.0a0+7c8ec84dab.nv25.3`), `--upgrade-strategy
only-if-needed`, no `--no-deps`/`--ignore-installed`/`--force-reinstall`, no full lockfile
install. Torch verified unchanged (version, file path outside venv site-packages, CUDA
availability) before and after **every** install step in this run, including the two escalations
described below.

```
python:       3.12.3
platform:     Linux-5.10.134-013.9.kangaroo.al8.x86_64-x86_64-with-glibc2.39
torch:        2.7.0a0+7c8ec84dab.nv25.03  (unchanged throughout)
torch path:   /usr/local/lib/python3.12/dist-packages/torch/__init__.py  (system, not venv)
cuda:         available=True, version=12.8
gpu:          NVIDIA A100-SXM4-80GB, 81920 MiB, driver 570.133.20
diffusers:    0.39.0
transformers: 5.13.1
accelerate:   1.14.0
peft:         0.20.0
imageio:      2.37.4
decord:       0.6.0
deepspeed:    0.19.5
```

Full details: `artifacts/environment_summary.txt` / `.json`, `artifacts/venv_pip_freeze.txt`
(no secrets/credentials). Server-direct API used throughout; `localhost:15721` was never
referenced.

**Two additional dependency escalations were required beyond the initial `diffusers` /
`transformers` / `accelerate` install** (both performed under the same constraint-file +
`--upgrade-strategy only-if-needed` rigor, both verified not to touch torch):
1. Importing `rynnworld_teleop_trainer` (needed to trigger the `wan_forward` monkey-patch)
   transitively requires `peft` and `termcolor` — small, pure top-level packages used for
   LoRA config types and colored logging in the trainer module, unrelated to torch.
2. `core/finetune/models/__init__.py` eagerly imports **every** `.py` file under every
   subdirectory of `core/finetune/models/` (not just the Wan I2V trainer), which cascades into
   the full dataset stack (`core/finetune/datasets/utils.py`) and pulled in `imageio`,
   `decord` (video I/O) and `deepspeed` (used inside `core/finetune/trainer.py`) as further
   transitive requirements. Each was added the same way, one probe-and-install cycle at a
   time, confirming torch integrity before and after each.

`pip check` reports no broken requirements after all installs. `torch_constraint.txt` pinned
`torch==2.7.0a0+7c8ec84dab.nv25.3` throughout; pip never proposed replacing or upgrading torch
at any step.

## 6. Monkey-patch verification (bug found and fixed during this run)

**Bug**: the first version of `gate_b_smoke.py` computed the repo-root path for `sys.path`
insertion as `Path(__file__).parents[4]`, which actually resolves to `reports/` (one level too
shallow — the file lives 5 levels below repo root, not 4). This caused
`import core.finetune...` to fail with `ModuleNotFoundError`, which was — in the first version
of the script — caught by a bare `except Exception` and treated as a non-fatal "partial
import warning," allowing the script to silently continue with `WanTransformer3DModel.forward`
**unpatched**. A first full run completed "successfully" under this bug; its results were
discarded because the C0/C3 comparison would have been meaningless (both would have run the
unpatched base forward, so a match wouldn't distinguish anything).

**Fix**: corrected `parents[4]` → `parents[5]`, and — more importantly — replaced the
swallowed exception with an explicit check that `WanTransformer3DModel.forward` actually
changed identity after the import, raising a hard `RuntimeError` if not. This is now asserted
every run:

```
[PATCH] applying wan_forward monkey-patch ...
  [PATCH] ✅ [Monkey Patch Applied] `WanTimeTextImageEmbedding.forward` has been replaced to ensure float32 stability during mixed-precision training.
✅ [Monkey Patch Applied] `WanTimeTextImageEmbedding.forward` has been replaced to ensure float32 stability during mixed-precision training.
✅ [Monkey Patch Applied] `WanTransformer3DModel.forward` has been added control video latent.
  [PATCH] confirmed WanTransformer3DModel.forward is patched
```

All results reported below are from the **corrected** run only.

## 7. Model load

`WanTransformer3DModel` instantiated from `pretrained/Wan2.2-TI2V-5B-Diffusers/transformer`
config (30 layers, 24 attention heads, 128 head dim, in/out channels 48), then loaded with
`pretrained/RynnWorld-Teleop-Causal/ema_weights.bin` (12 GB checkpoint, smallest available
SFT-family checkpoint; the streaming/causal-distilled variant of Wan2.2-TI2V-5B). 5.00B
parameters, bfloat16, on `cuda` (A100-SXM4-80GB). Load succeeded without error.

```
[MODEL] loading RynnWorld-Teleop-Causal checkpoint ...
  loaded 5.00B params in 7.8s on cuda
```

(First cold-cache load took ~99s; the run reported above benefited from filesystem cache from
the prior discarded run — both loads succeeded identically otherwise.)

## 8. Adapter construction

Fresh `NativeTrajectoryEncoder(input_dim=33)` constructed and attached to the model as
`model.native_trajectory_encoder`. Confirmed `output_projection.weight.norm() == 0.0` and
`output_projection.bias.norm() == 0.0` at construction (zero-init invariant, asserted, not
assumed) — this is the mechanism by which the native-trajectory pathway contributes exactly
zero to `hidden_states` at initialization; unlike the skeleton-video `control_video_latent`
pathway, there is no independent `control_scale` scalar on this branch (confirmed by direct
code inspection of `wan_forward` — see prior investigation notes), so Gate B's
"control_scale=0" requirement is satisfied structurally by the adapter's own zero-init weights,
and the requested sweep collapses to the two-point zero-init/non-zero-init comparison
documented below.

## 9. C0 vs C3 forward-pass smoke test

Minimal legal input shapes derived from model config (`patch_size=[1,2,2]`, `in_channels=48`,
`text_dim=4096`): `hidden_states` `[1,48,1,4,4]`, `timestep` `[1,1]`,
`encoder_hidden_states` `[1,16,4096]`. C3's D-action-core input window: 21 frames × 33D from
episode 8 (confirmed train-split-only), shape `[1,21,33]`.

```
[FWD C0] running C0 (no control) forward ...
  C0: shape=(1, 48, 1, 4, 4), finite=True, time=0.35s
[FWD C3-zero] running C3 zero-init (must == C0) ...
  C3-zero == C0: True (max_abs_diff=0.00e+00)
[FWD C3-perturb] perturbing output_projection, expect C3 != C0 ...
  C3-perturb != C0: True (max_abs_diff=4.1992e-02)
[DETERMINISM] running C0 twice with same seed ...
  Deterministic (same seed): True
```

- **C0**: forward pass completes without crash/NaN, output shape and dtype as expected.
- **C3 at zero-init**: output is **byte-identical** to C0 (`max_abs_diff = 0.0`, exact
  floating-point equality, not just close) — confirms the D-action-core conditioning pathway
  contributes exactly zero at `control_scale=0`, within the tightest possible tolerance
  (exact equality, tolerance=0.0), satisfying the "must reproduce C0 within a documented
  tolerance" requirement.
- **C3 after perturbation**: `output_projection` weight set to a uniform constant (0.01),
  bias to zero (a controlled, non-training, post-hoc edit — not an optimizer step). Output
  diverges from C0 by `max_abs_diff ≈ 0.042`, confirming the pathway is genuinely wired into
  the forward computation and is sensitive to the D-action-core input. **This is reported as
  plumbing/sensitivity evidence only — it is explicitly NOT a semantic-success claim**, since
  `output_projection` was set to an arbitrary constant, not trained weights, and the rest of
  the adapter (attention layers, projections) remains random-init.
- Weights were restored to zero-init immediately after the perturbation check; no
  optimizer step was ever taken anywhere in this run (`torch.no_grad()` used throughout;
  the `.fill_()`/`.zero_()` calls are direct in-place weight edits under `no_grad`, not
  gradient-based updates).
- **Determinism**: repeating the C0 forward pass with the same seed produced a bit-identical
  output tensor.

Full metrics: `artifacts/gate_b_metrics.json` → `.forward_passes`, `.determinism`.

## 10. Video generation / output evidence

**Not attempted and out of scope for this run.** The forward-pass smoke test intentionally
uses the smallest legal single-denoising-step latent shape (`[1,48,1,4,4]`) to validate
plumbing cheaply; it does not run the multi-step diffusion sampling loop or VAE decode needed
to produce a viewable video. No video output or video-generation-blocked evidence is produced
by this run — this is a deliberate scope choice (single forward-pass structural check), not a
blocked/failed video pipeline. Video-quality evaluation is out of scope for Gate B by
definition (Gate B is zero-training structural/plumbing validation) and remains a Gate C/D
concern once weights are actually trained.

## Gate B checklist status

- [x] Legacy `data/agibot_362_native_smoke` / episode IDs 649616, 650872, 650989 never used
      as Gate B input, split evidence, metrics, or PASS evidence.
- [x] All Gate B data traced to the audited task_3400 shard and cross-checked against the
      canonical split manifest; every evaluated episode confirmed non-held-out.
- [x] 33D channel order locked to raw index order (not joint-first), asserted via self-test,
      and used consistently by loader and model code via a single shared module.
- [x] No duplication of the 110 parquet files into the repo/report tree (scratch-only).
- [x] C1, C2 explicitly marked UNAVAILABLE with reasons — no fake/stub controls used.
- [x] Loader shape/dtype/order/timestamp/split-isolation/finite/determinism validated on
      real data.
- [x] No accidental future-state reads at the loader level.
- [x] Smallest bounded pretrained-checkpoint inference smoke test run
      (RynnWorld-Teleop-Causal, 12GB, 5B params).
- [x] At control_scale=0 (zero-init adapter), C3 reproduces C0 exactly (tolerance=0.0).
- [x] At nonzero scale (perturbed adapter), only plumbing/sensitivity reported — no semantic
      success claim.
- [x] No weights optimized anywhere in this run.
- [x] Exact config, commands, environment summary (no secrets), logs, machine-readable
      metrics saved under this run directory.
- [x] No crash, NaN, shape/time mismatch, unavailable-required-checkpoint, or unverifiable
      control semantics encountered in the final (corrected) run.
- [x] All pre-existing dirty repo changes preserved; nothing committed or pushed.

## Artifacts

- `artifacts/action_core_schema.py` — 33D channel-order lock module + self-test.
- `artifacts/build_split_manifest.py` — split-manifest builder/cross-checker.
- `artifacts/split_manifest.json` — canonical Gate B split manifest (adopted from Gate A
  run_002, integrity re-verified).
- `artifacts/gate_b_smoke.py` — loader validation + model load + C0/C3 forward-pass smoke
  test (this run's main script).
- `artifacts/gate_b_metrics.json` — machine-readable metrics, full run output.
- `artifacts/environment_summary.txt` / `.json` — environment, package versions, GPU info
  (no secrets).
- `artifacts/venv_pip_freeze.txt` — full venv pip freeze (filtered of local/editable paths).
- `extracted/info.json` — task_3400 `meta/info.json` (small metadata only, no parquet payload).
- `logs/gate_b_smoke.log` — full stdout/stderr of the corrected run.
- `logs/split_manifest.log`, `logs/action_core_schema_selftest.log` — supporting tool logs.

## Changed files

None in `core/` or other training source files. `reports/direct_action/gate_b/run_001/` was
created new. All pre-existing dirty changes in the repo (`core/finetune/datasets/wan_dataset.py`
and others) were left untouched and unexamined beyond what was already investigated before
this run began.

## Next action

Gate B PASS on the narrowed C0/C3 scope, with C1/C2 explicitly unavailable. Per instruction,
proceed automatically to **Gate C**: single-example then tiny-set overfit, with recoverable
checkpoints and logs. **Do not start Gate D or full training until Gate C is independently
reviewed.**
