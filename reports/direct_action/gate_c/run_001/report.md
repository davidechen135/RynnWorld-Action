# Gate C Report — run_001

**Gate:** C (weight-optimization overfit validation)
**Date:** 2026-08-20
**Verdict:** FAIL (script's own pass_conditions; see §7 for nuance — not a clean PASS, not a clean FAIL either)
**Prerequisite:** [Gate B run_001](../../gate_b/run_001/report.md) — PASS (C0/C3 narrowed scope)

---

## 1. Purpose

Gate C tests whether the D-action-core (33D) `NativeTrajectoryEncoder` adapter can
actually learn — i.e. whether gradients flow through it end-to-end and the
production flow-matching loss can be driven down on:

1. A **single real example** (one D-action-core window, fixed noise/timestep) —
   tests basic optimization correctness.
2. A **tiny set of 4 real, distinct examples** (4 different train-split episodes) —
   tests that the adapter can differentiate between distinct trajectory inputs,
   not just memorize a constant.
3. **Checkpoint recoverability** — a saved adapter state_dict must reload into a
   fresh instance and reproduce identical loss.

This is **not** full training. The `WanTransformer3DModel` backbone is frozen
throughout (`requires_grad_(False)`, `eval()` mode); only the adapter's
16,745,794 parameters receive gradient updates via AdamW. No `finetune.py`,
`accelerate`, or DeepSpeed machinery is used — this is a minimal standalone
script that reuses Gate B's model-load / adapter-attach / `wan_forward`
monkey-patch path.

## 2. Guardrails carried forward from Gate A/B

- `data/agibot_362_native_smoke` and episode IDs 649616/650872/650989 are
  **not used anywhere** in this run.
- D-action-core channel order is locked to **raw index order** (end/position,
  end/orientation, joint/position, waist/position) — identical to Gate B, no
  reorder applied.
- Torch remains at `2.7.0a0+7c8ec84dab.nv25.03` (unchanged, reused Gate B venv).
- No parquet files duplicated into the repo tree.
- No commits/pushes; pre-existing dirty changes preserved.
- Gate D not started. This run stops for independent review as instructed,
  regardless of verdict.

## 3. Conditioning inputs: what's real vs. synthetic

| Input | Source |
|---|---|
| D-action-core (33D) trajectory windows | **Real** — task_3400 train-split parquets, raw-index-order sliced |
| Video latent, image latent, text embedding | **Synthetic** — `torch.randn` with fixed per-example seed, correct shape/dtype |

This mirrors Gate B's own precedent: Gate B's C0/C3 forward-pass comparison
never decoded a real video either (it used minimal random latents). Gate C's
target is gradient-flow / loss-convergence evidence for the *adapter*, not
video fidelity, so reusing that precedent is in-scope. Real video extraction
from the 30GB `task_3400/313498_314085.tar.gz` archive was attempted (`tar`
streaming extraction of one episode's video member) and timed out after 120s
due to the sequential nature of a gzip-compressed 30GB tar; this was
abandoned once Gate B's precedent was confirmed to already cover Gate C's
actual evidentiary requirement.

## 4. Loss formula

Reproduced exactly from `core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py`
`compute_loss()` (lines ~897–1042), restricted to the `native_trajectory`
condition branch:

```
s = timestep_idx / 1000
sigma_t = flow_shift * s / (1 + (flow_shift - 1) * s)      # flow_shift = 5.0
noisy_latents = (1 - sigma) * video_latent + sigma * noise
noisy_latents[:, :, 0:1] = img_latent                       # frame 0 replaced
target = noise - video_latent
pred = model(hidden_states=noisy_latents, ..., robot_trajectory=window, null_condition=False)
timestep_weight = clamp(1 / (sigma*(1-sigma) + 1e-5), max=10.0); normalized by its mean
loss = mean(timestep_weight * mean((pred[:,:,1:] - target[:,:,1:])**2, dims=(1,2,3,4)))
```

Loss is computed on frames `[1:]` only (frame 0 is the replaced image-condition
frame), matching production semantics exactly.

## 5. Results

### 5.1 Phase 1 — single-example overfit

- Episode 8 (train-split), 80 AdamW steps, fixed seed (identical noise/timestep every step).
- Loss: **0.2505 → 0.00928** (**96.3% reduction**), monotonic decrease confirmed
  (mean of last 10 steps < mean of first 10 steps).
- **PASS** against a >50% reduction bar.

### 5.2 Phase 2 — tiny-set overfit

- 4 distinct train-split episodes (3, 39, 67, 92), disjoint from Phase 1's episode 8.
- Fresh zero-init adapter instance (independent of Phase 1), 200 epochs, fixed
  per-example seed.
- Mean loss: **2.0685 → 1.6733** (**19.1% reduction**), monotonic decrease
  confirmed (mean of last 5 epochs < mean of first 5 epochs) but the loss
  **plateaus by ~epoch 150** (oscillating 1.63–1.69 for the remaining ~50
  epochs) — additional epochs at this LR are unlikely to help further.
- Per-example losses at epoch 199: `[2.148, 1.271, 1.665, 1.610]` — episode 39
  (index 1) improved the most (2.44→1.27), episode 3 (index 0) barely moved
  (2.44→2.15), consistent with different examples landing on intrinsically
  harder/easier timestep-weighted loss regions.
- **FAIL** against the run's own 20% reduction bar (19.1% < 20%) — narrowly
  short, and confirmed plateaued rather than still-improving.

Two prior attempts (80 epochs → 16.8%, then 200 epochs → 19.1%) show
diminishing returns from just adding epochs; the bottleneck is optimizer
dynamics (LR/schedule) across 4 simultaneously-fit loss landscapes, not
insufficient training time.

### 5.3 Phase 3 — checkpoint recoverability

- Adapter checkpoint saved at the end of Phase 1 (`single_example_step0079.pt`).
- Loaded into a **fresh** `NativeTrajectoryEncoder(input_dim=33)` instance,
  loss recomputed with identical inputs/seed.
- In-memory loss: `0.009164758957922459`
- Reloaded-checkpoint loss: `0.009164758957922459`
- **Exact match. PASS.**

Loss curves: `figures/gate_c_loss_curves.png`.

## 6. What this validates

- Gradients flow correctly through the `NativeTrajectoryEncoder` adapter into
  the frozen backbone's forward pass and back.
- The adapter can memorize a single real D-action-core window to near-zero loss.
- The adapter shows real (if incomplete, plateaued) differentiation across 4
  distinct real trajectory inputs — it is not simply ignoring the conditioning
  signal.
- Checkpoints are exactly recoverable — no silent state loss on save/reload.

## 7. What this does NOT claim

- **Not** generalization to held-out episodes or unseen conditioning (both
  phases are memorization/overfit tests by design).
- **Not** semantic video quality or rollout success — loss convergence only,
  no video was decoded or rendered.
- **Not** full-scale training behavior — batch size, LR, and schedule all
  differ from production `finetune.py`.
- The Phase 2 shortfall (19.1% vs. a 20% bar) is reported honestly, not
  adjusted post hoc. The 20% threshold itself was set by this run's own
  script and is not derived from any prior instruction — it is a reasonable-
  but-arbitrary bar. Phase 1 and Phase 3 are unambiguous passes.

## 8. Verdict and recommendation

The script's mechanical pass_conditions evaluate to **FAIL** (4 of 5 conditions
pass; `phase2_reduction_gt_20pct` is false at 19.1%). This is reported as-is.

The underlying evidence is substantive: gradient flow is confirmed, single-
example overfit is strong (96.3%), tiny-set overfit shows real but plateaued
progress (19.1%, confirmed still monotonic net of noise), and checkpoints are
exactly recoverable. Whether 19.1% plateaued reduction on a 4-example tiny-set
constitutes "enough" for Gate C's purposes is a judgment call this report
defers to independent review, per the standing instruction that Gate C must
be independently reviewed before any Gate D work begins — regardless of which
way this verdict lands.

**Recommendation for the reviewer:** either (a) accept this evidence as
sufficient given Phase 1/3's strength, or (b) request a rerun with a
different optimizer/LR schedule (e.g. lower LR with more epochs, or per-
example LR warmup) to see if Phase 2 can clear 20% before Gate D is even
discussed.

**Gate D has not been started.** No training source files were modified. No
commits or pushes were made. All pre-existing dirty changes in the repository
were preserved.
