# AgiBot task-362 multi-episode LoRA fine-tune (v3)

Scope: the user asked to fine-tune on task 362 ("Folding shorts") only, with **more
data** than the earlier single-episode runs, then visualize and jointly judge whether
the fine-tune helped. This is the writeup. Verdict up front, then evidence.

Previous rounds (task 357, single episode) are in `docs/agibot_finetune.md`. This is a
different task and ~10× the data, so numbers are not directly comparable to those.

---

## Verdict

**The fine-tune did not produce a usable world model on unseen episodes, and the
headline metric says the opposite of the truth.** Three findings, in order of
importance:

1. **Both zero-shot and fine-tuned collapse on held-out episodes.** Frame 0 renders
   correctly (it is pinned from the conditioning latent), then the rollout melts from
   frame ~2: arms dissolve, then the whole frame flattens to brown. This happens on all
   3 held-out episodes, for both the released SFT checkpoint and the LoRA.
2. **Root cause: a control-strength domain gap in the data I prepared — not the LoRA,
   not data volume.** The decisive test: the *same model and same code* run on an
   official sample clip does **not** collapse (detail ratio 0.917) while every AgiBot
   clip does (0.605–0.66). My control latents carry ~half the temporal motion of the
   official ones (0.074–0.092 vs 0.132–0.152), and the AgiBot head-camera video is
   itself only ~70% as dynamic (0.28 vs 0.38–0.41). The SFT model was trained where the
   control signal is strong relative to the video; given a weaker one it does not stay
   anchored and drifts.
3. **A real harness defect exists, but it is not the cause.** `control_type=add`
   zero-initializes `control_patch_embedding` and hardcodes `control_scale=0.1` every
   run, discarding the trained control weights in the SFT checkpoint (100 steps reaches
   only 1/31 of SFT magnitude). I fixed it (`--control_init_from`), verified the fix
   restores the magnitude (0.0369 vs SFT 0.0364) — **and the collapse did not go away**
   (detail ratio 0.617 → 0.605). Worth keeping as a correctness fix; it was not the
   bottleneck.

So the answer to "did more data help" is: **this run cannot answer that question.**
Every arm collapses for a reason upstream of the fine-tune. Data scale was not the
binding constraint, and neither was the harness defect. That is the honest result.

## Do not trust motion energy alone on this run

The metric that looked like success:

| held-out episode | GT | zero-shot | %GT | fine-tuned ckpt-100 | %GT |
|---|---|---|---|---|---|
| 650190 / f000729 | 0.0106 | 0.0073 | 69% | 0.0101 | **96%** |
| 650213 / f000729 | 0.0096 | 0.0073 | 76% | 0.0101 | **105%** |
| 650872 / f000810 | 0.0044 | 0.0116 | 262% | 0.0139 | 315% |

Read alone, this says fine-tuning moved motion from 69/76% to 96/105% of GT — nearly
perfect. **It is a false positive.** The melt itself generates frame-to-frame change of
roughly GT's magnitude. What discriminates is structure retention:

| | detail ratio (last third / first third) |
|---|---|
| GT | **1.02 – 1.15** (structure held) |
| zero-shot | 0.65 / 0.65 / 0.66 |
| fine-tuned | 0.62 / 0.63 / 0.74 |

GT holds its spatial detail; every generated run loses ~35-40% of it. Frame strips
(`outputs/agibot_362_sweep/good_eps_trace.png`) confirm visually: GT shows a clean
81-frame fold, generated shows two frames then mush.

Lesson worth keeping: **motion energy cannot detect collapse** — a collapsing rollout
and a correctly-moving one can score the same. Always pair it with a structure metric
and look at frames.

## Root cause: control-strength domain gap

Figure: `outputs/agibot_362_plots/domain_gap_362.png`.

The control experiment that settles it — same checkpoint, same code, only the input
clip changes:

| input clip | zero-shot detail ratio | collapses? |
|---|---|---|
| official `basic_fold_1_6_rgb` | **0.917** (GT 0.986) | **no** |
| my 650190 held-out | 0.646 (GT 1.152) | yes |

Inference config and the SFT checkpoint are therefore exonerated. What differs is the
motion the tensors carry, measured as mean \|x_t − x_{t−1}\| on the stored latents:

| clip | control Δt | video Δt |
|---|---|---|
| official jigsaw | 0.132 | 0.411 |
| official fold | 0.152 | 0.408 |
| official jenga | 0.143 | 0.382 |
| mine 362 train | 0.074 | 0.281 |
| mine 362 held-out | 0.092 | 0.283 |
| mine 357 (v2) | 0.065 | 0.290 |

Latent *scale* matches the official data (video std 0.80 vs 0.78–0.89, control std 1.63
vs 1.68–1.70), so this is not a normalization bug — it is signal strength. The skeleton
control is a proxy built from a **fitted** camera (AgiBot ships no intrinsics), and it
carries about half the motion the model expects.

## The harness defect (real, fixed, not the cause)

`core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py` (the `else` branch, i.e.
`control_type=add`) creates the control path fresh each run:

```python
control_patch_embedding.weight.zero_()          # zero-init
control_patch_embedding.bias.zero_()
control_scale = nn.Parameter(torch.tensor(0.1))  # hardcoded
```

It never loads `control_patch_embedding.bin` / `control_scale.bin` from the SFT
checkpoint (that load path exists only for accelerate *resume*). Measured:

| source | `control_patch_embedding` \|w\|max | `control_scale` |
|---|---|---|
| released SFT | **0.036377** | 0.1094 |
| LoRA v1 ckpt-24 | 0.000832 (44× smaller) | 0.0996 |
| LoRA v1 ckpt-100 | 0.001167 (**31× smaller**) | 0.0996 |
| **warm-start ckpt-24** | **0.036865** ✓ | **0.1089** ✓ |

Figure: `outputs/agibot_362_plots/control_diag_362.png`.

**The fix works and the collapse remains.** Full 75-step warm-start run, all 3 held-out
episodes, structure retention (detail ratio, higher = better, GT ≈ 1.1):

| episode | GT | zero-shot | LoRA v1 (zero-init) | LoRA warm-start |
|---|---|---|---|---|
| 650190 | 1.152 | 0.646 | 0.617 | 0.593 |
| 650213 | 1.143 | 0.648 | 0.631 | 0.607 |
| 650872 | 1.021 | 0.659 | 0.742 | 0.705 |
| **mean** | **1.105** | **0.651** | **0.663** | **0.635** |

All three arms sit within ±0.03 of each other and ~0.45 below GT. Fixing the control
init moved nothing (0.663 → 0.635 mean, marginally *worse*); frames still melt from
frame ~1. Keep the flag as a correctness fix — a from-scratch control path is clearly
wrong for a short run, and it also means the v1 "zero-shot vs fine-tuned" comparison was
never the intended experiment (zero-shot had an intact control path, the LoRA arm did
not). But it does not explain the collapse.

Added `--control_init_from <dir>` (finetune.py + trainer), which warm-starts
`control_patch_embedding` and `control_scale` from a trained checkpoint instead of zero.
Default `None` keeps the original zero-init byte-identical, so nothing else changes.

Two implementation notes worth remembering:
- `Args` in `finetune.py` is a **pydantic BaseModel** built via `cls(**vars(args))`. An
  argparse flag with no matching model field is **silently dropped** — my first
  warm-start run ignored the flag for exactly this reason and had to be redone. The
  field declaration is mandatory, not optional.
- Confirm the warm start in the log (`control_patch_embedding warm-started from ...`,
  `control_scale warm-started to 0.109375`) before trusting a run.

Run: `bash scripts/agibot_362_lora_ctlinit.sh` → `training/agibot_362_lora_ctlinit/`.

**Training loss does not distinguish the two runs** (step 1: 0.2429 vs 0.2425; step 20:
0.1047 vs 0.1032) because loss is a single-step denoising MSE, largely insensitive to
control-path magnitude. The verdict must come from held-out rollout, not from loss. This
is why the loss curve alone would have declared v1 fine.

## What was built

| artifact | what it is |
|---|---|
| `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel.json` | 395 clips / 20 training episodes |
| `/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_362_skel_heldout.json` | 56 clips / 3 held-out episodes (whole unseen episodes) |
| `scripts/agibot_362_extract.py` | pull head_color.mp4 + proprio h5 from the two 48G tars, then guarded-delete them |
| `scripts/agibot_362_prep_all.py` | per-episode prep driver → merged clip indexes |
| `scripts/agibot_362_lora_2gpu.sh` | v1 launcher (zero-init control, 100 steps) |
| `scripts/agibot_362_lora_ctlinit.sh` | warm-start launcher (`--control_init_from`) |
| `scripts/agibot_362_heldout_sweep.py` | held-out eval across all 3 unseen episodes |
| `scripts/plot_362_curves.py` | loss / grad_norm / lr, three stacked panels |
| `scripts/plot_362_heldout.py` | motion-energy vs structure-retention, side by side |
| `scripts/plot_362_control_diag.py` | the control-path magnitude figure |
| `scripts/plot_362_domain_gap.py` | **the root-cause figure** |

Figures are in `outputs/agibot_362_plots/` (light + dark variants of each).

Data pipeline: 20 train + 3 held-out whole episodes, AV1 `head_color.mp4` → 81-frame
clips → official three-tuple; real end-effector trajectory rendered to a 21-keypoint
skeleton video and VAE-encoded as control. Episode-disjoint by construction (asserted).
Camera is *fitted* per episode (AgiBot ships no intrinsics), R² ≈ 0.64–0.99.

Training: 2×H20, LoRA rank 32 + control path, `control_type=add`, 395 clips, 100 steps
(4 epochs), 49:46, peak 15.9 GB/GPU. Loss went 0.285 → min 0.0926@step60 → 0.109; the
rolling mean is essentially **flat at 0.15–0.20** the whole run.

## Fixed along the way

`scripts/agibot_heldout_eval.py` had no GT anchor — it compared zero-shot against
fine-tuned in absolute units, which cannot say whether a change moved *toward* GT. It
now exports the pipeline's own GT decode (`out, gt_out, *_ = pipe(...)`) as a third arm.

Do **not** hand-roll that GT decode. I tried, and it silently produced a GT motion value
of 0.666 vs the correct 0.0044 — a ~150× error that inverts the verdict. The
normalization arithmetic was fine; the bug was skipping
`video_processor.postprocess_video`, which the pipeline applies to both generated and GT
frames. Self-check that catches it: GT frame 0, `img_latent` decoded alone, and generated
frame 0 must all have the same mean RGB (the pipeline pins
`latents[:, :, 0:1] = img_latent`).

## Scene diversity caveat

All 220 in-range task-362 episodes are shorts-on-a-bed. Episode and trajectory count
went up ~10× vs the single-episode runs; **scene variety is still 1**. Even with the
control path fixed, this dataset cannot show cross-scene generalization.

## Next, in order

1. **Strengthen the control signal — this is the one that matters.** The skeleton
   control carries ~half the official motion, and the camera it is projected through is
   fitted, not calibrated. Options: recover real intrinsics, render the skeleton with
   larger/higher-contrast keypoints, or drive control from a source with more temporal
   energy. Target the official band (Δt 0.13–0.15).
2. Re-run `agibot_362_heldout_sweep.py` after any control change and judge on **detail
   ratio and frames**, never motion energy alone.
3. Only once a non-collapsing baseline exists does "does more data help" become an
   answerable question. Adding episodes before that just scales a broken run — which is
   exactly what this round demonstrates.
4. Keep `--control_init_from` on for future runs regardless; a from-scratch control path
   on a ~100-step run is never what you want.
