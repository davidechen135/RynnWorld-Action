# RynnWorld-Action

### Action-Conditioned Robot World Modeling from a Single Image

RynnWorld-Action is a research extension of
[RynnWorld-Teleop](https://github.com/alibaba-damo-academy/RynnWorld-Teleop)
that replaces human skeletal control with native robot action sequences.
Given a robot's first-person observation and a future action trajectory, the
model generates the corresponding egocentric manipulation video.

> **Research goal:** turn robot control logs into a predictive visual world
> model without requiring a rendered skeleton or an additional control video.

This repository contains the action representation, conditioning modules,
training recipes, fixed-noise evaluation tools, and architecture diagnostics
developed for this direction.

## Overview

The original RynnWorld-Teleop pipeline conditions a Wan video diffusion model
on a first frame and a temporal hand-skeleton representation. Our extension
keeps the strong visual prior of the pretrained Wan DiT while replacing the
skeleton input with robot-native state and action signals.

```text
first-person frame ---------------------------> Wan I2V latent stream
                                                       |
robot state + future action sequence                    |
        |                                              |
        v                                              v
native action features -> temporal conditioner -> action-conditioned Wan DiT
                                                       |
                                                       v
                                  future egocentric robot video
```

The central design objective is not merely to generate plausible motion, but
to make the generated future respond to the **identity, magnitude, and temporal
order** of the supplied action sequence.

## Highlights

- **Native robot action conditioning.** Uses robot proprioception and commands
  directly instead of converting actions into a human-style skeleton video.
- **Leakage-aware temporal features.** The current V10/V11 representation
  combines observed state, target state, relative motion, and velocity while
  anchoring relative quantities to the observed initial state.
- **First-frame conditioned generation.** Preserves the original scene and
  robot appearance through the Wan2.2 TI2V image-conditioning path.
- **Multiple conditioning paths.** Supports temporal residual conditioning,
  adaptive normalization, visual gating, and optional spatial action control.
- **Controlled evaluation.** Includes fixed-seed comparisons for correct,
  held, shifted, reversed, swapped, and zero-action conditions.
- **Mechanism-level diagnostics.** Provides layer-wise probes for temporal
  energy, projection spectrum, conditioning magnitude, spatial gates, and DiT
  block response.
- **Reproducible experiment recipes.** Training and evaluation scripts cover
  single-clip fitting, temporal alignment, spatial control, batch-size sweeps,
  and LoRA ablations.

## Action Representation

The current native representation is built from a dual-arm robot trajectory:

```text
state(37) + target(37) + relative(37) + velocity(37) = 148 dimensions / step
```

The 37-dimensional state uses translation and 6D rotation features for both
arms. Relative and velocity terms expose motion explicitly, while the observed
initial state provides the causal reference used to construct the sequence.

Implementation:

- `core/control/native_action_features.py`
- `core/control/native_trajectory_encoder.py`
- `core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py`

## Architecture

The action conditioner maps `[B, T, D_action]` trajectories into the hidden
width of the Wan DiT. The repository currently supports four complementary
interfaces:

1. **Temporal residual path** — projects per-step action features into the DiT
   hidden space.
2. **Adaptive normalization path** — produces action-dependent modulation for
   transformer activations.
3. **Visual action gate** — couples the action sequence to visual features from
   the first-person observation.
4. **Spatial control path** — injects image-space action maps at selected DiT
   blocks when spatial annotations are available.

These interfaces are intentionally instrumented so that action selectivity can
be measured at the encoder output, injection point, and intermediate DiT
blocks—not inferred from training loss alone.

## Repository Layout

```text
core/control/
  native_action_features.py          native V10/V11 feature construction
  native_trajectory_encoder.py       temporal, adaptive, and spatial encoders

core/finetune/
  datasets/wan_dataset.py            action-aware dataset and collation
  models/wan_i2v/
    rynnworld_teleop_trainer.py       Wan action injection and training path

scripts/
  prepare_native_action_*.py         dataset/cache preparation
  train_native_action_*.sh           experiment recipes
  eval_native_action_v2_rot6d37.py   fixed-noise action evaluation
  diagnose_native_action_signal.py   conditioning-path diagnostics
  ablate_native_action_injection.py  architecture ablations
  official_control_positive_control.py

reports/
  action_conditioning_diagnosis/     reusable causal probes
  native_action_arch_ab/             architecture A/B measurements
  native_action_v10_v11_execution/   experiment plans and analysis

docs/
  upstream_baseline.md               upstream reproduction notes
  agibot_native_action_plan.md       native action project plan
  model_forward.md                   model-forward documentation
```

## Setup

The project builds on the environment and checkpoints released by
RynnWorld-Teleop.

```bash
conda create -n rynnworld-action python=3.10 -y
conda activate rynnworld-action
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the Wan2.2 TI2V backbone and the upstream RynnWorld-Teleop weights as
described in the
[official repository](https://github.com/alibaba-damo-academy/RynnWorld-Teleop).
Paths can be supplied through the existing environment variables or adjusted
in the experiment launch scripts.

## Preparing Native Action Data

The preparation scripts produce cached action-conditioned training windows
from the robot dataset. Choose the representation required by the experiment:

```bash
# 148D state/target/relative/velocity representation
python scripts/prepare_native_action_v8_features.py --help

# State-based V10 cache
python scripts/prepare_native_action_v10_state_256.py --help

# V10 cache with image-space action control
python scripts/prepare_native_action_v10_spatial.py --help

# Delay-aligned temporal windows
python scripts/prepare_native_action_delay_aligned.py --help
```

Dataset locations are intentionally kept outside the repository. Do not commit
raw videos, cached latents, checkpoints, or private robot data.

## Training Recipes

The repository provides small-scale architecture checks as well as longer
native-action runs:

```bash
# Single-clip fitting: verify that one action/video pair is learnable
bash scripts/train_native_action_v8_single_clip.sh

# Spatially gated single-clip experiment
bash scripts/train_native_action_v9_spatial_gate_single_clip.sh

# V10 state-conditioned training
bash scripts/train_native_action_v10b_state_256.sh

# Temporal alignment experiment
bash scripts/train_native_action_v11_temporal_alignment.sh

# Controlled optimization ablations
bash scripts/train_native_action_v10b_batch_sweep.sh
bash scripts/train_native_action_v10b_lora_unfreeze.sh
```

Each launcher exposes the important paths and training settings near the top of
the script. Review them before starting a run on a new machine.

## Evaluation

Action controllability is evaluated with the first frame, diffusion noise,
seed, sampler, and checkpoint held fixed. Only the action sequence changes.

Recommended condition set:

| Condition | Purpose |
|---|---|
| `correct` | Intended action trajectory |
| `held` | Constant action/state control |
| `shifted` | Temporal alignment sensitivity |
| `reversed` | Temporal-order sensitivity |
| `swapped` | Action-identity sensitivity |
| `zero` | Conditioning-presence reference |

Useful entry points:

```bash
python scripts/eval_native_action_v2_rot6d37.py --help
python scripts/diagnose_native_action_signal.py --help
python scripts/ablate_native_action_injection.py --help
python scripts/official_control_positive_control.py --help
```

We recommend reporting video quality and action selectivity together. A useful
model should preserve the pretrained visual prior while producing differences
that are consistent with the requested action rather than sampling noise.

## Current Research Focus

The current codebase is focused on three questions:

1. How should continuous robot actions be represented so that temporal order
   remains accessible to a large video diffusion transformer?
2. Where should the action signal enter the DiT to preserve image quality while
   enabling strong controllability?
3. Which fixed-noise counterfactual evaluations best distinguish true action
   understanding from generic motion generation?

The included probes and experiment recipes make these questions measurable at
both the representation and generated-video levels.

## Roadmap

- [x] Reproduce the upstream RynnWorld-Teleop training and inference paths
- [x] Add native dual-arm action feature construction
- [x] Add temporal, adaptive-normalization, and visual-gating interfaces
- [x] Add optional spatial action conditioning
- [x] Build fixed-noise counterfactual evaluation tools
- [x] Add layer-wise conditioning and projection diagnostics
- [ ] Consolidate the best-performing conditioning interface
- [ ] Release a clean training manifest and compact example dataset
- [ ] Publish quantitative multi-task action-controllability benchmarks
- [ ] Release pretrained native-action checkpoints

## Relationship to RynnWorld-Teleop

This is an independent research extension built on the open-source
RynnWorld-Teleop codebase. The upstream project conditions video generation on
human hand-pose/skeleton sequences; this repository investigates direct robot
action conditioning for egocentric robot video prediction.

Upstream resources:

- [RynnWorld-Teleop repository](https://github.com/alibaba-damo-academy/RynnWorld-Teleop)
- [Project page](https://alibaba-damo-academy.github.io/RynnWorld-Teleop.github.io/)
- [Technical report](https://arxiv.org/abs/2607.06558)
- [Model checkpoints](https://huggingface.co/Alibaba-DAMO-Academy/RynnWorld-Teleop)

## License

This repository follows the upstream RynnWorld-Teleop license. Third-party
models, datasets, and checkpoints remain subject to their respective licenses.
See [LICENSE](LICENSE) and the upstream project for details.

## Acknowledgements

This work builds on
[RynnWorld-Teleop](https://github.com/alibaba-damo-academy/RynnWorld-Teleop),
[Wan2.2](https://github.com/Wan-Video/Wan2.2), and the AgiBot robot dataset and
tooling used in our experiments. We thank the respective authors for releasing
their code, models, and data resources.
