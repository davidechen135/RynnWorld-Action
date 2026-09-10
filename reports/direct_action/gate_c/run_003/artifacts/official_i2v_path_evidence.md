# Gate C official first-frame I2V path evidence

Date: 2026-08-20

The repository's Wan/RynnWorld I2V implementation uses the first-frame latent as a clean condition, rather than as an ordinary noisy video target:

- `core/streaming/train_distill.py:7296-7318` sets `img_latent_x = gt_lat[:, :, :1]`, creates a latent noise tensor, creates `first_frame_mask` filled with one, then sets `first_frame_mask[:, :, 0] = 0` and `condition = img_latent_x`.
- `core/streaming/train_distill.py:7331-7334` feeds the transformer `latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents` at every denoising step.
- `core/streaming/train_distill.py:7347-7350` advances the scheduler and reapplies `(1 - first_frame_mask) * condition + first_frame_mask * latents` after the step.
- `core/streaming/train_distill.py:7326-7329` constructs the configured scheduler and sets its inference timesteps. The same scheduler/timestep sequence is shared by all controls in the audit.
- `core/streaming/model.py:872-890` uses the same clean-condition mask in the training pipeline.
- `reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py:397-399` already preserves frame 0 during training; its rollout implementation is being audited separately because the prior rollout used a simplified Euler loop.

Protocol decision: the corrected audit uses one encoded real source first-frame latent, one prompt embedding, one fixed initial noise tensor, one scheduler/timestep sequence, one CFG setting, one inference-step count, one resolution, and one duration for all four conditions. Only the 33D action tensor differs: correct, all-zero, deterministic shuffled, and deterministic reversed. The clean first-frame mask is applied before every model call and after every scheduler step.
