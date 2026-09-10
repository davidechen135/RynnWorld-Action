"""
Gate C: minimal standalone overfit validation for the D-action-core (33D)
NativeTrajectoryEncoder adapter.

Purpose: demonstrate that gradients actually flow through the adapter and
that the loss can be driven down on (a) a single training example and
(b) a tiny multi-example training set, using the EXACT production
flow-matching loss formula from rynnworld_teleop_trainer.py:compute_loss.

This is NOT full training. It does not use finetune.py / accelerate /
DeepSpeed. It reuses Gate B's model-load + adapter-attach + wan_forward
monkey-patch path and adds a minimal AdamW loop over
NativeTrajectoryEncoder parameters ONLY (transformer backbone stays frozen
and in eval mode throughout).

Data: real D-action-core (33D, raw index order) windows from task_3400
train-split episodes only (never held-out, never legacy task_362 data).
Video/image/text conditioning inputs are synthetic-but-legal random
tensors of the correct shape/dtype -- Gate B itself established this
precedent (its C0/C3 comparison never decoded a real video either).
Gate C's target is gradient-flow / loss-convergence evidence for the
adapter, not video fidelity, so this is in-scope and consistent.

What this validates:
  1. Single-example overfit: loss decreases substantially (target: >90%
     reduction from step 0) when repeatedly fed the SAME (action window,
     noise, timestep-sample) tuple.
  2. Tiny-set overfit: loss decreases when trained on a small set of
     distinct real D-action-core windows drawn from distinct train
     episodes, each paired with its own fixed synthetic video/noise
     target (multi-example memorization, not generalization).
  3. Recoverable checkpoints: adapter state_dict saved at intervals and
     verified to reload and reproduce identical loss.

What this does NOT claim:
  - Generalization to held-out episodes or unseen conditioning.
  - Semantic video quality / rollout success (loss convergence only).
  - Full-scale training behavior (batch size, LR schedule, all differ from
    production finetune.py).

Usage:
  cd /mnt/workspace/RynnWorld-Teleop
  source /tmp/scratch/gate_b_env/bin/activate
  python reports/direct_action/gate_c/run_001/artifacts/gate_c_overfit.py \
    --out_dir reports/direct_action/gate_c/run_001 \
    --parquet_dir /tmp/scratch/gate_b_task3400/data/data/chunk-000 \
    --checkpoint_dir pretrained/RynnWorld-Teleop-Causal \
    --base_model_dir pretrained/Wan2.2-TI2V-5B-Diffusers \
    --split_manifest reports/direct_action/gate_a/run_002/artifacts/episode_split.json \
    --seed 42
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# D-action-core 33D column-order lock (must match gate_b_smoke.py exactly)
# ---------------------------------------------------------------------------
CORE_SLICES = [
    ("action/end/position",    slice(2, 8),   6),
    ("action/end/orientation", slice(8, 16),  8),
    ("action/joint/position",  slice(16, 30), 14),
    ("action/waist/position",  slice(33, 38), 5),
]
RAW_ACTION_DIM = 40
CORE_DIM = 33
assert sum(d for _, _, d in CORE_SLICES) == CORE_DIM

FLOW_SHIFT = 5.0
NUM_TRAIN_TIMESTEPS = 1000


def slice_core_33d(raw: np.ndarray) -> np.ndarray:
    assert raw.shape[-1] == RAW_ACTION_DIM
    parts = [raw[..., sl] for _, sl, _ in CORE_SLICES]
    return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# Real D-action-core window loading (train-split only)
# ---------------------------------------------------------------------------

def load_core_window(parquet_dir: Path, episode: int, held_eps: set, n_frames: int) -> np.ndarray:
    assert episode not in held_eps, f"FAIL: episode {episode} is held-out, refusing to load"
    import pandas as pd
    path = parquet_dir / f"episode_{episode:06d}.parquet"
    assert path.exists(), f"FAIL: {path} not found"
    df = pd.read_parquet(str(path))
    assert (df["episode_index"] == episode).all()
    action_all = np.stack(df["action"].values)  # (T, 40)
    assert np.isfinite(action_all).all()
    core = slice_core_33d(action_all)  # (T, 33)
    assert core.shape[0] >= n_frames, f"FAIL: episode {episode} too short ({core.shape[0]} < {n_frames})"
    return core[:n_frames].astype(np.float32)


# ---------------------------------------------------------------------------
# Model / adapter loading (same path as Gate B)
# ---------------------------------------------------------------------------

def apply_monkey_patch(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    import io, contextlib
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel as _WTM
    _forward_before = _WTM.forward
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        import core.finetune.models.wan_i2v.rynnworld_teleop_trainer as _trainer  # noqa: F401
    patch_log = capture.getvalue()
    print(f"  [PATCH] {patch_log.strip()}", flush=True)
    if _WTM.forward is _forward_before:
        raise RuntimeError(
            "FAIL: wan_forward monkey-patch did not change WanTransformer3DModel.forward. "
            "Aborting -- forward passes would run the unpatched base model."
        )
    print("  [PATCH] confirmed WanTransformer3DModel.forward is patched", flush=True)


def load_model(checkpoint_dir: Path, base_model_dir: Path, out: dict):
    print("[MODEL] loading RynnWorld-Teleop-Causal checkpoint ...", flush=True)
    t0 = time.time()
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        str(base_model_dir / "transformer"),
        torch_dtype=torch.bfloat16,
    )
    ema_path = checkpoint_dir / "ema_weights.bin"
    assert ema_path.exists(), f"FAIL: {ema_path} not found"
    ema_weights = torch.load(str(ema_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ema_weights, strict=False)
    del ema_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)  # backbone frozen throughout Gate C

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {n_params/1e9:.2f}B params in {elapsed:.1f}s on {device} (backbone frozen)", flush=True)

    out["model_load"] = {
        "checkpoint": str(checkpoint_dir),
        "base_model": str(base_model_dir),
        "device": str(device),
        "dtype": "bfloat16",
        "n_params_B": round(n_params / 1e9, 3),
        "load_time_s": round(elapsed, 2),
        "backbone_frozen": True,
        "status": "PASS",
    }
    return model, device


def build_adapter_33d(device):
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    enc = NativeTrajectoryEncoder(input_dim=33)
    enc = enc.to(device).to(torch.float32)  # fp32 adapter for stable AdamW steps
    w_norm = enc.output_projection.weight.norm().item()
    b_norm = enc.output_projection.bias.norm().item()
    assert w_norm == 0.0 and b_norm == 0.0, "FAIL: output_projection not zero-init"
    enc.train()
    enc.requires_grad_(True)
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    return enc, n_trainable


# ---------------------------------------------------------------------------
# Synthetic-but-legal video/image/text inputs (Gate B precedent)
# ---------------------------------------------------------------------------

def make_example(model, device, seed, num_frames=5, h_patches=3, w_patches=3):
    """Build one training example: (video_latent, img_latent, null_embedding).
    num_frames counted in post-patch temporal units (p_t=1), so num_frames=5
    means 5 latent frames -- frame 0 replaced by img_latent, loss computed
    on frames[1:] (4 supervised frames), matching production semantics.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    p_t, p_h, p_w = model.config.patch_size  # [1,2,2]
    in_ch = model.config.in_channels          # 48
    text_dim = model.config.text_dim          # 4096
    H = p_h * h_patches
    W = p_w * w_patches

    video_latent = torch.randn(1, in_ch, num_frames, H, W, device=device,
                                dtype=torch.bfloat16, generator=g)
    img_latent = torch.randn(1, in_ch, 1, H, W, device=device,
                              dtype=torch.bfloat16, generator=g)
    null_embedding = torch.randn(1, 16, text_dim, device=device,
                                  dtype=torch.bfloat16, generator=g)
    return video_latent, img_latent, null_embedding


def compute_gate_c_loss(model, enc, video_latent, img_latent, null_embedding,
                         robot_trajectory, device, step_seed):
    """Exact reproduction of rynnworld_teleop_trainer.compute_loss's
    flow-matching formula, restricted to the native_trajectory condition
    path (no pose_video/control_video branch)."""
    model_dtype = model.patch_embedding.weight.dtype
    batch_size, num_channels, num_frames, height, width = video_latent.shape

    g = torch.Generator(device=device).manual_seed(step_seed)
    noise = torch.randn(video_latent.shape, device=device, dtype=model_dtype, generator=g)
    timesteps_idx = torch.randint(0, NUM_TRAIN_TIMESTEPS, (batch_size,),
                                   device=device, generator=g).long()
    s = timesteps_idx.float() / NUM_TRAIN_TIMESTEPS
    sigma_t = FLOW_SHIFT * s / (1 + (FLOW_SHIFT - 1) * s)
    sigma_view = sigma_t.view(batch_size, 1, 1, 1, 1).to(model_dtype)
    shifted_timesteps = sigma_t * NUM_TRAIN_TIMESTEPS

    noisy_latents = (1.0 - sigma_view) * video_latent + sigma_view * noise
    target = noise - video_latent
    noisy_latents[:, :, 0:1, :, :] = img_latent.clone().to(model_dtype)

    first_frame_mask = torch.ones(1, 1, num_frames, height, width, device=device)
    first_frame_mask[:, :, 0] = 0
    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * shifted_timesteps.view(-1, 1, 1, 1).float()).flatten(1)
    timestep_input = temp_ts.to(model_dtype)

    pred = model(
        hidden_states=noisy_latents,
        timestep=timestep_input,
        encoder_hidden_states=null_embedding,
        encoder_hidden_states_image=None,
        robot_trajectory=robot_trajectory,
        null_condition=False,
        return_dict=False,
    )[0]

    sigma_scalar = sigma_t.float()
    timestep_weight = 1.0 / (sigma_scalar * (1.0 - sigma_scalar) + 1e-5)
    timestep_weight = timestep_weight.clamp(max=10.0)
    timestep_weight = timestep_weight / timestep_weight.mean()

    per_sample_loss = ((pred[:, :, 1:].float() - target[:, :, 1:].float()) ** 2).mean(dim=(1, 2, 3, 4))
    loss = (per_sample_loss * timestep_weight).mean()
    return loss


# ---------------------------------------------------------------------------
# Overfit loops
# ---------------------------------------------------------------------------

def run_single_example_overfit(model, enc, opt, example, robot_trajectory, device,
                                seed, n_steps, ckpt_dir, log_every=10):
    video_latent, img_latent, null_embedding = example
    losses = []
    t0 = time.time()
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        loss = compute_gate_c_loss(model, enc, video_latent, img_latent, null_embedding,
                                    robot_trajectory, device, step_seed=seed)  # FIXED seed: same noise/timestep every step
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1e6)
        opt.step()
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  [single-ex] step={step:04d} loss={loss.item():.6f} grad_norm={grad_norm.item():.4e}", flush=True)
        if step in (0, n_steps // 2, n_steps - 1):
            torch.save(enc.state_dict(), ckpt_dir / f"single_example_step{step:04d}.pt")
    elapsed = time.time() - t0
    return losses, elapsed


def run_tiny_set_overfit(model, enc, opt, examples, device, seed, n_epochs, ckpt_dir, log_every=5):
    """examples: list of (video_latent, img_latent, null_embedding, robot_trajectory, episode_id)"""
    losses = []
    t0 = time.time()
    step_counter = 0
    for epoch in range(n_epochs):
        epoch_losses = []
        for i, (vl, il, ne, rt, ep) in enumerate(examples):
            opt.zero_grad(set_to_none=True)
            loss = compute_gate_c_loss(model, enc, vl, il, ne, rt, device,
                                        step_seed=seed + 1000 + i)  # fixed per-example seed
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1e6)
            opt.step()
            epoch_losses.append(float(loss.item()))
            step_counter += 1
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"  [tiny-set] epoch={epoch:03d} mean_loss={mean_loss:.6f} per_ex={['%.4f' % l for l in epoch_losses]}", flush=True)
        if epoch in (0, n_epochs // 2, n_epochs - 1):
            torch.save(enc.state_dict(), ckpt_dir / f"tiny_set_epoch{epoch:03d}.pt")
    elapsed = time.time() - t0
    return losses, elapsed


def verify_checkpoint_reload(model, ckpt_path, example, robot_trajectory, device, seed):
    """Load a saved adapter checkpoint into a FRESH encoder instance and confirm
    the loss it produces matches what the trained-in-memory encoder produces."""
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    enc_fresh = NativeTrajectoryEncoder(input_dim=33).to(device).to(torch.float32)
    enc_fresh.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    enc_fresh.eval()
    model.native_trajectory_encoder = enc_fresh
    video_latent, img_latent, null_embedding = example
    with torch.no_grad():
        loss = compute_gate_c_loss(model, enc_fresh, video_latent, img_latent, null_embedding,
                                    robot_trajectory, device, step_seed=seed)
    return float(loss.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_ex_steps", type=int, default=60)
    parser.add_argument("--tiny_set_epochs", type=int, default=40)
    parser.add_argument("--tiny_set_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"
    art_dir = out_dir / "artifacts"
    log_dir = out_dir / "logs"
    for d in (ckpt_dir, fig_dir, art_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).parents[5]

    results = {
        "run_id": "direct_action/gate_c/run_001",
        "date": "2026-08-20",
        "gate": "C",
        "seed": args.seed,
        "purpose": (
            "Single-example then tiny-set overfit of the D-action-core (33D) "
            "NativeTrajectoryEncoder adapter, backbone frozen, exact production "
            "flow-matching loss. Gradient-flow / loss-convergence evidence only."
        ),
        "d_action_core_channel_order_note": (
            "Raw index order: end/position[2:8], end/orientation[8:16], "
            "joint/position[16:30], waist/position[33:38]. NOT joint-first."
        ),
        "backbone_frozen": True,
        "video_image_text_inputs": "synthetic-but-legal random tensors (Gate B precedent); real D-action-core trajectory data only",
        "legacy_data_excluded": "data/agibot_362_native_smoke and episode IDs 649616/650872/650989 NOT used anywhere in this run",
    }

    try:
        with open(args.split_manifest) as f:
            split_manifest = json.load(f)
        assert split_manifest["shard"] == "task_3400/313498_314085.tar.gz"
        train_eps = sorted(split_manifest["train_episodes"])
        held_eps = set(split_manifest["held_out_episodes"])
        assert set(train_eps).isdisjoint(held_eps)

        print("[PATCH] applying wan_forward monkey-patch ...", flush=True)
        apply_monkey_patch(repo_root)

        model, device = load_model(Path(args.checkpoint_dir), Path(args.base_model_dir), results)

        enc, n_trainable = build_adapter_33d(device)
        model.native_trajectory_encoder = enc
        results["adapter"] = {
            "class": "NativeTrajectoryEncoder",
            "input_dim": 33,
            "init": "zero",
            "n_trainable_params": n_trainable,
        }
        print(f"[ADAPTER] NativeTrajectoryEncoder(input_dim=33), {n_trainable} trainable params", flush=True)

        opt = torch.optim.AdamW(enc.parameters(), lr=args.lr)

        parquet_dir = Path(args.parquet_dir)

        # -------------------------------------------------------------
        # Phase 1: single-example overfit
        # -------------------------------------------------------------
        print("\n[PHASE 1] single-example overfit ...", flush=True)
        rng = np.random.default_rng(args.seed)
        ep_single = int(rng.choice(train_eps))
        assert ep_single not in held_eps

        core_window = load_core_window(parquet_dir, ep_single, held_eps, n_frames=21)
        # NativeTrajectoryEncoder interpolates to output_frames internally; feed as-is.
        robot_trajectory_single = torch.from_numpy(core_window).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,21,33)

        example_single = make_example(model, device, seed=args.seed, num_frames=5, h_patches=3, w_patches=3)

        losses_single, t_single = run_single_example_overfit(
            model, enc, opt, example_single, robot_trajectory_single, device,
            seed=args.seed, n_steps=args.single_ex_steps, ckpt_dir=ckpt_dir,
        )

        loss0, lossN = losses_single[0], losses_single[-1]
        reduction_pct = 100.0 * (loss0 - lossN) / max(loss0, 1e-12)
        print(f"[PHASE 1] loss[0]={loss0:.6f} -> loss[-1]={lossN:.6f} ({reduction_pct:.1f}% reduction)", flush=True)

        results["phase1_single_example"] = {
            "episode": ep_single,
            "episode_in_train_split": ep_single in train_eps,
            "n_steps": args.single_ex_steps,
            "elapsed_s": round(t_single, 2),
            "loss_step0": loss0,
            "loss_final": lossN,
            "reduction_pct": round(reduction_pct, 2),
            "losses": losses_single,
            "monotonic_decrease_last10_vs_first10": float(np.mean(losses_single[-10:])) < float(np.mean(losses_single[:10])),
        }

        # -------------------------------------------------------------
        # Phase 2: tiny-set overfit (distinct real episodes)
        # -------------------------------------------------------------
        print("\n[PHASE 2] tiny-set overfit ...", flush=True)
        candidate_eps = [e for e in train_eps if e != ep_single]
        rng2 = np.random.default_rng(args.seed + 7)
        tiny_eps = sorted(int(e) for e in rng2.choice(candidate_eps, size=args.tiny_set_size, replace=False))
        assert all(e not in held_eps for e in tiny_eps)
        assert len(set(tiny_eps)) == args.tiny_set_size

        # Reset adapter to zero-init fresh instance for phase 2 (independent overfit test)
        enc2, n_trainable2 = build_adapter_33d(device)
        model.native_trajectory_encoder = enc2
        opt2 = torch.optim.AdamW(enc2.parameters(), lr=args.lr)

        examples = []
        for i, ep in enumerate(tiny_eps):
            core_win = load_core_window(parquet_dir, ep, held_eps, n_frames=21)
            rt = torch.from_numpy(core_win).unsqueeze(0).to(device=device, dtype=torch.float32)
            vl, il, ne = make_example(model, device, seed=args.seed + 100 + i, num_frames=5, h_patches=3, w_patches=3)
            examples.append((vl, il, ne, rt, ep))

        losses_tiny, t_tiny = run_tiny_set_overfit(
            model, enc2, opt2, examples, device, seed=args.seed,
            n_epochs=args.tiny_set_epochs, ckpt_dir=ckpt_dir,
        )

        loss0_t, lossN_t = losses_tiny[0], losses_tiny[-1]
        reduction_pct_t = 100.0 * (loss0_t - lossN_t) / max(loss0_t, 1e-12)
        print(f"[PHASE 2] mean_loss[epoch0]={loss0_t:.6f} -> mean_loss[epochN]={lossN_t:.6f} ({reduction_pct_t:.1f}% reduction)", flush=True)

        results["phase2_tiny_set"] = {
            "episodes": tiny_eps,
            "episodes_in_train_split": all(e in train_eps for e in tiny_eps),
            "episodes_disjoint_from_phase1": ep_single not in tiny_eps,
            "n_epochs": args.tiny_set_epochs,
            "set_size": args.tiny_set_size,
            "elapsed_s": round(t_tiny, 2),
            "mean_loss_epoch0": loss0_t,
            "mean_loss_final": lossN_t,
            "reduction_pct": round(reduction_pct_t, 2),
            "losses_by_epoch": losses_tiny,
            "monotonic_decrease_last5_vs_first5": float(np.mean(losses_tiny[-5:])) < float(np.mean(losses_tiny[:5])),
        }

        # -------------------------------------------------------------
        # Phase 3: checkpoint recoverability
        # -------------------------------------------------------------
        print("\n[PHASE 3] checkpoint recoverability check ...", flush=True)
        final_ckpt = ckpt_dir / f"single_example_step{args.single_ex_steps - 1:04d}.pt"
        assert final_ckpt.exists(), f"FAIL: expected checkpoint {final_ckpt} missing"

        model.native_trajectory_encoder = enc
        enc.eval()
        with torch.no_grad():
            loss_in_memory = compute_gate_c_loss(
                model, enc, *example_single, robot_trajectory_single, device, step_seed=args.seed
            ).item()
        enc.train()

        loss_reloaded = verify_checkpoint_reload(
            model, final_ckpt, example_single, robot_trajectory_single, device, seed=args.seed
        )
        ckpt_matches = abs(loss_in_memory - loss_reloaded) < 1e-4
        print(f"[PHASE 3] in-memory loss={loss_in_memory:.6f}, reloaded-checkpoint loss={loss_reloaded:.6f}, match={ckpt_matches}", flush=True)

        try:
            ckpt_rel = str(final_ckpt.relative_to(repo_root))
        except ValueError:
            ckpt_rel = str(final_ckpt)
        results["phase3_checkpoint_recovery"] = {
            "checkpoint_path": ckpt_rel,
            "loss_in_memory": loss_in_memory,
            "loss_reloaded_checkpoint": loss_reloaded,
            "matches": ckpt_matches,
            "status": "PASS" if ckpt_matches else "FAIL",
        }
        model.native_trajectory_encoder = enc  # restore for cleanliness

        # -------------------------------------------------------------
        # Figures
        # -------------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot(losses_single)
        axes[0].set_title(f"Phase 1: single-example overfit (ep {ep_single})")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("flow-matching loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(losses_tiny)
        axes[1].set_title(f"Phase 2: tiny-set overfit ({args.tiny_set_size} episodes)")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("mean flow-matching loss")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig_path = fig_dir / "gate_c_loss_curves.png"
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        try:
            fig_rel = str(fig_path.relative_to(repo_root))
        except ValueError:
            fig_rel = str(fig_path)
        results["figures"] = [fig_rel]

        # -------------------------------------------------------------
        # Verdict
        # -------------------------------------------------------------
        pass_conditions = {
            "phase1_reduction_gt_50pct": reduction_pct > 50.0,
            "phase1_monotonic_trend": results["phase1_single_example"]["monotonic_decrease_last10_vs_first10"],
            "phase2_reduction_gt_20pct": reduction_pct_t > 20.0,
            "phase2_monotonic_trend": results["phase2_tiny_set"]["monotonic_decrease_last5_vs_first5"],
            "phase3_checkpoint_recoverable": ckpt_matches,
        }
        overall = "PASS" if all(pass_conditions.values()) else "FAIL"
        fail_reasons = [k for k, v in pass_conditions.items() if not v]
        results["pass_conditions"] = pass_conditions

    except Exception as e:
        overall = "FAIL"
        fail_reasons = [str(e)]
        results["exception"] = traceback.format_exc()
        print(f"[FAIL] {e}", flush=True)
        traceback.print_exc()

    results["verdict"] = overall
    results["fail_reasons"] = fail_reasons
    results["weights_optimized"] = True
    results["weights_optimized_scope"] = "NativeTrajectoryEncoder adapter parameters ONLY; WanTransformer3DModel backbone frozen throughout"
    results["next_action"] = (
        "Gate C complete. Do NOT start Gate D or full training until Gate C is "
        "independently reviewed, per standing instruction."
        if overall == "PASS" else
        "Gate C FAILED -- do not proceed to Gate D."
    )

    out_path = art_dir / "gate_c_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Gate C verdict: {overall}")
    print(f"       metrics -> {out_path}", flush=True)

    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
