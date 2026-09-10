"""
Gate B zero-training structural/inference smoke test.

Controls:
  C0 = no control (null condition, baseline)
  C1 = UNAVAILABLE (no trained oracle; zero-init encoder would be byte-identical to C0)
  C2 = UNAVAILABLE (no genuine causal history-only implementation exists in repo)
  C3 = D-action-core 33D raw columns, fed through a freshly constructed
       zero-init NativeTrajectoryEncoder(input_dim=33). Since output_projection
       is zero-init, C3 output is byte-identical to C0 at initialization --
       this tests plumbing only, not semantic conditioning. Sensitivity is
       checked by perturbing the output_projection away from zero.

What this validates:
  1. Data loader: shape, dtype, column order, timestamp monotonicity,
     finite values, episode-level split isolation (no held-out leaks into
     the evaluated sample), deterministic same-seed behavior.
  2. Pretrained checkpoint forward pass: C0 runs without crash/NaN on a
     minimal latent shape using RynnWorld-Teleop-Causal (12 GB, smallest
     available).
  3. C3 plumbing: NativeTrajectoryEncoder(input_dim=33) forward pass
     produces the correct output shape, and at zero-init its residual
     contribution to hidden_states is all-zeros (C3 == C0 exactly).
  4. Sensitivity: after a controlled unit-perturbation of output_projection,
     C3 diverges from C0 (confirms the adapter is actually wired into the
     forward path and the perturbation reaches the output).

What this does NOT claim:
  - Semantic rollout quality (weights are random-init for the adapter).
  - Correct action conditioning (that requires Gate C / trained weights).
  - C1 or C2 validity (both marked UNAVAILABLE, see below).

Usage:
  cd /mnt/workspace/RynnWorld-Teleop
  source /tmp/scratch/gate_b_env/bin/activate
  python reports/direct_action/gate_b/run_001/artifacts/gate_b_smoke.py \
    --out_dir reports/direct_action/gate_b/run_001 \
    --parquet_dir /tmp/scratch/gate_b_task3400/data/data/chunk-000 \
    --checkpoint_dir pretrained/RynnWorld-Teleop-Causal \
    --base_model_dir pretrained/Wan2.2-TI2V-5B-Diffusers \
    --split_manifest reports/direct_action/gate_a/run_002/artifacts/episode_split.json \
    --seed 42
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# D-action-core 33D column-order lock (must match action_core_schema.py)
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


def slice_core_33d(raw: np.ndarray) -> np.ndarray:
    assert raw.shape[-1] == RAW_ACTION_DIM
    parts = [raw[..., sl] for _, sl, _ in CORE_SLICES]
    return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------

def validate_loader(parquet_dir: Path, split_manifest: dict, seed: int, out: dict):
    """Load one training episode (by split_manifest) and validate shape/dtype/
    timestamp/finite/order/no-held-out-leakage/determinism."""
    print("[LOADER] validating task_3400 parquet loader ...", flush=True)
    train_eps = sorted(split_manifest["train_episodes"])
    held_eps = set(split_manifest["held_out_episodes"])

    # Assert split is internally disjoint and covers all 110 episodes
    assert set(train_eps).isdisjoint(held_eps), "FAIL: train/held_out overlap"
    assert len(train_eps) + len(held_eps) == 110, "FAIL: split does not cover 110 episodes"
    assert all(ep not in held_eps for ep in train_eps), "FAIL: held-out episode found in train set"

    # Select two train episodes deterministically for same-seed check
    rng = np.random.default_rng(seed)
    ep_a = int(rng.choice(train_eps))
    ep_b = int(rng.choice([e for e in train_eps if e != ep_a]))

    results = {}
    for ep in [ep_a, ep_b]:
        parquet_path = parquet_dir / f"episode_{ep:06d}.parquet"
        assert parquet_path.exists(), f"FAIL: {parquet_path} not found"
        assert ep not in held_eps, f"FAIL: selected episode {ep} is held_out"

        import pandas as pd
        df = pd.read_parquet(str(parquet_path))

        # Schema checks
        for col in ["action", "observation.state", "timestamp", "frame_index", "episode_index"]:
            assert col in df.columns, f"FAIL: missing column {col}"

        # episode_index consistency
        assert (df["episode_index"] == ep).all(), \
            f"FAIL: episode_index mismatch in parquet (ep={ep})"

        # dtype
        ts = df["timestamp"].values
        assert ts.dtype == np.float32 or np.issubdtype(ts.dtype, np.floating), \
            f"FAIL: timestamp dtype {ts.dtype}"

        # shape and dtype of action / state
        a0 = np.array(df["action"].iloc[0])
        s0 = np.array(df["observation.state"].iloc[0])
        assert a0.shape == (40,) and a0.dtype == np.float32, \
            f"FAIL: action shape/dtype {a0.shape} {a0.dtype}"
        assert s0.shape == (169,) and s0.dtype == np.float32, \
            f"FAIL: state shape/dtype {s0.shape} {s0.dtype}"

        # stack all rows
        action_all = np.stack(df["action"].values)  # (T, 40)
        state_all  = np.stack(df["observation.state"].values)  # (T, 169)
        assert action_all.shape == (len(df), 40)
        assert state_all.shape  == (len(df), 169)

        # finite values
        assert np.isfinite(action_all).all(), f"FAIL: NaN/Inf in action (ep={ep})"
        assert np.isfinite(state_all).all(),  f"FAIL: NaN/Inf in state  (ep={ep})"

        # D-action-core slice
        core = slice_core_33d(action_all)
        assert core.shape == (len(df), CORE_DIM), \
            f"FAIL: core slice shape {core.shape}"
        assert np.isfinite(core).all(), "FAIL: NaN/Inf in D-action-core slice"

        # Timestamp: strictly increasing and dt ≈ 1/30
        dt = np.diff(ts)
        assert (dt > 0).all(), f"FAIL: timestamps not strictly increasing (ep={ep})"
        assert np.allclose(dt, 1.0/30.0, atol=1e-3), \
            f"FAIL: unexpected timestamp dt (mean={dt.mean():.5f}, ep={ep})"

        # frame_index: 0,1,2,...,T-1
        fi = df["frame_index"].values
        assert list(fi) == list(range(len(df))), \
            f"FAIL: frame_index not contiguous from 0 (ep={ep})"

        # Confirm this episode is NOT a held-out episode
        assert ep not in held_eps, f"FAIL: evaluated ep={ep} is in held_out"

        results[ep] = {
            "n_frames": int(len(df)),
            "action_shape": list(action_all.shape),
            "state_shape":  list(state_all.shape),
            "core_shape":   list(core.shape),
            "ts_dt_mean":   float(dt.mean()),
            "ts_dt_std":    float(dt.std()),
            "action_finite": bool(np.isfinite(action_all).all()),
            "state_finite":  bool(np.isfinite(state_all).all()),
            "core_finite":   bool(np.isfinite(core).all()),
            "in_train_split": ep in train_eps,
            "in_held_out": ep in held_eps,
        }
        print(f"  ep={ep:03d}: {len(df)} frames, action OK, state OK, core OK, "
              f"dt={dt.mean():.5f}±{dt.std():.2e}", flush=True)

    # Determinism: reload ep_a with same seed, confirm identical arrays
    rng2 = np.random.default_rng(seed)
    ep_a2 = int(rng2.choice(train_eps))
    assert ep_a2 == ep_a, "FAIL: non-deterministic episode selection under same seed"

    import pandas as pd
    df_a1 = pd.read_parquet(str(parquet_dir / f"episode_{ep_a:06d}.parquet"))
    df_a2 = pd.read_parquet(str(parquet_dir / f"episode_{ep_a:06d}.parquet"))
    arr1 = np.stack(df_a1["action"].values)
    arr2 = np.stack(df_a2["action"].values)
    assert np.array_equal(arr1, arr2), "FAIL: parquet reads are not deterministic"
    print(f"  Determinism: ep_a={ep_a} reads identically across two loads. PASS", flush=True)

    out["loader"] = {
        "status": "PASS",
        "evaluated_episodes": [ep_a, ep_b],
        "per_episode": results,
        "split_disjoint": True,
        "split_covers_110": True,
        "determinism": True,
        "seed": seed,
    }
    print("[LOADER] PASS\n", flush=True)
    return ep_a, df_a1  # return one episode for inference


# ---------------------------------------------------------------------------
# Checkpoint / model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: Path, base_model_dir: Path, out: dict):
    """Load WanTransformer3DModel from pretrained/RynnWorld-Teleop-Causal and
    apply the wan_forward monkey-patch from rynnworld_teleop_trainer.py."""
    print("[MODEL] loading RynnWorld-Teleop-Causal checkpoint ...", flush=True)
    t0 = time.time()

    from safetensors.torch import load_file
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    # Load base model config from the Wan2.2-TI2V-5B-Diffusers transformer
    model = WanTransformer3DModel.from_pretrained(
        str(base_model_dir / "transformer"),
        torch_dtype=torch.bfloat16,
    )

    # Load RynnWorld-Teleop-Causal EMA weights (full fine-tuned model)
    ema_path = checkpoint_dir / "ema_weights.bin"
    assert ema_path.exists(), f"FAIL: {ema_path} not found"
    ema_weights = torch.load(str(ema_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ema_weights, strict=False)
    del ema_weights

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {n_params/1e9:.2f}B params in {elapsed:.1f}s on {device}", flush=True)

    out["model_load"] = {
        "checkpoint": str(checkpoint_dir),
        "base_model": str(base_model_dir),
        "device": str(device),
        "dtype": "bfloat16",
        "n_params_B": round(n_params / 1e9, 3),
        "load_time_s": round(elapsed, 2),
        "status": "PASS",
    }
    return model, device


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

def build_adapter_33d(device):
    """Construct a fresh zero-init NativeTrajectoryEncoder(input_dim=33)."""
    sys.path.insert(0, str(Path(__file__).parents[5]))  # repo root
    from core.control.native_trajectory_encoder import NativeTrajectoryEncoder
    enc = NativeTrajectoryEncoder(input_dim=33)
    enc = enc.to(device).to(torch.bfloat16)
    enc.eval()

    # Confirm output_projection is all-zeros (zero-init adapter invariant)
    w_norm = enc.output_projection.weight.norm().item()
    b_norm = enc.output_projection.bias.norm().item()
    assert w_norm == 0.0 and b_norm == 0.0, \
        f"FAIL: output_projection not zero-init (w_norm={w_norm}, b_norm={b_norm})"
    return enc


# ---------------------------------------------------------------------------
# Minimal latent construction
# ---------------------------------------------------------------------------

def make_minimal_inputs(model, device):
    """Build the smallest legal inputs for a single wan_forward call.

    Config: in_channels=48, patch_size=[1,2,2], text_dim=4096.
    Minimal spatial: height=p_h*1=2, width=p_w*1=2.
    Minimal temporal: num_frames=p_t*1=1 (patch_size[0]=1).
    """
    p_t, p_h, p_w = model.config.patch_size  # [1, 2, 2]
    in_ch = model.config.in_channels          # 48
    text_dim = model.config.text_dim          # 4096
    B = 1

    # Latent: [B, in_ch, num_frames, H, W]
    num_frames = p_t  # 1 temporal frame
    H = p_h * 2       # minimal spatial: 2 patches high
    W = p_w * 2       # minimal spatial: 2 patches wide
    hidden_states = torch.randn(B, in_ch, num_frames, H, W,
                                device=device, dtype=torch.bfloat16)

    # Timestep: integer in [0, 1000)
    # Wan2.2-TI2V uses 2D timestep [B, seq_len] where seq_len = num_frames
    post_frames = num_frames // p_t  # 1
    timestep = torch.randint(0, 1000, (B, post_frames), device=device)

    # Text encoder hidden states: [B, seq_len_text, text_dim]
    encoder_hidden_states = torch.randn(B, 16, text_dim,
                                        device=device, dtype=torch.bfloat16)

    # First-frame image condition (encoder_hidden_states_image): [B, 1, inner_dim]
    # For minimal test pass None (handled by wan_forward when has_control=False)
    encoder_hidden_states_image = None

    return hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

def run_forward(model, hidden_states, timestep, enc_hs, enc_hs_img,
                robot_trajectory=None, null_condition=False, label=""):
    """Run wan_forward (already monkey-patched onto model) and return output."""
    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=enc_hs,
            encoder_hidden_states_image=enc_hs_img,
            robot_trajectory=robot_trajectory,
            null_condition=null_condition,
            return_dict=False,
        )
    sample = out[0]
    assert torch.isfinite(sample).all(), f"FAIL: NaN/Inf in {label} output"
    return sample


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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    results = {
        "run_id": "direct_action/gate_b/run_001",
        "date": "2026-08-20",
        "seed": args.seed,
        "gate": "B",
        "controls_defined": {
            "C0": "no control (null_condition=True), baseline",
            "C1": "UNAVAILABLE -- no trained oracle; zero-init adapter gives byte-identical output to C0",
            "C2": "UNAVAILABLE -- no genuine causal history-only implementation in repo",
            "C3": "D-action-core 33D raw columns through zero-init NativeTrajectoryEncoder(input_dim=33); plumbing + sensitivity only",
        },
        "d_action_core_channel_order": [
            {"field": name, "raw_indices": f"{sl.start}:{sl.stop}", "dim": dim,
             "output_position": f"{sum(d for _, _, d in CORE_SLICES[:i])}:{sum(d for _, _, d in CORE_SLICES[:i+1])}"}
            for i, (name, sl, dim) in enumerate(CORE_SLICES)
        ],
        "channel_order_note": (
            "Raw index order as stored in 40D parquet action column "
            "(end/position, end/orientation, joint/position, waist/position). "
            "NOT joint-first. No reorder applied."
        ),
    }

    try:
        # ------------------------------------------------------------------
        # 1. Load split manifest
        # ------------------------------------------------------------------
        with open(args.split_manifest) as f:
            split_manifest = json.load(f)
        assert split_manifest["shard"] == "task_3400/313498_314085.tar.gz"
        assert split_manifest["total_episodes"] == 110

        # ------------------------------------------------------------------
        # 2. Loader validation
        # ------------------------------------------------------------------
        parquet_dir = Path(args.parquet_dir)
        ep_a, df_a = validate_loader(parquet_dir, split_manifest, args.seed, results)

        # Extract a short D-action-core window for C3 from ep_a (train-split only)
        action_all = np.stack(df_a["action"].values)   # (T, 40)
        core_all   = slice_core_33d(action_all)         # (T, 33)
        # Use first 21 frames (≥ one temporal patch for a minimal window)
        window = torch.from_numpy(core_all[:21]).unsqueeze(0).to(torch.bfloat16)  # (1,21,33)
        results["loader"]["c3_window_shape"] = list(window.shape)
        results["loader"]["c3_episode_used"] = ep_a

        # Assert no held-out episode ID was used
        held_eps = set(split_manifest["held_out_episodes"])
        assert ep_a not in held_eps, "FAIL: C3 data window drawn from held-out episode"

        # ------------------------------------------------------------------
        # 3. Apply wan_forward monkey-patch
        # ------------------------------------------------------------------
        print("[PATCH] applying wan_forward monkey-patch ...", flush=True)
        import sys
        sys.path.insert(0, str(Path(__file__).parents[5]))
        # Importing rynnworld_teleop_trainer applies the monkeypatch as a side-effect.
        # This import must succeed -- a silently-failed patch means the C0/C3 forward
        # passes below would run the UNPATCHED base model, which would corrupt every
        # downstream result without raising. Do not swallow this exception.
        import io, contextlib
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel as _WTM
        _forward_before = _WTM.forward
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            import core.finetune.models.wan_i2v.rynnworld_teleop_trainer as _trainer
        patch_log = capture.getvalue()
        print(f"  [PATCH] {patch_log.strip()}", flush=True)
        patch_applied = _WTM.forward is not _forward_before
        if not patch_applied:
            raise RuntimeError(
                "FAIL: wan_forward monkey-patch did not change WanTransformer3DModel.forward "
                "after importing rynnworld_teleop_trainer -- forward passes below would run "
                "the unpatched base model, invalidating C0/C3 comparison."
            )
        print(f"  [PATCH] confirmed WanTransformer3DModel.forward is patched", flush=True)

        # ------------------------------------------------------------------
        # 4. Load model
        # ------------------------------------------------------------------
        model, device = load_model(
            Path(args.checkpoint_dir), Path(args.base_model_dir), results
        )

        # ------------------------------------------------------------------
        # 5. Build zero-init 33D adapter and attach to model
        # ------------------------------------------------------------------
        enc = build_adapter_33d(device)
        model.native_trajectory_encoder = enc
        results["adapter"] = {
            "class": "NativeTrajectoryEncoder",
            "input_dim": 33,
            "init": "zero (output_projection weight/bias all-zeros)",
            "attached_to_model": True,
            "output_projection_norm_at_init": 0.0,
        }

        # ------------------------------------------------------------------
        # 6. Build minimal inputs (same for all forward calls, fix seed)
        # ------------------------------------------------------------------
        torch.manual_seed(args.seed)
        hs, ts, enc_hs, enc_hs_img = make_minimal_inputs(model, device)
        window_dev = window.to(device)

        # ------------------------------------------------------------------
        # 7. C0 forward pass (null_condition = no control)
        # ------------------------------------------------------------------
        print("[FWD C0] running C0 (no control) forward ...", flush=True)
        t0 = time.time()
        out_c0 = run_forward(model, hs, ts, enc_hs, enc_hs_img,
                              robot_trajectory=None, null_condition=True, label="C0")
        t_c0 = time.time() - t0
        print(f"  C0: shape={tuple(out_c0.shape)}, "
              f"finite={torch.isfinite(out_c0).all().item()}, "
              f"time={t_c0:.2f}s", flush=True)

        # ------------------------------------------------------------------
        # 8. C3 forward pass at zero-init (output must equal C0 exactly)
        # ------------------------------------------------------------------
        print("[FWD C3-zero] running C3 zero-init (must == C0) ...", flush=True)
        out_c3_zero = run_forward(model, hs, ts, enc_hs, enc_hs_img,
                                  robot_trajectory=window_dev, null_condition=False,
                                  label="C3-zero")
        c3_zero_matches_c0 = torch.allclose(out_c3_zero, out_c0, atol=0.0)
        print(f"  C3-zero == C0: {c3_zero_matches_c0} "
              f"(max_abs_diff={( out_c3_zero - out_c0).abs().max().item():.2e})", flush=True)
        if not c3_zero_matches_c0:
            results["c3_zero_match_c0"] = False
            results["c3_zero_max_diff"] = float((out_c3_zero - out_c0).abs().max().item())
            raise AssertionError(
                f"FAIL: C3 zero-init output differs from C0 by "
                f"{(out_c3_zero - out_c0).abs().max().item():.2e}. "
                f"Zero-init adapter should contribute exactly 0 to hidden_states."
            )

        # ------------------------------------------------------------------
        # 9. C3 sensitivity: perturb output_projection, confirm C3 != C0
        # ------------------------------------------------------------------
        print("[FWD C3-perturb] perturbing output_projection, expect C3 != C0 ...", flush=True)
        torch.manual_seed(args.seed + 1)
        with torch.no_grad():
            enc.output_projection.weight.fill_(0.01)
            enc.output_projection.bias.zero_()

        out_c3_perturb = run_forward(model, hs, ts, enc_hs, enc_hs_img,
                                     robot_trajectory=window_dev, null_condition=False,
                                     label="C3-perturb")
        c3_perturb_differs = not torch.allclose(out_c3_perturb, out_c0, atol=1e-4)
        max_diff_perturb = (out_c3_perturb - out_c0).abs().max().item()
        print(f"  C3-perturb != C0: {c3_perturb_differs} "
              f"(max_abs_diff={max_diff_perturb:.4e})", flush=True)
        if not c3_perturb_differs:
            raise AssertionError(
                f"FAIL: C3 after perturbation is still identical to C0 "
                f"(max_diff={max_diff_perturb:.2e}). "
                "The adapter is not wired into the forward path."
            )

        # Restore zero-init
        with torch.no_grad():
            enc.output_projection.weight.zero_()
            enc.output_projection.bias.zero_()

        # ------------------------------------------------------------------
        # 10. Same-seed determinism: run C0 twice, confirm identical
        # ------------------------------------------------------------------
        print("[DETERMINISM] running C0 twice with same seed ...", flush=True)
        torch.manual_seed(args.seed)
        hs2, ts2, enc_hs2, enc_hs_img2 = make_minimal_inputs(model, device)
        out_c0_b = run_forward(model, hs2, ts2, enc_hs2, enc_hs_img2,
                               robot_trajectory=None, null_condition=True,
                               label="C0-repeat")
        deterministic = torch.allclose(out_c0, out_c0_b, atol=0.0)
        print(f"  Deterministic (same seed): {deterministic}", flush=True)

        # ------------------------------------------------------------------
        # 11. Collect metrics
        # ------------------------------------------------------------------
        results["forward_passes"] = {
            "C0": {
                "status": "PASS",
                "output_shape": list(out_c0.shape),
                "output_finite": bool(torch.isfinite(out_c0).all().item()),
                "output_norm": float(out_c0.float().norm().item()),
                "time_s": round(t_c0, 3),
            },
            "C1": {
                "status": "UNAVAILABLE",
                "reason": (
                    "No trained oracle checkpoint exists. Zero-init NativeTrajectoryEncoder "
                    "would produce byte-identical output to C0 regardless of input data. "
                    "A meaningful leakage upper-bound measurement requires trained weights."
                ),
            },
            "C2": {
                "status": "UNAVAILABLE",
                "reason": (
                    "No genuine causal history-only conditioning pathway implemented in repo. "
                    "The only action/trajectory encoder (NativeTrajectoryEncoder) uses the "
                    "full window (not causal/history-only). Per Gate B rules, an unavailable "
                    "required control marks C2 as unavailable, not as a stub PASS."
                ),
            },
            "C3": {
                "status": "PASS",
                "output_shape": list(out_c3_zero.shape),
                "output_finite": bool(torch.isfinite(out_c3_zero).all().item()),
                "zero_init_matches_c0_exactly": c3_zero_matches_c0,
                "sensitivity_perturbed_diverges": c3_perturb_differs,
                "sensitivity_max_diff_after_perturbation": float(max_diff_perturb),
                "adapter_input_dim": 33,
                "adapter_init": "zero",
                "data_window_shape": list(window.shape),
                "data_window_episode": ep_a,
                "data_window_is_train_only": ep_a in set(split_manifest["train_episodes"]),
                "note": (
                    "Plumbing-only test: zero-init adapter exactly reproduces C0. "
                    "Sensitivity confirmed: non-zero output_projection perturbs output. "
                    "No semantic claim -- adapter weights are untrained."
                ),
            },
        }
        results["determinism"] = {
            "same_seed_c0_identical": bool(deterministic),
        }

        overall = "PASS"
        fail_reasons = []

    except Exception as e:
        overall = "FAIL"
        fail_reasons = [str(e)]
        results["exception"] = traceback.format_exc()
        print(f"[FAIL] {e}", flush=True)
        traceback.print_exc()

    results["verdict"] = overall
    results["fail_reasons"] = fail_reasons

    # Write machine-readable metrics
    out_path = Path(args.out_dir) / "artifacts" / "gate_b_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Gate B verdict: {overall}")
    print(f"       metrics -> {out_path}", flush=True)

    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
