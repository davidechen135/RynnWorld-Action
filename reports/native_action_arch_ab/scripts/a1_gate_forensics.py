#!/usr/bin/env python3
"""A1: confirm the real computation of the spatial gate, on the V9 checkpoint.

The gate (``spatialize_residual``) exists only in the V9 lineage: V10/V11
subclass V8, so their state dicts carry no gate parameters. A1 therefore runs
on ``native_action_v9_spatial_gate_single_clip_fixed_300``, which is the only
gate-bearing checkpoint in the repo.

Measured on the 16 real windows:
  - real shapes through the whole chain
  - gate std along T, and mean |gate[t+1] - gate[t]|
  - gate variation across H,W within a frame
  - gate under correct / reversed / held / swapped / zero
  - whether reversed is merely the time-reversal of the correct gate
  - whether the query sees only the first visual frame
  - whether the gate depends on a per-frame action key

Writes metrics/a1_gate_forensics.json. Does not modify model code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "reports/native_action_arch_ab"
DATA = Path(
    "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/"
    "agibot_action_v10_state_spatial_16_v1"
)
HELPERS = REPO / "reports/direct_action/gate_c/run_003/artifacts/gate_c_visual_v2.py"
BASE = REPO / "pretrained/Wan2.2-TI2V-5B-Diffusers"
RYNN = REPO / "pretrained/RynnWorld-Teleop-Causal"
TRAIN = REPO / "training/native_action_v9_spatial_gate_single_clip_fixed_300"
CKPT_STEP = 300
SEED = 42
DTYPE = torch.bfloat16


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shapes_of(t: torch.Tensor) -> str:
    return "x".join(str(d) for d in t.shape)


def main() -> None:
    from safetensors.torch import load_file
    from core.control.native_action_features import build_v8_features
    from core.control.native_trajectory_encoder import NativeTrajectoryConditionerV9

    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    helpers = load_module(HELPERS, "gate_c_visual_v2")
    helpers.apply_monkey_patch(REPO)
    model = helpers.load_model(RYNN, BASE, sys.stdout)
    patch = model.patch_embedding
    print(f"[A1] patch_embedding kernel={tuple(patch.weight.shape)} "
          f"in={patch.in_channels} out={patch.out_channels}")

    encoder = NativeTrajectoryConditionerV9(input_dim=111).to(device=device, dtype=DTYPE)
    state_dict = torch.load(TRAIN / f"checkpoint-{CKPT_STEP}" / "native_trajectory_encoder.bin",
                            map_location="cpu", weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    akw = encoder.action_key_projection.weight.float()
    print(f"[A1] encoder=NativeTrajectoryConditionerV9 step={CKPT_STEP}")
    print(f"[A1] spatial_gate_dim={encoder.spatial_gate_dim} "
          f"action_key_projection rms={akw.pow(2).mean().sqrt().item():.6g} "
          f"absmax={akw.abs().max().item():.6g}")
    print(f"[A1] visual_query_projection rms="
          f"{encoder.visual_query_projection.weight.float().pow(2).mean().sqrt().item():.6g}")
    print(f"[A1] spatial_gate_bias={encoder.spatial_gate_bias.item():.6g}")
    print(f"[A1] base_residual={shapes_of(encoder.base_residual)} "
          f"rms={encoder.base_residual.float().pow(2).mean().sqrt().item():.6g}")

    files = sorted(DATA.glob("task*.safetensors"))
    print(f"[A1] windows={len(files)}")

    rows = []
    shape_report: dict = {}
    for path in files:
        packed = load_file(str(path))
        raw = packed["robot_trajectory_raw37"].unsqueeze(0).float().to(device)
        mean = packed["robot_trajectory_mean37"].float().to(device)
        std = packed["robot_trajectory_std37"].float().to(device)
        rel_s = packed["robot_relative_scale37"].float().to(device)
        vel_s = packed["robot_velocity_scale37"].float().to(device)
        state = packed["robot_observed_state37"].unsqueeze(0).float().to(device)

        def feat(src):
            return build_v8_features(src, mean, std, rel_s, vel_s)

        correct = feat(raw)
        swapped = correct.clone()
        swapped[:, [0, 1]] = correct[:, [1, 0]]

        img = packed["img_latent"].unsqueeze(0).to(device=device, dtype=DTYPE)
        latent = packed["video_latents"].unsqueeze(0).to(device=device, dtype=DTYPE)
        shape = latent.shape
        T = shape[2] // model.config.patch_size[0]
        noise = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(SEED),
                            device=device, dtype=DTYPE)
        mask = torch.ones((shape[0], 1, shape[2], shape[3], shape[4]),
                          device=device, dtype=DTYPE)
        mask[:, :, 0] = 0
        model_input = (1 - mask) * img + mask * noise

        with torch.no_grad():
            visual = patch(model_input)
            if not shape_report:
                centered0 = encoder.encode_centered(correct, T)
                proj0 = encoder.input_residual_projection(centered0)
                key0 = encoder.action_key_projection(
                    centered0.to(encoder.action_key_projection.weight.dtype))
                q0 = encoder.visual_query_projection(visual[:, :, :1]).squeeze(2)
                s0 = torch.einsum("bchw,btc->bthw", q0, key0)
                shape_report = {
                    "latent_video": shapes_of(latent),
                    "img_latent": shapes_of(img),
                    "patch_input": shapes_of(model_input),
                    "patch_kernel": list(patch.weight.shape),
                    "model_patch_size": list(model.config.patch_size),
                    "post_patch_visual_tokens": shapes_of(visual),
                    "post_patch_num_frames": T,
                    "post_patch_grid": [visual.shape[3], visual.shape[4]],
                    "encode_centered_out": shapes_of(centered0),
                    "action_key_projection_out": shapes_of(key0),
                    "visual_query_projection_out": shapes_of(q0),
                    "score_einsum_out": shapes_of(s0),
                    "gate_out": shapes_of(2.0 * torch.sigmoid(s0)),
                    "last_spatial_gate": shapes_of((2.0 * torch.sigmoid(s0)).unsqueeze(1)),
                    "input_residual_projection_before_transpose": shapes_of(proj0),
                    "action_residual": shapes_of(
                        proj0.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)),
                    "gate_dim_source": "visual_query_projection Conv3d output channels",
                }

            # residuals (gate does not depend on the residual argument)
            full = encoder(correct, T)
            residual = full[0] if isinstance(full, tuple) else full
            centered = encoder.encode_centered(correct, T)
            act = encoder.input_residual_projection(centered).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
            base = encoder.base_residual.to(act.dtype)

            gates = {}
            for name, src in (("correct", correct), ("reversed", feat(raw.flip(1))),
                              ("held", feat(raw[:, :1].expand_as(raw))),
                              ("swapped", swapped), ("zero", torch.zeros_like(correct))):
                encoder.spatialize_residual(src, T, residual, visual)
                gates[name] = encoder.last_spatial_gate.float().clone()

            visual_perturbed = visual.clone()
            visual_perturbed[:, :, 1:] = torch.randn_like(visual_perturbed[:, :, 1:])
            encoder.spatialize_residual(correct, T, residual, visual_perturbed)
            gate_query_perturbed = encoder.last_spatial_gate.float().clone()

            # action-key test: hold the key at its t=0 value, re-score
            key_real = encoder.action_key_projection(
                centered.to(encoder.action_key_projection.weight.dtype))
            key_frozen = encoder.action_key_projection(
                centered[:, :1].expand_as(centered).to(
                    encoder.action_key_projection.weight.dtype))
            q = encoder.visual_query_projection(visual[:, :, :1]).squeeze(2).float()
            score_real = torch.einsum("bchw,btc->bthw", q, key_real.float())
            score_frozen = torch.einsum("bchw,btc->bthw", q, key_frozen.float())

        gc = gates["correct"]                       # [1,1,T,H,W]
        row = {
            "window": path.stem,
            "gate_T": int(gc.shape[2]),
            "gate_grid": [int(gc.shape[3]), int(gc.shape[4])],
            "gate_mean": gc.mean().item(),
            "gate_std_T": gc.std(dim=2).mean().item(),
            "gate_mean_adjacent_absdiff": (gc[:, :, 1:] - gc[:, :, :-1]).abs().mean().item(),
            "gate_T_range": (gc.amax(dim=2) - gc.amin(dim=2)).mean().item(),
            "gate_std_over_HW": gc.std(dim=(3, 4)).mean().item(),
            "gate_HW_range": (gc.amax(dim=(3, 4)) - gc.amin(dim=(3, 4))).mean().item(),
            "gate_per_frame_mean": gc.squeeze(0).squeeze(0).mean(dim=(1, 2)).tolist(),
            "gate_per_frame_stdHWC": gc.squeeze(0).squeeze(0).std(dim=(1, 2)).tolist(),
            "diff_abs_vs_correct": {n: (gates[n] - gc).abs().mean().item() for n in gates},
            "reversed_vs_flip_of_correct": (gates["reversed"] - gc.flip(2)).abs().mean().item(),
            "reversed_vs_correct": (gates["reversed"] - gc).abs().mean().item(),
            "query_firstframe_only_maxdiff": (gate_query_perturbed - gc).abs().max().item(),
            "query_firstframe_only_meandiff": (gate_query_perturbed - gc).abs().mean().item(),
            "score_stdT_real_key": score_real.std(dim=2).mean().item(),
            "score_stdT_frozen_key": score_frozen.std(dim=2).mean().item(),
            "action_key_effect_on_score": (score_real - score_frozen).abs().mean().item(),
            "action_residual_rms": act.float().pow(2).mean().sqrt().item(),
            "base_residual_rms": base.float().pow(2).mean().sqrt().item(),
        }
        rows.append(row)
        print(json.dumps({
            "window": path.stem,
            "std_T": round(row["gate_std_T"], 8),
            "adj": round(row["gate_mean_adjacent_absdiff"], 8),
            "std_HW": round(row["gate_std_over_HW"], 8),
            "T_range": round(row["gate_T_range"], 8),
            "rev_vs_flip": round(row["reversed_vs_flip_of_correct"], 8),
            "q_perturb_max": round(row["query_firstframe_only_maxdiff"], 8),
            "key_effect": round(row["action_key_effect_on_score"], 8),
        }, ensure_ascii=False), flush=True)

    payload = {
        "checkpoint": str(TRAIN / f"checkpoint-{CKPT_STEP}"),
        "encoder": "NativeTrajectoryConditionerV9",
        "shape_report": shape_report,
        "windows": rows,
    }
    (OUT / "metrics" / "a1_gate_forensics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"shape_report": shape_report}, indent=2))
    print(f"[A1] wrote {OUT / 'metrics' / 'a1_gate_forensics.json'}")


if __name__ == "__main__":
    main()
