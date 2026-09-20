"""Measure the actual magnitude of the native action conditioning signal."""
from __future__ import annotations

import argparse
import torch
from core.control import (
    NativeTrajectoryConditionerV8,
    NativeTrajectoryConditionerV10,
    NativeTrajectoryConditionerV11,
    NativeTrajectoryConditionerV7,
)


def build(version: str, dim: int):
    common = dict(hidden_dim=768, output_dim=3072, num_layers=2, num_heads=12)
    if version == "v7":
        return NativeTrajectoryConditionerV7(input_dim=dim, **common)
    if version == "v8":
        return NativeTrajectoryConditionerV8(input_dim=dim, **common)
    if version == "v10":
        return NativeTrajectoryConditionerV10(input_dim=dim, **common)
    if version == "v11":
        return NativeTrajectoryConditionerV11(input_dim=dim, **common)
    raise ValueError(version)


def load(module, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"  loaded {path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"    missing sample: {missing[:3]}")
    if unexpected[:3]:
        print(f"    unexpected sample: {unexpected[:3]}")
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v11")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--source-frames", type=int, default=61)
    args = ap.parse_args()

    torch.manual_seed(0)
    dim = {"v7": 74, "v8": 111, "v10": 148, "v11": 148}[args.version]
    encoder = build(args.version, dim).eval()
    if args.checkpoint:
        encoder = load(encoder, args.checkpoint)

    # Simulate the post-patch-embedding hidden state RMS. Wan patch_embedding on
    # a VAE latent with unit-ish std produces roughly unit-scale tokens.
    target_hidden_rms = 1.0

    action_correct = torch.randn(1, args.source_frames, 37) * 1.0
    action_reversed = action_correct.flip(1).clone()
    action_zero = torch.zeros_like(action_correct)
    perm = torch.randperm(args.source_frames)
    action_permuted = action_correct[:, perm].clone()

    def feats(a):
        layout = {
            "v7": a,
            "v8": torch.cat([a, torch.zeros_like(a), torch.zeros_like(a)], -1),
            "v10": torch.cat([a, a, torch.zeros_like(a), torch.zeros_like(a)], -1),
            "v11": torch.cat([a, a, torch.zeros_like(a), torch.zeros_like(a)], -1),
        }[args.version]
        with torch.no_grad():
            out = encoder(layout, args.frames)
        return out

    ref = feats(action_correct)
    print(f"\n=== {args.version}  frames={args.frames} source={args.source_frames} ===")
    if isinstance(ref, tuple):
        residual, modulation = ref
        print(f"residual shape {tuple(residual.shape)}  modulation shape {tuple(modulation.shape)}")
        print(f"residual RMS   = {residual.float().square().mean().sqrt().item():.6e}")
        print(f"residual absmax= {residual.abs().max().item():.6e}")
        print(f"modulation RMS = {modulation.float().square().mean().sqrt().item():.6e}")
        print(f"modulation absmax = {modulation.abs().max().item():.6e}")
        print(f"\nratio residual_RMS / hidden_RMS({target_hidden_rms}) = "
              f"{residual.float().square().mean().sqrt().item() / target_hidden_rms:.6e}")
        for name, a in (
            ("zero", action_zero),
            ("reversed", action_reversed),
            ("permuted", action_permuted),
        ):
            other = feats(a)
            dres = (other[0] - residual).abs().mean().item()
            dmod = (other[1] - modulation).abs().mean().item()
            rel = dres / (residual.abs().mean().item() + 1e-12)
            print(f"  {name:9s} |Δresidual|={dres:.6e}  |Δmod|={dmod:.6e}  "
                  f"|Δres|/|res|={rel:.4f}")
    else:
        print(f"features shape {tuple(ref.shape)}")
        print(f"features RMS = {ref.float().square().mean().sqrt().item():.6e}")
        for name, a in (
            ("zero", action_zero),
            ("reversed", action_reversed),
            ("permuted", action_permuted),
        ):
            other = feats(a)
            d = (other - ref).abs().mean().item()
            print(f"  {name:9s} |Δ|={d:.6e}  |Δ|/|f|={d / (ref.abs().mean().item() + 1e-12):.4f}")


if __name__ == "__main__":
    main()
