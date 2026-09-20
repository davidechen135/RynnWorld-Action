"""Verify the four structural claims against the shipped V10b/V11 checkpoints."""
import sys, torch
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV10, NativeTrajectoryConditionerV11

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"

def load(cls, path):
    m = cls(input_dim=148)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"  loaded {path}\n    missing={len(miss)} unexpected={len(unexp)}")
    return m, sd

def spectrum(name, w):
    s = torch.linalg.svdvals(w.float())
    e = s.square() / s.square().sum()
    print(f"  {name:34s} shape={tuple(w.shape)} top1={e[0]:.4f} top10={e[:10].sum():.4f} "
          f"rank99={(e.cumsum(0) < 0.99).sum().item()+1}")

for tag, cls, path in [
    ("V10b", NativeTrajectoryConditionerV10, f"{REPO}/training/native_action_v10b_state_256_600/checkpoint-400"),
    ("V11",  NativeTrajectoryConditionerV11, f"{REPO}/training/native_action_v11_temporal_alignment_300/checkpoint-150"),
]:
    print(f"\n{'='*70}\n{tag}  {path}\n{'='*70}")
    m, sd = load(cls, f"{path}/native_trajectory_encoder.bin")
    m = m.eval()

    print("\n-- spectra --")
    spectrum("input_residual_projection", m.input_residual_projection.weight)
    spectrum("adaln_projection", m.adaln_projection.weight)

    print("\n-- frozen base vs action magnitude --")
    print(f"  base_residual   rms={m.base_residual.float().square().mean().sqrt().item():.6f} "
          f"absmax={m.base_residual.abs().max().item():.6f} "
          f"requires_grad={m.base_residual.requires_grad}")
    print(f"  base_modulation rms={m.base_modulation.float().square().mean().sqrt().item():.6f} "
          f"absmax={m.base_modulation.abs().max().item():.6f}")
    print(f"  spatial_stem[-1] |w|max={m.spatial_stem[-1].weight.abs().max().item():.6e} "
          f"|w|mean={m.spatial_stem[-1].weight.abs().mean().item():.6e}")

    print("\n-- live residual vs a zero action --")
    torch.manual_seed(0)
    # realistic-ish feature magnitudes: state/target/rel/vel stacked 4x37
    a = torch.randn(1, 33, 148) * 0.5
    z = torch.zeros_like(a)
    with torch.no_grad():
        r_correct, mod_correct = m(a, 9)
        r_zero, mod_zero = m(z, 9)
    def stats(t, label):
        t = t.float()
        print(f"  {label:22s} shape={tuple(t.shape)} rms={t.square().mean().sqrt().item():.6e} "
              f"mean={t.mean().item():.6e}")
    stats(r_correct, "residual(correct)")
    stats(r_zero, "residual(zero)")
    stats(mod_correct, "modulation(correct)")
    stats(mod_zero, "modulation(zero)")

    # temporal structure of the residual: variance across time index / total energy
    r = r_correct.float()
    # residual shape (B, C, T, 1, 1) -> time is dim 2
    tmean = r.mean(dim=2, keepdim=True)
    tvar = (r - tmean).square().mean()
    print(f"\n  residual temporal variance fraction = "
          f"{tvar.item() / r.square().mean().item():.6e}")

    print("\n-- dead-module check (present but bypassed) --")
    for n in ("position_embedding", "temporal_transformer"):
        if hasattr(m, n):
            mod = getattr(m, n)
            params = list(mod.parameters()) if not isinstance(mod, torch.Tensor) else [mod]
            tot = sum(p.numel() for p in params)
            print(f"  {n:22s} present, {tot} params")

    # Is the residual purely a function of the time-mean action?
    a2 = a.clone()
    a2[:, 2, :] += 40.0  # large single-frame perturbation
    with torch.no_grad():
        r2, _ = m(a2, 9)
    d = (r2.float() - r_correct.float()).abs().mean().item()
    rel = d / (r_correct.float().abs().mean().item() + 1e-12)
    print(f"\n  single-frame spike -> |delta residual|={d:.6e}  relative={rel:.6f}")
