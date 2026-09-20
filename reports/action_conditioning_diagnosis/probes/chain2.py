"""Where in the encoder chain does the time-varying (motion) structure die?

And: is the low-rank input_residual_projection the thing killing it?
"""
import sys, glob
import torch, torch.nn.functional as F
from safetensors.torch import load_file
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV11

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
DS   = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop"

enc = NativeTrajectoryConditionerV11(input_dim=148).eval()
sd  = torch.load(f"{REPO}/training/native_action_v11_temporal_alignment_300/checkpoint-300/native_trajectory_encoder.bin",
                 map_location="cpu", weights_only=False)
enc.load_state_dict(sd, strict=False)

def tvfrac(m):
    """m: [B,T,C] or [C,T]. Fraction of energy that varies across time."""
    x = m.float()
    if x.ndim == 3: x = x[0]
    if x.shape[0] > x.shape[1]:      # [C,T] -> want time on dim0 for mean
        x = x.T
    v = x - x.mean(dim=0, keepdim=True)     # [T,C] minus time-mean
    return (v.square().mean().sqrt() / (x.square().mean().sqrt() + 1e-30)).item()

traj = None
for p in sorted(glob.glob(f"{DS}/agibot_action_v10_state_256_v1/task*.safetensors")):
    t = load_file(p)["robot_trajectory"].unsqueeze(0).float()
    if t.shape[-1] == 148 and t.abs().sum() > 0: traj = t; break

print("=== stage-by-stage tv_frac (V11, real window, output_frames=9) ===")
with torch.no_grad():
    # reproduce _encode manually so we can read every intermediate
    state, target, relative, velocity = traj.split(37, dim=-1)
    reference = state[:, :1].expand_as(state)
    cat = torch.cat((enc.reference_projection(state),
                     enc.target_projection(target),
                     enc.relative_projection(relative),
                     enc.velocity_projection(velocity)), dim=-1)
    print(f"  branch outputs concat      tv_frac={tvfrac(cat)*100:7.3f}%   rms={cat.square().mean().sqrt():.5f}")
    fused = enc.feature_fusion(cat)
    print(f"  after feature_fusion       tv_frac={tvfrac(fused)*100:7.3f}%   rms={fused.square().mean().sqrt():.5f}")
    # temporal compressor 1+4*(T-1) -> T
    first = fused[:, :1]
    rest  = enc.temporal_compressor(fused[:, 1:].transpose(1, 2)).transpose(1, 2)
    pooled = torch.cat((first, rest), dim=1)
    print(f"  after temporal_compressor  tv_frac={tvfrac(pooled)*100:7.3f}%   rms={pooled.square().mean().sqrt():.5f}")
    normed = enc.output_norm(pooled)
    print(f"  after output_norm          tv_frac={tvfrac(normed)*100:7.3f}%   rms={normed.square().mean().sqrt():.5f}")
    print(f"  --> pre-projection signal carries {tvfrac(normed)*100:.3f}% time-varying energy")

    W = enc.input_residual_projection.weight                 # [3072, 768]
    trained_proj = F.linear(normed, W).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
    print(f"  after TRAINED projection   tv_frac={tvfrac(trained_proj.squeeze().reshape(3072,9))*100:7.3f}%")

    # rank of W
    s = torch.linalg.svdvals(W.float())
    e = s.square(); e = e / e.sum()
    r99 = int((e.cumsum(0) < 0.99).sum()) + 1
    print(f"  input_residual_projection: rank@99%={r99} of {min(W.shape)}, top1={e[0]:.4f}")

    # random full-rank map, RMS-matched to the trained one
    torch.manual_seed(0)
    Wr = torch.randn_like(W)
    Wr = Wr / Wr.square().mean().sqrt() * W.square().mean().sqrt()
    rand_proj = F.linear(normed, Wr).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
    print(f"  after RANDOM full-rank map tv_frac={tvfrac(rand_proj.squeeze().reshape(3072,9))*100:7.3f}%"
          f"   (rms matched: {trained_proj.square().mean().sqrt():.5f} vs {rand_proj.square().mean().sqrt():.5f})")

print("\n=== counterfactual on the FINAL injected residual (what the DiT receives) ===")
with torch.no_grad():
    acts = {
      "correct": traj,
      "reversed": torch.flip(traj, dims=[1]),
      "shift+8": torch.cat((traj[:, 8:], traj[:, -1:].expand(-1, 8, -1)), dim=1),
      "zero": torch.zeros_like(traj),
    }
    key = "correct"
    r = {k: enc(v, 9)[0] for k, v in acts.items()}
    print(f"  {'pair':<26}{'|dr| rms':>12}{'tv-part dist':>15}{'tv dist / |dr|':>17}")
    b = r[key].squeeze().reshape(3072, 9)
    bt = b - b.mean(dim=1, keepdim=True)
    for k in ("reversed", "shift+8", "zero"):
        x = r[k].squeeze().reshape(3072, 9)
        d = x - b
        dt = (x - x.mean(dim=1, keepdim=True)) - bt
        print(f"  correct vs {k:<15}{d.square().mean().sqrt().item():>12.6f}"
              f"{dt.square().mean().sqrt().item():>15.6f}"
              f"{dt.square().mean().sqrt().item()/(b.square().mean().sqrt().item()+1e-30)*100:>16.3f}%")
