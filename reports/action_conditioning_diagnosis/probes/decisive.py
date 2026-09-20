"""Decisive: is the collapse the MAP or the FEATURES?  Real data, real checkpoint."""
import sys, glob
import torch
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV10
from safetensors.torch import load_file

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
D = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1"

def build():
    m = NativeTrajectoryConditionerV10(input_dim=148).eval()
    sd = torch.load(f"{REPO}/training/native_action_v10b_state_256_600/checkpoint-400/native_trajectory_encoder.bin",
                    map_location="cpu", weights_only=False)
    m.load_state_dict(sd.get("state_dict", sd), strict=False)
    return m

m = build()
C = m.input_residual_projection.weight.clone()      # [3072,768]
out_rms = 0.0350  # measured action-only residual rms on real data

def total_residual(m, traj, proj=None):
    """Full injected residual = action part + frozen base."""
    with torch.no_grad():
        c = m.encode_centered(traj, 9)
        r = (C if proj is None else proj) @ c.transpose(1,2)   # [1,3072,9]
        r = r.unsqueeze(-1).unsqueeze(-1)
        return (r + m.base_residual).float()

def tf(t):  # time-varying energy fraction, time is dim 2
    mu = t.mean(dim=2, keepdim=True)
    return ((t-mu).square().mean() / (t.square().mean()+1e-30)).item()

files = sorted(glob.glob(f"{D}/task*.safetensors"))[:8]
conds = ["correct","reversed","shifted","held"]
agg = {k:[] for k in conds}
print(f"{len(files)} real windows\n")
print(f"{'map':<22}{'|c-rev|/|c-zero|':>18}{'|c-shf|/|c-zero|':>18}{'timevar frac':>14}{'base share':>12}")

for tag, proj in [("trained (rank~18)", None)] + [
        (f"random full-rank s={s}", torch.randn_like(C)*s) for s in (1.0, 0.0)
    ]:
    if proj is not None:
        # rescale to match the trained map's output rms
        with torch.no_grad():
            a = torch.randn(1,33,148)*0.5
            ref = (C @ m.encode_centered(a,9).transpose(1,2)).square().mean().sqrt()
            proj = proj * (ref / (proj @ m.encode_centered(a,9).transpose(1,2)).square().mean().sqrt())
    vals = {k:[] for k in conds}; tfs=[]; bases=[]
    for p in files:
        traj = load_file(p)["robot_trajectory"].unsqueeze(0).float()
        if traj.shape[-1]!=148: continue
        R = {k: total_residual(m, traj, proj) for k in conds}
        R["zero"] = total_residual(m, torch.zeros_like(traj), proj)
        base = R["zero"]  # zero action -> pure base
        denom = (R["correct"]-base).abs().mean().item()+1e-12
        for k in conds:
            vals[k].append((R[k]-R["correct"]).abs().mean().item()/denom)
        tfs.append(tf(R["correct"]))
        bases.append((base.square().mean()/R["correct"].square().mean()).item())
    print(f"{tag:<22}{sum(vals['reversed'])/len(vals['reversed']):>18.4f}"
          f"{sum(vals['shifted'])/len(vals['shifted']):>18.4f}"
          f"{sum(tfs)/len(tfs):>14.6f}{sum(bases)/len(bases):>12.4f}")

print("\nratio >1 means the perturbation moves the residual MORE than the correct-vs-zero gap.")
print("A temporal map should give reversed/shifted well above 1; ~0 means time-blind.")
