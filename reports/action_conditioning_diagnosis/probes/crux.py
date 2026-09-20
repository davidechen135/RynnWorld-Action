"""Unambiguous: how different are the residual's 9 time slices from each other?"""
import sys, glob
import torch
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV10
from safetensors.torch import load_file

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
D = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1"
m = NativeTrajectoryConditionerV10(input_dim=148).eval()
m.load_state_dict(torch.load(f"{REPO}/training/native_action_v10b_state_256_600/checkpoint-400/native_trajectory_encoder.bin",
                  map_location="cpu", weights_only=False), strict=False)

def time_structure(res5d):
    """res5d: [1,C,T,1,1]. Report directly, no ratios of ratios."""
    r = res5d.float().squeeze(0).squeeze(-1).squeeze(-1)   # [C,T]
    T = r.shape[1]
    mu = r.mean(dim=1, keepdim=True)
    tv = r - mu                                            # time-varying part
    return dict(
        total_rms = r.square().mean().sqrt().item(),
        tv_rms    = tv.square().mean().sqrt().item(),
        tv_frac   = tv.square().mean().sqrt().item() / (r.square().mean().sqrt().item()+1e-12),
        # mean pairwise distance between time slices, normalised by slice rms
        slice_dist= torch.cdist(r.T, r.T).mean().item() / (r.square().mean().sqrt().item()+1e-12),
    )

print("A) random gaussian action (what probe.py used):")
torch.manual_seed(0)
a = torch.randn(1,33,148)*0.5
with torch.no_grad(): ra,_ = m(a,9)
s = time_structure(ra)
for k,v in s.items(): print(f"     {k:12s} = {v:.6f}")

print("\nB) real windows (what decisive2 used):")
print(f"     {'file':<24}{'total_rms':>11}{'tv_rms':>11}{'tv_frac':>10}{'slice_dist':>12}")
rows=[]
for p in sorted(glob.glob(f"{D}/task*.safetensors"))[:12]:
    traj = load_file(p)["robot_trajectory"].unsqueeze(0).float()
    if traj.shape[-1]!=148: continue
    with torch.no_grad(): rc,_ = m(traj,9)
    s = time_structure(rc); rows.append(s)
    print(f"     {p.split('/')[-1]:<24}{s['total_rms']:>11.6f}{s['tv_rms']:>11.6f}"
          f"{s['tv_frac']:>10.6f}{s['slice_dist']:>12.6f}")
med = lambda k: sorted(r[k] for r in rows)[len(rows)//2]
print(f"\n     median tv_frac = {med('tv_frac'):.6f}   median slice_dist = {med('slice_dist'):.6f}")

print("\nC) decompose the real-data residual: base vs action part")
p = sorted(glob.glob(f"{D}/task*.safetensors"))[0]
traj = load_file(p)["robot_trajectory"].unsqueeze(0).float()
with torch.no_grad():
    rc,_ = m(traj,9); rz,_ = m(torch.zeros_like(traj),9)
for label, t in (("total (base+action)", rc), ("base only (zero act)", rz),
                 ("action only (rc-rz)", rc-rz)):
    s = time_structure(t)
    print(f"     {label:22s} total_rms={s['total_rms']:.6f} tv_rms={s['tv_rms']:.6f} "
          f"tv_frac={s['tv_frac']:.6f} slice_dist={s['slice_dist']:.6f}")
