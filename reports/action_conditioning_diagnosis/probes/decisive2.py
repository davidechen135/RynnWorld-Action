"""Corrected: time-axis fractions + does a full-rank map rescue reversal sensitivity?"""
import sys, glob
import torch
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV10
from safetensors.torch import load_file

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
D = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1"

m = NativeTrajectoryConditionerV10(input_dim=148).eval()
sd = torch.load(f"{REPO}/training/native_action_v10b_state_256_600/checkpoint-400/native_trajectory_encoder.bin",
                map_location="cpu", weights_only=False)
m.load_state_dict(sd.get("state_dict", sd), strict=False)

def tf_time(t):
    """energy fraction that varies along TIME. t is [B,T,C] -> dim 1."""
    t = t.float(); mu = t.mean(dim=1, keepdim=True)
    return ((t-mu).square().mean()/(t.square().mean()+1e-30)).item()

def tf_time_5d(t):
    t = t.float(); mu = t.mean(dim=2, keepdim=True)
    return ((t-mu).square().mean()/(t.square().mean()+1e-30)).item()

print("=== A. time-axis temporal fraction through the encoder stages (real windows) ===")
files = sorted(glob.glob(f"{D}/task*.safetensors"))[:10]
stages = {k: [] for k in ("raw","branches","fusion","compressed","direction","final","act_residual")}
for p in files:
    traj = load_file(p)["robot_trajectory"].unsqueeze(0).float()
    if traj.shape[-1] != 148: continue
    with torch.no_grad():
        state, target, relative, velocity = traj.split(37, dim=-1)
        br = torch.cat((m.reference_projection(state), m.target_projection(target),
                        m.relative_projection(relative), m.velocity_projection(velocity)), -1)
        fused = m.feature_fusion(br)
        comp = torch.cat((fused[:, :1],
                          m.temporal_compressor(fused[:, 1:].transpose(1,2)).transpose(1,2)), 1)
        direction = m.output_norm(comp)
        mr = torch.cat((relative, velocity), -1)
        mr_p = torch.cat((mr[:, :1],
                          mr[:, 1:].reshape(mr.shape[0],8,4,mr.shape[-1]).square().mean(2).sqrt()), 1)
        mag = (1.0 + mr_p.float().square().mean(-1, keepdim=True)).sqrt()
        final = direction * mag.to(direction.dtype)
        c = m.encode_centered(traj, 9)
        res = m.input_residual_projection(c).unsqueeze(-1).unsqueeze(-1)
    stages["raw"].append(tf_time(traj));     stages["branches"].append(tf_time(br))
    stages["fusion"].append(tf_time(fused)); stages["compressed"].append(tf_time(comp))
    stages["direction"].append(tf_time(direction)); stages["final"].append(tf_time(final))
    stages["act_residual"].append(tf_time_5d(res))
for k,v in stages.items():
    print(f"  {k:14s} median tf = {sorted(v)[len(v)//2]:.5f}")

print("\n=== B. feature-level reversal / shift / held sensitivity (time structures) ===")
print("   ratio = ||enc(cond)-enc(correct)|| / ||enc(correct)-enc(zero)||   (>1 = distinguishable)")
files = sorted(glob.glob(f"{D}/task*.safetensors"))[:10]
enc = {k: [] for k in ("reversed","shifted","held")}
for p in files:
    traj = load_file(p)["robot_trajectory"].unsqueeze(0).float()
    if traj.shape[-1] != 148: continue
    with torch.no_grad():
        c   = m.encode_centered(traj, 9)
        z   = m.encode_centered(torch.zeros_like(traj), 9)
        rev = m.encode_centered(traj.flip(1), 9)
        shf = m.encode_centered(traj.roll(traj.shape[1]//2, 1), 9)
        hl  = m.encode_centered(traj[:, :1].expand_as(traj), 9)
        d = c - z
        den = d.norm().item() + 1e-12
        enc["reversed"].append((rev-c).norm().item()/den)
        enc["shifted"].append((shf-c).norm().item()/den)
        enc["held"].append((hl-c).norm().item()/den)
for k,v in enc.items():
    print(f"  {k:10s} median ratio = {sorted(v)[len(v)//2]:.5f}")

print("\n=== C. does a full-rank OUTPUT map change any of this? ===")
print("   A linear W scales every direction; the ratio is bounded by cond(W).")
C = m.input_residual_projection.weight.clone()
W_rand = torch.linalg.qr(torch.randn(3072,768))[0] * (768/3072)**0.5

def residual(traj, traj_cond, W):
    with torch.no_grad():
        a = m.encode_centered(traj, 9); b = m.encode_centered(traj_cond, 9)
        ra = (W @ a.transpose(1,2)).unsqueeze(-1).unsqueeze(-1)
        rb = (W @ b.transpose(1,2)).unsqueeze(-1).unsqueeze(-1)
        return (ra+rb)
files = sorted(glob.glob(f"{D}/task*.safetensors"))[:10]
for tag, W in (("trained rank-18", C), ("random orthogonal", W_rand)):
    acc = []
    for p in files:
        traj = load_file(p)["robot_trajectory"].unsqueeze(0).float()
        if traj.shape[-1] != 148: continue
        with torch.no_grad():
            a = m.encode_centered(traj,9); z = m.encode_centered(torch.zeros_like(traj),9)
            r = m.encode_centered(traj.flip(1),9)
            da = (W @ a.transpose(1,2)); dz = (W @ z.transpose(1,2)); dr = (W @ r.transpose(1,2))
            acc.append(((dr-da).norm()/ (da-dz).norm().clamp_min(1e-12)).item())
    print(f"  {tag:18s} reversed/correct-zero ratio = {sorted(acc)[len(acc)//2]:.5f}")
