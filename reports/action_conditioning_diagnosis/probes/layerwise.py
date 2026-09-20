"""Where does temporal variance die? Trace the V10b encoder on real data."""
import sys, json, glob
import torch
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV10
from safetensors.torch import load_file

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
D = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1"
m = NativeTrajectoryConditionerV10(input_dim=148).eval()
sd = torch.load(f"{REPO}/training/native_action_v10b_state_256_600/checkpoint-400/native_trajectory_encoder.bin",
                map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
m.load_state_dict(sd, strict=False)

def tfrac(t, tdim):
    """fraction of energy that varies over the time axis"""
    t = t.float()
    mu = t.mean(dim=tdim, keepdim=True)
    return ((t-mu).square().mean() / (t.square().mean()+1e-30)).item()

def rms(t): return t.float().square().mean().sqrt().item()

samples = sorted(glob.glob(f"{D}/task*.safetensors"))[:6]
print(f"probing {len(samples)} real windows\n")
rows=[]
for p in samples:
    b = load_file(p)
    traj = b["robot_trajectory"].unsqueeze(0).float()   # expected [1,T,148]
    if traj.shape[-1] != 148:
        print(f"  {p.split('/')[-1]}: unexpected dim {traj.shape}"); continue
    with torch.no_grad():
        state, target, relative, velocity = traj.split(37, dim=-1)
        # stage 0: raw features
        s_in   = traj
        # stage 1: four branch projections
        br = torch.cat((m.reference_projection(state),
                        m.target_projection(target),
                        m.relative_projection(relative),
                        m.velocity_projection(velocity)), dim=-1)
        # stage 2: fusion
        fused = m.feature_fusion(br)
        # stage 3: temporal compressor
        first = fused[:, :1]
        rest  = m.temporal_compressor(fused[:, 1:].transpose(1,2)).transpose(1,2)
        comp  = torch.cat((first, rest), dim=1)
        # stage 4: output_norm(direction)
        direction = m.output_norm(comp)
        # stage 5: magnitude factor
        mr = torch.cat((relative, velocity), dim=-1)
        r2 = mr[:, 1:].reshape(mr.shape[0], 8, 4, mr.shape[-1]).square().mean(dim=2).sqrt()
        mr_p = torch.cat((mr[:, :1], r2), dim=1)
        mag = (1.0 + mr_p.float().square().mean(dim=-1, keepdim=True)).sqrt()
        final = direction * mag.to(direction.dtype)
        # full residual path
        resid = m.input_residual_projection(m.encode_centered(traj, 9))
    row = {
      "file": p.split("/")[-1],
      "in_dim":   (traj.shape[-1], f"{tfrac(traj,-1):.4f}", f"{rms(traj):.4f}"),
      "branches": (br.shape[-1], f"{tfrac(br,-1):.4f}", f"{rms(br):.4f}"),
      "fusion":   (fused.shape[-1], f"{tfrac(fused,-1):.4f}", f"{rms(fused):.4f}"),
      "compressed":("", f"{tfrac(comp,-1):.4f}", f"{rms(comp):.4f}"),
      "direction":("", f"{tfrac(direction,-1):.4f}", f"{rms(direction):.4f}"),
      "final":    ("", f"{tfrac(final,-1):.4f}", f"{rms(final):.4f}"),
      "residual": ("", f"{tfrac(resid,2):.6f}", f"{rms(resid):.4f}"),
    }
    rows.append(row)
    print(f"  {row['file']}")
    for k in ("in_dim","branches","fusion","compressed","direction","final","residual"):
        print(f"      {k:11s} dim={str(row[k][0]):6s} tf={row[k][1]:>10s} rms={row[k][2]}")
    print()
