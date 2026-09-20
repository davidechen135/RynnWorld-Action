"""One clean table: what the DiT actually receives."""
import sys, glob
import torch, torch.nn.functional as F
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import NativeTrajectoryConditionerV10
from safetensors.torch import load_file

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
D = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1"
m = NativeTrajectoryConditionerV10(input_dim=148).eval()
m.load_state_dict(torch.load(f"{REPO}/training/native_action_v10b_state_256_600/checkpoint-400/native_trajectory_encoder.bin",
                  map_location="cpu", weights_only=False).get("state_dict", {}) or
                  torch.load(f"{REPO}/training/native_action_v10b_state_256_600/checkpoint-400/native_trajectory_encoder.bin",
                  map_location="cpu", weights_only=False), strict=False)

files = sorted(glob.glob(f"{D}/task*.safetensors"))[:16]
acc = {k: [] for k in ("base_rms","act_rms","total_rms","act_share","share_rev","share_shf","share_held")}
for p in files:
    traj = load_file(p)["robot_trajectory"].unsqueeze(0).float()
    if traj.shape[-1] != 148: continue
    with torch.no_grad():
        rc, _ = m(traj, 9)
        rz, _ = m(torch.zeros_like(traj), 9)
        rr, _ = m(traj.flip(1), 9)
        rs, _ = m(traj.roll(traj.shape[1]//2, 1), 9)
        rh, _ = m(traj[:, :1].expand_as(traj), 9)
        rc, rz = rc.float(), rz.float()
        base_r = rz.square().mean().sqrt().item()
        act_energy = (rc - rz).square().mean().item()
        act_r = act_energy ** 0.5
        tot_r = rc.square().mean().sqrt().item()
        acc["base_rms"].append(base_r); acc["act_rms"].append(act_r); acc["total_rms"].append(tot_r)
        acc["act_share"].append(act_r / (base_r + 1e-12))
        for k, r in (("share_rev",rr),("share_shf",rs),("share_held",rh)):
            acc[k].append(((r.float()-rc).square().mean().item()) / (act_energy + 1e-12))
med = lambda v: sorted(v)[len(v)//2]
print(f"over {len(acc['base_rms'])} real windows (V10b ckpt-400)\n")
print(f"  frozen base_residual rms        = {med(acc['base_rms']):.6f}")
print(f"  action-only residual rms        = {med(acc['act_rms']):.6f}")
print(f"  total injected residual rms     = {med(acc['total_rms']):.6f}")
print(f"  action share of injection       = {med(acc['act_share']):.4f}  ({med(acc['act_share'])*100:.2f}%)")
print(f"  patch_embedding rms (reference) ~ 0.43")
print()
print(f"  ||correct-reversed|| / ||correct-zero|| = {med(acc['share_rev']):.6f}")
print(f"  ||correct-shifted || / ||correct-zero|| = {med(acc['share_shf']):.6f}")
print(f"  ||correct-held    || / ||correct-zero|| = {med(acc['share_held']):.6f}")
