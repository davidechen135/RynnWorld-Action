"""Is the motion-carrying part of the injected residual destroyed before the DiT sees it?

Injection site: core/finetune/models/wan_i2v/rynnworld_teleop_trainer.py:202
    hidden_states = hidden_states + native_features.to(hidden_states.dtype)
At eval, hidden_states is bf16 (scripts/eval_native_action_v2_rot6d37.py:146,156).

The residual enters DiT tokens as [B, T, C] with T the OUTER index (trainer:226
flattens (B,C,T,H,W) after injection).  So a 3072-vector per frame = a token in
the "time" role, broadcast over all spatial positions.

base_residual is a nn.Parameter(torch.zeros(...)) that is frozen
(rynnworld_teleop_trainer.py:869-871) -- never trained, exactly zero.
The DiT therefore receives ONLY the action residual.

Of that residual, the component constant across frames is identical for every
frame and cannot encode motion.  Only tv = r - mean_t(r) can.  Measure tv,
then measure what bf16 casting does to it.
"""
import sys, glob
import torch
from safetensors.torch import load_file
sys.path.insert(0, "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi")
from core.control import (NativeTrajectoryConditionerV6,
                          NativeTrajectoryConditionerV9,
                          NativeTrajectoryConditionerV11)

REPO = "/mnt/workspace/umi-world-model-lab/third_party/RynnWorld-Teleop-umi"
DS   = "/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop"
BF16_EPS = torch.finfo(torch.bfloat16).eps

CASES = [
 ("V6  causal-4x",     NativeTrajectoryConditionerV6(input_dim=37),
   f"{REPO}/training/native_action_v6_causal_rot6d37_300/checkpoint-300",
   f"{DS}/agibot_action_rot6d37_v2_smoke", 37),
 ("V9  spatial-gate",  NativeTrajectoryConditionerV9(input_dim=111),
   f"{REPO}/training/native_action_v9_spatial_gate_single_clip_fixed_300/checkpoint-300",
   f"{DS}/agibot_action_v8_factorized_v1", 111),
 ("V11 temporal-align",NativeTrajectoryConditionerV11(input_dim=148),
   f"{REPO}/training/native_action_v11_temporal_alignment_300/checkpoint-300",
   f"{DS}/agibot_action_v10_state_256_v1", 148),
]

def tv(m):                       # m: [C,T] -> time-varying part
    return m - m.mean(dim=1, keepdim=True)

def report(r, label):
    x  = r.float().squeeze(0).squeeze(-1).squeeze(-1)      # [C,T]
    xq = x.to(torch.bfloat16).float()
    T  = x.shape[1]
    a, aq = tv(x), tv(xq)
    rms   = x.square().mean().sqrt().item()
    tv32  = a.square().mean().sqrt().item()
    tvq   = aq.square().mean().sqrt().item()
    step  = BF16_EPS * rms                                  # bf16 rounding step
    # consecutive-frame step size of the tv part: the per-step "motion"
    d32 = torch.linalg.vector_norm(a[:,1:] - a[:,:-1], dim=0).mean().item()
    dq  = torch.linalg.vector_norm(aq[:,1:] - aq[:,:-1], dim=0).mean().item()
    frozen = int((torch.linalg.vector_norm(aq[:,1:] - aq[:,:-1], dim=0) == 0).sum())
    print(f"  {label}")
    print(f"     residual rms /element            {rms:.6f}")
    print(f"     tv rms (fp32)                    {tv32:.8f}")
    print(f"     tv rms after bf16 cast           {tvq:.8f}   keeps {tvq/(tv32+1e-30)*100:6.1f}%")
    print(f"     consecutive-frame step fp32      {d32:.8f}")
    print(f"     consecutive-frame step bf16      {dq:.8f}   keeps {dq/(d32+1e-30)*100:6.1f}%")
    print(f"     bf16 rounding step eps*rms       {step:.8f}")
    print(f"     step / rounding  (want >>1)      {d32/step:8.4f}")
    print(f"     frame pairs frozen by the cast   {frozen}/{T-1}")
    return dict(rms=rms, tv32=tv32, tvq=tvq, d32=d32, dq=dq, step=step, ratio=d32/step)

for name, enc, ckpt, data, dim in CASES:
    print(f"\n{'='*72}\n{name}\n  ckpt {ckpt.split('training/')[-1]}")
    sd = torch.load(f"{ckpt}/native_trajectory_encoder.bin", map_location="cpu", weights_only=False)
    res = enc.load_state_dict(sd, strict=False)
    if res.missing_keys:   print(f"  MISSING   {res.missing_keys[:4]}")
    if res.unexpected_keys: print(f"  UNEXPECTED {res.unexpected_keys[:4]}")
    print(f"  base_residual absmax = {enc.base_residual.abs().max().item():.8f} "
          f"(frozen parameter; eval feeds it from this ckpt)")
    enc = enc.eval()

    traj = None
    for p in sorted(glob.glob(f"{data}/task*.safetensors")):
        t = load_file(p)["robot_trajectory"].unsqueeze(0).float()
        if t.shape[-1] == dim and t.abs().sum() > 0:
            traj = t; fname = p.split("/")[-1]; break
    if traj is None:
        print(f"  !! no non-trivial {dim}-dim window in {data}"); continue
    print(f"  window {fname}   traj absmax={traj.abs().max().item():.3f}")

    with torch.no_grad():
        ra, _ = enc(traj, 9)
        rz, _ = enc(torch.zeros_like(traj), 9)
    if ra is None:
        print("  residual is None (use_input_residual off)"); continue
    print(f"  injected residual shape {tuple(ra.shape)}  (= [B,C,T,1,1], one C-vector per frame)")
    report(ra, "TOTAL injected residual  r(a)")
    report(ra - rz, "ACTION-ONLY part   r(a) - r(0)")
