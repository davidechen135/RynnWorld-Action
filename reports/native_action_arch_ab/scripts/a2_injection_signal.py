#!/usr/bin/env python3
"""A2: measure the action signal at each real injection point (V11 checkpoint).

Architecture note established by reading the code, not assumed:

  V11 -> V10 -> V8 -> ... -> V2 -> V3.forward
  * V11 has NO spatial gate: ``spatialize_residual`` / ``last_spatial_gate``
    exist only in V9, which V11 does not inherit from.
  * ``NativeTrajectoryConditionerV3.forward`` returns
    ``action_residual + base_residual``, i.e. the frozen base is *inside* the
    single additive channel.  It is a hard 20x offset, not a post-hoc bias.
  * A second path exists: per-frame AdaLN ``native_modulation`` added to
    ``timestep_proj`` (time on dim 1).
  * A third path exists: a 4-block spatial branch (blocks 0/10/20/30) fed by
    ``encode_spatial_control``.  Its stem is zero-init; it is dead in this
    checkpoint and is reported as such.

Six measured locations:
  L1 input_residual_projection output            (action only)
  L2 action_residual + base_residual             (V3.forward, grid space)
  L3 patch_embedding output                      (video, no action)
  L4 hidden_states after L3 + L2                 (the real injection)
  L5 flattened DiT tokens before block 0
  L6 block 0/10/20/30 outputs

Writes metrics/a2_injection_signal.json.  Does not modify model code.
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
TRAIN = REPO / "training/native_action_v11_temporal_alignment_300"
CKPT_STEP = 300
SEED = 42
DTYPE = torch.bfloat16


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def as_time_major(x: torch.Tensor, T: int) -> torch.Tensor:
    """Return x as [B, T, F].

    Locates the axis equal to T and moves it to position 1.  A token stream
    ``[B, N, C]`` has no axis equal to T, so time is recovered from the known
    leading layout ``N = T*S`` with t-major ordering.  Ambiguity here is what
    silently produced tv_frac = 0 in an earlier revision, so the fallbacks are
    explicit rather than guessed.
    """
    x = x.float()
    if x.ndim < 2:
        raise ValueError(f"unsupported layout {tuple(x.shape)}")
    exact = [i for i in range(1, x.ndim) if x.shape[i] == T]
    if len(exact) == 1:
        t_axis = exact[0]
    elif x.ndim == 3 and x.shape[1] % T == 0:
        t_axis = 1          # [B, T*S, C], t-major
    elif x.ndim == 3 and x.shape[2] % T == 0:
        t_axis = 2
    else:
        raise ValueError(f"cannot locate T={T} in {tuple(x.shape)}")
    order = [0, t_axis] + [i for i in range(1, x.ndim) if i != t_axis]
    return x.permute(*order).reshape(x.shape[0], T, -1)


def tv_share_tokens(x: torch.Tensor, T: int, S: int) -> float | None:
    """One-way temporal variance share for a token stream ``[B, T*S, C]``.

    Requires genuine within-frame replication (S > 1).  The residual path is
    ``[B,C,T,1,1]`` -- one vector per frame broadcast over H*W -- so its spatial
    extent is 1 and this quantity is *undefined* there; it returns None rather
    than a number that would look like a measurement.

    Decomposition per channel c, over the S spatial slots of each frame:
        between[c] = Var_T( mean_S x )      temporal signal
        within[c]  = mean_T( Var_S x )      within-frame spread
        share      = between / (between + within)
    """
    if S <= 1:
        return None
    b, n, c = x.shape
    if n != T * S:
        return None
    f = x.float().reshape(b, T, S, c)
    between = f.mean(dim=2).var(dim=1, unbiased=False)     # [C]
    within = f.var(dim=2, unbiased=False).mean(dim=1)      # [C]
    denom = between + within
    m = denom > 1e-12
    if int(m.sum()) == 0:
        return None
    return float((between[m] / denom[m]).mean().item())


def to_tokens(x: torch.Tensor) -> torch.Tensor:
    """Return x as a 3-dim token stream ``[B, T*S, C]``, t-major.

    Accepts either an already-flattened token stream or a ``[B,C,T,H,W]`` grid.
    ``tv_share_tokens`` unpacks three values; handing it a 5-dim grid raises
    ``ValueError: too many values to unpack (expected 3)``, which is how this
    came up.
    """
    if x.ndim == 3:
        return x
    if x.ndim == 5:
        return x.flatten(2).transpose(1, 2)
    raise ValueError(f"cannot read {tuple(x.shape)} as a token stream")


def rel_tv(x: torch.Tensor, T: int) -> float:
    """Relative temporal variation: mean_t ||x_t - xbar||^2 / mean_t ||x_t||^2.

    Scale-free, per-frame-vector based, so it is not washed out by averaging
    over many features the way a feature-averaged variance decomposition is.
    """
    f = as_time_major(x, T)
    bar = f.mean(dim=1, keepdim=True)
    denom = float(f.pow(2).mean().item())
    if denom <= 1e-12:
        return 0.0
    return float((f - bar).pow(2).mean().item() / denom)


def rms(x: torch.Tensor) -> float:
    return float(x.float().square().mean().sqrt().item())


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    return float((x @ y / (x.norm() * y.norm() + 1e-12)).item())


def main() -> None:
    from safetensors.torch import load_file
    from core.control.native_action_features import build_v10_features
    from core.control.native_trajectory_encoder import NativeTrajectoryConditionerV11

    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    helpers = load_module(HELPERS, "gate_c_visual_v2")
    helpers.apply_monkey_patch(REPO)
    model = helpers.load_model(RYNN, BASE, sys.stdout)
    tokenizer, text_encoder = helpers.load_text_encoder(BASE, device, DTYPE)
    prompt = helpers.encode_text(tokenizer, text_encoder, "", device)
    del tokenizer, text_encoder
    torch.cuda.empty_cache()
    encoder = NativeTrajectoryConditionerV11(input_dim=148).to(device=device, dtype=DTYPE)
    encoder.load_state_dict(
        torch.load(TRAIN / f"checkpoint-{CKPT_STEP}" / "native_trajectory_encoder.bin",
                   map_location="cpu", weights_only=True)
    )
    encoder.eval()
    model.native_trajectory_encoder = encoder
    patch = model.patch_embedding

    block_ix = list(getattr(encoder, "spatial_block_indices", (0, 10, 20, 30)))
    block_ix = [i for i in block_ix if i < len(model.blocks)]
    stem_w = encoder.spatial_stem[-1].weight
    print(f"[A2] checkpoint step={CKPT_STEP} blocks={block_ix}")
    print(f"[A2] spatial_stem[-1].weight rms={rms(stem_w):.6g} "
          f"(0 => spatial branch dead)")
    print(f"[A2] spatial_block_scales="
          f"{[round(float(v), 6) for v in encoder.spatial_block_scales.detach().cpu()]}")
    print(f"[A2] base_modulation rms={rms(encoder.base_modulation):.6g}")
    print(f"[A2] base_residual rms={rms(encoder.base_residual):.6g}")

    captured: dict[str, torch.Tensor] = {}
    handles = [
        patch.register_forward_hook(
            lambda m, i, o: captured.__setitem__("L3_patch_embedding", o.detach())),
    ]

    def mk(i):
        def fn(m, inp, out):
            captured[f"L6_block_{i}"] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    for i in block_ix:
        handles.append(model.blocks[i].register_forward_hook(mk(i)))

    files = sorted(DATA.glob("task*.safetensors"))
    print(f"[A2] windows={len(files)}", flush=True)

    rows = []
    for path in files:
        packed = load_file(str(path))
        raw = packed["robot_trajectory_raw37"].unsqueeze(0).float().to(device)
        mean = packed["robot_trajectory_mean37"].float().to(device)
        std = packed["robot_trajectory_std37"].float().to(device)
        rel_s = packed["robot_relative_scale37"].float().to(device)
        vel_s = packed["robot_velocity_scale37"].float().to(device)
        state = packed["robot_observed_state37"].unsqueeze(0).float().to(device)

        def feat(src):
            return build_v10_features(src, state, mean, std, rel_s, vel_s)

        correct = feat(raw)
        swapped = correct.clone()
        swapped[:, [0, 1]] = correct[:, [1, 0]]
        cond = {
            "correct": correct,
            "reversed": feat(raw.flip(1)),
            "held": feat(state[:, None].expand_as(raw)),
            "swapped": swapped,
            "zero": torch.zeros_like(correct),
        }

        img = packed["img_latent"].unsqueeze(0).to(device=device, dtype=DTYPE)
        latent = packed["video_latents"].unsqueeze(0).to(device=device, dtype=DTYPE)
        shape = latent.shape
        T = shape[2] // model.config.patch_size[0]
        gH, gW = shape[3] // model.config.patch_size[1], shape[4] // model.config.patch_size[2]
        S = gH * gW
        noise = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(SEED),
                            device=device, dtype=DTYPE)
        mask = torch.ones((shape[0], 1, shape[2], shape[3], shape[4]),
                          device=device, dtype=DTYPE)
        mask[:, :, 0] = 0
        model_input = (1 - mask) * img + mask * noise

        per: dict[str, dict] = {}
        with torch.no_grad():
            for name, src in cond.items():
                captured.clear()
                l3 = patch(model_input)                       # L3 [B,3072,T,gH,gW]
                residual, modulation = encoder(src, T)         # L2 and AdaLN
                centered = encoder.encode_centered(src, T)
                act = encoder.input_residual_projection(centered)          # L1 [B,T,3072]
                act_grid = act.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # L2 action part
                base_grid = encoder.base_residual.to(act_grid.dtype)

                l2 = residual.to(l3.dtype)                     # [B,3072,T,1,1]
                l4 = l3 + l2                                   # L4
                l5 = l4.flatten(2).transpose(1, 2)             # L5 [B,N,3072] t-major
                l3_tok = l3.flatten(2).transpose(1, 2)         # same layout, no action
                l4_tok = l5                                      # L4 flattened

                # Invariance check, NOT a null model.  ``tv_share_tokens``
                # decomposes per-channel variance into a between-frame and a
                # within-frame term; both depend only on the *multiset* of
                # per-frame statistics, so permuting whole frames cannot change
                # either one.  The shuffled value is therefore expected to equal
                # the unshuffled value exactly, and it does -- which confirms the
                # estimator is well defined, but means it supplies no floor.
                # The real floor has to come from a different estimator
                # (``rel_tv``, which is order-sensitive because it uses a mean
                # over t as its reference point).
                perm = torch.randperm(T, device=l3.device)
                nb = l3_tok.shape[0]
                l3_shuf = l3_tok.reshape(nb, T, S, -1)[:, perm].reshape(l3_tok.shape)
                l4_shuf = l4_tok.reshape(nb, T, S, -1)[:, perm].reshape(l4_tok.shape)

                # real DiT forward to reach L6 (needs prompt embeddings and,
                # because V11 emits per-frame modulation, per-token timesteps)
                ts = (mask[0, 0, :, ::model.config.patch_size[1],
                          ::model.config.patch_size[2]] * 0).flatten().unsqueeze(0)
                model(hidden_states=model_input,
                      timestep=ts,
                      encoder_hidden_states=prompt,
                      encoder_hidden_states_image=None,
                      robot_trajectory=src,
                      robot_spatial_control=None,
                      null_condition=False)

                per[name] = {
                    "L1_action_proj_rms": rms(act),
                    "L2_action_rms": rms(act_grid),
                    "L2_base_rms": rms(base_grid),
                    "L2_action_over_base": rms(act_grid) / max(rms(base_grid), 1e-12),
                    "L2_action_plus_base_rms": rms(l2),
                    "L3_patch_embedding_rms": rms(l3),
                    "L4_post_add_rms": rms(l4),
                    "L4_action_share": rms(l2) / max(rms(l4), 1e-12),
                    "L5_token_rms": rms(l5),
                    "tv_share_L1": tv_share_tokens(act, T, 1),
                    "tv_share_L2": tv_share_tokens(l2, T, 1),
                    "tv_share_L3": tv_share_tokens(l3_tok, T, S),
                    "tv_share_L4": tv_share_tokens(l4_tok, T, S),
                    "tv_share_L5": tv_share_tokens(l5, T, S),
                    "L3_tv_share_floor": tv_share_tokens(l3_shuf, T, S),
                    "L4_tv_share_floor": tv_share_tokens(l4_shuf, T, S),                    "rel_tv_L1_action": rel_tv(act, T),
                    # Same quantity one stage earlier.  ``encode_centered`` is
                    # _encode(traj) - _encode(zeros), so it genuinely carries
                    # action information; measuring its temporal variation is
                    # what separates "the encoder never produced a per-frame
                    # signal" from "the projection collapsed the per-frame
                    # signal that was there".  Without this the two are
                    # indistinguishable from the L1 number alone.
                    "rel_tv_centered": rel_tv(centered, T),
                    "centered_rms": rms(centered),
                    "centered_frame_norm_cv": float(
                        centered.float().norm(dim=-1).std(unbiased=False)
                        / (centered.float().norm(dim=-1).mean() + 1e-12)),
                    "rel_tv_L2_residual": rel_tv(l2, T),
                    "rel_tv_L3_video": rel_tv(l3, T),
                    "rel_tv_L4_post_add": rel_tv(l4, T),
                    "rel_tv_L5_tokens": rel_tv(l5, T),
                    "rel_tv_L2_action_part": rel_tv(act_grid, T),
                    "adaln_modulation_rms": rms(modulation),
                    "adaln_rel_tv": rel_tv(modulation, T),
                    "L6_tv_share": {k: tv_share_tokens(to_tokens(v), T, S)
                                    for k, v in captured.items()
                                    if k.startswith("L6")},
                    "L6_rms": {k: rms(v) for k, v in captured.items()
                               if k.startswith("L6")},
                    "_act_grid": act_grid.detach().float().cpu(),
                    "_l2": l2.detach().float().cpu(),
                    "_l4": l4.detach().float().cpu(),
                    "_l6": {k: v.detach().float().cpu() for k, v in captured.items()
                            if k.startswith("L6")},
                }

        c = per["correct"]
        by_cond = {}
        for n, v in per.items():
            by_cond[n] = {
                "L2_rms_ratio_vs_correct": v["L2_action_plus_base_rms"] / max(c["L2_action_plus_base_rms"], 1e-12),
                "L4_rms_ratio_vs_correct": v["L4_post_add_rms"] / max(c["L4_post_add_rms"], 1e-12),
                "cos_L1_action_vs_correct": cos(v["_act_grid"], c["_act_grid"]),
                "cos_L2_vs_correct": cos(v["_l2"], c["_l2"]),
                "cos_L4_vs_correct": cos(v["_l4"], c["_l4"]),
                "normdiff_L4_vs_correct": float(
                    ((v["_l4"] - c["_l4"]).norm() / (c["_l4"].norm() + 1e-12)).item()),
                "L4_tv_share": v["tv_share_L4"],
                "L6_block_cos_vs_correct": {
                    k: cos(v["_l6"][k], c["_l6"][k]) for k in c["_l6"]},
                "L6_block_normdiff_vs_correct": {
                    k: float(((v["_l6"][k] - c["_l6"][k]).norm()
                              / (c["_l6"][k].norm() + 1e-12)).item()) for k in c["_l6"]},
            }

        row = {
            "window": path.stem,
            "T": int(T), "grid": [int(gH), int(gW)], "N_tokens": int(T * gH * gW),
            "L1_action_proj_rms": c["L1_action_proj_rms"],
            "L2_action_rms": c["L2_action_rms"],
            "L2_base_rms": c["L2_base_rms"],
            "L2_action_over_base": c["L2_action_over_base"],
            "L2_action_plus_base_rms": c["L2_action_plus_base_rms"],
            "L3_patch_embedding_rms": c["L3_patch_embedding_rms"],
            "L4_post_add_rms": c["L4_post_add_rms"],
            "L4_action_share": c["L4_action_share"],
            "L5_token_rms": c["L5_token_rms"],
            "tv_share_L1_action": c["tv_share_L1"],
            "tv_share_L2_residual": c["tv_share_L2"],
            "tv_share_L3_video": c["tv_share_L3"],
            "tv_share_L4_post_add": c["tv_share_L4"],
            "tv_share_L5_tokens": c["tv_share_L5"],
            "tv_share_L3_floor": c["L3_tv_share_floor"],
            "tv_share_L4_minus_L3": (
                None if c["tv_share_L4"] is None else
                c["tv_share_L4"] - c["L3_tv_share_floor"]),
            "rel_tv_L1_action": c["rel_tv_L1_action"],
            "rel_tv_centered": c["rel_tv_centered"],
            "centered_rms": c["centered_rms"],
            "centered_frame_norm_cv": c["centered_frame_norm_cv"],
            "rel_tv_L2_action_part": c["rel_tv_L2_action_part"],
            "rel_tv_L2_residual": c["rel_tv_L2_residual"],
            "rel_tv_L3_video": c["rel_tv_L3_video"],
            "rel_tv_L4_post_add": c["rel_tv_L4_post_add"],
            "rel_tv_L5_tokens": c["rel_tv_L5_tokens"],
            "adaln_modulation_rms": c["adaln_modulation_rms"],
            "adaln_rel_tv": c["adaln_rel_tv"],
            "L6_tv_share": c["L6_tv_share"],
            "L6_rms": c["L6_rms"],
            "by_condition": by_cond,
        }
        rows.append(row)
        print(json.dumps({
            "window": path.stem,
            "act/base": round(row["L2_action_over_base"], 6),
            "L2_rms": round(row["L2_action_plus_base_rms"], 5),
            "L3_rms": round(row["L3_patch_embedding_rms"], 5),
            "L4_rms": round(row["L4_post_add_rms"], 5),
            "tv_L1": round(row["tv_share_L1_action"], 6) if row["tv_share_L1_action"] is not None else None,
            "tv_L2": round(row["tv_share_L2_residual"], 6) if row["tv_share_L2_residual"] is not None else None,
            "tv_L3": round(row["tv_share_L3_video"], 6) if row["tv_share_L3_video"] is not None else None,
            "tv_L4": round(row["tv_share_L4_post_add"], 6) if row["tv_share_L4_post_add"] is not None else None,
            "tv_L3_floor": round(row["tv_share_L3_floor"], 6) if row["tv_share_L3_floor"] is not None else None,
            "rtv_L1": round(row["rel_tv_L1_action"], 6),
            "rtv_L2": round(row["rel_tv_L2_residual"], 6),
            "rtv_L3": round(row["rel_tv_L3_video"], 6),
            "rtv_L4": round(row["rel_tv_L4_post_add"], 6),
            "rtv_adaln": round(row["adaln_rel_tv"], 6),
            "normdiff_rev_L4": round(by_cond["reversed"]["normdiff_L4_vs_correct"], 6),
            "normdiff_held_L4": round(by_cond["held"]["normdiff_L4_vs_correct"], 6),
        }, ensure_ascii=False), flush=True)

    for h in handles:
        h.remove()
    payload = {
        "checkpoint": str(TRAIN / f"checkpoint-{CKPT_STEP}"),
        "encoder": "NativeTrajectoryConditionerV11",
        "spatial_branch_alive": rms(stem_w) > 0,
        "token_order": "t-major: index = t*(H*W) + h*W + w",
        "locations": {
            "L1": "input_residual_projection output (action only)",
            "L2": "action_residual + base_residual (V3.forward, grid space)",
            "L3": "patch_embedding output (video, no action)",
            "L4": "hidden_states after L3 + L2 (the real injection)",
            "L5": "flattened DiT tokens before block 0",
            "L6": "block 0/10/20/30 outputs",
        },
        "windows": rows,
    }
    (OUT / "metrics" / "a2_injection_signal.json").write_text(json.dumps(payload, indent=2))
    print(f"[A2] wrote {OUT / 'metrics' / 'a2_injection_signal.json'}")


if __name__ == "__main__":
    main()
