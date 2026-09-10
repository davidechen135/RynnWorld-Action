"""Stage-A ablation: single-layer (off) vs concat (A) vs token (B) three-layer heads.

Forward-only smoke on the REAL transformer config (Wan2.2-TI2V-5B: inner_dim=3072,
out_ch=48, patch (1,2,2), 30 layers). Reports per interface:
  * output tensor shapes (3 layer latents / alpha / recomposed)
  * added parameter count vs single-layer
  * forward peak CUDA memory
  * forward speed (ms/step, median of N)
  * a dummy 3-layer loss value (MSE of recomposed vs a target + per-layer reg)
  * equivalence check: layered recomposed ~= single-layer output at init (warm start)

Writes outputs/layered/ablation.json + per-interface layer_manifest placeholder.

Usage:
  python scripts/layered_ablation.py --config pretrained/Wan2.2-TI2V-5B-Diffusers/transformer/config.json \
    --frames 21 --height 30 --width 52 --iters 5 --output outputs/layered
Notes:
  * latent grid defaults to the real inference shape [48,21,30,52]; shrink --frames
    for a lighter smoke. This is a forward/interface ablation, NOT training.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from termcolor import cprint

from core.streaming.model import WanCausalTransformer3DModel
from core.layered import LAYER_NAMES, write_manifest


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def build_model(config_path, dtype, device):
    with open(config_path) as f:
        cfg = json.load(f)
    # keep only ctor-relevant keys (diffusers configs carry extra metadata)
    keys = ["patch_size", "num_attention_heads", "attention_head_dim", "in_channels",
            "out_channels", "text_dim", "freq_dim", "ffn_dim", "num_layers",
            "cross_attn_norm", "qk_norm", "eps", "image_dim", "added_kv_proj_dim",
            "rope_max_seq_len", "pos_embed_seq_len"]
    ctor = {k: cfg[k] for k in keys if k in cfg}
    model = WanCausalTransformer3DModel.from_config(ctor).to(device=device, dtype=dtype)
    model.eval()
    return model, cfg


def make_inputs(model, cfg, frames, height, width, dtype, device):
    b = 1
    in_ch = cfg["in_channels"]
    hidden = torch.randn(b, in_ch, frames, height, width, device=device, dtype=dtype)
    # text cond: [B, seq, text_dim]
    enc = torch.randn(b, 12, cfg["text_dim"], device=device, dtype=dtype)
    # expand_timesteps path uses per-token timestep [B, seq_len]; use scalar-per-batch
    p_t, p_h, p_w = cfg["patch_size"]
    seq = (frames // p_t) * (height // p_h) * (width // p_w)
    timestep = torch.full((b, seq), 500, device=device, dtype=torch.long)
    return dict(hidden_states=hidden, timestep=timestep, encoder_hidden_states=enc,
                attention_kwargs=None, encoder_hidden_states_image=None,
                control_video_latent=None, return_dict=False)


@torch.no_grad()
def run_forward(model, inputs):
    out = model(**inputs)
    return out[0], model.last_layered_output


def measure(model, inputs, iters):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # warmup
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    times.sort()
    return times[len(times)//2], peak


def dummy_loss(single_out, layered):
    """A stand-in 3-layer loss to report a number (NOT a trained objective):
    recomposition MSE to single-layer output + light per-layer L2 spread."""
    rec = layered["recomposed"].float()
    recon = F.mse_loss(rec, single_out.float())
    spread = sum(l.float().pow(2).mean() for l in layered["layers"]) / len(layered["layers"])
    return float(recon), float(spread), float(recon + 0.01 * spread)


def main(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    os.makedirs(args.output, exist_ok=True)
    results = {}

    for mode in ["off", "concat", "token"]:
        cprint(f"\n=== layer_mode = {mode} ===", "cyan")
        model, cfg = build_model(args.config, dtype, device)
        base_params = count_params(model)
        model.init_layered_head(mode)
        total_params = count_params(model)
        inputs = make_inputs(model, cfg, args.frames, args.height, args.width, dtype, device)

        single_out, layered = run_forward(model, inputs)
        ms, peak = measure(model, inputs, args.iters)

        entry = {
            "single_out_shape": list(single_out.shape),
            "added_params": total_params - base_params,
            "added_params_M": round((total_params - base_params) / 1e6, 3),
            "forward_ms_median": round(ms, 2),
            "forward_peak_mem_MB": round(peak, 1),
        }
        if layered is not None:
            entry["layer_latent_shapes"] = [list(l.shape) for l in layered["layers"]]
            entry["alpha_shape"] = list(layered["alpha"].shape)
            entry["recomposed_shape"] = list(layered["recomposed"].shape)
            recon, spread, loss = dummy_loss(single_out, layered)
            entry["dummy_loss"] = {"recomposition_mse": round(recon, 6),
                                    "layer_spread": round(spread, 6),
                                    "total": round(loss, 6)}
            # warm-start equivalence: at init recomposed should be close to single-layer
            entry["init_recomp_vs_single_mse"] = round(recon, 6)
            # write placeholder manifest
            man_path = os.path.join(args.output, f"manifest_{mode}.json")
            write_manifest(man_path, interface=mode,
                           layer_paths=[{"latent": f"layer{i}_{LAYER_NAMES[i]}.safetensors"}
                                        for i in range(len(LAYER_NAMES))],
                           recomposed_rgb_path="recomposed.mp4",
                           source={"note": "stage-A forward smoke", "config": args.config})
            entry["manifest"] = man_path
        results[mode] = entry
        cprint(f"  {json.dumps(entry, ensure_ascii=False)}", "green")
        del model; torch.cuda.empty_cache()

    out_path = os.path.join(args.output, "ablation.json")
    with open(out_path, "w") as f:
        json.dump({"config": args.config,
                   "latent_grid": [cfg["in_channels"], args.frames, args.height, args.width],
                   "dtype": args.dtype, "results": results}, f, indent=2, ensure_ascii=False)
    cprint(f"\nDone. -> {out_path}", "green")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="pretrained/Wan2.2-TI2V-5B-Diffusers/transformer/config.json")
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--width", type=int, default=52)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--output", default="outputs/layered")
    main(p.parse_args())
