#!/usr/bin/env python3
"""
RynnWorld 批量推理 + WandB 上传脚本
对 8 个 example case 分别推理，并上传骨骼视频 + 生成结果到 WandB
"""

import os
import sys
import subprocess
import glob
import json
import time
import wandb

# ============================================================
# 配置
# ============================================================
API_KEY = os.environ["WANDB_API_KEY"]
ENTITY    = "lv-robotics-lab-nus"
PROJECT   = "rynnworld"
BASE_DIR  = "/root/autodl-tmp/RynnWorld-Teleop"
CKPT      = "pretrained/RynnWorld-Teleop"
OUTPUT_ROOT = "outputs/batch_demo"

# 8 个 demo case
CASES = [
    {"name": "assemble_jenga",        "folder": "assemble_jenga_001"},
    {"name": "basic_fold",            "folder": "basic_fold_009"},
    {"name": "basic_pick_place",      "folder": "basic_pick_place_000"},
    {"name": "clean_surface",         "folder": "clean_surface_001"},
    {"name": "clip_unclip_papers",    "folder": "clip_unclip_papers_006"},
    {"name": "color_task",            "folder": "color_004"},
    {"name": "flip_pages",            "folder": "flip_pages_008"},
    {"name": "fold_unfold_paper",     "folder": "fold_unfold_paper_basic_008"},
]

# ============================================================
# 清除代理环境变量（避免 WandB 连接失败）
# ============================================================
for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(var, None)

# ============================================================
# 登录 WandB
# ============================================================
print("=" * 60)
print("Logging in to WandB...")
wandb.login(key=API_KEY)
print(f"Entity: {ENTITY}, Project: {PROJECT}")
print("=" * 60)

os.chdir(BASE_DIR)

results_summary = []

for idx, case in enumerate(CASES):
    name   = case["name"]
    folder = case["folder"]
    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(CASES)}] Processing: {name} ({folder})")
    print(f"{'='*60}")

    image_path   = f"example/{folder}/first_frame.png"
    control_path = f"example/{folder}/control_video.mp4"
    text_emb     = f"example/{folder}/text_embedding.safetensors"
    exp_output   = f"{OUTPUT_ROOT}/{name}"

    # 检查输入文件
    for fpath in [image_path, control_path, text_emb]:
        if not os.path.isfile(fpath):
            print(f"  WARNING: Missing file {fpath}, skipping...")
            continue

    # --------------------------------------------------------
    # 运行推理
    # --------------------------------------------------------
    cmd = [
        sys.executable, "inference_benchmark.py",
        "--image",          image_path,
        "--control_video",  control_path,
        "--text_embedding", text_emb,
        "--experiment_name", name,
        "--checkpoint",     CKPT,
        "--disable_cpu_offload",
        "--seeds",          "42",
    ]

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ERROR (code={proc.returncode}):")
        print(proc.stderr[-2000:])
        results_summary.append({"name": name, "success": False, "error": proc.stderr[-500:]})
        continue

    print(f"  Inference done in {elapsed:.1f}s")

    # --------------------------------------------------------
    # 找到输出目录（最新的时间戳目录）
    # --------------------------------------------------------
    pattern = f"outputs/{name}/*"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"  ERROR: No output dir found at {pattern}")
        results_summary.append({"name": name, "success": False, "error": "output dir not found"})
        continue

    out_dir = dirs[-1]
    print(f"  Output dir: {out_dir}")

    # --------------------------------------------------------
    # 上传到 WandB
    # --------------------------------------------------------
    print(f"  Uploading to WandB (entity={ENTITY}, project={PROJECT})...")

    try:
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            name=f"demo_{name}",
            tags=["demo", "inference", "teleop"],
            config={
                "task": name,
                "example_folder": folder,
                "checkpoint": CKPT,
                "seed": 42,
                "inference_time_s": elapsed,
            },
        )

        log_data = {}

        # 骨骼控制视频（输入）
        if os.path.isfile(control_path):
            log_data["control_skeleton_video"] = wandb.Video(
                control_path, fps=16, format="mp4",
                caption=f"[INPUT] Skeleton Control Video — {name}"
            )

        # 生成的 rollout 视频（输出）
        rollout_mp4 = os.path.join(out_dir, "rollout.mp4")
        if os.path.isfile(rollout_mp4):
            log_data["rollout_video"] = wandb.Video(
                rollout_mp4, fps=16, format="mp4",
                caption=f"[OUTPUT] Generated Rollout — {name}"
            )

        # rollout GIF
        rollout_gif = os.path.join(out_dir, "rollout.gif")
        if os.path.isfile(rollout_gif):
            log_data["rollout_gif"] = wandb.Image(
                rollout_gif, caption=f"Rollout GIF — {name}"
            )

        # 第一帧
        if os.path.isfile(image_path):
            log_data["input_first_frame"] = wandb.Image(
                image_path, caption=f"[INPUT] First Frame — {name}"
            )

        # 图表
        for key, fname, caption in [
            ("action_curve",  "action_curve.png",  "Action Curves"),
            ("trajectory",    "trajectory.png",    "3D Trajectory"),
            ("feature_pca",   "feature_pca.png",   "Feature PCA"),
            ("feature_tsne",  "feature_tsne.png",  "Feature t-SNE"),
            ("benchmark",     "benchmark.png",     "Performance Benchmark"),
            ("attention",     "attention_layer0.png", "Attention Heatmap"),
        ]:
            p = os.path.join(out_dir, fname)
            if os.path.isfile(p):
                log_data[key] = wandb.Image(p, caption=f"{caption} — {name}")

        # 指标
        metrics_path = os.path.join(out_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"metrics/{k}"] = v

        run.log(log_data)
        run_url = run.url
        run.finish()

        print(f"  ✅ Uploaded! Run: {run_url}")
        results_summary.append({
            "name": name,
            "success": True,
            "run_url": run_url,
            "inference_time": elapsed,
            "files_uploaded": len(log_data),
        })

    except Exception as e:
        print(f"  ❌ WandB upload failed: {e}")
        results_summary.append({"name": name, "success": False, "error": str(e)})

# ============================================================
# 汇总报告
# ============================================================
print("\n" + "=" * 60)
print("BATCH INFERENCE SUMMARY")
print("=" * 60)
for r in results_summary:
    status = "✅" if r.get("success") else "❌"
    url    = r.get("run_url", r.get("error", ""))
    files  = r.get("files_uploaded", 0)
    t      = r.get("inference_time", 0)
    print(f"  {status} {r['name']:30s}  {files:2d} files  {t:5.0f}s  {url}")

print(f"\nProject: https://wandb.ai/{ENTITY}/{PROJECT}")
print("Done!")
