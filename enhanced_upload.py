#!/usr/bin/env python3
"""
增强版 WandB 上传：
- 移除黑屏 GIF
- 移除假的 trajectory/PCA
- 新增：帧网格、输入输出对比、时序差值热图、真实视频 PCA
"""

import os, glob, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import wandb

# 清除代理
for var in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy"]:
    os.environ.pop(var, None)

API_KEY = os.environ["WANDB_API_KEY"]
ENTITY  = "lv-robotics-lab-nus"
PROJECT = "rynnworld"
BASE    = "/root/autodl-tmp/RynnWorld-Teleop"

CASES = [
    {"name": "assemble_jenga",      "folder": "assemble_jenga_001",         "out_name": "assemble_jenga"},
    {"name": "basic_fold",          "folder": "basic_fold_009",             "out_name": "basic_fold"},
    {"name": "basic_pick_place",    "folder": "basic_pick_place_000",       "out_name": "basic_pick_place"},
    {"name": "clean_surface",       "folder": "clean_surface_001",          "out_name": "clean_surface"},
    {"name": "clip_unclip_papers",  "folder": "clip_unclip_papers_006",     "out_name": "clip_unclip_papers"},
    {"name": "color_task",          "folder": "color_004",                  "out_name": "color_task"},
    {"name": "flip_pages",          "folder": "flip_pages_008",             "out_name": "flip_pages"},
    {"name": "fold_unfold_paper",   "folder": "fold_unfold_paper_basic_008","out_name": "fold_unfold_paper"},
]

# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────

def read_video_frames(path, max_frames=81):
    """读取视频所有帧，返回 uint8 RGB list"""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_frame_grid(frames, title, out_path, n_cols=9, target_w=200):
    """把 frames 均匀采样排成网格图"""
    n_cols = min(n_cols, len(frames))
    indices = np.linspace(0, len(frames)-1, n_cols, dtype=int)
    selected = [frames[i] for i in indices]

    h0, w0 = selected[0].shape[:2]
    th = int(target_w * h0 / w0)

    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * target_w / 100, th / 100 + 1.2))
    if n_cols == 1:
        axes = [axes]
    for ax, (idx, f) in zip(axes, zip(indices, selected)):
        ax.imshow(cv2.resize(f, (target_w, th)))
        ax.set_title(f"f{idx}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def make_comparison_grid(ctrl_frames, gen_frames, title, out_path, n_cols=8, target_w=180):
    """骨骼控制帧(上) vs 生成帧(下) 对比网格"""
    n_avail = min(len(ctrl_frames), len(gen_frames))
    n_cols  = min(n_cols, n_avail)
    indices = np.linspace(0, n_avail-1, n_cols, dtype=int)

    h0, w0 = gen_frames[0].shape[:2]
    th = int(target_w * h0 / w0)

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * target_w / 80, 2 * th / 80 + 1.4))
    fig.suptitle(title, fontsize=10, fontweight="bold")

    row_labels = ["Control (Skeleton)", "Generated (Rollout)"]
    for row, (label, flist) in enumerate([(row_labels[0], ctrl_frames), (row_labels[1], gen_frames)]):
        for col, idx in enumerate(indices):
            ax = axes[row][col]
            f = cv2.resize(flist[idx], (target_w, th))
            ax.imshow(f)
            if col == 0:
                ax.set_ylabel(label, fontsize=8, rotation=90, labelpad=4)
            ax.set_title(f"f{idx}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def make_temporal_diff_heatmap(frames, title, out_path, n_samples=9):
    """相邻帧差值 → 热图，显示运动区域"""
    indices = np.linspace(1, len(frames)-1, n_samples, dtype=int)

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2.2, 2.8))
    fig.suptitle(title, fontsize=10, fontweight="bold")

    for ax, idx in zip(axes, indices):
        diff = np.mean(np.abs(
            frames[idx].astype(np.float32) - frames[idx-1].astype(np.float32)
        ), axis=2)
        im = ax.imshow(diff, cmap="hot", vmin=0, vmax=60)
        ax.set_title(f"Δf{idx-1}→{idx}", fontsize=7)
        ax.axis("off")

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Pixel Δ")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def make_real_pca(frames, title, out_path):
    """从实际视频帧计算真实 PCA，每个 case 结果不同"""
    from sklearn.decomposition import PCA as skPCA
    h, w = 64, 64
    feats = np.array([cv2.resize(f, (w, h)).flatten().astype(np.float32) / 255.0
                      for f in frames])
    # 归一化
    feats -= feats.mean(axis=0)

    pca = skPCA(n_components=2)
    proj = pca.fit_transform(feats)

    n = len(proj)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n - 1):
        ax.plot(proj[i:i+2, 0], proj[i:i+2, 1], color=colors[i], linewidth=1.8)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(n), cmap="viridis", s=30, zorder=5)
    plt.colorbar(sc, ax=ax, label="Frame index")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_motion_intensity(frames, title, out_path):
    """每帧的运动强度曲线（帧差均值）"""
    diffs = [np.mean(np.abs(
        frames[i].astype(np.float32) - frames[i-1].astype(np.float32)
    )) for i in range(1, len(frames))]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(range(len(diffs)), diffs, alpha=0.3, color="royalblue")
    ax.plot(diffs, color="royalblue", linewidth=2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Mean Pixel Difference")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────

wandb.login(key=API_KEY)
print(f"Logged in → {ENTITY}/{PROJECT}\n")

summary = []

for case in CASES:
    name     = case["name"]
    folder   = case["folder"]
    out_name = case["out_name"]
    img_path = os.path.join(BASE, "example", folder, "first_frame.png")
    ctrl_vid = os.path.join(BASE, "example", folder, "control_video.mp4")

    dirs = sorted(glob.glob(os.path.join(BASE, "outputs", out_name, "*")))
    if not dirs:
        print(f"[SKIP] {name} — no output dir")
        continue
    out_dir    = dirs[-1]
    rollout_mp4 = os.path.join(out_dir, "rollout.mp4")
    if not os.path.isfile(rollout_mp4):
        print(f"[SKIP] {name} — no rollout.mp4")
        continue

    print(f"\n[{name}]  {out_dir}")

    # 读帧
    gen_frames  = read_video_frames(rollout_mp4)
    ctrl_frames = read_video_frames(ctrl_vid) if os.path.isfile(ctrl_vid) else []
    print(f"  gen frames: {len(gen_frames)},  ctrl frames: {len(ctrl_frames)}")

    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # 生成新图表
    frame_grid_path   = os.path.join(viz_dir, "frame_grid.png")
    compare_path      = os.path.join(viz_dir, "ctrl_vs_gen.png")
    temp_diff_path    = os.path.join(viz_dir, "temporal_diff.png")
    real_pca_path     = os.path.join(viz_dir, "real_pca.png")
    motion_path       = os.path.join(viz_dir, "motion_intensity.png")

    make_frame_grid(gen_frames, f"Generated Frame Sequence — {name}", frame_grid_path)
    print("  + frame_grid")

    if ctrl_frames:
        make_comparison_grid(ctrl_frames, gen_frames, f"Skeleton Control vs Generated — {name}", compare_path)
        print("  + ctrl_vs_gen comparison")

    make_temporal_diff_heatmap(gen_frames, f"Temporal Pixel Difference — {name}", temp_diff_path)
    print("  + temporal_diff heatmap")

    make_real_pca(gen_frames, f"Video Frame PCA Trajectory — {name}", real_pca_path)
    print("  + real PCA")

    make_motion_intensity(gen_frames, f"Frame-wise Motion Intensity — {name}", motion_path)
    print("  + motion_intensity")

    # WandB 上传
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=f"demo_{name}_v2",
        tags=["demo", "v2", "inference", "teleop", "skeleton"],
        config={"task": name, "example_folder": folder, "out_dir": out_dir},
    )
    print(f"  Run: {run.url}")

    log_data = {}

    # 骨骼控制视频
    if os.path.isfile(ctrl_vid):
        log_data["control_skeleton_video"] = wandb.Video(ctrl_vid, fps=16, format="mp4",
            caption=f"[INPUT] Skeleton Control — {name}")

    # 生成视频
    log_data["rollout_video"] = wandb.Video(rollout_mp4, fps=16, format="mp4",
        caption=f"[OUTPUT] Generated Rollout — {name}")

    # 输入第一帧
    if os.path.isfile(img_path):
        log_data["input_first_frame"] = wandb.Image(img_path, caption=f"[INPUT] First Frame — {name}")

    # Action curve（已有，保留）
    ac_path = os.path.join(out_dir, "action_curve.png")
    if os.path.isfile(ac_path):
        log_data["action_curve"] = wandb.Image(ac_path, caption=f"Control Action Curves — {name}")

    # 新图表
    log_data["frame_grid"]        = wandb.Image(frame_grid_path, caption=f"Generated Frame Sequence — {name}")
    log_data["motion_intensity"]  = wandb.Image(motion_path,     caption=f"Frame-wise Motion Intensity — {name}")
    log_data["temporal_diff"]     = wandb.Image(temp_diff_path,  caption=f"Temporal Pixel Diff Heatmap — {name}")
    log_data["real_pca"]          = wandb.Image(real_pca_path,   caption=f"Video Frame PCA (per-case) — {name}")
    if os.path.isfile(compare_path):
        log_data["ctrl_vs_gen"]   = wandb.Image(compare_path,    caption=f"Skeleton vs Generated Frames — {name}")

    # 指标
    for mf in ["metrics.json", "benchmark.json"]:
        p = os.path.join(out_dir, mf)
        if os.path.isfile(p):
            with open(p) as f:
                metrics = json.load(f)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"perf/{k}"] = v
            break

    run.log(log_data)
    run.finish()
    print(f"  ✅  {len(log_data)} items → {run.url}")
    summary.append({"name": name, "url": run.url, "files": len(log_data)})

print(f"\n{'='*60}")
print(f"DONE — {len(summary)}/8 runs uploaded")
for s in summary:
    print(f"  {s['name']:30s}  {s['files']:2d} files  {s['url']}")
print(f"\n➡  https://wandb.ai/{ENTITY}/{PROJECT}")
