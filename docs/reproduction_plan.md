# RynnWorld-Teleop 复现计划（待审核）

目标：以官方仓库为直接基线，**先复现、后改造**。本阶段只做"复现 + 结构验证"，
产出一个可复用官方 checkpoint / 数据格式 / 控制注入路径的骨架，为后续三层视频模型改造铺路。

---

## 0. 现状盘点（已完成的部分，来自审核）

- ✅ 8 个官方 case 的 SFT 推理已跑通，产出有效 rollout（832×480 / 81 帧 / H.264），画面写实连贯。
- ✅ 同 case 同 seed 字节级可复现。
- ✅ checkpoint（`RynnWorld-Teleop`、`RynnWorld-Teleop-Causal`）、`Wan2.2-TI2V-5B-Diffusers` base、
  `data/video_latents`、`data/text_embeddings` 均在位。
- ⚠️ 未跑 streaming/causal；benchmark 用自写封装而非官方入口；缺 upstream SHA / 环境 / seeds 记录；
  控制注入路径尚未形成工程结论。

本计划要补齐 ⚠️ 部分，达到任务验收标准 (1)~(3)，并为 (4)~(7) 搭好观测骨架。

---

## 1. 冻结 upstream 基线（验收 1：git SHA / 环境 / checkpoint / seeds）

1. 记录官方 upstream commit SHA（当前 remote 挂在 `ghfast.top` 镜像，需对齐 `alibaba-damo-academy/RynnWorld-Teleop` 官方 HEAD）。
   - 输出：`docs/upstream_baseline.md`，含官方 repo URL、pinned SHA、本地 SHA (`503333b`) 差异说明。
2. 冻结环境：导出 `pip freeze` → `docs/env/requirements.lock.txt`；记录 CUDA / torch / diffusers 版本、GPU 型号（H20 96G）。
3. 记录 checkpoint 来源与校验：`RynnWorld-Teleop`（Stage 1 SFT）、`RynnWorld-Teleop-Causal`（Stage 2 streaming）、
   `Wan2.2-TI2V-5B-Diffusers`（base）的下载源与文件清单 + sha256。
4. 固定 seeds 清单（默认 `42`），记录到复现文档。

## 2. 用官方入口跑通 SFT + streaming（验收 2）

> 关键修正：改用**官方脚本**对齐 baseline，而非 `inference_benchmark.py`（其 latent 初始化路径与官方不同）。

1. **SFT（`inference_user.py`）** —— 跑通官方 8-case（`example/example_cases.json`）。
   - 每个 case：`--image`、`--control_video`、`--text_embedding`、`--checkpoint pretrained/RynnWorld-Teleop`、`--mode sft`。
   - 产出 rollout + control.mp4，归档到 `outputs/repro_sft/<case>/`。
2. **Streaming（`inference_streaming.py`）** —— 至少跑通 1 个 causal streaming case。
   - 用 `pretrained/RynnWorld-Teleop-Causal` + 自动检测的 `control_running_stats.bin`（`--use_control_norm`）。
   - 产出归档到 `outputs/repro_streaming/<case>/`。
3. 记录每个 case 的输入/输出/命令/显存/耗时到 `docs/inference_reproduce.md`（覆盖现有描述，补齐官方入口结果）。

## 3. 记录可复用模块（验收：改造骨架）

梳理 Stage 1 控制注入与 Stage 2 训练入口中**可复用**的模块，落到 `docs/reusable_modules.md`：
- `control_patch_embedding`（零初始化 Conv3d）如何把 hand-pose control video 注入 DiT。
- `control_scale`（可学习标量）注入公式与实际加载值。
- causal cache / `control_running_stats`（streaming 归一化）机制。
- Stage 2 MSE warm-up + DMD 蒸馏训练入口中的可复用组件（先只做代码路径梳理，不训练）。
- 明确标注：现单 RGB denoising head 的位置，即后续扩成 background/scene / object/contact / robot/actor 三层输出的挂载点。

## 4. 单 clip reconstruction smoke（验收：dataloader + 重建冒烟）

1. 在 1~3 个共享分层 robot clip 上，跑 dataloader → VAE encode → decode 的重建冒烟，确认数据格式与官方一致。
2. 报告重建 tensor shape / loss（若有 GT）/ 显存 / 速度。

## 5. 复现文档与可视化交付（验收 1 收口）

统一整理 `docs/`：命令、环境、git SHA、checkpoint、输入输出、可视化（rollout / trajectory / action curve）。
纠正现有 `docs/model_forward.md` 中描述性/推测性内容，改为与代码核对过的 tensor 流。

---

## 明确不在本阶段范围（后续任务）

- 不实现三层 decoder/latent（VAE 通道拼接 vs token fusion + layer embedding 的两种最小接口设计另立文档）。
- 不申请 32×A800 微调 / 不做 DMD 训练（本阶段只推理 + 结构验证）。
- 不做逐层 mask/recomposition、layer IoU、VBench 等指标评测。

## 待你确认的选择点

1. **官方 upstream SHA**：是否直接 pin 官方 `main` 当前 HEAD，还是你指定某个 commit？
2. **streaming case 数量**：先跑 1 个（满足验收下限）还是 8 个都跑？
3. **是否保留 `inference_benchmark.py`**：作为附加可视化工具保留，但复现结论以官方入口为准——同意否？
