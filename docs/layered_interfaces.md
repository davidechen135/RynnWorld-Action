# 三层输出接口设计与 ablation — Stage A

日期：2026-07-27 · 分支 `three-layer-transform`（基线快照 `core/streaming/model_singlelayer_baseline.py`）
硬件：NVIDIA H20 96G · Base 模型：Wan2.2-TI2V-5B-Diffusers（`inner_dim=3072`, `out_ch=48`, `patch=(1,2,2)`, `num_layers=30`）

目标：把官方单 RGB denoising head 扩成 **background/scene + object/contact + robot/actor 三层输出**，
设计两种最小接口（A=通道拼接 concat，B=token fusion + layer embedding），做 tensor/参数量/显存/速度/loss ablation，
给出 adopt / cherry-pick / reject 结论。

**本阶段范围**：只做接口设计 + forward-only ablation。逐层指标框架（IoU/leakage 等，工单 B）与 `layer_manifest` 字段定稿**暂缓**——
无真实分层 GT，等团队统一 schema。当前 manifest 为占位（见 [manifest.py](../core/layered/manifest.py)）。

---

## 一句话总结

两种接口都能在真实 5B config 上跑通，输出 3 层 latent + alpha + 可重组 latent，形状全部正确；
显存/速度相对单层**无可测差异**（forward 由 30 层 attention 主导）。
唯一实质差异是**新增参数量**：concat **+1.77M** vs token **+0.009M**（≈190×）。
两者均做到 **warm-start：init 时重组 latent 与单层输出逐字节等价**（`recomp_vs_single_mse=3e-6`，bf16 舍入下限）。
**结论：adopt 接口 B（token）为默认，接口 A（concat）作为可选高容量变体保留。**

---

## 挂载点（复用单层路径，不重造）

单 RGB head 在 [model.py:698-705](../core/streaming/model.py#L698-L705)：

```
norm_out → normed_tokens → proj_out: Linear(3072, 48·prod(1,2,2)=192) → unpatchify → [B,48,21,30,52]
```

三层改造只在 `proj_out` 之后加**旁路**（[model.py](../core/streaming/model.py) forward 尾部 `if self.layer_mode != "off"`），
单层输出路径逐字节不变，结果 stash 在 `self.last_layered_output`（不改返回签名，不破坏任何 caller）。
开关 `layer_mode ∈ {off, concat, token}`，默认 `off`。
两个 head 定义在 [core/layered/layered_head.py](../core/layered/layered_head.py)，共享同一个 `AlphaCompositor`（保证重组路径与后续指标可比）。

---

## 接口 A：通道拼接 concat（[LayeredConcatHead](../core/layered/layered_head.py#L90)）

- 并联宽输出头 `proj_out_layered = Linear(3072, 3·192)`，unpatchify 后 split 成 3 个 `[B,48,F,H,W]`。
- **Warm start**：`init_from_proj_out` 把原 `proj_out` 权重复制进 3 个 slot，init 时每层 = 单层预测。
- 每层 latent 各自 VAE decode → 3 路 RGB（decode 在 ablation 外，本阶段只验证 latent）。

## 接口 B：token fusion + layer embedding（[LayeredTokenHead](../core/layered/layered_head.py#L128)）

- `layer_embedding = nn.Embedding(3, 3072)`，零初始化；每层把 embedding 加到 token 上，过**共享的原 `proj_out`**。
- **Warm start**：embedding 零初始化 ⇒ init 时 3 层 token 相同、共享 proj_out ⇒ 3 层输出相同 = 单层预测。
- 代价：3 次 proj_out 前向（但 proj_out 相对 30 层 backbone 可忽略）。

## 共享重组：[AlphaCompositor](../core/layered/layered_head.py#L51)

- 每层 `alpha_head: Conv3d(48→3)` → mean → 3 层 softmax，得到逐 voxel 的 partition-of-unity alpha `[B,3,F,H,W]`。
- 重组 = 凸和 `Σ αᵢ · latentᵢ`。softmax 保证这是**忠实且与顺序无关**的重组；
  init 时（3 层 = L，alpha=1/3）恰好返回 L。
- **注意**：迭代式 over-compositing `recomposed*(1-a)+L*a` 在 partition-of-unity 下**不等于**凸和，
  init 会残留 ~0.7·L（首版 bug，已修正为凸和）。z 序 **robot > object > background** 用于下游 RGB 空间合成，写入 manifest；
  latent 空间用凸和。

---

## Ablation 结果（[outputs/layered/ablation.json](../outputs/layered/ablation.json)）

真实 config，latent grid `[48,21,30,52]`，bf16，forward-only，5 次取中位数：

| 维度 | off（单层） | concat (A) | token (B) |
|---|---|---|---|
| single_out shape | `[1,48,21,30,52]` | 同 | 同 |
| 3 层 latent shape | — | 3×`[1,48,21,30,52]` | 3×`[1,48,21,30,52]` |
| alpha shape | — | `[1,3,21,30,52]` | `[1,3,21,30,52]` |
| recomposed shape | — | `[1,48,21,30,52]` | `[1,48,21,30,52]` |
| **新增参数量** | 0 | **+1,770,195 (1.77M)** | **+9,363 (0.009M)** |
| forward 中位 (ms) | 1035.6 | 1029.2 | 1029.5 |
| forward peak VRAM (MB) | 11131.1 | 11161.8 | 11158.1 |
| dummy loss: recomp_mse | — | 3e-6 | 3e-6 |
| dummy loss: layer_spread | — | 0.335 | 0.341 |
| **init recomp vs single MSE** | — | **3e-6** | **3e-6** |

读数：
- **参数量**：concat 比 token 多 ≈190×（宽 3× 输出头 vs 一个 3×3072 embedding + 小 alpha head），但两者相对 5B backbone 都可忽略（<0.04‰）。
- **速度/显存**：三种模式 forward 中位数在 ~6ms 抖动内（噪声），peak VRAM 差 ~30MB（alpha/head 激活）——**无实质差异**，forward 由 30 层 attention 主导。
- **Warm start**：两接口 `init recomp vs single MSE = 3e-6`（bf16 舍入下限，等价于 0），训练从"重组=单层"的安全状态起步，不破坏已复现的单层能力。

---

## 回退验证 ✅

`layer_mode="off"` 与快照 [model_singlelayer_baseline.py](../core/streaming/model_singlelayer_baseline.py) 同权重同输入：
`max abs diff = 0.0`，`torch.equal = True`，`last_layered_output = None`。**单层路径逐字节不变、可完全回退。**

---

## Physics forcing 每层接入点（不预设有效，B 阶段验证）

工单要求"physics forcing 逐层"。当前只梳理**接入点**，不做收敛验证（无分层 GT）：

- **robot/actor 层**：可控层。已有的 control 注入（`control_patch_embedding` 零初始化 Conv3d + `control_scale`，见
  [model.py:475-546](../core/streaming/model.py#L475-L546)）语义上对应机器人动作 → 只对 robot 层施加 action-conditioned forcing 最自然；
  接口 B 可让 robot 层 embedding 额外拼接 action token，接口 A 需在宽头对应 slot 前融合。
- **object/contact 层**：接触/受力约束（如接触时 object 层与 robot 层 alpha 的互斥/接触一致性 loss）应作用在 alpha 与该层 latent 上。
- **background/scene 层**：静态先验（时间一致性 / 低 motion 正则）作用在该层 latent，抑制背景被动作带偏（leakage）。

具体 loss 形式与是否有效留待 B 阶段（有真实分层数据后）。

---

## 结论：adopt / cherry-pick / reject

- **Adopt — 接口 B（token fusion + layer embedding）作为默认三层接口。**
  参数量近乎为零（+0.009M）、与单层权重完全共享（proj_out）、warm-start 天然（embedding 零初始化），
  在无分层 GT、需从复现基线平滑演进的当前阶段风险最低。
- **Cherry-pick — 接口 A（concat）作为可选高容量变体保留。**
  当出现真实分层数据、且各层需要相互解耦的独立表达能力时，A 的独立宽头（+1.77M）提供更大容量；
  代价（参数/显存/速度）经 ablation 证明可忽略，故保留而非删除。共享 `AlphaCompositor` 使 A/B 指标可比、可热切换。
- **Reject — 迭代 over-compositing 的 latent 空间重组。**
  在 partition-of-unity alpha 下不等于凸和，会破坏 warm-start（init 残留 ~0.7·L）；latent 空间统一用凸和 `Σαᵢlatentᵢ`。

**默认配置**：`layer_mode="token"`；`layer_mode="off"` 保持为单层可复现基线的完全等价回退。

---

## 复现命令

```bash
# 三模式 forward ablation → outputs/layered/ablation.json + manifest_{off,concat,token}.json
python scripts/layered_ablation.py --frames 21 --height 30 --width 52 --iters 5 --dtype bf16 --output outputs/layered
```

## 明确不做（本阶段边界）

- ⏸ 逐层指标框架（IoU / Boundary-F / recomposition / leakage / motion，工单 B）—— 无真实分层 GT，暂缓。
- ⏸ `layer_manifest` 字段定稿 —— 等 Lai/Ziyang/Jiahao 统一 schema；当前为占位。
- ⏸ 收敛训练 / VAE decode 逐层 RGB 质量 —— 本阶段只做 forward 接口 ablation。
