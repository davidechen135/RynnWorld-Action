# 跨域 zero-shot 测试 — 真实智元 AgiBot World 数据（无微调）

用**真实智元机器人 ego 数据**测官方 SFT 模型的跨域泛化：真实 head-camera 首帧 + 真实双臂 20 维 action，经零训练 encoder 注入，冻结模型 zero-shot 出 rollout。
脚本：[scripts/agibot_zeroshot.py](../scripts/agibot_zeroshot.py)。产物：`outputs/agibot_zeroshot/`。

## 数据来源
- 数据集：**AgiBotWorld-Alpha**（智元 / OpenDriveLab），经 **ModelScope** 下载（`sample_dataset.tar` 7.1GB；huggingface.co 与 hf-mirror 在本机均不通，ModelScope 通畅 ~16MB/s）。
- 取 3 个 episode 的 `head_color.mp4`（ego 头部视角，AV1 编码，用 libdav1d 软解）+ `proprio_stats.h5`（20 维 action）+ `task_info`。

### 真实双臂 20 维 action（智元官方定义）
| 分量 | 字段 | 维度 |
|---|---|---|
| 双臂关节角 | `/action/joint/position` | 14（左7+右7） |
| 夹爪 | `/action/effector/position` | 2（0开1合） |
| 头部 | `/action/head/position` | 2（yaw/pitch） |
| 腰部 | `/action/waist/position` | 2（pitch/lift） |
| **合计** | | **20** ✓ |

与官方"人手 pose"是**完全不同的域**：智元是真实双臂机器人做家务（洗瓶/洗碗/叠衣），官方训练是桌面人手操作。

## 结果（诚实：跨域泛化**不稳定，看 case**）

| case | 任务 | 首帧场景 | 首帧重建 | 时序保持 |
|---|---|---|---|---|
| 357 | 洗碗 | 灶台+盘子+窗格（纹理丰富） | ✅ | **✅ 全程保持** |
| 392 | 洗瓶子 | 深色水槽 | ✅ | ❌ ~1s 后坍缩 |
| 362 | 叠短裤 | 深色床面 | ✅ | ❌ ~1s 后坍缩 |

**两个稳定规律：**
1. **首帧重建 3/3 成功** —— 模型能吃下 OOD 的真实双臂机器人 ego 图像，rollout 第 0 帧几乎完美复刻真实首帧（VAE 编码/解码保真，模型不排斥新本体）。
2. **时序稳定性 1/3，取决于首帧纹理** —— 崩掉的 362、392 首帧是**大面积低纹理深色区**（深色床/深色水槽），模型缺视觉锚点 → 去噪推进中内容漂移、坍缩成模糊深色纹理；保住的 357 **纹理丰富、结构清晰**（灶台边缘/盘子/窗格），有足够锚点撑住整个 81 帧。

## 结论
1. **"智元 ego 数据 zero-shot 一定崩"是错的。** 真实情况：**首帧总能接住，时序稳定性看场景纹理**——纹理丰富的场景（357 洗碗）能全程保持，低纹理深色场景会坍缩。
2. **action 仍不精确**（与合成 action 实验一致）：encoder 未训练，20 维真实 action 只是分布对齐注入，不构成精确语义驱动。所以即使 357 时序稳住，画面内容也不是被 action 精确控制的。
3. **这是有价值的跨域观测**：世界模型对"新机器人本体 + 新场景"的 zero-shot 迁移，瓶颈不在"能不能接收"（首帧 OK），而在"能不能维持时序"（低纹理场景崩）。要稳定跨域，需在目标域数据上继续训练（越过"不微调"边界）。

## 局限
- encoder 无学习，20 维 action 是分布对齐注入而非语义解码。
- 每 case 单 seed；3 episode 样本小，纹理规律是观察性结论，非统计显著。
- text prompt 用智元任务描述现编码（洗碗/洗瓶等），与官方训练文本分布不同，也是 OOD 因素之一。
