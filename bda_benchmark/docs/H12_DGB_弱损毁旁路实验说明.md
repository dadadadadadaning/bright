# H12-DGB：弱损毁证据旁路

## 1. 结论

H11-T1E 已经排除“冻结 SARMAE 深层 Block 可以提供中度损毁证据”。下一步不延长 T1E、不解冻 Block10/11、不换更大 SAR 主干。

当前主瓶颈是第一级 `Intact → Affected`。H12 只做一件事：把 Stage2/3 的建筑区域 RGB–SAR 弱变化证据，以零初始化残差的方式旁路到 Affected gate。

## 2. 必须先做的零训练审计

比较：

```text
A2 Stage1 / Stage2 / Stage3
SARMAE Block3 / Block6 / Block8 / Block11（可选）
```

任务：

- Intact vs Damaged
- A2 对 vs 错
- Damaged gate-miss
- P3 错 A2 对 / A2 错 P3 对（提供 P3 权重时）

按事件报告 AUROC/AUPRC、event-macro、event-worst。测试集入口不存在。

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H12_DGB_5090.sh preflight
bash run/run_H12_DGB_5090.sh audit_smoke

SARMAE_CKPT=/path/to/official/SARMAE_ViT_B_checkpoint.pth \
bash run/run_H12_DGB_5090.sh audit_val
```

自动建议：

| 建议 | 含义 |
|---|---|
| `IMPLEMENT_H12_D1_STAGE2_BYPASS` | 使用 A2 浅层旁路，进入 D1 |
| `KEEP_SHALLOW_SARMAE_ADAPTER` | Block3/6 明显优于 8/11 和 A2 浅层，才保留浅层 SARMAE |
| `STOP_PIXEL_BYPASS_USE_BUILDING_POOLING` | 建筑池化明显强于像素，改轻量建筑头 |
| `STOP_SARMAE_AND_BYPASS` | 停止 SARMAE 和像素旁路 |

## 3. 模型

```text
灾前 RGB ── 冻结 P3/A2 定位（不改）
       └── Stage2/3 RGB common ─┐
                                ├─ |R-S|、R×S、3×3 局部统计
灾后 SAR ── 冻结 A2 SAR ────────┘
                                ↓
                    1×1 投影 + 3×3 depthwise
                                ↓ 腐蚀后的建筑 ROI
              tanh(γ) · candidate，γ=0
                                ↓
            只改 Intact vs Affected；严重度 odds 不变
```

D0 与 H6-A2 状态字典严格一致。训练开始时 D1/D2/D3 的预测逐元素等于 A2。最后一层卷积使用 Kaiming 初始化，只把 `γ` 置零；两者同时为零会导致第一步梯度消失，这与 H6/H10 的安全残差约定相同。旁路残差在 fp32 中计算，并对 candidate 做 `8·tanh(x/8)` 软截断：否则 AMP bf16 会在第 2 个 optimizer step（`γ` 离开 0 之后）把 Damaged/Destroyed logits 加到 Inf，`logsumexp` 得到 `Inf-Inf=NaN`，smoke 报 `non-finite H4 loss before optimizer step 2`。冻结的 H6-A2 前向放在 `torch.no_grad()` 里，反向只经过旁路。

| 变体 | 唯一变量 |
|---|---|
| D0 | 精确 H6-A2 |
| D1 | Stage2 旁路，原层级 NLL |
| D2 | Stage2+3 旁路，原层级 NLL |
| D3 | 最佳旁路 + 用 FN-aware focal **替换** Affected 项 |

D1 失败则不跑 D2/D3。不要直接跑 5000 步。

## 4. 训练与选择协议

冻结完整 H6-A2。只训练旁路和 `affected_transition`。

同时启用：

- 事件均衡采样
- Damaged 中心 crop（`--focus_probability 0.5`，Damaged 权重 4）

Checkpoint 顺序：

```text
Event worst → Event mean → Damaged IoU → Valid mIoU
```

进入下一阶段必须同时满足：

```text
Damaged IoU       ≥ +0.30 pp
Affected Recall   ≥ +0.50 pp
Damaged gate-miss ≤ -0.50 pp
Event mean / worst 不下降
至少 10/14 事件非负
Final mIoU 下降不超过 0.10 pp
```

只有验证锁定的一个候选允许一次 `eval_test`。

## 5. 训练命令

正式训练只跑 D1、1500 步、单种子。不要直接跑 5000 步，也不要碰测试集。

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H12_DGB_5090.sh preflight

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H12_DGB
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/H12_D1_train1500_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  SKIP_AUDIT=1 \
  VARIANT=d1 \
  OPTIMIZER_STEPS=1500 \
  bash run/run_H12_DGB_train.sh \
  > "$LOG" 2>&1 &

echo $!
tail -F "$LOG"
```

这条命令会依次执行：D0 精确复现 H6-A2 → D1 smoke 20 步 → D1 训练 1500 步 → 独立验证。权重写到：

```text
/root/shared-nvme/BRIGHT-checkpoints/H12_DGB/H12_D1_screen1500_seed3407/best_val_miou.pth
```

如果只想训某一个变体、不要控制器：

```bash
VARIANT=d1 EXP_NAME=H12_D1_screen1500_seed3407 \
bash run/run_H12_DGB_5090.sh train
```

D1 通过后再跑 D2：

```bash
VARIANTS="d2" bash run/run_H12_DGB_sequence.sh screen
```

若 D1 优于 D2，D3 只使用 Stage2：

```bash
VARIANT=d3 H12_BYPASS_STAGES=stage2 \
EXP_NAME=H12_D3_stage2_screen1500_seed3407 \
bash run/run_H12_DGB_5090.sh screen
```

独立验证：

```bash
EXP=/root/shared-nvme/BRIGHT-checkpoints/H12_DGB/H12_D1_screen1500_seed3407
H12_CKPT="$EXP/best_val_miou.pth" \
H12_BASELINE_JSON=/root/shared-nvme/BRIGHT-checkpoints/H12_DGB/H12_D0_guard_seed3407/initial_validation.json \
EVAL_OUTPUT_JSON="$EXP/independent_val.json" \
bash run/run_H12_DGB_5090.sh eval_val
```

## 6. 暂时停止的方向

不要继续：T1E 到 5000 步、T2E 解冻深层、更大 SAR 基础模型、FlowMamba、DSTCD、局部匹配/全图 FCA、单图传感器原型、更复杂 Router/MoE、再增加 3D 卷积尺度。

若 D1/D2 都失败，但审计中建筑池化 AUROC 明显高于像素，再做轻量建筑 mean/max/boundary pooling 头，不要直接上 Mask2Former。
