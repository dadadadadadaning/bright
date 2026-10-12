# H10-H3D：层级式 Lite-3D-STA 损毁评估

## 1. 研究问题

HeteCD 的 3D-STA 原本解决二值异构变化检测。BRIGHT 的目标却是四类像素分割，其中建筑内部又包含 Intact、Damaged、Destroyed。直接让 3D-STA 同时判断三级损毁，会把“有没有变化”和“变化有多严重”混在一起。

H10 将建筑损毁分成两个条件决策：

1. `Intact` 与 `Affected = Damaged + Destroyed`；
2. 已知 `Affected` 后，区分 `Damaged` 与 `Destroyed`。

最终仍输出四类像素概率，不使用硬阈值级联。

## 2. 概率与安全继承

锁定 H6-A2 的建筑类别 logits 为 `z_i, z_d, z_x`：

```text
z_aff = logsumexp(z_d, z_x) - z_i
z_sev = z_x - z_d
```

H2/H3 的受损残差同时加到 `z_d` 与 `z_x`，因此只改变 `Intact vs Affected`，不会改变 `Damaged vs Destroyed`。新分支由 `tanh(gamma_aff)` 控制，`gamma_aff=0` 时输出与 A2 完全一致。

H4 的严重度残差只重新分配 Damaged/Destroyed 的条件概率，并保持 Affected 总质量不变。`gamma_sev=0` 时与输入 H3 完全一致。

## 3. 消融链

| 变体 | 内容 | 科学问题 |
|---|---|---|
| H0 | 锁定 H6-A2 | 原始基线 |
| H1 | 只暴露层级 log-odds，输出不重建 | 层级解释是否严格等价 |
| H2 | Stage4 参数匹配 2D 差异头 | 普通新增参数是否有效 |
| H3 | Stage4 Lite-3D-STA | 3D 跨模态建模是否优于 2D |
| H4 | 冻结通过验证的 H3，再训练条件严重度头 | 严重度细分是否继续提升 |

默认序列只运行 H0/H1 Guard 与 H2/H3。H4 不会被自动启动。

## 4. Lite-3D-STA

```text
RGB Stage4 common ─→ RGB专属1×1投影 ─┐
                                     ├─ stack为长度2的伪时间轴
SAR Stage4 common ─→ SAR专属1×1投影 ─┘
       ↓
深度可分离 Conv3D：(2×3×3)、(2×5×5)、(2×7×7)
       ↓
通道注意力 + 空间注意力
       ↓
受损发生 residual
       ↓
乘冻结P3建筑概率，仅作用于建筑区域
```

它保留 HeteCD 的多尺度 3D 时空差异思想，但只在 Stage4 使用，投影到 64 通道，并针对单通道 SAR 与小样本 BRIGHT 降低参数量。

## 5. 训练目标

只使用一个层级负对数似然：

```text
Intact    : softplus(z_aff)
Damaged   : softplus(-z_aff) + softplus(z_sev)
Destroyed : softplus(-z_aff) + softplus(-z_sev)
```

背景不训练损毁头；P3 建筑定位完全冻结。训练命令将历史 CE/Lovasz、SAR auxiliary、representation、flow、hierarchy auxiliary 权重全部设为 0，避免重复监督。

## 6. 训练阶段

### Phase A：H2/H3

- 冻结完整 H6-A2；
- 只训练 affected head 与 `affected_transition`；
- H2 和 H3 使用同一数据、步数、学习率与选择器；
- step 0 参与最佳权重选择，保证最坏可退回 A2。

### Phase B：H4

- 必须从验证选出的 H3 权重加载；
- 冻结 H3 occurrence head；
- 只训练 severity head 与 `severity_transition`。

## 7. 指标解释

除了最终四类 IoU，新增：

- `affected_iou/f1/recall`：是否检测到真实受损；
- `damaged_gate_miss`：GT Damaged 在第一层被判回 Intact；
- `damaged_severity_to_destroyed`：通过第一层后被严重度头推成 Destroyed；
- `destroyed_gate_miss`：GT Destroyed 在第一层被判回 Intact；
- `destroyed_severity_to_damaged`：通过第一层后被判成 Damaged；
- `conditional_destroyed_f1`：只看受损内部的严重度判断。

## 8. 决策门槛

只有 H3 相对 H2 同时满足以下条件，才允许 H4：

- `Affected F1` 提高；
- `Affected Recall` 提高；
- `Damaged gate miss` 下降；
- Final mIoU 不下降；
- 14 个事件中至少 10 个非负。

否则结论是：3D-STA 具有变化检测能力，但不能稳定转化为 BRIGHT 中度损毁收益，应停止扩展。

## 9. 服务器运行

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H10_H3D_5090.sh preflight

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H10_H3D
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/controller_h2_h3_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  VARIANTS="h2 h3" \
  OPTIMIZER_STEPS=3000 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_H10_H3D_sequence.sh pilot \
  > "$LOG" 2>&1 &

echo $!
tail -F "$LOG"
```

H4 只有在比较报告中的 `proceed_to_h4=true` 后运行：

```bash
H3=/root/shared-nvme/BRIGHT-checkpoints/H10_H3D/H10_H3_pilot3000_seed3407/best_val_miou.pth

VARIANT=h4 \
H10_H3_CKPT="$H3" \
EXP_NAME=H10_H4_pilot3000_seed3407 \
bash run/run_H10_H3D_5090.sh pilot
```

独立测试只对验证锁定的最终候选执行一次：

```bash
EXP=/root/shared-nvme/BRIGHT-checkpoints/H10_H3D/H10_H3_pilot3000_seed3407
H10_CKPT="$EXP/best_val_miou.pth" \
EVAL_OUTPUT_JSON="$EXP/independent_test.json" \
bash run/run_H10_H3D_5090.sh eval_test
```
