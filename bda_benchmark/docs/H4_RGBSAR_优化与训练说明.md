# H4 RGB–SAR 优化与训练说明

## 1. 目标与边界

H4 在已经验证有效的 H3 旁边建立一条独立实验路线，不修改 H3 模型、训练脚本和已有权重，也不会影响正在进行的 H3 续训。

H3 已经具备：

- 灾前 RGB 与灾后单通道 SAR 的独立编码器；
- 仅由 RGB 分支完成建筑定位；
- common/unique 表征分离；
- `3×3 Top-k` 局部稀疏对应；
- common 对齐损失只用于背景和完好建筑；
- 软建筑先验约束损毁解码。

H4 集中处理四个剩余问题：

1. H3 的 reliability 只是拼接特征，没有直接控制 SAR 注入强度；
2. 固定局部窗口无法主动校正小范围 RGB–SAR 错位；
3. SAR 后两级缺少训练集内的雷达自监督适配；
4. Damaged 与 Intact/Destroyed 的类别边界仍不稳定。

## 2. 受控变体

| 变体 | 名称 | 相对前一变体的新增变量 |
|---|---|---|
| A0 | H4-A0-H3-Control | 严格加载 H3，用于验证模型迁移完全等价 |
| A1 | H4-A1-Reliability-Gated-Fusion | 显式可靠性门控 SAR 候选融合 |
| A2 | H4-A2-Unchanged-Flow-Alignment | A1 + 有界局部流对齐 |
| A3 | H4-A3-SAR-Masked-Pretraining | A2 + 训练集 SAR 掩码重建初始化 |
| A4 | H4-A4-Hierarchical-Balanced-Severity | A3 + 层级损毁头和训练集先验均衡监督 |

A1 的门控输入由局部匹配可靠性、RGB–SAR 差异、余弦相似度、SAR 幅值、RGB 建筑先验和 SAR 辅助分类置信度组成。建筑先验和 SAR 置信度在进入门控前停止梯度，避免门控反向操纵两个监控分支。

A1/A2 使用零初始化过渡：

```text
joint = H3_joint + tanh(alpha) × (gated_joint - H3_joint)
alpha 初始值 = 0
```

因此 A1 加载 H3 后的损毁输出与 H3 等价。A2 的流预测器最后一层也初始化为零，初始位移严格为零；位移限制在每个特征层的 ±2 像素。

## 3. SAR 掩码预训练的安全策略

SAR-MAE 不从随机编码器开始，也不覆盖 H3 的全部 SAR 表征。它执行：

```text
H3 SAR 编码器权重
  → 冻结 stem、stage1、stage2 及其连接下采样
  → 只适配 stage3、stage4 与重建解码器
  → 保存 SAR encoder 权重
  → A3/A4 加载该权重后进行监督训练
```

训练只读取 `train_set.txt`，不访问验证集和测试集。该设计保留 H3 的低层散射统计，同时让高层 SAR 表征学习训练域内的空间上下文。A3 加载 SAR-MAE 后不再要求初始输出严格等于 H3，因此必须先让 A1/A2 通过，再单独判断 A3 是否带来净收益。

## 4. 论文依据与代码映射

- CROMA（NeurIPS 2023）：独立雷达/光学编码与掩码建模。H4 只在 BRIGHT 训练集 SAR 上进行掩码重建，不直接迁移不同传感器域权重。
- CMX：跨模态特征校正与分阶段融合。H4 将隐式可靠性改成显式、可监控的门控残差。
- Keep the Balance（CVPR 2025）：动态稀疏 RGB-X 融合。H4 保留局部 Top-k，并限制 SAR 的注入幅度。
- FlowMamba：利用流对齐处理建筑损毁中的空间错位。H4 只允许小范围特征位移，且流正则只作用于背景和完好建筑，避免抹平真实变化。
- Balanced Softmax（NeurIPS 2020）：利用训练类频率修正长尾分类梯度。A4 对 building、affected、destroyed 三个条件二分类头使用固定训练集先验。

这些模块不是机械堆叠，而是分别对应“可靠性、错位、SAR 表征、类别长尾”四个可独立验证的问题。

## 5. 冻结、学习率和损失

全程冻结：

- RGB 编码器；
- 建筑定位解码器和定位头；
- SAR stem、stage1、stage2 及相邻下采样；
- 上述冻结模块的归一化状态。

前 500 个 optimizer steps 只允许新增门控、流和层级头更新。之后：

- 新模块学习率：`1e-4`；
- SAR stage3/stage4 学习率：`1e-5`；
- 原损毁融合与解码核心学习率：`1e-6`；
- AdamW，weight decay `5e-3`；
- 默认有效 batch size 为 16。

总损失包括：

- Damage CE + Lovasz；
- SAR 辅助分类；
- unchanged common 对齐和 common/unique 正交；
- 仅 unchanged 区域的流幅值与平滑正则；
- A4 层级先验均衡损失。

定位损失只计算和记录，不加入反向传播。

## 6. 指标与模型保存

除 Valid mIoU、四类 mIoU、各类 IoU、定位 F1 和错误流外，新增记录：

- Event mean / worst / std mIoU；
- 每阶段 correspondence reliability；
- 每阶段 SAR gate 总体及四类分组均值；
- 每阶段 flow mean / P90；
- gate transition 与 hierarchy transition；
- damage balance score。

默认只保留：

```text
best_val_miou.pth
latest_resume.pth
training_summary.json
latest_validation.json
```

只有设置 `SAVE_SECONDARY_BEST=1` 才额外保存 event/balance 最佳权重。

## 7. 服务器运行顺序

先把本地新增文件上传到服务器相同相对路径，然后执行：

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

export SOURCE_H3_CKPT=/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/HDF_H3_full8000_seed3407_fullv1/best_val_miou.pth

bash run/run_H4_RGBSAR_5090.sh preflight
bash run/run_H4_RGBSAR_5090.sh stats
bash run/run_H4_RGBSAR_5090.sh priors
VARIANT=a0 bash run/run_H4_RGBSAR_5090.sh guard
```

A0 完整验证结果必须与 `SOURCE_H3_CKPT` 一致。若不一致，不运行后续实验。

第一阶段只比较 A1/A2：

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H4_RGBSAR
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/controller_a1_a2_pilot_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  SOURCE_H3_CKPT="$SOURCE_H3_CKPT" \
  VARIANTS="a1 a2" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_H4_RGBSAR_sequence.sh pilot \
  > "$LOG" 2>&1 &

echo $!
tail -f "$LOG"
```

只有 A2 通过后，才运行 SAR 自监督和 A3/A4：

```bash
VARIANT=a3 SOURCE_H3_CKPT="$SOURCE_H3_CKPT" \
  bash run/run_H4_RGBSAR_5090.sh sar_smoke

nohup env SOURCE_H3_CKPT="$SOURCE_H3_CKPT" \
  bash run/run_H4_RGBSAR_5090.sh sar_pretrain \
  > /root/shared-nvme/BRIGHT-train-logs/H4_RGBSAR/H4_SAR_MAE.log 2>&1 &
```

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H4_RGBSAR
LOG="$LOG_ROOT/controller_a3_a4_pilot_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  SOURCE_H3_CKPT="$SOURCE_H3_CKPT" \
  VARIANTS="a3 a4" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_H4_RGBSAR_sequence.sh pilot \
  > "$LOG" 2>&1 &

echo $!
tail -f "$LOG"
```

Pilot 完成后可随时重新生成统一对比表：

```bash
bash run/run_H4_RGBSAR_5090.sh summary
cat /root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/H4_validation_comparison.json
```

## 8. Pilot 决策规则

相对于最终选定的 H3 source 完整验证结果：

```text
Valid mIoU       至少 +0.30 pp
Damaged IoU      至少 +0.50 pp
Event mean mIoU  不下降超过 0.20 pp
Localization F1  不下降超过 0.20 pp
```

还应排除以下异常：

- gate 在所有类别都饱和到 0 或 1；
- flow P90 长期贴近最大边界 2.0；
- Damaged 类 gate 明显低于 Intact 类；
- 平均指标提高但 Event worst 显著恶化；
- A3/A4 在初始验证或前 1000 步出现超过 1 pp 的结构性退化。

只有通过 Pilot 的单个最佳变体进入 8000 步 Full；所有结构和超参数锁定前不访问测试集。
