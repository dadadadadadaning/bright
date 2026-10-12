# H6-B：P3安全锚定的原生SAR残差路由

## 目标

H6-A2虽然提高了Destroyed IoU，但验证集仍未超过P3，测试集和事件均值也下降。
H6-B不再替换P3主路径，而是保留完整、冻结的P3推理路径，并让原生单通道SAR只学习增量损毁证据。

## 变体

- `B0`：完整P3安全对照。验证和测试输出必须复现锁定P3。
- `B1`：P3 + A2原生SAR专家，仅stage4传感器差异原型和类别残差路由。
- `B2`：P3 + A2原生SAR专家，stage3和stage4传感器差异原型和类别残差路由。

## 数据契约

H6-B使用`h4`数据视图，其中：

- 通道0：训练集统计归一化、p01/p99截断的原生SAR，送入原生SAR编码器；
- 通道1：未截断SAR原始显示值，只用于严格重建P3历史三通道ImageNet输入。

两个通道来自同一SAR文件和同一次几何增强，不存在空间错位。P3不会读取原生SAR通道，原生SAR编码器也不会读取P3审计通道。

## 融合方式

最终损毁logits为：

```text
P3 logits + spatial_gate × class_transition × (native logits - P3 logits)
```

`class_transition`只有intact、damaged、destroyed三个参数，初始化严格为0；背景和建筑定位不允许被原生SAR修改。因此初始化输出严格等于P3，同时transition在第一步具有非零梯度。

## 训练约束

- P3完整模型、RGB定位、BatchNorm全程冻结；
- B1/B2从锁定的H6-A2最佳验证权重初始化原生SAR专家；
- 前1000步只训练传感器适配器和安全路由；
- 不加入KD、教师损失、频率损失、层级辅助损失或表征辅助损失；
- 训练损失仅使用原有Damage CE + Lovasz；
- checkpoint只保留`best_val_miou.pth`与覆盖式`latest_resume.pth`。

## 测试集监控

按照当前实验要求，初始评估和每次验证后都评估测试集。测试指标写入：

```text
test_monitor_history.json
latest_test_monitor.json
round_comparison.json
```

测试指标只记录，绝不调用checkpoint选择器。`best_val_miou.pth`仍只能由验证集`valid_miou`更新。

## 服务器运行

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H6_SDCR_5090.sh preflight
bash run/run_H6_SDCR_5090.sh safe_guard

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H6_SDCR
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/H6B_controller_pilot_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  VARIANTS="b1 b2" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_H6B_safe_residual_sequence.sh pilot \
  > "$LOG" 2>&1 &

echo $!
tail -f "$LOG"
```

## 每轮结果汇总

```bash
EXP=/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_B1_pilot3000_seed3407

python script/standard_ML/summarize_H6B_rounds.py \
  --experiment_dir "$EXP"
```

表格中的`Selected=*`表示验证集选择的最佳步。`dP3-4C`只描述测试集相对P3的变化，不参与选择。

## 决策标准

B1/B2进入Full前至少满足：

- Validation Valid mIoU高于60.5019%；
- Validation Damaged IoU不低于43.0310%；
- Validation Destroyed IoU不低于62.8957%；
- Validation Event mean不低于52.7280%；
- 测试监控不得出现连续两轮整体下降；
- B0验证必须复现P3，B0测试Final mIoU应接近67.2725%。

如果B0不能复现P3，禁止开始B1/B2，优先检查P3权重哈希、双SAR视图和测试列表。
