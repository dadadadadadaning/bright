# S8-P5-C2 Recovery 中文接入说明

## 1. 目标

本分支不覆盖原有 P5。它复用已经训练好的 C2 原生 SAR 分支，并以冻结
P3 作为稳定锚点，验证原生 SAR 是否能提供正向残差，而不是重新替代 P3。

变体严格嵌套：

| 变体 | 内容 |
|---|---|
| R0 | 原 C2 长训练对照 |
| R1 | P3 + 零初始化标量门控 C2 残差 |
| R2 | R1 + Damaged/Destroyed→Intact 间隔损失 |
| R3 | R2 + 建筑实例级监督对比损失 |

R1/R2/R3 在门控初值为零时满足：

```text
DamageOutput = P3DamageOutput
```

完整验证集第 0 步必须复现 `60.5019%` Valid mIoU，容差为 `0.05 pp`。

## 2. 关键文件

```text
model/DamageFormer_S8_P5_C2_Recovery.py
script/standard_ML/train_S8_P5_dcmf.py
tests/test_s8_p5_c2_recovery.py
run/run_S8_P5_C2_recovery_5090.sh
run/run_S8_P5_C2_recovery_sequence.sh
```

## 3. 固定输入权重

```text
P3:
/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth

C2:
/root/shared-nvme/BRIGHT-checkpoints/S8_P5_DCMF/S8_P5_C2_pilot5000_seed3407/best_val_miou.pth

SAR statistics:
/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json
```

P3 教师、RGB 编码器、定位头及所有 BatchNorm 始终冻结。新恢复头学习率为
`1e-4`，C2 SAR/融合学习率为 `1e-5`，损毁核心学习率为 `5e-7`。

## 4. 推荐执行顺序

先运行统一预检：

```bash
bash run/run_S8_P5_C2_recovery_5090.sh preflight
```

然后只验证 R1 的严格 P3 锚定：

```bash
RECOVERY_VARIANT=r1 bash run/run_S8_P5_C2_recovery_5090.sh guard
```

通过后依次执行 Smoke：

```bash
for V in r0 r1 r2 r3; do
  RECOVERY_VARIANT="$V" \
  bash run/run_S8_P5_C2_recovery_5090.sh smoke || exit 1
done
```

首轮建议只跑 R0/R1/R2。R2 有正收益后再运行 R3：

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/S8_P5_C2_Recovery
mkdir -p "$LOG_ROOT"

nohup env \
  VARIANTS="r0 r1 r2" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_S8_P5_C2_recovery_sequence.sh pilot \
  > "$LOG_ROOT/controller_pilot.log" 2>&1 &

tail -f "$LOG_ROOT/controller_pilot.log"
```

R2 通过后单独运行 R3：

```bash
nohup env \
  VARIANTS="r3" \
  MAX_RETRIES=1 \
  CONTINUE_ON_FAILURE=1 \
  bash run/run_S8_P5_C2_recovery_sequence.sh pilot \
  > "$LOG_ROOT/controller_r3_pilot.log" 2>&1 &
```

## 5. Pilot 协议

```text
Optimizer steps       : 6000
Recovery-only warmup  : 500
Validation            : every 500
Checkpoint            : every 1000
Guard starts          : 5000
Guard patience        : 3
Effective batch size  : 16
Test                   : not accessed
```

R0 不执行恢复头 warmup，直接作为原 C2 长训练对照。

## 6. 新增监控指标

训练日志新增：

```text
error_flow_loss
error_flow_damaged
error_flow_destroyed
supcon_loss
supcon_anchor_count
recovery_scale
recovery_gate
```

`recovery_gate` 初始必须为 `0`。若始终接近零，表示 C2 没有提供超过 P3 的
稳定增量；若快速饱和到 `±1`，需要降低恢复头学习率。

## 7. 通过标准

```text
Valid mIoU              >= 60.5019
Damaged IoU             > 43.0309
Destroyed IoU           >= 61.5
Damaged -> Intact       < 30.2791%
Destroyed -> Intact     < 12.3192%
Event mIoU              >= 52.7280
Localization F1         = 90.5704 +/- 0.05
```

同时需要检查 Intact→Damaged 与 Intact→Destroyed，防止错误流约束导致过度
预测损毁。未通过验证前不得运行测试集。
