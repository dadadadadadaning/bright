# H6-SDCR 中文接入与实验说明

## 1. 研究问题

H6 不再继续扩展实例 Query。它把 RGB–SAR 观测差异建模为：

```text
观测跨模态差异 = 灾害变化 + 传感器差异 + 局部几何误差
```

模型使用高置信度背景和完好建筑估计当前图像的传感器差异原型，再从
深层 RGB–SAR 差异中减去该原型。这里只改变网络表征，不增加新的监督损失。

## 2. 变体

| 变体 | 唯一新增变量 | 初始安全性 |
|---|---|---|
| A0 | 无，严格 H4-A4 对照 | 状态字典严格一致 |
| A1 | P3 RGB encoder、fusion_layer_1、clf_1 权重迁移 | 无教师、无蒸馏 |
| A2 | stage 3/4 事件条件传感器差异原型 | 残差过渡为零 |
| A3 | A2 + 低频传感器伪变化抑制 | 频率过渡为零 |

A1 会使用 P3 RGB 权重覆盖 A4 的 RGB 编码器，同时使用 P3 的多尺度定位融合。
RGB、定位头和全部 BatchNorm 在训练中保持冻结。A2/A3 的未变化权重由 RGB
建筑概率和 SAR 完好概率共同计算并 `detach`，避免置信度分支投机。

## 3. 必需文件

```text
/root/shared-nvme/BRIGHT-checkpoints/H5_BIQ/source/H4_A4_step5500_valid602784_locked.pth
/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth
/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json
/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/damage_train_priors.json
dataset/splitname/standard_ML/train_set.txt
dataset/splitname/standard_ML/val_set.txt
```

P3 默认 SHA256 必须为：

```text
45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7
```

## 4. 运行顺序

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_H6_SDCR_5090.sh preflight
bash run/run_H6_SDCR_5090.sh guard
bash run/run_H6_SDCR_5090.sh anchor_guard

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H6_SDCR
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/controller_pilot_$(date +%Y%m%d_%H%M%S).log"
nohup env VARIANTS="a1 a2 a3" MAX_RETRIES=1 CONTINUE_ON_FAILURE=1 \
  bash run/run_H6_SDCR_sequence.sh pilot > "$LOG" 2>&1 &
echo $!
tail -F "$LOG"
```

控制器统一执行预检、A0 全验证保护、A1 P3 定位保护、20 步 Smoke 和 3000
步 Pilot。每个变体保存 `best_val_miou.pth` 和 `latest_resume.pth`，不会生成大量
里程碑 checkpoint。

## 5. 实验判定

建议进入 Full 的必要条件：

```text
Valid mIoU  > 60.5019
Damaged IoU > 43.0310
Event mIoU  > 52.7280
Loc F1      接近或不低于 90.50
Destroyed IoU 不出现明显下降
```

A1 只回答“P3 空间锚点是否消除定位上限”；A2 只回答“事件条件传感器差异是否
有效”；A3 只回答“低频伪变化抑制是否提供额外收益”。不得在 Pilot 前加入实例
解码、知识蒸馏或新的辅助损失。

## 6. 迁移检查表

- A0 严格加载无 missing/unexpected keys。
- A1/A2/A3 仅允许缺少 H6 新模块键。
- P3 encoder、fusion_layer_1、clf_1 分别严格加载。
- P3 SHA256 与锁定值一致。
- RGB、定位头、BatchNorm 全程冻结并保持 eval。
- SAR stages 1/2 冻结，stages 3/4 在新模块 warmup 后训练。
- A2/A3 初始因果残差为零；A3 初始频率残差为零。
- 仅访问 train/validation，`test_accessed=false`。
