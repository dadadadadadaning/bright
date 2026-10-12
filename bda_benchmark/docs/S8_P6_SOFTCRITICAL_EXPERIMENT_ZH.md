# S8-P6 A4 Soft-Critical MICB受控实验

## 研究问题

在保持P3定位、ROI、像素分类头和测试路径不变的条件下，可微分的多关键
实例聚合能否比单一Hard-Argmax Critical更稳定地改善弱Damaged？

## 唯一结构变化

P3使用：

```text
local occurrence logits -> argmax -> one critical token
```

S8-P6使用：

```text
local occurrence logits -> temperature softmax -> soft critical token
```

并将局部标量证据改为按Bag大小归一化的LogSumExp。默认温度为0.5。该模块
不增加可训练参数，参数命名与P3 MICB头完全一致。

## 明确排除

- 不加入N1-shared Cross-Attention；
- 不加入像素Occurrence新头；
- 不启用MIL logits广播；
- 不访问Test；
- 不改变数据划分、损失权重或checkpoint选择指标。

## 4000步协议

```text
Step 0       : A4函数保持初始化，必须精确复现P3
Step 1-1000  : 只适配A4原生单通道SAR PatchEmbed，MICB warm-up
Step 1001-4000: 开启原P3 Bag损失与Soft-Critical聚合；
                锁定Step-1000 A4输入头；
                只选择性训练fusion_layer_3和clf_2
每500步      : 完整Validation
```

P3初始状态参与checkpoint选择。若连续两次Validation低于P3 0.3 pp，提前
停止。`mil_residual_scale=0`，因此Soft-Critical仍是训练期辅助监督。

## 准入门槛

相对P3至少同时满足：

```text
Valid mIoU        不下降
Damaged IoU       +0.30 pp
Affected Recall   +0.50 pp
Damaged gate-miss -0.50 pp
Event mean        不下降
Event worst       不下降
非负事件          >= 10/14
```

当前P4训练器只直接选择Valid mIoU checkpoint。训练完成后必须使用独立汇总
脚本检查完整准入条件；未通过时禁止Test和下一阶段像素Occurrence头。

## 执行顺序

```bash
bash run/run_S8_P6_SoftCritical_5090.sh preflight
bash run/run_S8_P6_SoftCritical_5090.sh stats
bash run/run_S8_P6_SoftCritical_5090.sh exact_guard
bash run/run_S8_P6_SoftCritical_5090.sh smoke
bash run/run_S8_P6_SoftCritical_5090.sh pilot4000
bash run/run_S8_P6_SoftCritical_5090.sh gate
```
