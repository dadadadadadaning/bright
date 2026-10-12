# H7-CER：P3/A2 互补路由与灾害对比

## 1. 实验结论

H7 的 Oracle 审计证明 P3 与 H6-A2 存在真实互补，但 A2 并非整体优于 P3：

- 全部有效像素中，P3 独占正确 3,470,795，A2 独占正确 3,030,885；
- 只有 damaged 类呈现净 A2 纠错：499,204 对 282,705；
- background、intact、destroyed 均由 P3 占优；
- Oracle 的 Valid mIoU 为 66.3547%，比 P3 高 5.8528 pp，14/14 个事件均有正上限。

因此 H7-CER 的第一轮只允许一种动作：当 P3 与 A2 分歧、A2 建议 damaged、且位于 P3 建筑区域时，学习是否接受 A2。它不允许 A2 全类别覆盖 P3。

## 2. 模型结构

```text
冻结 P3 ───────────────→ 稳定损毁 logits ───────────────┐
                                                       ├─ damaged-only 低开启率路由 → 输出
冻结 H6-A2 原生 SAR 专家 → 原生 SAR 损毁 logits ────────┘
                                 ↑
                P3/A2 概率、分歧、熵、margin、SAR 场景统计
```

路由器初始化输出严格等于 P3。P3、A2、RGB 定位头和全部 BatchNorm 均冻结，只训练小型路由器。事件名称不会输入模型；SAR 场景均值/标准差仅提供可部署的传感器上下文。

## 3. 为什么要先重新运行 Oracle

旧报告按真实类别统计，能证明“真实 damaged 像素上 A2 有净收益”，却不能直接证明“凡是 A2 建议 damaged 都应采纳”。增强版 Oracle 会额外输出 `[真实类别, P3预测, A2预测]` 转移立方体，必须先检查 P3→A2 damaged 转移的净纠错数。

## 4. 服务器运行顺序

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

# 代码与合成测试
bash run/run_H7_P3_A2_oracle.sh preflight
bash run/run_H7_CER_5090.sh preflight

# 重新生成增强版验证集 Oracle、类别转移、事件和灾害类型报告
bash run/run_H7_P3_A2_oracle.sh val

# 路由器初始化必须严格复现 P3
bash run/run_H7_CER_5090.sh guard

# 20 步冒烟
bash run/run_H7_CER_5090.sh smoke

# 1500 步受控 Pilot；不访问测试集
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/H7_CER
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/H7_CER_Damaged_pilot1500_seed3407.log"
nohup bash run/run_H7_CER_5090.sh pilot > "$LOG" 2>&1 &
echo $!
tail -f "$LOG"
```

## 5. Pilot 后逐事件/灾害类型评估

```bash
EXP=/root/shared-nvme/BRIGHT-checkpoints/H7_CER/H7_CER_Damaged_pilot1500_seed3407
H7_CKPT="$EXP/best_val_miou.pth" \
bash run/run_H7_CER_5090.sh eval_val
```

结果保存为：

- `eventwise_val/H7_CER_eventwise_val.json`
- `eventwise_val/H7_CER_eventwise_val.tsv`

报告同时列出每个事件及 earthquake、wildfire、hurricane、volcano、explosion、flood、conflict 七类灾害的 P3、A2、CER 对比。

## 6. 决策规则

第一轮不进入 Full，除非同时满足：

- 验证 Valid mIoU 高于 P3 60.5019%；
- Damaged IoU 高于 P3 43.0309%；
- Event mean 不低于 P3 52.7280%；
- wildfire 不再出现明显整体退化；
- 至少 8/14 个事件相对 P3 为正，且最差事件退化不超过 0.3 pp。

若总体提高但仅火山/爆炸受益，应继续做事件留一验证或事件均衡训练，不能直接宣称跨灾害泛化提升。测试集只在验证集锁定最终候选后评估一次。
