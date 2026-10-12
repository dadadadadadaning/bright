# DamageFormer Region Evidence Readout V1

该实验只回答一个问题：冻结的 V2 Stage1 区域证据，怎样用于最终像素决策才不会破坏 A1？它不重新训练 A1，不改变区域形成方式，也不访问 Test。

## 固定输入

- 主模型：A1 `best_val_miou.pth`，完全冻结。
- 区域证据：V2 `p_sc_residual/stage1_change` 的 3000 步检查点，完全冻结。
- 监督：原四分类像素标签；门控标签只在“基础模型预测 Intact 的建筑像素”上构造。
- 训练：2000 optimizer steps，batch 2，累积 8，学习率 `2e-5`，每 250 步完整 Val450。

## 受控变量

| 变体 | 作用 | 关键约束 |
|---|---|---|
| Base | 原 A1 | 不训练；由每次审计中的 `base_metrics` 给出 |
| W0 | 旧式全四类区域残差 | 无门控、无蒸馏的负对照 |
| W1 | 全局校准 Intact/Damaged | 不读取区域证据，排除“只是校准有效” |
| W2 | 区域条件反对称 I/D logit 残差 | 只修改基础模型判为 Intact 的建筑候选；一减一加 |
| W3 | 区域条件 I/D 概率质量重分配 | Background、Destroyed、I+D 总概率严格不变 |
| W4 | W2/W3 的 1/2/5/10% 选择性使用 | 阈值只由训练集 gate 分数分位数确定，不读 GT |
| W5 | 四分类区域 adapter + KD | 检查“更自由的写回”是否必要 |

W6 低秩特征注入故意未放进第一轮：只有 W2/W3/W5 证明决策级证据有价值后才值得增加自由度。

## 安全协议

1. Stage1 检查点必须为 `change + p_sc_residual` 且 `step >= 3000`；可用 `EXPECTED_STAGE1_SHA256` 再锁定文件。
2. 所有新增头零初始化。训练前运行完整验证集，要求预测与 A1 完全一致，并立即保存 step-0 best。
3. 若后续训练始终变差，`best_val_miou.pth` 自动保留 step 0，而不是强迫选择负收益模型。
4. W4 阈值在完整 Train 上按建筑 gate 均值定覆盖率，然后原样用于 Val；Val 不参与阈值选择。
5. 结果同时报告 mIoU、Damaged IoU/F1、Damaged→Intact、Destroyed IoU、Rescue、Harm 和净修正数。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/damageformer_region_evidence_readout_v1/run.sh preflight
bash experiments/damageformer_region_evidence_readout_v1/run.sh smoke w3_id_mass

mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1
nohup bash experiments/damageformer_region_evidence_readout_v1/run_sequential.sh \
  > /root/data/BRIGHT-train-logs/DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1/sequence.log 2>&1 &
```

若 V2 Stage1 不在默认位置：

```bash
export STAGE1_CKPT=/实际路径/stage1_change/latest.pth
export EXPECTED_STAGE1_SHA256=你的锁定SHA256
```

查看进度：

```bash
tail -f /root/data/BRIGHT-train-logs/DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1/sequence.log
bash experiments/damageformer_region_evidence_readout_v1/run.sh status
```

最终汇总位于：

```text
/root/data/BRIGHT-eval-results/DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1/summary.json
```

## 判断规则

- `W2 > W1`：区域证据优于单纯全局校准。
- `W3 > W2`：保持 I/D 概率质量的写回更安全。
- `W4-selective > W3-global`：问题主要在“何时使用证据”，选择性机制成立。
- `W5 > W3` 且 Destroyed 不退化：才说明四分类自由写回值得保留。
- 若最佳文件仍为 step 0：该变体按严格协议等价于“没有通过”，不要访问 Test。
