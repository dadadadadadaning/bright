# BRIGHT受控实验记录模板

每个实验目录建议固定保存以下文件：

```text
experiment_dir/
├─ run_manifest.json
├─ hypothesis.md
├─ trainable_parameters.txt
├─ metrics_history.jsonl
├─ independent_val.json
├─ decision.json
├─ best_*.pth
├─ latest_resume.pth
└─ command.txt
```

## 1. 实验身份

- 实验ID：
- 日期：
- 随机种子：
- 代码版本/文件哈希：
- 父实验与初始化权重：
- 是否访问Test：`false`

## 2. 单一研究问题

只写一句可以被结果证伪的话，例如：

> 在保持RGB、融合和损毁头冻结时，解冻SAR Stage4能否稳定提高Damaged IoU且不降低Event worst？

## 3. 唯一变化与固定项

### 唯一变化

- 本实验相对基线只改变：

### 固定项

- 数据划分：
- 数据增强：
- SAR统计文件及哈希：
- 损失函数：
- 优化器与学习率：
- 训练步数与有效batch：
- 验证频率：
- checkpoint选择规则：

## 4. 参数权限审计

- 可训练参数总数：
- 总参数数：
- 可训练参数名前缀：
- 冻结参数名前缀：
- BatchNorm/LayerNorm状态：
- step0是否严格复现父模型：

`trainable_parameters.txt`必须列出每个可训练参数的完整名称和形状，不能只写“解冻Stage4”。

## 5. 预注册准入门槛

在训练前填写，不允许看完结果后修改：

- Valid mIoU：
- Damaged IoU：
- Destroyed IoU：
- Damaged gate-miss：
- Event mean：
- Event worst：
- 非负事件数：
- 允许的计算/显存增量：

## 6. 结果表

| 指标 | 基线 | 候选 | Delta (pp) | 门槛 | PASS/FAIL |
|---|---:|---:|---:|---:|---|
| Valid mIoU | | | | | |
| Damaged IoU | | | | | |
| Destroyed IoU | | | | | |
| Damaged gate-miss | | | | | |
| Event mean | | | | | |
| Event worst | | | | | |

另存14个事件的mIoU4与Delta，不得只报告平均值。

## 7. 机制审计

- 输入分布：均值、标准差、分位数、非有限值数。
- 梯度：每个可训练阶段的梯度范数。
- 参数变化：相对初始化的L2变化。
- SAR贡献：消融/置零/打乱后的性能变化。
- 错误迁移：Damaged→Intact、Damaged→Destroyed、Destroyed→Damaged。
- 计算量：峰值显存、训练吞吐、参数量。

## 8. 决策

- `PROCEED`：预注册门槛全部通过。
- `REPEAT_SEEDS`：单种子通过，补多种子确认。
- `STOP_THIS_HYPOTHESIS`：只否定本实验假设，不扩大为整条方向失败。
- `IMPLEMENTATION_INVALID`：守卫、数据契约或恢复协议失败，结果不得用于论文。

## 9. 下一实验

- 下一步唯一变化：
- 为什么由本实验结果推出：
- 明确不做的变化：
