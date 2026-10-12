# P6-C1与RM7受控实验设计

## 1. P6-C1：只验证弱损毁发生监督

```text
P6/A4 Step 1000安全权重
        ↓
冻结Native SAR输入头、双编码器、clf_2和MICB评分器
        ↓
只训练fusion_layer_3（LR=5e-7）
        ↓
Soft-Critical Occurrence辅助损失（0.05）
+ Intact保护（0.02）
        ↓
500步线性升温，三类MIL和Severity均关闭
        ↓
每250步验证；相对Step 1000连续两次下降0.10 pp即停止
```

该实验不使用Memory，不修改Test预测协议。它只回答：Soft-Critical能否在
不重新训练像素分类器的情况下，改善Intact/Affected发生表示。

## 2. RM7：Memory只做可拒绝建议，不做第二分类器

```text
                        ┌─ P3区域语义Key ── Visual D/I LLR ─┐
P3预测Intact的建筑区域 ┤                                  ├─ 全部通过才接受
                        └─ 场景相对SAR Key ─ SAR D/I LLR ──┘
                                      + P3边界不确定性
                                      + Damaged支持域检查
                                      + 多事件邻居共识
```

检索Key不再把256维视觉特征和2维SAR统计直接拼成一个向量。两个Key独立
检索、独立计算Damaged相对Intact的对数似然比，避免某一种证据掩盖另一种。

拒绝机制遵循AND规则：任一证据不足就保持P3 Intact。阈值只由train-only
Memory的留一事件检索和自然分布权重确定；若训练阶段找不到精度至少67%、
净纠正为正的规则，RM7自动关闭，验证输出精确等于P3。

## 3. 两条路线的隔离

- P6-C1研究训练期梯度，不读取Memory。
- RM7冻结P3，只做无训练验证审计。
- RM7通过前禁止Cross-Attention、Memory残差和Test。
- 两者都通过后也不能立即叠加；先分别完成多种子验证，再设计二维消融。

