# 超像素 Memory 结果诊断与重设计

- 用户研究方向：灾前 RGB 与灾后 SAR 的建筑损毁评估
- 用户研究问题：为什么 SLIC 超像素与类别 Memory 裸检索较强，却无法稳定提高冻结 P3 的像素 mIoU；下一版应如何设计
- 分析日期：2026-09-16
- 证据：当前 P3、Memory V2、Oracle、M2/M2-Union 实验；BRIGHT 官方论文；SRAC 公开录用与成果介绍

## 核心判断

当前方案验证了“干净超像素具有可检索的类别信息”，但没有验证“检索建议能够安全修正 P3”。问题不是 Memory 容量太小，而是训练目标、查询样本和最终评价目标错位。

```text
当前Memory学习：这个超像素属于什么损毁类别？
真正需要学习：在P3当前判断下，采用这个修改能否提高最终mIoU？
```

## 为什么裸检索高、回写增益低

1. 裸检索审计主要面对可靠、纯度较高的区域；真正回写面对的是 P3 最不确定、配准最差、类别最混杂的困难区域。
2. 固定 SLIC 依据灾前 RGB 边界形成，不一定对应灾后 SAR 散射区域、整栋建筑或损毁标注单位。
3. 类别 Memory 与 Direct Head 学习近似相同的三分类信息，容易重复 P3 已经知道的证据。
4. 类别检索正确不等于修改 P3 有益；一个区域类别大体正确，硬回写仍可能伤害大量原本正确的像素。
5. 现有 Oracle 表明候选覆盖和选择均有限：候选 SLIC 只覆盖约 59.25% 的 P3 affected 漏检，Memory 建议覆盖约 21.48%，Memory Oracle recovery 约 17.43%。
6. 当前跨事件 Logistic Verifier 几乎不增益，说明现有统计特征不能稳定预测“修改效用”。

## 与 SRAC 的可比性边界

公开信息确认 SRAC 被 IGARSS 2026 接收，并宣称使用 VMambaSSN 超像素与外部知识检索，在 BRIGHT 达到 75.65% mIoU。但目前可检索材料主要是成果介绍和会议题录，缺少完整论文、公式、训练细节、评价代码和公开实现。因此不能假定其“超像素”和当前固定 SLIC 相同，也不能把其 75.65% 与当前 foreground-only valid mIoU 60.50%直接相减。必须先统一是否包含背景、数据划分、训练步数、测试集和后处理。

## 下一版建议结构

```text
冻结P3
  -> 预测建筑区域/建筑bag
  -> 建筑内部多尺度局部Token（SLIC仅作为可选划分）
  -> Direct Head提出Occurrence修改
  -> Residual Utility Memory检索历史“接受/拒绝修改”案例
  -> 输出期望效用、风险、跨事件多样性和OOD距离
  -> 保守Verifier
  -> 安全软残差回写；不可靠时保持P3
```

Memory 不再存储“Intact/Damaged/Destroyed类别原型”作为主决策依据，而存储：P3状态、候选方向、跨模态关系、区域质量、真实效用、接受/拒绝结果和来源事件。

## 推荐逐级实验

1. C0：当前 SLIC 候选 Oracle。
2. C1：建筑 mask 内规则多尺度 Token，检查候选覆盖是否高于 C0。
3. C2：建筑 bag + 学习式 soft region，检查是否优于固定 SLIC。
4. U0：Direct Utility Verifier，无 Memory。
5. U1：类别 Memory 证据。
6. U2：Residual Utility Memory。
7. U3：打乱 Utility 标签 Memory。
8. W0：硬区域翻转；W1：置信度控制的软残差回写。

只有 C2 提高候选 Oracle、U2 超过 U0/U1、U2 明显超过打乱对照，才可把 Memory 写成有效创新。

## 建议题目

1. **Cross-Event Residual Utility Memory for RGB-SAR Building Damage Assessment**
   - 跨事件检索历史修改效用，而不是检索类别。
2. **Building-Conditioned Multi-Scale Token Retrieval for All-Weather Damage Mapping**
   - 以建筑 bag 和多尺度 Token 替代单一固定 SLIC。
3. **Risk-Aware Selective Retrieval for Multimodal Building Damage Segmentation**
   - 以不确定性、检索分歧和 OOD 风险控制回写。

## 证据来源

- BRIGHT 官方论文指出跨事件泛化、SAR散斑和几何畸变是核心困难；其物体多数投票使集合级 mIoU 下降 0.41pp，却提高事件级 mIoU 0.69pp，说明区域一致性不必然提高像素指标。
- SRAC 的会议录用和 75.65% mIoU 来自中国科学院大学成果介绍；完整论文和代码尚未在本次检索中找到。
- 当前项目 Oracle 与 M2 结果来自本地严格隔离实验，Test 未访问。
