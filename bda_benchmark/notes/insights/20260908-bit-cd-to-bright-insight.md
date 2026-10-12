# BIT-CD → BRIGHT 迁移判断（2026-09-08）

## 结论

不建议用 BIT 替换 P3，也不建议把原仓库直接接到 RGB–SAR 输入。建议提取其中“少量语义 token + token 到像素回写”的机制，做成建筑局部、异构模态、零初始化门控的独立残差模块，并且只有在冻结特征探针通过后才训练。

## 迁移对象

### A. 立即可做：token 可靠性证据

在冻结 P3 的每个 LocCC 伪实例内部，把 RGB、SAR、融合特征分别软聚合成 4 个 token，不训练完整 Transformer。提取：跨模态 token 距离、token 分配熵、TTA 后 token 稳定性、修改簇与稳定区 token 差异。把这些统计加入 Metric-Aligned Cluster Verifier。

这一步回答的是“BIT 风格上下文是否能预测一次 S2 修改的真实 mIoU 效用”，而不是直接追求最终分割增益。

### B. 第二阶段：Building-Local BIT residual

```text
冻结或半冻结 P3
  ├─ RGB P3 特征 → RGB 专属软 tokenizer ─┐
  ├─ SAR P3 特征 → SAR 专属软 tokenizer ─┼→ 每栋建筑 4–8 个 token
  └─ LocCC/P3 建筑掩膜限制取样区域 ──────┘
                              ↓
                    1 层跨模态 token encoder
                              ↓
                 token-to-pixel cross-attention
                              ↓
        z_new = z_P3 + gamma_B × delta_z（gamma 初始为 0）
```

只在建筑内部回写，不在整幅图上做全局平滑；小簇仍可直接采用 fixed S2，大簇由 verifier 决定接受该残差还是回退 P3。

## 为什么不是直接复制 BIT

1. P3 已用 Swin，单纯再加 Transformer 会成为重复堆模块。
2. RGB 与 SAR 的统计分布不同，共享 tokenizer/backbone 会强迫两种模态使用同一表示。
3. 三分类损毁需要方向性和等级信息，不能只使用绝对差。
4. 全图 token 容易把不同建筑和背景混合；当前 S2 的有效作用发生在建筑局部。
5. 当前最重要的科学问题是 fixed S2 修改何时可信，BIT 本身没有回答这个决策问题。

## 最小隔离实验

1. E0：P3。
2. E1：P3 + fixed S2。
3. E2：现有 Metric-Aligned M3 verifier。
4. E3：M3 + 参数冻结的 BIT-style token 统计。
5. E4：建筑局部 soft tokenizer，无 encoder、无 decoder。
6. E5：E4 + 1 层 token encoder。
7. E6：E5 + 零门控 token-to-pixel residual。

每一步只增加一个因素。E3 若在 meta_calibration 上 utility AUROC < 0.60、事件宏平均 AUROC < 0.58，或事件方向正确数 < 10/14，则停止 BIT 路线。E6 进入 meta_confirm 前，至少需要相对 fixed S2 提升 0.30pp、Damaged F1 不下降、拒绝率不超过 70%。最终用按事件留一和 bootstrap CI 判断，而不能只看随机 patch 切分。

## 创新性判断

- 原样复现 BIT 或在 P3 后再接一个 Transformer：约 3/10，已有工作明显，且与 P3 重叠。
- 建筑局部的异构模态 tokenizer：约 5/10，需要实验证明它解决 RGB–SAR 对齐与局部污染。
- “建筑局部非对称 token + 零门控回写 + metric-aligned 修改簇选择 + 未见事件验证”的完整闭环：约 7/10。创新点不是某一个算子，而是针对 BRIGHT 失效模式建立了可拒绝、可回退、按指标训练和按事件验证的系统。

## 风险与停止条件

- Token 统计可能重复已有 RGB/SAR relation 特征；先做冻结探针，失败就停止。
- 伪实例粘连会使一组 token 混入多栋建筑；需报告边界/距离切分对照。
- 零门控可能一直学不到有效残差；记录 `gamma`、改动像素比例和每类净效用。
- 不允许同时加入 GroupDRO、Lovász、边界头和 BIT，否则无法归因。

