# RGB–SAR建筑局部高层比较与安全回写路线

- 用户方向：灾前 RGB、灾后 SAR 的建筑损毁语义分割与未见事件泛化。
- 用户问题：如何把 Metric-Aligned Verifier、BIT token、建筑局部上下文和近期模块组成可验证的改进。
- 分析日期：2026-09-08
- 主要趋势依据：BRIGHT 2025、DFC 2025冠军方案、ChangeMamba 2024、2026 BRIGHT Challenge outcome、FSTS-Net 2026、DRUA 2026。

## 机制解释

整个系统相当于“P3初诊、S2提出复诊意见、Verifier判断要不要改病历”。Metric-Aligned Utility给每个修改簇计算真实得失；M2/M3学习哪些修改特征通常可靠；BIT-style token把一栋建筑的大量像素压缩成少量高层摘要；soft tokenizer进一步学习摘要方式；zero-gated decoder只把通过验证的上下文小幅写回P3，且可以回退。

## 推荐结构

```text
灾前RGB ─→ RGB专属P3特征 ─→ 建筑几何/材质token ─┐
                                                  ├→ 局部跨模态比较 → 损毁残差
灾后SAR ─→ SAR专属P3特征 ─→ 散射/结构token ─────┘
       P3/LocCC建筑mask ─→ 限制比较范围

最终logit = P3 logit + 建筑门控 × Verifier接受概率 × 损毁残差
```

比较不能只用绝对差，应同时保留 `[rgb, sar, |rgb-sar|, rgb* sar, cosine, uncertainty]`，并通过类别方向头分别预测 intact→damaged、intact→destroyed、damaged↔destroyed 的证据。

## 近期论文迁移结论

| 来源 | 可借鉴模块 | 使用方式 | 难度/风险 |
|---|---|---|---|
| DFC 2025冠军 | 光学建筑定位、低不确定性伪标签、TTA概率 | 作为P3/S2概率质量与建筑先验对照 | 中；其竞赛式多模型融合成本高 |
| 2026 BRIGHT Challenge | 模态专属编码、后期融合、定位/分类解耦、光学直通初始化 | 作为主结构原则 | 中；优胜代码链接目前一项404、一项空仓库 |
| ChangeMamba 2024 | 长距离时空扫描及BRIGHT脚本 | 只作独立主干对照或替换一个融合块 | 高；CUDA selective-scan环境复杂 |
| CROMA | 雷达/光学独立编码与对比对齐 | 冻结特征探针，测试建筑token可分性 | 中；Sentinel-1/2与VHR分布不同 |
| AnySat/SkySense | 多模态预训练高层表示 | 先线性探针，禁止直接替换P3 | 高；尺度、通道和显存不匹配 |
| DRUA 2026 | 内容/风格解耦 | 在token层增加共享内容与模态风格统计 | 中高；不要先做整图生成式转换 |
| FSTS-Net 2026 | 频率、结构、时空证据及自适应融合 | 拆成三个独立消融，不照搬整网 | 高；代码可用性尚需确认 |

## 本地“即插即用”模块审计

| 本地模块 | 建议 | 原因 |
|---|---|---|
| 2025 Lightweight Cross Attention | 优先改造 | 接口直接接收两个特征图，最接近RGB/SAR后期交互；需改为双向、建筑mask和零门控 |
| 2024 RCM | 第二优先 | 残差与极小层缩放较安全，适合建筑形状上下文；需限制在建筑区域 |
| 2026 CAFR | 只做融合对照 | 能做高低层/双特征通道交互，但当前整理版无归一化安全门，不能直接接主干 |
| 2025 DSA | 边界阶段使用 | 可处理几何形变，但依赖自定义编译算子，不是真正即插即用 |
| 2025 TSSA | 暂不使用 | 适合大量token的线性注意力；本方案每栋仅4–8个token，没有复杂度收益 |
| 2025 ARConv | 暂缓 | 接口依赖epoch和动态核选择，侵入训练循环且很难归因 |
| 2025 FAM整理版 | 不使用 | 文件注释与模块来源不一致，来源可追溯性不足 |
| 2026 TFFA整理版 | 不使用 | 所谓小波分支只是普通卷积，三个分支共用同一门控，不宜当论文忠实实现 |
| 2024 ELA / 2025 GBC | 仅作轻量控制组 | 容易接入，但不解决跨模态比较，论文创新贡献弱 |

## 最小实验顺序

1. 固定P3和fixed S2，建立M2/M3 metric-utility基线。
2. 从P3已有RGB/SAR/fused特征做每建筑4-token无训练统计，验证utility AUROC和事件宏平均。
3. 加两个独立1×1投影与soft tokenizer，只训练投影/tokenizer，P3冻结。
4. 加双向轻量cross-attention，输出方向性损毁残差；残差门从0初始化。
5. 只有以上有效，才分别测试RCM建筑结构上下文、频率特征或GroupDRO；每次只增加一项。

## 论文题目候选

1. **Metric-Aligned Building-Local Cross-Modal Token Verification for All-Weather Damage Mapping**：以指标效用训练修改簇选择器，并在建筑内部比较RGB/SAR token。
2. **Safe Cross-Modal Context Injection for Building Damage Assessment**：以光学直通、零门控残差和拒绝回退保证跨模态注入安全。
3. **Disentangled Structure–Damage Tokens for Cross-Event RGB–SAR Building Damage Mapping**：把建筑结构token、SAR损毁token与事件风格token分开。
4. **From Pixel Smoothing to Instance-Aware Token Refinement**：把fixed S2提升为可学习的建筑局部token回写，并完整比较规则、线性Verifier和学习模块。

## 准入标准

- 冻结token统计相对现有M3必须提高事件宏平均utility AUROC，且至少10/14事件方向正确。
- 学习模块相对fixed S2至少提升0.30pp，Damaged F1不下降，拒绝率不超过70%。
- meta_confirm bootstrap CI下界大于0，再进入Val450。
- 任何基础模型只先做冻结线性探针；不能因模型更新、更大或论文年份更新就直接进入主实验。

