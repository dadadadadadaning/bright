# 面向RGB–SAR建筑损毁评估的可学习超像素设计

- 用户研究方向：灾前RGB、灾后SAR的建筑损毁评估
- 用户研究问题：将固定SLICO升级为任务适配的可学习区域，同时避免GT泄漏、区域塌缩和Memory归因混乱
- 分析日期：2026-09-17
- 论文来源：SSN（ECCV 2018）、SpixelFCN（CVPR 2020）以及用户提供的SRAC方法图和说明

## 核心结论

不建议直接复制“VMamba + SSN + Memory”的完整结构。最稳妥的第一版是建筑条件的软超像素网络：只输入灾前RGB结构特征和冻结P3建筑概率，输出像素到附近规则种子的软分配。先用建筑边界、重构、紧凑度和连通性训练区域；区域本身通过消融后，再加入冻结ChangeMamba变化特征作为弱引导；最后才替换现有SLICO并复用相同的区域池化、Direct、Memory和Router协议。

## 建议结构

```text
灾前RGB -> 轻量结构编码器 -> 像素嵌入 E_rgb
冻结P3建筑概率 -----------> 建筑先验 P_build
规则网格种子 -------------> 初始中心 C0
E_rgb + 坐标 + P_build ----> 局部软分配 A(i,k)
                            -> 可学习软超像素
冻结ChangeMamba变化证据 ----> 只用于第二阶段弱边界/异质性引导
```

第一阶段禁止把P3损毁概率和GT损毁类别作为超像素生成输入。GT建筑标签只进入训练损失，推理不可见。

## 训练损失

1. RGB特征/颜色重构损失：同一区域保持结构一致。
2. 建筑边界保持损失：建筑内外像素不能被高概率分到同一区域。
3. 紧凑度损失：避免区域碎裂和跨越过远位置。
4. 面积均衡损失：避免所有像素集中到少数区域。
5. 邻接平滑与连通性近似：避免孤立小岛。
6. 第二阶段可选的变化异质性损失：强变化边界两侧减少软分配共享，但权重必须很小，避免SAR散斑支配区域。

推荐初始权重：重构1.0、建筑边界1.0、紧凑度0.1、面积均衡0.05、连通0.05；变化引导从0开始，最高不超过0.1。

## 分级实验

- LSP0：固定SLICO，现有基线。
- LSP1：可学习超像素，仅RGB重构和紧凑度。
- LSP2：LSP1 + 建筑边界保持。
- LSP3：LSP2 + 冻结ChangeMamba弱变化引导。
- LSP4：LSP3 + 下游区域分类损失小权重联合微调。
- LSP5：打乱建筑概率或变化图，机制负对照。

先比较区域质量，再比较下游分类。只有LSP2在建筑边界泄漏、Boundary Recall、Undersegmentation Error和区域纯度上优于LSP0，才进入Memory。只有LSP3在不破坏建筑边界的同时提高P3错误区域覆盖，才保留变化引导。

## 关键风险

- 损毁标签直接训练区域会形成类别标签泄漏和类别形状过拟合。
- SAR直接参与第一阶段划分会把散斑和错位当边界。
- 只有下游mIoU、没有区域质量对照，无法证明可学习超像素有效。
- 区域数量不受约束时会退化为像素级碎片或一个大区域。
- 过早接入Memory会把区域变化、Key变化和检索变化混在一起。

## 建议题目方向

1. Building-Constrained Learnable Region Tokens for RGB–SAR Disaster Damage Assessment
2. Boundary-Preserving Soft Superpixels with Selective Retrieval for Cross-Modal Building Damage Mapping
3. Alignment-Risk-Aware Learnable Regions and Safe Memory Routing for Multimodal Damage Assessment

## 证据

- SSN把SLIC的离散最近邻约束放松为连续软分配，使超像素可通过下游任务损失端到端学习。
- SpixelFCN直接预测局部像素—超像素关联图，是更容易迁移到PyTorch的实现路线。
- 用户现有实验显示固定SLIC区域关系没有稳定超过规则区域，因此新模块必须先证明边界和区域纯度，而不能默认“可学习”就有效。
