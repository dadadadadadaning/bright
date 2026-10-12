# 修改簇、RGB–SAR骨干与即插即用模块迁移设计

- 日期：2026-09-07
- 任务：灾前 RGB + 灾后 SAR 的建筑损毁评估
- 当前父系统：冻结 P3 → LocCC → A6 → fixed S2

## 1. 当前问题的准确表述

P3 给出基础四类像素预测，fixed S2 在 LocCC 伪建筑内提出上下文修改。已有实验说明 fixed S2 总体有效，但部分修改有害；Building Oracle 又说明按区域选择 P3/S2 存在较大理论空间。失败的 PI-M0 则说明整栋建筑平均特征无法预测修改效用。

因此下一步的目标不是再次构造一个全局 RGB–SAR 融合网络，而是把 S2 的局部修改变成可验证提案：

```text
P3 base prediction
       +
fixed S2 candidate prediction
       ↓
同一 LocCC 内的 changed pixels
       ↓ 按 P3→S2 方向分组
       ↓ 每组做 8 连通域
Modification Clusters
       ↓
角色 token + 局部跨模态证据 + 跨事件 Memory
       ↓
接受该簇的 S2，或回退该簇的 P3
```

## 2. 修改簇定义

修改簇同时满足三个条件：

1. 属于同一个 LocCC；
2. 具有相同修改方向，例如 Intact→Damaged；
3. 在图像中 8 邻域连通。

一栋伪建筑可以包含多个修改簇，例如中央的 Intact→Damaged、边缘的 Background→Damaged、角落的 Damaged→Intact。修改簇不是新建筑实例，不是 SLIC，也不使用 GT。它只是 fixed S2 相对 P3 的一个局部候选修改单元。

训练标签才使用 GT 计算 corrected、harmed 与 net utility；推理时 Key 只能由 RGB、SAR、P3、S2、空间位置和检索统计构成。

## 3. 当前 P3 接口边界

锁定 P3 使用两个独立 Swin-T 编码器，分别处理灾前 RGB 和灾后输入。Stage3/Stage4 通道数分别是 384/768，随后四尺度上采样拼接并压缩为每模态 256 通道，再生成融合损毁特征。

锁定 P3 的 `return_aux=True` 主要暴露融合后的特征，没有直接暴露原始 Stage3/4。因此不能直接假设图中的 Stage3/4 已经可读。正确做法是使用独立 wrapper 复用冻结 encoder，或复用项目中已经实现的 Stage3/4 提取模型；不要改写锁定 P3 文件和 checkpoint。

## 4. 三种骨干修改等级

### L0：冻结特征读取，当前首选

- P3 权重和预测完全不变；
- 只读取 RGB/SAR Stage3/4；
- 在修改簇及其 halo 内池化 change、anchor、boundary 三类角色 token；
- 用于无参数审计和 Memory key。

这是当前唯一应该立即进入主线的方式。

### L1：冻结骨干 + 小型模态适配器

- RGB：边界/高频/跨尺度结构适配器；
- SAR：原生单通道、稳健纹理/频率或轻量专家；
- 输出只进入 verifier，或以零初始化残差进入 logits；
- P3 编码器本体继续冻结。

只有 L0 证明某类证据具有跨事件信号后，才能训练对应适配器。

### L2：替换或解冻整个编码器

此时它已是新的父模型，必须重新训练并重新生成 LocCC、S2、Oracle 标签与 Memory。不能沿用旧 P3 的效用标签，也不能把结果称为同一 P3 上的插件增益。

## 5. RGB 与 SAR 不应做对称处理

灾前 RGB 更适合提供屋顶形状、轮廓、邻接和建筑支持域；灾后 SAR 更适合提供散射、纹理异常与损毁变化。两者成像机制不同，且时间也不同，因此不能在全图强制特征相等。

推荐：

```text
RGB branch: Stage3 detail + Stage4 semantics → geometry/anchor token
SAR branch: Stage3 texture + Stage4 semantics → damage/change token
cross-modal: local shiftable matching + confidence/entropy
relation: compare within-RGB and within-SAR relations
```

跨模态模块只在修改簇及其 halo 内运行，并允许拒绝；低置信度时回退 fixed S2 或 P3。

## 6. 本地 2024–2026 插件分级

### A：适合改造为局部证据

1. **HFP（2025）**：用于 RGB Stage3 的小建筑高频与边界 token；不直接覆盖 P3 特征。
2. **Edge-Gaussian/Scharr（2025）**：RGB 侧提取结构边缘，SAR 侧提取稳健梯度/残差，适合构成 boundary token。
3. **RelationAwareAttention（2025）**：只用于数量很少的修改簇/角色 token；须在无参数 relation difference 先通过后训练。
4. **CrossScaleBoundaryFusion（项目安全版）**：Stage3 细节对 Stage4 语义做零初始化残差；只在边界诊断支持时使用。

### B：只做独立对照

1. **FSA（2024）**：低/高频和横/纵分解合理，但原融合后接口已出现增益约零且门几乎未打开。只能在 SAR Stage3 或修改簇局部 token 上作为全新接口重测，不能把旧实验延长训练。
2. **RCM（2024）**：矩形/轴向上下文适合建筑，但原接口已显著下降。若再测，只能作为 RGB 几何分支对照，不能与 FSA 联合。
3. **KernelSelectiveFusionAttention（2025）**：可作为 SAR 多尺度专家对照，需改为 GroupNorm 和零初始化残差。
4. **DiffAttention（2025）**：只允许在 Top-R Memory token 上运行，避免全图二次复杂度与错误全局交互。

### C：当前不采用

- Lightweight Cross Attention：原任务是低照度亮度/色彩互补，不具备光学–SAR配准或变化假设；
- 本地 DeformableInteractiveAttention：代码没有完整可变形采样偏移机制；
- MaskAttention：随机掩码不适合作为稳定、可复现的科学证据；
- TFFA：本地单文件的 DoG/小波分支实质是普通可学习卷积，且没有恒等残差；
- CAFR：全局池化会丢失修改簇位置，且原实现没有安全恒等初始化；
- StripModule：已用于 RGID 几何分支，但 Split Oracle 上限近零，不应继续进入主线。

## 7. 外部论文与代码的可迁移部分

### M2CD

官方实现把光学–SAR变化检测建模为模态专家与自蒸馏。可迁移的是“模态专用专家”，不是再次蒸馏 S2；项目已经证明 S2 蒸馏不值得继续。

### HFA-PANet

官方实现使用非孪生、层级特征对齐。可迁移的是独立编码和逐层证据；本项目应把对齐限制在修改簇邻域，并用置信度门控，避免把真实损毁当成域差异消掉。

### FlowMamba / MCD-Mamba / ChangeMamba

这些方法说明状态空间模型可用于长距离遥感结构和多模态变化建模。当前更适合作为第二父模型或外部基线，不适合作为 P3 后的小插件，因为替换骨干会同时改变定位、候选生成和选择标签。

### 3MOS

其价值是检验多源、多分辨率、多场景的光学–SAR匹配泛化。可借鉴其跨源/跨场景意识来设计 Memory 的留一事件检索与位移置信度，而不是直接迁移类别头。

### C-DiffSET

可迁移不确定性/方差抑制思想：当 SAR–RGB对应不可靠时降低证据权重。完整 SAR→EO 扩散转换计算量高、任务链过长，不作为首轮方案。

### RAC、Retrieval-Augmented Image Recognition 与 RIO

- RAC 支持把外部非参数 Memory 用于长尾分类；
- Retrieval-Augmented Image Recognition 支持学习过滤无关邻居；
- RIO 的对象级 episodic memory 更适合训练期的稀有类别回放。

对本项目而言，应先做精确 kNN 的无参数审计；只有检索本身跨事件有效，才增加单层 reranker 或训练期队列。

## 8. 推荐架构

```text
Frozen P3
 ├─ RGB Stage3/4 ── role pooling ─┐
 ├─ SAR Stage3/4 ── role pooling ─┤
 └─ logits → LocCC → fixed S2     │
                    ↓             │
             modification clusters
                    ├─ change token
                    ├─ anchor token
                    └─ boundary token
                          │
              local matching / relation
                          │
              cross-event direction-balanced Memory
                          │
       utility score + uncertainty + abstention
                          │
        accept S2 cluster / revert cluster to P3
```

Memory 必须排除同图像和同事件，并按修改方向平衡；邻居不足、相似度低、标签熵高或事件意见不一致时 abstain。第一版只训练 Utility，不同时训练 Category Memory。

## 9. 实验顺序

1. **C0 Cluster Oracle**：判断修改簇粒度是否明显超过 Building Oracle。
2. **CM-E0 spatial**：方向、面积、边界距离、change/anchor概率差。
3. **CM-E0 RGB**：增加 RGB 边界/高频 token。
4. **CM-E0 SAR**：增加 SAR 稳健纹理 token。
5. **CM-E0 matching**：局部可位移匹配的 residual/confidence/entropy/offset。
6. **CM-E0 relation**：模态内关系差。
7. **Memory-U0**：留一事件、方向平衡、同事件排除的无参数检索。
8. **Memory-U1**：仅在 U0 通过后训练 Logistic/两层 MLP；默认保留 fixed S2，只高置信拒绝有害簇。
9. **第二父模型**：使用 H6-A2 native SAR 或独立 FlowMamba/M2CD 设置复验机制，不与主线联合调参。

若 C0 相对 Building Oracle不足 +0.10pp，停止修改簇选择，不建设 Utility Memory；转而改进 P3 的原生 Damaged 表征。若 C0 通过但 E0/U0 不通过，说明粒度正确但现有可观测特征不足，不能用大 Transformer 强行拟合。

## 10. 必须报告的对照

```text
B0 P3
B1 P3 + fixed S2
B2 Building Oracle
B3 Cluster Oracle
E0 spatial-only
E1 + RGB-only
E2 + SAR-only
E3 co-located direct difference（负对照）
E4 + local shiftable matching
E5 + relation difference
E6 shuffled labels / shuffled positions
M0 cross-event kNN
M1 same-event retrieval（仅泄漏诊断）
```

准入建议：formal AUROC ≥ 0.60、event-macro ≥ 0.58、至少 10/14 事件方向正确、置换 p<0.05，且超过 spatial-only/shuffle 至少 0.02 AUC。最终选择器还必须相对 fixed S2 至少 +0.30pp mIoU、bootstrap 下界大于 0、Damaged F1 不下降并恢复至少 25% Oracle 空间。

## 11. 论文创新定位

不能写成“加入 HFP/FSA/Attention/Memory”。更合理的主张是：

> 将建筑上下文产生的局部修改显式建模为方向条件化的候选簇，利用角色感知的光学–SAR局部证据与跨事件检索来估计修改效用，并以拒绝机制安全地在 P3 和 S2 之间选择。

候选题目：

1. Retrieve to Verify: Modification-Cluster Utility Memory for Multimodal Building Damage Assessment
2. Role-Aware Cross-Modal Memory for Safe Pseudo-Building Refinement
3. Direction-Conditioned Local Retrieval for Optical–SAR Damage Correction
4. From Context Propagation to Trusted Correction in Multimodal Building Damage Mapping
5. Abstaining Cross-Event Memory for Reliable Optical–SAR Building Damage Assessment

## 12. 依据与代码

- BRIGHT：<https://github.com/ChenHongruixuan/BRIGHT>
- M2CD：<https://github.com/circleLZY/M2CD>
- HFA-PANet：<https://github.com/TongfeiLiu/HFA-PANet-for-MCD>
- FlowMamba：<https://github.com/flying318/FlowMamba>
- MCD-Mamba：<https://github.com/Ci2Lab/MCD-Mamba>
- ChangeMamba：<https://github.com/VisionMamba/MambaCD>
- DSTCD：<https://github.com/Lucky-DW/DSTCD>
- 3MOS：<https://github.com/3M-OS/3MOS>
- C-DiffSET：<https://github.com/KAIST-VICLab/C-DiffSET>
- RAC：<https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html>
- Retrieval-Augmented Image Recognition：<https://openaccess.thecvf.com/content/CVPR2023/html/Iscen_Improving_Image_Recognition_by_Retrieving_From_Web-Scale_Image-Text_Data_CVPR_2023_paper.html>
- RIO：<https://github.com/NVlabs/RIO>

## 13. 新颖性、难度与风险

- 任务匹配度：高。直接针对 fixed S2 的有益/有害修改混合，而不是泛化地增加注意力。
- 新颖性：中高。关键在修改簇效用、角色 token、跨事件 Memory 和 abstention 的组合，不在单个插件。
- 实现难度：E0 中等，Memory-U0 中等，U1 中等，替换骨干高。
- 最大风险：Cluster Oracle空间不足、检索学到事件捷径、SAR斑点导致伪匹配、GT写入查询 Key、同时改骨干导致归因失效。
