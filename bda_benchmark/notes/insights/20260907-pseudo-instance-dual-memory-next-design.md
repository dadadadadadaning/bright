# 伪实例双 Memory 与跨模态实例库设计

- 日期：2026-09-07
- 方向：灾前 RGB、灾后 SAR 多模态建筑损毁评估
- 问题：在 P3、LocCC、A6 与 fixed S2 基础上，构造不会重复 PI-M0 整栋平均池化失败的实例库

## 当前实现边界

已验证有效的是：P3 建筑定位、四连通且面积不小于 32 的 LocCC、A6 区域可靠性、S1/fixed S2 建筑上下文，以及 Building Oracle 的理论选择空间。

已实现但未进入有效主线的是：RGID 分裂、SMP 修改模式、Cluster Oracle、Multi-Prototype Context、BGMP 类别原型与 CMR 检索。RGID 的 Split Oracle 上限近零；PI-M0 整栋 Utility Memory 的正式跨事件 AUC 为 0.5176。

当前 LocCC 只是预测建筑支持域，不是真实单栋实例。它没有稳定实例 ID、真实轮廓监督或已验证的实例分类头。

## 三类已有 Memory

1. PI-M0：整栋 LocCC 的 utility positive/negative 检索。失败原因是整栋均值丢失了 S2 修改的位置、方向和内部角色。
2. CMR：Damaged/Intact 案例检索，回答“更像哪种损毁”，不回答“S2 修改是否有益”。EC2 排除同事件后曾有较强诊断信号，但尚未安全转化为最终预测。
3. BGMP：训练期 I/D/X 类别 EMA 原型，适合长尾表征和辅助损失，不适合直接广播类别。

## 推荐方案：Modification-Cluster Dual Memory

### Memory-U：修改效用库

主任务是判断一个 S2 修改簇接受 S2 还是回退 P3。每条记录包括：

```text
identity: split_sha / p3_sha / event / image / LocCC / cluster
key:      P3->S2方向、空间拓扑、边界距离、概率行为、RGB/SAR角色token、局部匹配、关系差
value:    corrected / harmed / neutral / utility_ratio（仅训练库可见）
metadata: area_bin / aspect_bin / event / disaster_type / retrieval_quality
```

GT 只能写入 train-only value，不能进入查询 Key。

### Memory-C：损毁案例库

副任务是补充稀有类别证据，拆成：

```text
Occurrence Memory: Intact vs Affected
Severity Memory:   Damaged vs Destroyed
```

它不能直接覆盖 P3；只在 P3 低 margin、邻居高度一致且跨模态局部证据通过时给出小的零初始化 logit 残差。

## 角色 Token

每个修改簇不再保存一个整栋均值，而保存：

```text
change token  : S2 实际修改区
anchor token  : 同一 LocCC 内未修改且 P3 高置信区
boundary token: 修改区与建筑边界/邻域交界带
```

各角色分别池化 RGB/SAR Stage3/4 统计、P3/S2 概率、change-anchor 差、局部匹配置信度和模态内关系。这样 Memory 检索的是相似修改机制，而不是相似建筑外观。

## 检索与拒绝

正式检索必须排除同图像和同事件，按事件、效用标签和修改方向平衡，每个事件限制邻居数。输出 positive/negative affinity、margin、utility 均值/方差、标签熵、事件一致性和 availability。邻居不足、相似度低或意见不一致时 abstain，保持 fixed S2。

第一版不使用 Transformer，只做精确余弦检索和固定统计。

## 实验阶梯

1. C0 Cluster Oracle：无训练。只有显著超过 Building Oracle 才建设 Memory-U。
2. U-M0 无参数检索：比较 whole-building mean、修改模式、角色 token、local matching、relation difference、组合、shuffled label 和 same-event 泄漏诊断。
3. U-M1 低容量 Verifier：只有 U-M0 通过才训练 Logistic/两层 MLP，默认保留 S2，只高置信拒绝有害修改。
4. C-M0 类别库审计：Occurrence 与 Severity 分开报告跨事件 AUC。
5. C-M1 训练期动态队列：若推理 Memory 不安全，按事件×类别×角色维护 detached Teacher FIFO，用于跨事件正对和困难负对，不直接改预测。

U-M0 门槛：AUROC 不低于 0.60、event-macro 不低于 0.58、至少 10/14 事件方向正确、availability 不低于 0.95、超过 whole-building mean 至少 0.03、超过 geometry/shuffle 至少 0.02、置换检验 p 小于 0.05。

U-M1 最终要求：相对 fixed S2 至少增加 0.30pp mIoU、bootstrap 下界大于 0、Damaged F1 不下降、至少 10/14 事件不下降、恢复 Cluster Oracle 空间至少 25%。

## 可迁移工作

- SRAC（IGARSS 2026）：迁移局部区域样本库、Top-R 邻居和结构元数据。尚无正式全文/官方代码，不臆测具体损失和容量。
- RAC（CVPR 2022）：迁移非参数外部 Memory 服务长尾知识。
- Retrieval-Augmented Image Recognition（CVPR 2023）：迁移对无关邻居降权；仅在无参数检索通过后使用单层 reranker。
- RIO（ICML 2021）：迁移对象级 Memory replay，适合作为训练期事件均衡困难样本队列。
- M2CD、HFA-PANet、IM4CD/3MOS：用于构造模态专用、局部可位移匹配和关系感知的 Key，不直接替代损毁分类器。
- C-DiffSET：迁移方差/置信度拒绝思想，不迁移完整扩散模型。

本地插件中，RelationAwareAttention、HFP、Edge-Gaussian/Scharr、CrossScaleBoundaryFusion 可作为角色 token 或 verifier 候选；DiffAttention 只能用于少量 Top-R token。FSA/RCM 旧接口、随机 MaskAttention、低照度 Lightweight Cross Attention 和全图 Cross-Attention 当前不采用。

## 创新、难度与风险

- 任务匹配：高。
- 创新度：中高；核心是修改簇角色 token、效用/类别双 Memory、跨事件检索和拒绝，而不是简单 kNN。
- 难度：U-M0 中等，U-M1 中等，动态队列中高，完整 Cross-Attention 高。
- 风险：伪实例粘连、GT 泄漏、事件捷径、相似外观但不同效用、Destroyed 样本不足、Memory 覆盖正确 P3。

## 建议题目

1. Retrieve to Verify: Modification-Cluster Utility Memory for Multimodal Building Damage Assessment
2. Role-Aware Pseudo-Instance Memory for Safe RGB-SAR Damage Refinement
3. Dual Memory for Context Utility and Rare Damage Recognition under Noisy Building Instances
4. Cross-Event Local Evidence Retrieval for All-Weather Building Damage Mapping
5. From Pseudo Buildings to Trusted Cases: Abstaining Retrieval for Heterogeneous Damage Assessment

## 证据来源

- SRAC：<https://2026.ieeeigarss.org/papers/accepted_papers.php>
- BRIGHT：<https://github.com/ChenHongruixuan/BRIGHT>
- RAC：<https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html>
- Retrieval-Augmented Image Recognition：<https://openaccess.thecvf.com/content/CVPR2023/html/Iscen_Improving_Image_Recognition_by_Retrieving_From_Web-Scale_Image-Text_Data_CVPR_2023_paper.html>
- RIO：<https://github.com/NVlabs/RIO>
- M2CD：<https://github.com/circleLZY/M2CD>
- 3MOS：<https://github.com/3M-OS/3MOS>
- C-DiffSET：<https://github.com/KAIST-VICLab/C-DiffSET>

