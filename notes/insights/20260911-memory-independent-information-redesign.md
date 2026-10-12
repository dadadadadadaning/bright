# Memory独立信息重设计方案

## 研究问题

完整建筑Memory在建筑级分类上有效，但真实邻居不优于随机、打乱或零支持；P3特征Router也优于加入Memory的Router。目标不是继续增加Top-K，而是让Memory包含并利用P3自身表征之外的跨事件信息。

## 核心诊断

当前库已有约10,577条平衡病例，数量不是首要瓶颈。主要问题是：检索键大量复用P3特征；一栋建筑被压缩成均值或少量角色Token；类别对比目标压缩了Damaged/Destroyed类内差异；事件与类别双重均衡过强。独立贡献必须用条件增益定义：在已经知道P3输出后，Memory特征仍能提高留出事件性能。

## 数据量重设计

1. 先做容量曲线：每个事件-类别保留32、64、128、256、512及全部病例，Top-K固定为8。
2. 同容量比较随机选择、置信度选择和k-center多样性选择。若256以后饱和，停止扩容。
3. 不重复过采样少数事件。按灾害类型、类别、建筑面积、长宽比、SAR质量和配准质量建立覆盖单元。
4. 外部数据只有在标签体系、分辨率和灾前RGB/灾后SAR组合兼容时才加入；否则只能用于预训练，不直接进入标注Memory。

## 独立特征空间

检索键不再使用P3-global作为主成分，拆成三部分：

- Identity/Structure Key：灾前RGB轮廓、屋顶边缘、尺度、方向和形态；
- Damage-Evidence Key：灾后SAR log幅度、局部中位数/MAD、方向梯度、高低频散射及建筑内相对异常；
- Quality/Context Key：配准置信度、SAR质量、成像元数据、灾害类型和区域属性。

先按Structure和Quality粗筛可比建筑，再以Damage Evidence和角色对齐重排。训练目标预测P3的残余错误或条件效用，而不是重复预测完整类别。

## 建筑卡重设计

每张卡保留：

- 建筑级结构Token；
- core/ring/edge/stable/anomaly五类空间Token及坐标；
- RGB结构patch和SAR散射patch；
- P3预测仅作为查询状态，不进入独立检索键；
- GT类别分布、纯度和P3错误类型仅存在训练库Value中；
- 配准、遮挡、面积和SAR质量置信度。

检索后不广播类别概率，而是将对齐的支持Token作为K/V，通过逐Token置信门增强当前建筑特征。

## 实验顺序

### MI-D0 条件独立信息审计

比较P3、独立卡、P3+独立卡、P3+打乱独立卡。只有P3+独立卡稳定超过P3且打乱后消失，才证明新增信息存在。

### MI-D1 容量与选择策略

运行容量曲线和随机/k-center对照。选择第一个达到性能平台的最小容量，禁止按最终测试结果选容量。

### MI-D2 建筑卡消融

依次加入RGB结构、SAR物理证据、相对异常、质量元数据和空间Token，记录条件AUC、邻居纯度、稀有类Recall@8及真实-打乱差值。

### MI-E1 条件残差检索学习

使用自然分布采样，在batch内保证跨事件正样本和困难负样本，不使用强逆频率事件权重。目标包含P3残差分类、跨事件对比、类内多原型和支持多样性约束。

### MI-E2 角色对齐交叉注意力

仅在E1通过后，把支持Token接入P3特征；使用逐建筑/逐Token门和零初始化残差，比较真实、随机、打乱、self和zero支持。

## 准入标准

- P3+Memory相对P3的条件Macro-AUC至少提高0.01，建议目标0.02；
- 真实支持相对随机和打乱支持至少提高0.50个百分点；
- 邻居纯度至少0.70；
- Damaged与Destroyed Recall@8均不低于P3；
- 至少10/14事件不下降；
- 像素阶段相对P3+fixed S2的mIoU至少提高0.30个百分点，Damaged F1不下降；
- Val450和Test仅在全部开发门通过并冻结后访问。

## 风险

如果独立原始卡在P3条件下仍没有增益，则问题不是Memory容量，而是数据本身没有足够的可迁移跨事件相似性。此时应停止检索路线，转向已经被实验支持的P3困难感知Router。

## 可形成的题目

1. Cross-Event Structure-and-Scattering Memory for RGB-SAR Building Damage Assessment：以结构键、散射键和角色Token支持为主线。
2. Conditional Residual Retrieval for Multimodal Disaster Building Damage Assessment：以预测P3残余错误而非重复分类为主线。
3. Quality-Aware Hierarchical Building Memory under Cross-Event Domain Shift：以质量过滤、分层检索和跨事件验证为主线。
4. Role-Aligned Support Memory for Spatially Consistent Building Damage Segmentation：以角色patch对齐与逐Token交叉注意力为主线。

## 依据与迁移说明

- Pin the Memory（CVPR 2022）通过类别Memory、虚拟域偏移和类内聚合/Memory分离学习跨域类别知识；可迁移的是训练期域泛化与Memory约束，而不是直接概率写回：https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Pin_the_Memory_Learning_To_Generalize_Semantic_Segmentation_CVPR_2022_paper.html
- SCCA（ICCV 2023）指出单原型压缩会丢失空间细节，并使用对齐后的支持patch进行交叉注意力；可迁移的是角色patch对齐，而不是其few-shot任务设定：https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Self-Calibrated_Cross_Attention_Network_for_Few-Shot_Segmentation_ICCV_2023_paper.html
- Semantic-Aware Domain Generalized Segmentation（CVPR 2022）同时约束类内紧凑和类间可分性；可迁移的是类别条件的域对齐，风险是过强压缩Damaged/Destroyed类内结构：https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html
- Universal Domain Adaptation for Semantic Segmentation（CVPR 2025）采用域特定的多原型区分类内模式，支持用多原型替代每类单中心：https://openaccess.thecvf.com/content/CVPR2025/html/Choe_Universal_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2025_paper.html
