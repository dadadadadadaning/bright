# 超像素、图像匹配与检索增强在 RGB–SAR BDA 中的文献检索

- 检索日期：2026-08-26
- 项目：`backbone-memory-retrieval-bright`
- 研究问题：为什么 SRAC 风格检索可能有效；怎样把超像素和 IM4CD 的图像匹配思想迁移到当前 P3/RM6-P，而不重复硬 kNN 和模糊软分区的失败。
- 证据边界：SRAC 当前可核验的是 IGARSS 2026 录用信息、公开结构图和实验室新闻稿；未检索到可公开核验的全文或官方代码，因此不能把总增益归因给某个单独模块。

## P001 — SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment

- Authors: Siheng Ning, Yuan Li, Xiaodan Zhang, Jianbin Jiao
- Venue/Year: IGARSS 2026, accepted paper 2851
- Evidence: [IGARSS accepted papers](https://2026.ieeeigarss.org/papers/accepted_papers.php), [UCAS laboratory announcement](https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026)
- Verified facts: 官方页面确认论文被录用；公开新闻稿描述 VMambaSSN 超像素、外部标注库检索和 BRIGHT 上 75.65% mIoU、相对所述主流基线 +8.08 pp。
- Limitation: 未取得全文与官方代码，无法核验训练协议、对照基线、单模块消融和 8.08 pp 的具体来源。

## P002 — Multimodal Remote Sensing Change Detection: An Image Matching Perspective

- Authors: Hongruixuan Chen, Cuiling Lan, Jian Song, Damian Ibañez, Junshi Xia, Konrad Schindler, Naoto Yokoya
- Venue/Year: ISPRS Journal of Photogrammetry and Remote Sensing, 2026
- DOI: 10.1016/j.isprsjprs.2026.02.004
- Evidence: [publisher page](https://www.sciencedirect.com/science/article/pii/S0924271626000559)
- Code: 未核验到公开官方仓库
- Key point: 不再用共址像素差作为变化证据，而用局部模板匹配响应峰的位置偏移和稳健几何残差；再用超像素 CRF 的局部/非局部结构关系稠密化。
- Transfer value: 为当前“检索 Key 无法区分救援/伤害”提供新的、与事件外观弱耦合的匹配几何描述子。

## P003 — Superpixel Sampling Networks

- Authors: Varun Jampani, Deqing Sun, Ming-Yu Liu, Ming-Hsuan Yang, Jan Kautz
- Venue/Year: ECCV 2018
- Evidence: [paper](https://arxiv.org/abs/1807.10174), [official code](https://github.com/NVlabs/ssn_superpixels)
- Key point: CNN 像素嵌入、规则种子初始化、局部像素—种子关联、迭代种子更新、可选硬分配和连通性约束。
- Transfer value: 说明“可微超像素”不是一个 3×3 softmax；迭代、任务监督、硬化/连通性和下游目标共同决定区域质量。

## P004 — Retrieval Augmented Classification for Long-Tail Visual Recognition

- Authors: Alexander Long et al.
- Venue/Year: CVPR 2022
- Evidence: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html)
- Key point: 基础分类分支与外部记忆检索分支并行，检索负责补充尾部类别知识，而不是直接用 kNN 覆盖基础模型。
- Transfer value: 支持“P3 保留、Memory 是并行条件证据”的角色定义。

## P005 — Tip-Adapter: Training-Free Adaption of CLIP for Few-Shot Classification

- Authors: Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, Hongsheng Li
- Venue/Year: ECCV 2022
- Evidence: [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf), [code](https://github.com/gaopengcuhk/Tip-Adapter)
- Key point: 从训练样本构建 key-value cache，以残差方式更新已有模型知识，保留无训练安全基线。
- Transfer value: 支持 P3 锚定和零增量守卫，但不证明任意 cache key 都可靠。

## P006 — RIFT: Multi-Modal Image Matching Based on Radiation-Variation Insensitive Feature Transform

- Authors: Jiayuan Li, Qingwu Hu, Mingyao Ai
- Venue/Year: IEEE Transactions on Image Processing, 2020
- DOI: 10.1109/TIP.2019.2959244
- Evidence: [paper PDF](https://ljy-rs.github.io/web/Files/TIP2020.pdf)
- Key point: 用相位一致性检测结构点，用最大索引图描述非线性辐射差异下的多模态结构。
- Transfer value: 若冻结 P3 特征的局部余弦响应不足，可把 RIFT/相位结构描述子作为下一层受控替代，而不是立刻训练大网络。

## P007 — BRIGHT: A Globally Distributed Multimodal Building Damage Assessment Dataset

- Authors: Hongruixuan Chen et al.
- Venue/Year: Earth System Science Data, 2025
- DOI: 10.5194/essd-17-6217-2025
- Evidence: [official paper](https://essd.copernicus.org/articles/17/6217/2025/)
- Code/Data: [official repository](https://github.com/ChenHongruixuan/BRIGHT)
- Key point: 14 个灾害事件、跨地区和跨传感器差异显著；灾前光学适合建筑定位，灾后 SAR 提供结构损毁线索；跨事件泛化是核心挑战。
- Transfer value: 验证必须采用事件级审计，不能仅优化总体区域 Macro F1。

## 检索结论

1. 超像素的价值主要是形成结构一致的查询单元并减少检索数量，不会自动产生正确的损毁语义。
2. 成功的检索增强通常保留基础分类器，并把当前样本、支持样本及附加信息交给可学习的融合器；硬 kNN 替换不是代表性终态。
3. IM4CD 对当前最有价值的迁移不是 CRF 本身，而是“局部匹配响应峰与几何残差”这一 Key 证据。
4. 当前应先做无训练匹配证据审计；只有救援/伤害可分性和跨事件稳定性通过，才允许训练检索融合。
