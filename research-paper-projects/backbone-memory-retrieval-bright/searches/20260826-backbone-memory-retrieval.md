# 骨干网络与记忆检索增强分类论文检索结果

- 检索日期：2026-08-26
- 检索关键词：retrieval augmented classification、visual cache、long-tail recognition、selective classification、cross-event metric、RGB-SAR building damage
- 数据源：CVF Open Access、ECCV/ECVA、NeurIPS Proceedings、PMLR、OpenReview、arXiv、ScienceDirect、IGARSS 官方录用页、论文官方 GitHub

## P001 Retrieval Augmented Classification for Long-Tail Visual Recognition

- 年份：2022
- 收录于：CVPR 2022
- 作者：Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, Anton van den Hengel
- 中文摘要：提出 RAC，将标准图像编码器与查询非参数外部记忆的并行检索分支结合，用训练数据构建外部知识以增强长尾类别识别。
- 有无代码：未证实
- 原文链接：https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html
- PDF：开放获取
- 相关性说明：最直接支持“成熟骨干保留、检索作为并行分支、尾类知识显式存储”。

## P002 Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification

- 年份：2022
- 收录于：ECCV 2022
- 作者：Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, Hongsheng Li
- 中文摘要：从少样本训练集构建键值 Cache，以检索亲和度产生 Cache logits，并以残差方式更新冻结 CLIP 的先验预测；还提出仅微调 Cache keys 的版本。
- 有无代码：有：https://github.com/gaopengcuhk/Tip-Adapter
- 原文链接：https://arxiv.org/abs/2207.09519
- PDF：开放获取
- 相关性说明：说明 Memory 更适合作为受控残差证据，而不是直接替代强骨干。

## P003 Learning Customized Visual Models With Retrieval-Augmented Knowledge

- 年份：2023
- 收录于：CVPR 2023
- 作者：Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, Chunyuan Li
- 中文摘要：REACT 从大规模图文库检索目标域相关知识，在冻结原模型权重的同时只训练新增模块，使基础能力和领域定制解耦。
- 有无代码：未证实
- 原文链接：https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Learning_Customized_Visual_Models_With_Retrieval-Augmented_Knowledge_CVPR_2023_paper.html
- PDF：开放获取
- 相关性说明：支持冻结 P3、单独学习检索表征或校准器，而非反复解冻主干。

## P004 Supervised Contrastive Learning

- 年份：2020
- 收录于：NeurIPS 2020
- 作者：Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan
- 中文摘要：利用标签构造多正负对，使同类样本在嵌入空间聚集、异类样本分离，并报告对自然扰动和低数据条件的稳健性。
- 有无代码：有：https://github.com/HobbitLong/SupContrast
- 原文链接：https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
- PDF：开放获取
- 相关性说明：是 RM5 跨事件同类拉近的理论来源，但原论文并不保证保留既有检索邻域。

## P005 Similarity-Preserving Knowledge Distillation

- 年份：2019
- 收录于：ICCV 2019
- 作者：Frederick Tung, Greg Mori
- 中文摘要：不要求学生逐点模仿教师特征坐标，而是保持一个批次内样本两两相似关系，从而转移教师的关系结构。
- 有无代码：未证实
- 原文链接：https://arxiv.org/abs/1907.09682
- PDF：开放获取
- 相关性说明：提示 RM5-B 的逐点 L2 保持过于僵硬，更合理的是保持相似度矩阵或邻域排序。

## P006 Balanced Meta-Softmax for Long-Tailed Visual Recognition

- 年份：2020
- 收录于：NeurIPS 2020
- 作者：Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, Hongsheng Li
- 中文摘要：从训练与测试标签分布偏移出发提出 Balanced Softmax，并用 Meta Sampler 估计更合适的类别采样率。
- 有无代码：有：https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
- 原文链接：https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
- PDF：开放获取
- 相关性说明：直接说明“库容量均衡”不能自动等价为“预测后验均衡”，必须处理类别先验。

## P007 Long-Tail Learning via Logit Adjustment

- 年份：2021
- 收录于：ICLR 2021
- 作者：Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar
- 中文摘要：建立长尾学习的统计框架，根据标签频率进行训练期或后处理 logit 调整，以纠正类别分布导致的分类偏差。
- 有无代码：未证实
- 原文链接：https://openreview.net/forum?id=37nvvqkCo5
- PDF：开放获取
- 相关性说明：为 Memory 类别得分加入训练集先验提供直接依据。

## P008 SelectiveNet: A Deep Neural Network with an Integrated Reject Option

- 年份：2019
- 收录于：ICML 2019
- 作者：Yonatan Geifman, Ran El-Yaniv
- 中文摘要：把拒绝选项与分类器联合训练，直接优化给定覆盖率下的风险，而不是只对预训练模型置信度做阈值处理。
- 有无代码：有：https://github.com/geifmany/selectivenet
- 原文链接：https://proceedings.mlr.press/v97/geifman19a.html
- PDF：开放获取
- 相关性说明：对应你的核心需求——Memory 必须能够拒绝不可靠建议，而非每次都输出损毁类别。

## P009 Multimodal Remote Sensing Change Detection: An Image Matching Perspective

- 年份：2026
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
- 作者：Hongruixuan Chen, Cuiling Lan, Jian Song, Damian Ibañez, Junshi Xia, Konrad Schindler, Naoto Yokoya
- 中文摘要：IM4CD 放弃异构模态逐像素比较，使用局部模板匹配响应峰偏移表达变化强度，并把匹配、配准与 CRF 稠密变化估计结合。
- 有无代码：未证实
- 原文链接：https://doi.org/10.1016/j.isprsjprs.2026.02.004
- PDF：开放获取
- 相关性说明：支持把 RGB–SAR 异构性作为局部匹配问题，但你已有实验表明该信号暂不适合作为检索键增量。

## P010 SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment

- 年份：2026
- 收录于：IGARSS 2026
- 作者：Siheng Ning, Yuan Li, Xiaodan Zhang, Jianbin Jiao
- 中文摘要：公开录用信息表明该工作以可学习超像素组织建筑结构，并使用外部知识检索增强多模态灾害建筑损毁分类；完整论文细节尚未从公开正式全文核验。
- 有无代码：未证实
- 原文链接：https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026
- PDF：未获取
- 相关性说明：是任务最接近的工作，因此论文创新不能只写“超像素＋Memory”，必须突出跨事件先验校准与安全拒绝机制。

---

需要下载开放论文时，可提供编号，例如：`P001,P002,P005,P008`。

