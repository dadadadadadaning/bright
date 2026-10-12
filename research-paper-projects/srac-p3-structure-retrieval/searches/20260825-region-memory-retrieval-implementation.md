# 超像素区域记忆与检索分类实现论文

- 检索日期：2026-08-25
- 检索关键词：superpixel region memory bank retrieval augmented classification long-tail segmentation cross-domain memory
- 数据源：IGARSS 2026录用名单、CVF Open Access、arXiv、论文官方GitHub

## P001 SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment

- 年份：2026
- 收录于：IGARSS 2026
- 作者：Siheng Ning, Yuan Li, Xiaodan Zhang, Jianbin Jiao
- 中文摘要：面向灾前光学与灾后SAR建筑损毁评估，通过可学习超像素、区域嵌入、外部记忆检索与元数据条件分类处理结构不一致和稀有损毁类别。
- 有无代码：未证实
- 原文链接：https://2026.ieeeigarss.org/papers/accepted_papers.php
- PDF：未获取
- 相关性说明：与BRIGHT任务和用户提供结构图直接匹配，但全文和实现细节尚未公开。

## P002 Superpixel Sampling Networks

- 年份：2018
- 收录于：ECCV 2018
- 作者：Varun Jampani, Deqing Sun, Ming-Yu Liu, Ming-Hsuan Yang, Jan Kautz
- 中文摘要：提出可微分超像素采样网络，使任务特定的超像素划分可以端到端学习并接入下游视觉网络。
- 有无代码：未证实
- 原文链接：https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.pdf
- PDF：开放
- 相关性说明：是SRAC图中Differentiable SSN机制的直接基础。

## P003 Superpixel Segmentation With Fully Convolutional Networks

- 年份：2020
- 收录于：CVPR 2020
- 作者：Fengting Yang, Qian Sun, Hailin Jin, Zihan Zhou
- 中文摘要：以全卷积网络直接预测局部超像素关联，并用ASA、边界召回/精度和紧致度评估区域质量。
- 有无代码：有: https://github.com/fuy34/superpixel_fcn
- 原文链接：https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.pdf
- PDF：开放
- 相关性说明：提供了在没有SRAC代码时最可靠的公开超像素替代实现。

## P004 Inflated Episodic Memory With Region Self-Attention for Long-Tailed Visual Recognition

- 年份：2020
- 收录于：CVPR 2020
- 作者：Linchao Zhu, Yi Yang
- 中文摘要：为长尾视觉识别建立每类独立的代表特征记忆，并结合区域自注意力改善尾部类别表示。
- 有无代码：未证实
- 原文链接：https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_Inflated_Episodic_Memory_With_Region_Self-Attention_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.html
- PDF：开放
- 相关性说明：支撑每类等容量Memory Bank，而不是让Intact样本淹没Damaged样本。

## P005 Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation

- 年份：2022
- 收录于：CVPR 2022
- 作者：Tianfei Zhou, Meijie Zhang, Fang Zhao, Jianwu Li
- 中文摘要：以区域级语义记忆保存跨图像对象模式，通过区域对比和非参数注意力聚合数据集级上下文，避免昂贵且噪声敏感的像素关系建模。
- 有无代码：未证实
- 原文链接：https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Regional_Semantic_Contrast_and_Aggregation_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf
- PDF：开放
- 相关性说明：直接回答为什么区域记忆比像素记忆更高效、更稳健。

## P006 Retrieval Augmented Classification for Long-Tail Visual Recognition

- 年份：2022
- 收录于：CVPR 2022
- 作者：Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, Anton van den Hengel
- 中文摘要：为标准分类器增加非参数外部记忆和检索分支，使稀有类别知识显式保存在记忆中，再与主分类器联合预测。
- 有无代码：未证实
- 原文链接：https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html
- PDF：开放
- 相关性说明：是使用训练样本记忆改善Damaged长尾识别的直接分类依据。

## P007 Pin the Memory: Learning To Generalize Semantic Segmentation

- 年份：2022
- 收录于：CVPR 2022
- 作者：Jin Kim, Jiyoung Lee, Jungin Park, Dongbo Min, Kwanghoon Sohn
- 中文摘要：学习跨域稳定的类别记忆，在模拟未知域的元学习过程中训练记忆读取和更新，以降低目标域表征歧义。
- 有无代码：有: https://github.com/Genie-Kim/PintheMemory
- 原文链接：https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Pin_the_Memory_Learning_To_Generalize_Semantic_Segmentation_CVPR_2022_paper.pdf
- PDF：开放
- 相关性说明：为事件留出Memory和跨灾害稳定性验证提供公开实现依据。

## P008 Improving Image Recognition by Retrieving From Web-Scale Image-Text Data

- 年份：2023
- 收录于：CVPR 2023
- 作者：Ahmet Iscen, Alireza Fathi, Cordelia Schmid
- 中文摘要：使用近邻检索和注意力记忆模块对检索样本赋权，抑制无关邻居并保留对当前查询真正有帮助的支持样本。
- 有无代码：未证实
- 原文链接：https://openaccess.thecvf.com/content/CVPR2023/papers/Iscen_Improving_Image_Recognition_by_Retrieving_From_Web-Scale_Image-Text_Data_CVPR_2023_paper.pdf
- PDF：开放
- 相关性说明：对应SRAC图中Top-R邻居经过Cross-Attention重新加权，而不是简单kNN投票。

---

需要下载论文时，请输入编号，例如：`P002,P003,P005-P008`。

