# 面向 BRIGHT RGB–SAR 建筑损毁的 Memory 设计论文检索

- 检索日期：2026-08-28
- 检索方向：retrieval-augmented classification、key-value cache、training memory bank、multi-proxy metric learning、retrieve-and-rerank、domain generalization memory
- 数据源：CVF Open Access、ECVA、arXiv

1. **P001 Retrieval Augmented Classification for Long-Tail Visual Recognition**
   - 年份：2022
   - 收录于：CVPR 2022
   - 作者：Alexander Long 等
   - 中文摘要：使用标准分类编码器与并行非参数检索分支，从预编码图像和标签Memory中补充长尾类别知识。
   - 有无代码：未证实
   - 原文链接：https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html
   - PDF状态：开放获取
   - 相关性：代表推理阶段外部Memory与主模型并行融合。

2. **P002 Tip-Adapter: Training-Free Adaption of CLIP for Few-Shot Classification**
   - 年份：2022
   - 收录于：ECCV 2022
   - 作者：Renrui Zhang 等
   - 中文摘要：由少样本训练集构建Key-Value Cache，以查询—Key亲和度产生Cache logits，并残差式更新冻结模型的先验预测。
   - 有无代码：有：https://github.com/gaopengcuhk/Tip-Adapter
   - 原文链接：https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf
   - PDF状态：开放获取
   - 相关性：代表冻结主干、缓存亲和度和受控残差融合。

3. **P003 Cross-Batch Memory for Embedding Learning**
   - 年份：2020
   - 收录于：CVPR 2020
   - 作者：Xun Wang, Haozhi Zhang, Weilin Huang, Matthew R. Scott
   - 中文摘要：保存前序批次的近期嵌入，将困难样本挖掘扩展到当前mini-batch之外。
   - 有无代码：有：https://github.com/MalongTech/research-xbm
   - 原文链接：https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html
   - PDF状态：开放获取
   - 相关性：代表Train-only动态队列，而非推理分类器。

4. **P004 Semi-Supervised Semantic Segmentation With Pixel-Level Contrastive Learning From a Class-Wise Memory Bank**
   - 年份：2021
   - 收录于：ICCV 2021
   - 作者：Iñigo Alonso 等
   - 中文摘要：每类维护FIFO Memory，只写入教师产生的高置信度、高质量特征，并用于端到端像素级对比学习。
   - 有无代码：有：https://github.com/Shathe/SemiSeg-Contrastive
   - 原文链接：https://openaccess.thecvf.com/content/ICCV2021/html/Alonso_Semi-Supervised_Semantic_Segmentation_With_Pixel-Level_Contrastive_Learning_From_a_Class-Wise_ICCV_2021_paper.html
   - PDF状态：开放获取
   - 相关性：代表类别队列、质量过滤、教师更新和测试零Memory开销。

5. **P005 A Style and Semantic Memory Mechanism for Domain Generalization**
   - 年份：2021
   - 收录于：ICCV 2021
   - 作者：Yang Chen, Yu Wang, Yingwei Pan, Ting Yao, Xinmei Tian, Tao Mei
   - 中文摘要：分别构造语义Memory和域风格Memory，并使用动量Memory编码器缓解训练中特征快速漂移。
   - 有无代码：未证实
   - 原文链接：https://openaccess.thecvf.com/content/ICCV2021/html/Chen_A_Style_and_Semantic_Memory_Mechanism_for_Domain_Generalization_ICCV_2021_paper.html
   - PDF状态：开放获取
   - 相关性：代表把事件/成像风格与损毁语义分开保存。

6. **P006 SoftTriple Loss: Deep Metric Learning Without Triplet Sampling**
   - 年份：2019
   - 收录于：ICCV 2019
   - 作者：Qi Qian, Lei Shang, Baigui Sun, Juhua Hu, Hao Li, Rong Jin
   - 中文摘要：指出真实类别可包含多个局部簇，为每个类别学习多个中心而非单一中心。
   - 有无代码：未证实
   - 原文链接：https://openaccess.thecvf.com/content_ICCV_2019/html/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.html
   - PDF状态：开放获取
   - 相关性：代表多原型类别记忆，可描述Damaged内部多种损毁模式。

7. **P007 Hierarchical Proxy-Based Loss for Deep Metric Learning**
   - 年份：2022
   - 收录于：WACV 2022
   - 作者：Zhibo Yang, Muhammet Bastan, Xinliang Zhu, Douglas Gray, Dimitris Samaras
   - 中文摘要：构建多层代理金字塔，同时学习细粒度类别差异与高层共享属性。
   - 有无代码：未证实
   - 原文链接：https://openaccess.thecvf.com/content/WACV2022/html/Yang_Hierarchical_Proxy-Based_Loss_for_Deep_Metric_Learning_WACV_2022_paper.html
   - PDF状态：开放获取
   - 相关性：代表Occurrence与Severity分层关系，而非单一三类距离空间。

8. **P008 Correlation Verification for Image Retrieval**
   - 年份：2022
   - 收录于：CVPR 2022
   - 作者：Seongwon Lee, Hongje Seong, Suhyeon Lee, Euntai Kim
   - 中文摘要：先用全局表示粗检索，再构造Query与候选的多尺度稠密相关体，由相关性网络输出精细相似度用于重排序。
   - 有无代码：有：https://github.com/sungonce/CVNet
   - 原文链接：https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Correlation_Verification_for_Image_Retrieval_CVPR_2022_paper.html
   - PDF状态：开放获取
   - 相关性：直接对应Mean Top-20召回后使用局部Token验证。

9. **P009 LOCORE: Image Re-ranking with Long-Context Sequence Modeling**
   - 年份：2025
   - 收录于：CVPR 2025
   - 作者：Zilin Xiao, Pavel Suma, Ayush Sachdeva, Hao-Jen Wang, Giorgos Kordopatis-Zilos, Giorgos Tolias, Vicente Ordonez
   - 中文摘要：将Query与候选列表的局部描述子共同输入长上下文重排序器，建模候选间关系并输出列表相似度。
   - 有无代码：未证实
   - 原文链接：https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_LOCORE_Image_Re-ranking_with_Long-Context_Sequence_Modeling_CVPR_2025_paper.html
   - PDF状态：开放获取
   - 相关性：代表多支持案例联合判断，而非逐个候选独立打分。

10. **P010 Deep Meta Metric Learning**
   - 年份：2019
   - 收录于：ICCV 2019
   - 作者：Guangyi Chen, Tianren Zhang, Jiwen Lu, Jie Zhou
   - 中文摘要：在采样子任务中划分Support与Query，学习集合级距离而不是单样本距离，并结合困难样本挖掘。
   - 有无代码：未证实
   - 原文链接：https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.html
   - PDF状态：开放获取
   - 相关性：代表用多个支持建筑共同验证Query。

如需下载开放PDF，可指定编号，例如 `P004,P006,P008` 或 `P003-P009`。
