# Train-only Memory 与困难样本训练论文检索

- 检索日期：2026-08-28
- 研究问题：RGB–SAR 建筑损毁中，Memory 应继续在推理阶段改写预测，还是退回训练阶段用于困难样本选择？
- 数据源：CVF Open Access、PMLR、ScienceDirect

## P001 Cross-Batch Memory for Embedding Learning

- 年份：2020
- 收录于：CVPR 2020
- 作者：Xun Wang, Haozhi Zhang, Weilin Huang, Matthew R. Scott
- 中文摘要：保存此前批次的嵌入，在多个批次乃至近似全数据范围内挖掘困难负样本；Memory 服务于度量学习训练，而不是推理时直接输出类别。
- 有无代码：有：https://github.com/MalongTech/research-xbm
- 原文链接：https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html
- PDF：开放获取
- 相关性说明：支持把 BRIGHT Memory 改成动态训练队列，并要求控制陈旧特征造成的漂移。

## P002 Semi-Supervised Semantic Segmentation With Pixel-Level Contrastive Learning From a Class-Wise Memory Bank

- 年份：2021
- 收录于：ICCV 2021
- 作者：Iñigo Alonso, Alberto Sabater, David Ferstl, Luis Montesano, Ana C. Murillo
- 中文摘要：使用持续更新的类别级 Memory Bank 保存高质量标注特征，以像素级对比目标训练分割网络；测试阶段不需要 Memory。
- 有无代码：有：https://github.com/Shathe/SemiSeg-Contrastive
- 原文链接：https://openaccess.thecvf.com/content/ICCV2021/html/Alonso_Semi-Supervised_Semantic_Segmentation_With_Pixel-Level_Contrastive_Learning_From_a_Class-Wise_ICCV_2021_paper.html
- PDF：开放获取
- 相关性说明：提供“训练期类别队列、端到端更新、推理零开销”的直接范式。

## P003 Exploring Cross-Image Pixel Contrast for Semantic Segmentation

- 年份：2021
- 收录于：ICCV 2021
- 作者：Wenguan Wang, Tianfei Zhou, Fisher Yu, Jifeng Dai, Ender Konukoglu, Luc Van Gool
- 中文摘要：在分割训练中显式使用跨图像像素关系，组织类内与类间结构，且测试时没有额外开销。
- 有无代码：有：https://github.com/tfzhou/ContrastiveSeg
- 原文链接：https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.html
- PDF：开放获取
- 相关性说明：支持在 P3 的像素/区域发生表示上使用跨图像监督，但不能把所有异质 Damaged 强行压成单簇。

## P004 Training Region-Based Object Detectors With Online Hard Example Mining

- 年份：2016
- 收录于：CVPR 2016
- 作者：Abhinav Shrivastava, Abhinav Gupta, Ross Girshick
- 中文摘要：在线选择当前模型最难的训练实例，减少易样本主导和手工采样启发式。
- 有无代码：未证实
- 原文链接：https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html
- PDF：开放获取
- 相关性说明：支持先做不含跨事件正样本的事件上限 OHEM 对照，判断收益是否只是困难样本重采样。

## P005 Just Train Twice: Improving Group Robustness without Training Group Information

- 年份：2021
- 收录于：ICML 2021
- 作者：Evan Z. Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn
- 中文摘要：第一阶段模型识别高损失训练样本，第二阶段对这些样本加权重训，以提升最差子群表现。
- 有无代码：未证实
- 原文链接：https://proceedings.mlr.press/v139/liu21f.html
- PDF：开放获取
- 相关性说明：与你的 P3→筛选 D→I 困难建筑→再训练 P3 路径高度一致。

## P006 Examining and Combating Spurious Features under Distribution Shift

- 年份：2021
- 收录于：ICML 2021
- 作者：Chunting Zhou, Xuezhe Ma, Paul Michel, Graham Neubig
- 中文摘要：分析分布偏移下的伪相关，并讨论按预定义群组优化最差组损失的 Group DRO；同时提醒错误群组划分会削弱效果。
- 有无代码：未证实
- 原文链接：https://proceedings.mlr.press/v139/zhou21g.html
- PDF：开放获取
- 相关性说明：支持用“事件×类别”作为训练群组，直接优化最差灾害事件，而不是让 Ukraine/Turkey 数量支配训练。

## P007 Non-Isotropy Regularization for Proxy-Based Deep Metric Learning

- 年份：2022
- 收录于：CVPR 2022
- 作者：Karsten Roth, Oriol Vinyals, Zeynep Akata
- 中文摘要：指出单一类别代理可能抹掉类内局部结构，提出非各向同性约束保留代理周围的局部语义关系。
- 有无代码：有：https://github.com/ExplainableML/NonIsotropicProxyDML
- 原文链接：https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html
- PDF：开放获取
- 相关性说明：解释了你的统一 Damaged 嵌入为何反复伤害多形态损毁；若再做度量学习，应使用多原型或局部结构，而非单簇压缩。

## P008 Multimodal Remote Sensing Change Detection: An Image Matching Perspective

- 年份：2026
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing, 233:487–501
- 作者：Hongruixuan Chen, Cuiling Lan, Jian Song, Damian Ibañez, Junshi Xia, Konrad Schindler, Naoto Yokoya
- 中文摘要：IM4CD 放弃困难的逐像素跨模态特征直接比较，改用局部模板匹配、响应峰空间偏移、配准控制点残差和 CRF 推断变化。
- 有无代码：未证实
- 原文链接：https://www.sciencedirect.com/science/article/pii/S0924271626000559
- PDF：开放获取
- 相关性说明：它是独立的跨模态表示路线，不是当前 Train-only Memory 工程错误的修补项；只有单独立项时才应继续。

如需下载开放 PDF，可指定编号，例如 `P001,P004,P008`。
