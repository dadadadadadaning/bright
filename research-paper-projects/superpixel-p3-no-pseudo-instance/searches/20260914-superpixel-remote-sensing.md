# 超像素用于P3像素级建筑损毁：论文检索

1. **P001｜Superpixel Sampling Networks**
   - 年份：2018
   - 收录于：ECCV 2018
   - 作者：Varun Jampani, Deqing Sun, Ming-Yu Liu, Ming-Hsuan Yang, Jan Kautz
   - 原文：https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
   - 中文摘要：提出可微分的超像素采样网络，使超像素可在下游任务目标下端到端学习。
   - 有无代码：有：https://github.com/NVlabs/ssn_superpixels
   - 相关性：可借鉴“软像素—区域分配矩阵”，但不必把区域变成分类实例。
   - PDF状态：open

2. **P002｜Superpixel Segmentation With Fully Convolutional Networks**
   - 年份：2020
   - 收录于：CVPR 2020
   - 作者：Fengting Yang, Qian Sun, Hailin Jin, Zihan Zhou
   - 原文：https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.html
   - 中文摘要：用FCN预测像素到规则区域单元的关联概率，并给出基于超像素的下采样/上采样方式。
   - 有无代码：有：https://github.com/fuy34/superpixel_fcn
   - 相关性：适合第二阶段把固定SLIC替换成可学习的软区域分配。
   - PDF状态：open

3. **P003｜Semantic Segmentation by Early Region Proxy**
   - 年份：2022
   - 收录于：CVPR 2022
   - 作者：Yifan Zhang, Bo Pang, Cewu Lu
   - 原文：https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Semantic_Segmentation_by_Early_Region_Proxy_CVPR_2022_paper.pdf
   - 中文摘要：将图像表示为可学习区域代理，并使用Transformer建模区域间上下文；论文同时指出固定低层超像素噪声会限制较大模型。
   - 有无代码：有：https://github.com/YiF-Zhang/RegionProxy
   - 相关性：支持“区域上下文再广播回像素”，也提示不能硬制约最终标签。
   - PDF状态：open

4. **P004｜An Enhanced and Unsupervised Siamese Network With Superpixel-Guided Learning for Change Detection in Heterogeneous Remote Sensing Images**
   - 年份：2024
   - 收录于：IEEE JSTARS
   - 作者：Zhiyuan Ji, Xueqian Wang, Zhihao Wang, Gang Li
   - 原文：https://doi.org/10.1109/JSTARS.2024.3479703
   - 中文摘要：面向异构遥感影像变化检测，将超像素引导与孪生网络结合。
   - 有无代码：未证实
   - 相关性：与RGB–SAR异构变化任务接近，但其无监督标签构造不适合直接照搬到已有监督的BRIGHT。
   - PDF状态：unknown

5. **P005｜Operational earthquake-induced building damage assessment using CNN-based direct remote sensing change detection on superpixel level**
   - 年份：2022
   - 收录于：International Journal of Applied Earth Observation and Geoinformation
   - 作者：未获取
   - 原文：https://doi.org/10.1016/j.jag.2022.102899
   - 中文摘要：在地震建筑损毁评估中使用灾前超像素约束直接变化检测，以改善建筑边界和处理效率。
   - 有无代码：未证实
   - 相关性：直接支持“灾前RGB生成空间支持”的选择，但该工作是超像素级决策，本文不应照搬其硬区域分类。
   - PDF状态：unknown

6. **P006｜Edge-constrained temporal superpixel segmentation and graph-structured energy optimization for PolSAR change detection**
   - 年份：2025
   - 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
   - 作者：未获取
   - 原文：https://doi.org/10.1016/j.isprsjprs.2025.08.006
   - 中文摘要：针对PolSAR变化检测设计边缘约束超像素，并通过图结构关系增强弱散射变化、抑制跨边界合并。
   - 有无代码：未证实
   - 相关性：说明超像素是否跨越真实边界比“选哪一种复杂图网络”更先要验证。
   - PDF状态：unknown

7. **P007｜Hierarchical Histogram Threshold Segmentation – Auto-terminating High-detail Oversegmentation**
   - 年份：2024
   - 收录于：CVPR 2024
   - 作者：未获取
   - 原文：https://openaccess.thecvf.com/content/CVPR2024/html/Chang_Hierarchical_Histogram_Threshold_Segmentation_-_Auto-terminating_High-detail_Oversegmentation_CVPR_2024_paper.html
   - 中文摘要：提出自动终止的高细节过分割方法，并展示其对学习式语义分割掩膜的细化应用。
   - 有无代码：未证实
   - 相关性：可作为SLIC之外的边界敏感对照，但不适合第一版直接替换成熟SLIC。
   - PDF状态：open

可提供编号（如 `P001,P003`）以下载开放PDF；本轮未自动下载。

