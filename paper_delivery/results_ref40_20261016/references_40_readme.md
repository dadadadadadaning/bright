# 40篇学术文献核对说明

本包提供40篇互异论文的参考文献、BibTeX与逐条核验表。不包含新闻稿、会议日程，也不把同一工作的arXiv和正式版本计为两篇。

**这里的“核对”指逐条核对书目和适用范围，不能理解为40篇均已取得全文、逐页精读或复现实验。**

- L1：已读可用的正式/作者全文，或既有正式全文文本提取。
- L2：作者书目与已发表论文参考文献交叉核对。
- L3：作者/机构代码citation、作者主页书目核对。

本轮多个出版社及学术检索接口返回代理403；可用证据主要来自既有学术全文提取、作者托管PDF和正式代码citation。每条证据的URL、工作区路径、SHA256与限制均在核验表中列出。路径用于追溯，不保证外部读者拥有该本地文件；第三方全文和整份README不随包分发。

## 文件

- `references_40.json`：40条规范化记录、所有已核作者、论文链接、证据和格式化参考文献。
- `references_40.bib`：可导入文献工具的40条BibTeX。
- `references_40_audit.xlsx`：逐条核验、全部来源及SHA256、使用说明。
- 本说明：数量、证据等级和逐条索引。

## 使用规则

未核实的DOI、页码、卷期、论文直链保留空白；代码链接单列，不冒充论文全文。论文链接来自已核citation、作者PDF或已核DOI，并不表示每个出版社目标页本轮都能直接打开；实际访问的来源与响应另见核验表。作者名只到初字母时原样保留，不推测全名。引用中列出的书目均为学术论文；“作者仓库”只是核验来源，未作为参考文献计数。

题名和作者使用核验版：SSN、Superpixel FCN以作者PDF修正仓库拼写；Tip-Adapter使用2022最终作者页；MoCo的CVPR2020页码由现有SARMAE学术书目交叉核对；Attention Is All You Need按已核2017 arXiv记录列入。

## 逐条索引

| 编号 | 引用键 | 论文与年份 | 核验等级 | 论文链接 |
| --- | --- | --- | --- | --- |
| 1 | `bda01` | Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery (2019) | L2 | https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html |
| 2 | `bda02` | BRIGHT: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response (2025) | L1 | https://essd.copernicus.org/articles/17/6217/2025/ |
| 3 | `bda03` | Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters (2021) | L1 | https://www.sciencedirect.com/science/article/pii/S0034425721003564 |
| 4 | `bda04` | Dual-Tasks Siamese Transformer Framework for Building Damage Assessment (2022) | L2 | https://ieeexplore.ieee.org/document/9883139 |
| 5 | `bda05` | ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model (2024) | L2 | https://ieeexplore.ieee.org/document/10565926 |
| 6 | `bda06` | Towards transferable building damage assessment via unsupervised single-temporal change adaptation (2024) | L1 | https://www.sciencedirect.com/science/article/pii/S0034425724004425 |
| 7 | `bda09` | Assessing the transferability of post-disaster building damage assessment using synthetic aperture radar and machine learning (2026) | L2 | https://doi.org/10.1016/j.ijdrr.2026.106048 |
| 8 | `core07` | Fully Convolutional Networks for Semantic Segmentation (2015) | L3 | https://arxiv.org/abs/1411.4038 |
| 9 | `core01` | U-Net: Convolutional Networks for Biomedical Image Segmentation (2015) | L2 | https://doi.org/10.1007/978-3-319-24574-4_28 |
| 10 | `core02` | Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (2018) | L3 | https://arxiv.org/abs/1802.02611 |
| 11 | `core08` | Deep Residual Learning for Image Recognition (2016) | L2 | https://doi.org/10.1109/CVPR.2016.90 |
| 12 | `core03` | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (2021) | L3 | https://arxiv.org/abs/2103.14030 |
| 13 | `core04` | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (2021) | L3 | https://arxiv.org/abs/2105.15203 |
| 14 | `bda10` | Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network (2020) | L2 | https://ieeexplore.ieee.org/document/8937755 |
| 15 | `bda07` | MS4D-Net: Multitask-Based Semi-Supervised Semantic Segmentation Framework with Perturbed Dual Mean Teachers for Building Damage Assessment from High-Resolution Remote Sensing Imagery (2023) | L1 | https://www.mdpi.com/2072-4292/15/2/478 |
| 16 | `bda08` | Methodology for Object-Level Change Detection in Post-Earthquake Building Damage Assessment Based on Remote Sensing Images: OCD-BDA (2024) | L1 | https://www.mdpi.com/2072-4292/16/22/4263 |
| 17 | `reg01` | SLIC Superpixels Compared to State-of-the-Art Superpixel Methods (2012) | L3 | 未核实直链；代码来源另列 |
| 18 | `reg02` | Superpixel Sampling Networks (2018) | L1 | https://raw.githubusercontent.com/varunjampani/varunjampani.github.io/master/papers/jampani18_SSN.pdf |
| 19 | `reg03` | Superpixel Segmentation with Fully Convolutional Networks (2020) | L1 | https://arxiv.org/abs/2003.12929 |
| 20 | `reg04` | Superpixel Anything: A General Object-Based Framework for Accurate Yet Regular Superpixel Segmentation (2025) | L3 | https://arxiv.org/abs/2509.12791 |
| 21 | `reg05` | Object-Contextual Representations for Semantic Segmentation (2020) | L3 | https://arxiv.org/abs/1909.11065 |
| 22 | `reg06` | Pyramid Scene Parsing Network (2017) | L3 | https://arxiv.org/abs/1612.01105 |
| 23 | `reg07` | Dual Attention Network for Scene Segmentation (2019) | L3 | https://arxiv.org/abs/1809.02983 |
| 24 | `reg08` | Non-local Neural Networks (2018) | L3 | https://arxiv.org/abs/1711.07971 |
| 25 | `reg09` | Squeeze-and-Excitation Networks (2018) | L3 | http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf |
| 26 | `reg10` | Is Second-order Information Helpful for Large-scale Visual Recognition? (2017) | L3 | 未核实直链；代码来源另列 |
| 27 | `core10` | WANet: A Wavelet-Attention Network for Accurate Ditch Extraction in Remote Sensing Imagery (2026) | L1 | https://doi.org/10.1109/JSTARS.2026.3693082 |
| 28 | `ret05` | FiLM: Visual Reasoning with a General Conditioning Layer (2018) | L3 | 未核实直链；代码来源另列 |
| 29 | `ret03` | Exploring Cross-Image Pixel Contrast for Semantic Segmentation (2021) | L3 | https://arxiv.org/abs/2101.11939 |
| 30 | `ret04` | Rethinking Semantic Segmentation: A Prototype View (2022) | L3 | https://arxiv.org/abs/2203.15102 |
| 31 | `ret07` | Prototypical Networks for Few-shot Learning (2017) | L3 | http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf |
| 32 | `ret06` | Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions (2020) | L3 | https://arxiv.org/abs/1812.03664 |
| 33 | `ret01` | Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification (2022) | L3 | https://arxiv.org/abs/2207.09519 |
| 34 | `ret09` | Cross-Batch Memory for Embedding Learning (2020) | L3 | https://arxiv.org/abs/1912.06798 |
| 35 | `ret10` | Momentum Contrast for Unsupervised Visual Representation Learning (2020) | L2 | https://arxiv.org/abs/1911.05722 |
| 36 | `ret08` | Dense Contrastive Learning for Self-Supervised Visual Pre-Training (2021) | L3 | https://arxiv.org/abs/2011.09157 |
| 37 | `ret02` | Supervised Contrastive Learning (2020) | L3 | https://arxiv.org/abs/2004.11362 |
| 38 | `core09` | Attention Is All You Need (2017) | L3 | https://arxiv.org/abs/1706.03762 |
| 39 | `core05` | The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks (2018) | L2 | https://doi.org/10.1109/CVPR.2018.00464 |
| 40 | `core06` | Decoupled Weight Decay Regularization (2019) | L2 | https://arxiv.org/abs/1711.05101 |

## 参考文献文本

[1] Ritwik Gupta, Bryce Goodman, Nirav Patel, Rachel Hosfelt, Sajal Sajeev, Eric Heim, Jigar Doshi, Keane Lucas, Howie Choset, Matthew Gaston. Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2019.

[2] Hongruixuan Chen, Jian Song, Olivier Dietrich, Clifford Broni-Bediako, Weihao Xuan, Junjue Wang, Xinlei Shao, Yimin Wei, Junshi Xia, Cuiling Lan, Konrad Schindler, Naoto Yokoya. BRIGHT: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response. Earth System Science Data, 2025, 17(11): 6217–6253. DOI: 10.5194/essd-17-6217-2025.

[3] Zhuo Zheng, Yanfei Zhong, Junjue Wang, Ailong Ma, Liangpei Zhang. Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters. Remote Sensing of Environment, 2021, 265: 112636. DOI: 10.1016/j.rse.2021.112636.

[4] Hongruixuan Chen, Edoardo Nemni, Sofia Vallecorsa, X. Li, C. Wu, L. Bromley. Dual-Tasks Siamese Transformer Framework for Building Damage Assessment. IGARSS 2022 – 2022 IEEE International Geoscience and Remote Sensing Symposium, 2022: 1600–1603. DOI: 10.1109/IGARSS46834.2022.9883139.

[5] Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya. ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model. IEEE Transactions on Geoscience and Remote Sensing, 2024, 62: 1–20. DOI: 10.1109/TGRS.2024.3417253.

[6] Zhuo Zheng, Yanfei Zhong, Liangpei Zhang, Marshall Burke, David B. Lobell, Stefano Ermon. Towards transferable building damage assessment via unsupervised single-temporal change adaptation. Remote Sensing of Environment, 2024, 315: 114416. DOI: 10.1016/j.rse.2024.114416.

[7] James Currie, Karin Reinke. Assessing the transferability of post-disaster building damage assessment using synthetic aperture radar and machine learning. International Journal of Disaster Risk Reduction, 2026, 136: 106048. DOI: 10.1016/j.ijdrr.2026.106048.

[8] Jonathan Long, Evan Shelhamer, Trevor Darrell. Fully Convolutional Networks for Semantic Segmentation. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015: 3431–3440.

[9] Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, 2015: 234–241. DOI: 10.1007/978-3-319-24574-4_28.

[10] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. European Conference on Computer Vision (ECCV), 2018.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016: 770–778. DOI: 10.1109/CVPR.2016.90.

[12] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[13] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. Advances in Neural Information Processing Systems (NeurIPS), 2021.

[14] Hongruixuan Chen, Chen Wu, Bo Du, Liangpei Zhang, Le Wang. Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network. IEEE Transactions on Geoscience and Remote Sensing, 2020, 58(4): 2848–2864.

[15] Yongjun He, Jinfei Wang, Chunhua Liao, Xin Zhou, Bo Shan. MS4D-Net: Multitask-Based Semi-Supervised Semantic Segmentation Framework with Perturbed Dual Mean Teachers for Building Damage Assessment from High-Resolution Remote Sensing Imagery. Remote Sensing, 2023, 15(2): 478. DOI: 10.3390/rs15020478.

[16] Zhengtao Xie, Zifan Zhou, Xinhao He, Yuguang Fu, Jiancheng Gu, Jiandong Zhang. Methodology for Object-Level Change Detection in Post-Earthquake Building Damage Assessment Based on Remote Sensing Images: OCD-BDA. Remote Sensing, 2024, 16(22): 4263. DOI: 10.3390/rs16224263.

[17] R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua, S. Süsstrunk. SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2012, 34(11): 2274–2282.

[18] Varun Jampani, Deqing Sun, Ming-Yu Liu, Ming-Hsuan Yang, Jan Kautz. Superpixel Sampling Networks. European Conference on Computer Vision, 2018.

[19] Fengting Yang, Qian Sun, Hailin Jin, Zihan Zhou. Superpixel Segmentation with Fully Convolutional Networks. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[20] Julien Walther, Rémi Giraud, Michaël Clément. Superpixel Anything: A General Object-Based Framework for Accurate Yet Regular Superpixel Segmentation. British Machine Vision Conference, 2025.

[21] Yuhui Yuan, Xilin Chen, Jingdong Wang. Object-Contextual Representations for Semantic Segmentation. European Conference on Computer Vision, 2020.

[22] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia. Pyramid Scene Parsing Network. IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[23] Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, Hanqing Lu. Dual Attention Network for Scene Segmentation. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019: 3146–3154.

[24] Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He. Non-local Neural Networks. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

[25] Jie Hu, Li Shen, Gang Sun. Squeeze-and-Excitation Networks. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018.

[26] Peihua Li, Jiangtao Xie, Qilong Wang, Wangmeng Zuo. Is Second-order Information Helpful for Large-scale Visual Recognition?. IEEE International Conference on Computer Vision, 2017.

[27] Yanfeng Zhang, Zhiwu Gao, Ying Dai, Yu Wu, Lei Zhang, Xingxing Han, Zhenxuan Liu. WANet: A Wavelet-Attention Network for Accurate Ditch Extraction in Remote Sensing Imagery. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2026, 19: 17042–17054. DOI: 10.1109/JSTARS.2026.3693082.

[28] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville. FiLM: Visual Reasoning with a General Conditioning Layer. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2018.

[29] Wenguan Wang, Tianfei Zhou, Fisher Yu, Jifeng Dai, Ender Konukoglu, Luc Van Gool. Exploring Cross-Image Pixel Contrast for Semantic Segmentation. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021: 7303–7313.

[30] Tianfei Zhou, Wenguan Wang, Ender Konukoglu, Luc Van Gool. Rethinking Semantic Segmentation: A Prototype View. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[31] Jake Snell, Kevin Swersky, Richard Zemel. Prototypical Networks for Few-shot Learning. Advances in Neural Information Processing Systems (NIPS), 2017.

[32] Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, Fei Sha. Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020: 8808–8817.

[33] Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, Hongsheng Li. Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification. European Conference on Computer Vision (ECCV), 2022.

[34] Xun Wang, Haozhi Zhang, Weilin Huang, Matthew R. Scott. Cross-Batch Memory for Embedding Learning. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[35] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick. Momentum Contrast for Unsupervised Visual Representation Learning. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020: 9729–9738.

[36] Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, Lei Li. Dense Contrastive Learning for Self-Supervised Visual Pre-Training. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[37] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan. Supervised Contrastive Learning. Advances in Neural Information Processing Systems (NeurIPS), 2020.

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. Attention Is All You Need. arXiv preprint arXiv:1706.03762, 2017. https://arxiv.org/abs/1706.03762.

[39] Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko. The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018: 4413–4421. DOI: 10.1109/CVPR.2018.00464.

[40] Ilya Loshchilov, Frank Hutter. Decoupled Weight Decay Regularization. International Conference on Learning Representations (ICLR), 2019.

