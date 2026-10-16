# 40篇参考文献独立核对记录（2026-10-16）

## 总体结论

- 四份JSON合计40条、40个不同local_key、40个归一化后不同题名，未发现重复论文。12个已填DOI互不重复。未包含新闻稿或会议新闻；已核arXiv版本只计同一工作一次。
- 现有清单中8篇取得全文或作者托管PDF并用于核对，32篇主要依据作者/机构书目及学术参考文献交叉核对。这里的“取得全文”不等于本轮对全部实验完成重新复现，也不能表述为“40篇全文均逐页核对”。
- 8篇全文材料：bda02、bda06、bda07、bda08、bda09、reg02、reg03、core10。来源分别为现有出版社文本抽取（5篇）、作者托管PDF（2篇）、用户提供出版社PDF（1篇）。
- 原四份JSON记录58次本地证据引用，涉及48个不同文件；其中54次有登记SHA256，涉及44个不同文件。54/54登记哈希完全一致，58/58路径存在。其余4次是reg02/reg03补充作者页面/PDF，已有文件但JSON未登记SHA，不应说成“全部48份证据已有原始登记hash”。
- MoCo新增交叉来源后为59次本地证据引用、49个不同文件；55次登记SHA（45个不同文件），新增文件SHA亦已实算。原JSON未被本次审计修改。

## 必须注意或建议采用的修正

1. **MoCo（ret10）**：原README只确认2019预印本。现有SARMAE学术参考文献新增确认CVPR2020、9729–9738，已写入`reference40_patch.json`。采用patch后可保留CVPR2020，但证据等级必须为“官方作者仓库+学术参考文献交叉核对”，不能说直接取得CVF全文。
2. **Attention Is All You Need（core09）**：保留2017 arXiv:1706.03762版本。它本身是论文，非新闻；本轮不补未直接核验的NeurIPS卷页。不要与NeurIPS版本重复计数。
3. **SPAM（reg04）**：BMVC2025来自作者官方citation，未独立访问BMVC正式全文。可以按作者提供会议书目引用，交付审计清单明确此等级；如期刊要求出版社直核，后续补官网记录。
4. **Tip-Adapter（ret01）**：正确使用2022最终题名和主页显示的八位作者顺序。官方README末尾旧2021条目不应覆盖已核最终记录。
5. **DamageFormer（bda04）**：`authors`数组最后的`et al.`只是书目截断标记。正式参考文献可写前三作者后“et al.”；BibTeX不能把`et al.`当第四位作者。若要保留完整作者，可使用已核BRIGHT书目中的`H. Chen, E. Nemni, S. Vallecorsa, X. Li, C. Wu, L. Bromley`，不要猜测后三者全名。
6. **DeepLabv3+（core02）**：现有交叉参考文献页码冲突，留空是正确处理，不能选一个自认为正确的页码。
7. **论文用途**：WANet仅用于相关细结构工作和图表写法；MS4D/OCD/中分SAR迁移也不能直接列为BRIGHT同设置SOTA分数。MoCo/XBM/SupCon是训练表示学习背景，不能写成“测试时标注检索库的相同机制”。

## 逐条核对

| 键 | 论文（简称） | 当前核对等级 | 结论与用途边界 |
|---|---|---|---|
| bda01 | Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery | 作者/机构书目或学术参考文献交叉核对 | 作者数据集仓库及BRIGHT正式参考文献交叉核对；不另计xBD预印本；未核页码不填。 |
| bda02 | BRIGHT: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response | 全文/PDF材料核对 | 现有出版社全文抽取和作者BibTeX，题名/12作者/年/卷页/DOI一致。 |
| bda03 | Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters | 作者/机构书目或学术参考文献交叉核对 | 作者仓库与正式参考文献交叉核对；未直接取得ChangeOS全文。 |
| bda04 | Dual-Tasks Siamese Transformer Framework for Building Damage Assessment | 作者/机构书目或学术参考文献交叉核对 | 题名/年/刊会/页码/DOI由BRIGHT正式参考文献确认；authors中的et al.是显示截断，不是作者姓名。 |
| bda05 | ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model | 作者/机构书目或学术参考文献交叉核对 | 作者仓库和BRIGHT参考文献一致；不自行加未核文章编号。 |
| bda06 | Towards transferable building damage assessment via unsupervised single-temporal change adaptation | 全文/PDF材料核对 | 现有RSE全文抽取及作者书目；元数据一致。 |
| bda07 | MS4D-Net: Multitask-Based Semi-Supervised Semantic Segmentation Framework with Perturbed Dual Mean Teachers for Building Damage Assessment from High-Resolution Remote Sensing Imagery | 全文/PDF材料核对 | 现有Remote Sensing全文抽取，含题页引用和作者；可用。 |
| bda08 | Methodology for Object-Level Change Detection in Post-Earthquake Building Damage Assessment Based on Remote Sensing Images: OCD-BDA | 全文/PDF材料核对 | 按PDF题页/推荐引用作者顺序，优先于文档属性中的不同顺序。 |
| bda09 | Assessing the transferability of post-disaster building damage assessment using synthetic aperture radar and machine learning | 全文/PDF材料核对 | 现有IJDRR全文抽取；属于SAR迁移研究，不能直接用于BRIGHT先进性排名。 |
| bda10 | Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network | 作者/机构书目或学术参考文献交叉核对 | 作者仓库与BRIGHT书目；无DOI证据故留空。 |
| reg01 | SLIC Superpixels Compared to State-of-the-Art Superpixel Methods | 作者/机构书目或学术参考文献交叉核对 | 作者官方citation含完整刊会卷期页；未读出版社全文，不补DOI。 |
| reg02 | Superpixel Sampling Networks | 全文/PDF材料核对 | 作者托管PDF核对Sampling；官方README误写Samping已正确修正。 |
| reg03 | Superpixel Segmentation with Fully Convolutional Networks | 全文/PDF材料核对 | 作者托管PDF核对复数Networks；README单数Network未照抄。 |
| reg04 | Superpixel Anything: A General Object-Based Framework for Accurate Yet Regular Superpixel Segmentation | 作者/机构书目或学术参考文献交叉核对 | 作者官方citation确认BMVC2025；独立BMVC页面/全文未取得，等级须保留作者书目。 |
| reg05 | Object-Contextual Representations for Semantic Segmentation | 作者/机构书目或学术参考文献交叉核对 | 作者HRNet-OCR分支citation确认ECCV2020；不是本文方法全文核对。 |
| reg06 | Pyramid Scene Parsing Network | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；PSPNet是上下文聚合背景，不证明检索效果。 |
| reg07 | Dual Attention Network for Scene Segmentation | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation含CVPR2019及页3146–3154；不重复计TNNLS扩展。 |
| reg08 | Non-local Neural Networks | 作者/机构书目或学术参考文献交叉核对 | 作者机构仓库citation；属于非局部关联，不等同带标签历史库。 |
| reg09 | Squeeze-and-Excitation Networks | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；不自行补未核卷页DOI。 |
| reg10 | Is Second-order Information Helpful for Large-scale Visual Recognition? | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；协方差二阶表征与均值/标准差不能等同。 |
| ret01 | Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification | 作者/机构书目或学术参考文献交叉核对 | 最终题名、2022及八位作者顺序由第一作者主页纠正，不能用README末尾2021旧BibTeX覆盖。 |
| ret02 | Supervised Contrastive Learning | 作者/机构书目或学术参考文献交叉核对 | 作者及Google Research仓库书目；监督度量背景。 |
| ret03 | Exploring Cross-Image Pixel Contrast for Semantic Segmentation | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；训练期跨图像对比与推理历史读取须区分。 |
| ret04 | Rethinking Semantic Segmentation: A Prototype View | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；非参数原型背景，非本文Train案例效果验证。 |
| ret05 | FiLM: Visual Reasoning with a General Conditioning Layer | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；条件特征调制背景，未取全文。 |
| ret06 | Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation含CVPR2020及页8808–8817；支持集适配背景。 |
| ret07 | Prototypical Networks for Few-shot Learning | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；NIPS2017是当年会议名，可统一版式为NeurIPS但不改出版年。 |
| ret08 | Dense Contrastive Learning for Self-Supervised Visual Pre-Training | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；无DOI/页码证据不补。 |
| ret09 | Cross-Batch Memory for Embedding Learning | 作者/机构书目或学术参考文献交叉核对 | 作者仓库citation；训练队列与部署时支持库须区分。 |
| ret10 | Momentum Contrast for Unsupervised Visual Representation Learning | 作者/机构书目或学术参考文献交叉核对 | 历史官方README只写2019预印本；本次新增SARMAE学术参考文献[24]交叉确认CVPR2020、9729–9738；见patch。 |
| core01 | U-Net: Convolutional Networks for Biomedical Image Segmentation | 作者/机构书目或学术参考文献交叉核对 | BRIGHT正式参考文献核页码/DOI；独立实现链接不得称原作者官方仓库。 |
| core02 | Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation | 作者/机构书目或学术参考文献交叉核对 | 作者官方citation确认核心元数据；既有参考文献页码冲突，因此页码/DOI继续留空。 |
| core03 | Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | 作者/机构书目或学术参考文献交叉核对 | Microsoft作者仓库citation；未核页码DOI留空。 |
| core04 | SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | 作者/机构书目或学术参考文献交叉核对 | NVlabs作者仓库citation；未核页码DOI留空。 |
| core05 | The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks | 作者/机构书目或学术参考文献交叉核对 | 作者仓库与BRIGHT正式书目；核心字段一致。 |
| core06 | Decoupled Weight Decay Regularization | 作者/机构书目或学术参考文献交叉核对 | 作者仓库确认ICLR2019及正确arXiv1711.05101；不照抄WANet的错误号。 |
| core07 | Fully Convolutional Networks for Semantic Segmentation | 作者/机构书目或学术参考文献交叉核对 | 作者仓库与WANet正式书目；FCN与超像素FCN是两篇不同工作。 |
| core08 | Deep Residual Learning for Image Recognition | 作者/机构书目或学术参考文献交叉核对 | 2015预印本和2016出版版按同一工作计一次；正式出版字段有BRIGHT书目交叉证据。 |
| core09 | Attention Is All You Need | 作者/机构书目或学术参考文献交叉核对 | 现有证据能确认2017 arXiv及八位作者；采用arXiv版本合理，不声称已核NeurIPS卷页。 |
| core10 | WANet: A Wavelet-Attention Network for Accurate Ditch Extraction in Remote Sensing Imagery | 全文/PDF材料核对 | 用户提供JSTARS全文，七位作者/2026/卷19/页17042–17054/DOI一致；不能作为BDA直接基线。 |

## MoCo新增学术书目证据

- 文件：`/workspace/bright/research-paper-projects/post-disaster-building-damage/sources/extracted/C01-SARMAE-2026.txt`
- SHA256：`5cedd82f09a0c745e19d86e6bcab3c0e6efab1409f1caa2fcaae8981100b131f`
- 位置：PDF第9页，参考文献[24]；抽取文本第822–824行。
- 原文：

> [24] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, pages 9729–9738, 2020.

## 审计范围说明

本次独立检查四份JSON字段、对应已有来源片段、文件存在性、登记哈希和明显元数据冲突；未重复大量受网络限制的出版社请求。原始JSON、Word及只读sources均未修改。本审计不能代替投稿前对所引方法结论和完整实验条件的全文精读。
