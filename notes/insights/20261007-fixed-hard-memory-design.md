# RGB与SAR建筑损毁区域 Memory 设计与文献依据

当前最小实验应固定已核验hard区域表示，只验证历史Train区域是否提供额外分类信息。三个分区与SAR支持实验已经运行，但新增机制未表现出可靠增益，因此不在Memory训练时同时改变Q、建筑概率、Pooling或损失。以下为2026年10月7日核对的原始论文与作者代码，以及本项目的任务适配；不是SRAC的严格复现。

## 已有结果约束

研究任务为BRIGHT灾前RGB与灾后SAR的Background、Intact、Damaged、Destroyed四类分割。Train2918、Val450，不用Test选择方案。GT是建筑等级标签，连通域只作对象代理，区域软标签不代表楼内局部破损比例。

当前来源为A1初始化后3000步区域联合训练的前端，以及ROI_EXCHANGE的5000步hard区域读出。它不是历史A1。固定hard来源约61.5216%；实验二RGB/SAR参与Q的最终差约+0.0015pp；实验三SAR交换相对普通交换约−0.0005pp。它们不支持继续扩大这两种具体机制，也不能证明所有SAR分区或Memory无效。

本图跨组交换已发现发送端预算不限制接收端占比的设计边界，但尚未证明它解释全部伤害。此轮不修这个机制，以免与Memory收益混淆。

## 论文及可迁移机制

| 文献与作者代码 | 原任务与监督 | 可迁移部分 | 不可直接迁移及风险 |
|---|---|---|---|
| [Tip Adapter ECCV2022](https://arxiv.org/abs/2207.09519)，[官方代码](https://github.com/gaopengcuhk/Tip-Adapter) | CLIP自然图像少样本分类，训练样本特征与类别组成key-value缓存 | 保留原分类器，Train特征与标签提供补充证据 | CLIP文本分支、ImageNet参数和few-shot协议不同于RGB–SAR分割；本项目不复制仓库的超参数搜索入口，严格不用Test选择 |
| [CrossTransformers NeurIPS2020](https://arxiv.org/abs/2007.11498)，[作者代码](https://github.com/google-research/meta-dataset) | Meta-Dataset自然图像少样本任务，查询与标注支持做空间对应并按距离分类 | 让支持读出依赖当前query，不简单平均历史样本 | 原方法是空间匹配与类别原型，不等同于本项目的标签值嵌入Cross-Attention；TensorFlow episodic配置不照搬 |
| [Exploring Cross Image Pixel Contrast ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.pdf)，[官方代码](https://github.com/tfzhou/ContrastiveSeg) | RGB全监督语义分割，跨图像像素/区域记忆用于对比表征学习 | 提醒检索距离需要语义意义；若近邻不可靠，可之后单独研究度量学习 | 该Memory服务训练，测试不需要额外库；不是推理检索有效的直接证据。本轮不加对比损失 |
| [OCR ECCV2020作者论文页](https://www.microsoft.com/en-us/research/publication/object-contextual-representations-for-semantic-segmentation/)，[官方代码](https://github.com/openseg-group/openseg.pytorch) | RGB语义分割，在GT监督下学语义区域并聚合对象类别上下文 | 区域可以补充已有判别，而不必完全接管输出 | object region不是建筑实例或连通超像素，也不是外部Memory；本轮不替换为OCR解码器 |
| [Rethinking Semantic Segmentation A Prototype View CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.pdf)，[官方代码](https://github.com/tfzhou/ProtoSeg) | RGB语义分割，以多个非参数类原型与特征学习共同适配 | 后续可研究将真实Token库压缩为每类多个代表，而不是一个均值 | 原论文表征会共同学习，不能推导冻结本项目特征一定适合原型；首轮不同时改变真实支持为原型 |

这些是成熟、机制相关的代表工作，不以“最新”为选择理由。官方代码只作为机制核对，不作为新运行依赖。上述自然图像结果都不能推导BRIGHT必然提升或达到75%。

## 本轮落地

固定原图RGB与SAR融合、Soft ROI、软Q、预测建筑/背景硬分组、区域Embedding及原区域分类器。Train每事件×多数类别最多64条真实Token，保留四类软标签，不通过query GT或预测B筛选邻居类别。默认排除同图、同事件；地理事件信息只用于防止捷径，不作为标签嵌入。

用两个相同参数、相同初始化且独立训练的分类适配器：nomemory读取参数化类别槽，memory读取余弦Top-8真实Train支持。两者都在冻结区域logits上添加零初始化修正，step0严格复现来源。Memory由特征key与带标签value参与查询注意力，这是任务适配，不是原论文的逐层复刻。

等参数不等于等支持长度或计算量。每组新修正器有83,844个训练参数，原区域读出作为冻结公共部分保留。损失仍为原像素回投NLL+Lovasz，梯度只更新修正器；冻结模型及库不更新。最终除总体mIoU外审计Damaged P/R/F1、D到I伤害、背景误报、ROI错误、边界、对象代理尺寸、事件、修正幅度、邻居覆盖和标签匹配。

完整1024图像构建的支持与训练512查询存在上下文差异。多数类别平衡不等于软标签质量平衡；相似度高或标签低熵也不保证损毁判断正确。上述限制需保留，不能将弱邻居质量直接归咎于超像素。

## 匹配和创新边界

本轮迁移难度中等：前端不变，已有紧凑建库与检索可复用，主要增加可审计的残差分类对照。任务匹配属于机制相近而非输入模态相同。Train缓存、注意力、原型、残差均是已有机制，组合本身不足以声称原创。

可能形成贡献的是：在预测建筑分组不完美、RGB与SAR存在支持域差异的条件下，证明什么样的历史证据能提高建筑等级判别，并明确其伤害与泛化边界。前提是优于同容量对照，且经过多种子和事件分层验证。

## 三个可证伪题目方向

1. **固定建筑约束区域表示上的历史检索能否改善BRIGHT损毁分类**。本轮已实现。比较source_hard、nomemory、memory，其他固定。若memory不优于nomemory，不宣称历史检索有效。
2. **BRIGHT小对象区域的接收端跨组质量约束**。仅作为独立后续假设，未接入本轮。先关联自身质量、异组比例与错误；再匹配接收预算比较普通/SAR交换。若无关联或无配对增益，停止。
3. **跨事件损毁区域库的多代表压缩与可靠检索**。仅在真实Token库先有正向证据后考虑。比较同预算真实支持与多个类/事件代表，检查稀有类召回和计算量；不同时解冻Q或增加对比损失。

## 实现范围与验证状态

代码位于`bda_benchmark/experiments/damageformer_region_memory_v2`，只新增目录和本记录，旧实验文件、权重、缓存保持不变。训练入口与具体服务器命令见目录README。

本机验证为CPU模块及合成数据检查，不等于服务器真实Train/Val的CUDA成功。只有服务器smoke通过后才启动完整建库与正式训练。没有访问Test、没有替用户启动远程GPU训练，也没有承诺精度收益。
