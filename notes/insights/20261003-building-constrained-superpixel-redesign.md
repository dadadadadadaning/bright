# 建筑损毁区域表示：从全图软聚类转向受约束子区域的设计审查

## 摘要

本报告针对 BRIGHT 灾前 RGB 与灾后 SAR 的 DamageFormer 区域分支，审查可学习分区、掩膜内分区及密集像素—区域交互三条路线，并结合已有完整 Val 诊断提出受控重设计。文献支持将分区几何、语义保持和最终分类作为不同问题评价，而不支持把超像素图变得规整等同于损毁识别提升。当前工程确有建筑/背景混合和硬区域碎片风险，但已有概率投影及同特征分类头对照，不支持将全部下降归因于 Q。本报告建议保留 Soft ROI 与原像素路径，先用预测建筑范围内的 maskSLIC/网格建立可检验的几何隔离基线，再按需要引入可学习结构监督。所有新设计均为待验证方案，不是 SRAC 严格复现。本轮不修改或启动训练代码。

## 1. 研究问题与证据边界

- RQ1：如何降低跨背景、跨建筑对象代理的汇总，而不使用推理 GT？
- RQ2：如何保留建筑内部小损毁，而不是让一栋建筑只剩一个类别？
- RQ3：怎样将分区改善与区域数增加、图像调制、分类器容量的影响分开？

研究对象是当前 Swin/DamageFormer，而不是重新训练另一个 VMamba 主干。主要已有证据：pixel60.8998%，其概率经 Q 投影60.8982%，实际区域59.1447%；关图像调制60.3895%；同调制特征上新区域59.2255%高于新像素58.8414%。混合区域的伤害率高是定位证据，不是 Q 唯一因果解释。

## 2. 检索与核验方法

2026-10-03，采用三路独立定向检索：可学习超像素/Transformer tokenization，建筑对象约束/跨模态 BDA，区域上下文/反对过度归因；主审查补充 maskSLIC、SNIC 与评价基准。各路线进行通用关键词与具体方法追踪两轮。关键词包括 learned superpixel connectivity、building damage object constraints、maskSLIC、superpixel undersegmentation、pixel region context、SPFormer/SuiT。纳入论文原文/摘要、作者仓库和维护者文档；不以综述转述推断实现细节。范围从方法基础到本次检索可核验的近年工作，不宣称覆盖全部文献。原 SRAC 全文与训练配置仍未获得，因此不将宣传图当作公式规范。

## 3. 按设计责任分类

| 责任 | 代表工作 | 本项目应回答的问题 |
|---|---|---|
| 归属学习 | SSN、SCN、SPFormer | 哪些位置可以共享表示？明确训练了哪些保留属性？ |
| 区域空间约束 | maskSLIC、SNIC、ChangeOS | 是否限定可分配范围？是否连通？是否误把连通域当实例？ |
| 信息读出 | OCR、Mask2Former、SuiT | 汇总后是否仍保留每个位置的判别信息？ |
| 评价协议 | Stutz基准、Barcelos综述/基准 | 改善是否只是区域更多？是否只改善外观？ |

分类按本项目采用的功能划分，不意味着论文只能承担一种功能。尤其 SPFormer 同时涉及归属与像素回写。

## 4. 学习归属：不应只借鉴模块名称

SSN 使用可微聚类，而 SCN 直接预测局部归属；二者的重要共同点是允许按要保留的属性设计监督。当前 local9 与这些方法并非完全不同，但缺少独立结构保持目标，且未继承它们测试阶段的连通性处理。不能说当前完全没有空间约束：已有空间距离惩罚。参考 [SSN论文](https://arxiv.org/abs/1807.10174)、[SCN原文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.pdf)。

SPFormer 的像素—超像素交互与现有单次区域分类不同；它并不因使用 Transformer 就自动提供建筑隔离。作者公开文本可读，本次未核验其二维方法官方代码，勿误用同名三维 SPFormer 仓库。[SPFormer方法](https://arxiv.org/html/2401.02931v1)。官方 SSN 是旧 Caffe；SCN 是旧 PyTorch 工程，不能直接假定当前环境可运行。[SSN代码](https://github.com/NVlabs/ssn_superpixels)、[SCN代码](https://github.com/fuy34/superpixel_fcn)。

## 5. 空间约束：先限制范围，再讨论内容自适应

maskSLIC 在 ROI 内分区，而 SNIC 强调从生成过程中维持连通；前者直接对应建筑范围约束，后者说明连通性可显式设计而非期待 softmax 自动产生。二者均不是损毁语义识别器。[maskSLIC](https://arxiv.org/abs/1606.09518)、[维护者示例](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_mask_slic.html)、[SNIC论文](https://openaccess.thecvf.com/content_cvpr_2017/html/Achanta_Superpixels_and_Polygons_CVPR_2017_paper.html)、[SNIC作者代码](https://github.com/achanta/SNIC)。

ChangeOS 将对象化处理引入建筑损毁，但直接采用连通域统一投票，会与本项目保留内部损毁差异的目标发生冲突。借鉴预测对象约束，不照搬每域一个类别。RGB/RGB方法不自动适配RGB/SAR。[ChangeOS作者代码与论文入口](https://github.com/Z-Zheng/ChangeOS)。

预测建筑/背景二分只隔离两种类型；预测前景连通域可进一步隔离不相连的对象代理，却不能分开粘连建筑，且会把破碎预测拆成多个对象。报告中不可称这些为真实建筑实例。严格约束只能保证相对预测掩膜不混合，不能保证相对GT不混合。

## 6. 读出：区域不是必须替代像素

OCR 将对象上下文返回像素表示，而 Mask2Former 用掩膜限制注意力访问，二者都不等于沿固定 Q 给所有区域内像素同一个类别。借鉴的是上下文与空间约束的职责分离，不是把整个大模型一起移植。[OCR代码](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet_ocr.py)、[Mask2Former作者代码](https://github.com/facebookresearch/Mask2Former)。

SuiT 使用外部分区及区域平均/最大汇总，与可学习分区的 SPFormer 不同，说明 tokenization 与 aggregation 可独立对照。平均/最大或统计旁路不保留精确损毁位置，也不保证弥补混合。[SuiT作者代码](https://github.com/jangsoohyuk/SuiT)、[SuiT全文](https://arxiv.org/html/2412.04680v2)。本项目旧T3残差融合已失败，不能仅重命名残差作为新方案；必须明确更换了哪个约束并保持其他条件一致。

## 7. 重设计：预测建筑范围内的多子区域

### 7.1 固定内容，停止图像涂改干扰

以原A1融合后的 F_ROI 为区域内容，原RGB和SAR不压暗、不画边线。保留像素分支。第一阶段固定同一检查点、BN及建筑概率，只研究分区。六头对照若已启动可继续提供内容选择证据；不需要等另一个50k实验才能做几何诊断。

### 7.2 可解释的非学习基线

由预测建筑概率得到掩膜，在每个前景连通域代理内部生成多个子区域；背景另行生成局部区域，绝不合为一个巨大背景token。先固定阈值和空间尺度，并做小范围阈值敏感性，禁止按Test选择。小对象至少一个区域，大对象多个区域；不要无条件删除小连通域。

首选两种同范围基线：预测连通域内网格切分；预测连通域内maskSLIC。两者都使用预测掩膜，输出整数 region_id，并保证每个有效像素被覆盖。RGB用于几何分区，F_ROI用于损毁内容。掩膜分辨率、特征图下采样和映射必须统一；低分辨率下已粘连的建筑不能靠后面的约束恢复。

只按 B>=0.5 分组与按每个连通域独立分区不等价。为单独检验二者，增加“仅建筑/背景类型约束、不做对象代理隔离”这一对照。maskSLIC每个连通域分别运行与所有前景一次运行也不等价。

### 7.3 学习版仅在基线有信号后加入

为每个候选区域保存 group_id。关联定义为 q_ik = softmax_k(s_ik) over allowed candidates，allowed要求局部邻域且像素group与种子group一致。三轮中心更新、Pooling、Unpooling、mass计数必须共用同一套分组ID。仅给旧iterative_assignment传入建筑B、但仍用未分组ids进行后续汇总，会重新混合甚至索引错配。

每个位置必须有合法候选：小/狭长对象专属种子、空中心处理、全背景图回退均需显式实现。硬argmax分区可能不连通；拆分不连通块必须重新统计区域数，且Train/Val一致，不能只美化展示图片。

与“禁止跨边界”不同，软建筑兼容惩罚只能降低跨组倾向，不能宣称严格隔离；两者要用不同实验名。

### 7.4 显式教分区保留结构

先在未硬隔离的Q对照中加入建筑标签重建目标：训练GT建筑one-hot经Q汇总再回投，度量重建损失，单独更新分区器。GT仅在loss内使用，不作为输入、种子或Val推理特征。保持既有空间紧致先验；固定种子预算防止每像素一个区域的平凡解。

该目标是本项目参考SSN/SCN的改编，不是论文原配置。不必第一轮叠加四类损毁纯度、对比损失、边界损失。建筑二类纯度不能保证Damaged/Intact内部可分；若内部损伤依然明显，再单独检验四类语义重建，承认仅灾前RGB可能看不到灾后内部边界。低分辨率监督应妥善保留标签比例和ignore，而非粗糙缩放抹掉少数类。

### 7.5 不把错误掩膜变为硬输出标签

建筑组仍输出四类概率，背景组也可输出建筑类别。最终可保留像素读出用于边界/不可靠上下文；“像素fallback”在所有比较组都一致，不能只给新方法。若预测像素置信度用于门控，它是可验证启发式，不是正确性保证。Memory不在第一版中。

## 8. 最小实验阶梯与停止条件

先无训练比较区域几何；再同F_ROI/同读出/同预算比较。不能一组用新损失一组不用，却把所有收益都称为分区结构收益。

| 实验 | 唯一目标变化 | 主要问题 |
|---|---|---|
| P0 | 当前Q（原图F_ROI内容统一） | 当前分区参照 |
| P1 | 建筑/背景类型隔离 | 类型混合能否减少 |
| P2 | 预测连通域内网格，背景独立网格 | 对象代理约束是否足够 |
| P3 | 同范围maskSLIC | 内容自适应是否超越空间切分 |
| P4 | 受约束可学习Q | 学习是否超越非学习方法 |

P0-P3不需要新主干训练。分区实际区域数不能总是精确一致，需报告有效K、面积分布、运行时，并在相近K或质量-K曲线上比较，不能只匹配名义stride。P4才需要分区训练；结构监督应先单独开关对照。只有P2/P3有信号再投资P4。

第一关：GT前景/各类混合、跨GT连通域代理、硬分区连通性、建筑边界BR/BP，分别报告前景和全图，不用97%背景主导纯度掩盖风险。第二关：同像素概率投影的损失及同特征训练读出。第三关：完整前景mIoU、Damaged P/R/F1、小对象、边界与每事件表现。GT多数票ASA仅是硬区域常值分类的像素准确率参考，不是soft投影或mIoU严格上界。[评价定义](https://github.com/davidstutz/superpixel-benchmark/blob/master/docs/BENCHMARK.md)。

若几何改善但任务不涨，不继续追求超像素外观；若简单网格不弱于maskSLIC/学习Q，不强调可学习分区创新；若定位错误造成硬分组伤害，优先评估软约束或保留像素输出，不使用GT替代预测。已有长训练不自动停止、不覆盖、不删除。

## 9. 综合判断与开放问题

现有证据同时支持“混合区域值得修复”和“分类/内容路径仍有独立问题”。评价基准要求多属性和区域数控制，因而不能用纯度或可视化单独选择方案；较新的综述仍沿用多维评价，而非单一边界分数。[Stutz基准](https://github.com/davidstutz/superpixel-benchmark)、[Barcelos综述](https://arxiv.org/abs/2409.19179)、[配套评测代码](https://github.com/IMScience-PPGINF-PucMinas/superpixel-benchmark)。

任务语义需要纠正：BRIGHT按建筑标注损毁等级，多子区域的目的是保留用于判断等级的局部证据，不是要求同一真实建筑输出多种损毁标签。当前GT不能独立证明“屋顶哪一块物理损毁”的定位精度。可靠实例下的一致分类可以合理；问题是预测连通域可能混合多栋不同等级建筑，不应无条件统一投票。[BRIGHT正式论文](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html)。

剩余未知：预测建筑对象的可靠程度、特征分辨率对小建筑的上限、跨事件外观先验是否迁移。上述缺口不能靠增加Memory自动补足。可能的研究贡献是“结构约束与局部证据保留的结合及其严格验证”，并非发明maskSLIC或证明所有超像素无效。

## 10. 结论

RQ1：预测ROI内的多子区域可以显式控制空间访问，但只能相对预测掩膜保证隔离；先建立maskSLIC/网格参照。RQ2：一栋建筑多个子区域，像素分支继续保留局部判别；不做强制整栋同类投票。RQ3：内容固定、分区单变、有效区域数控制、几何与最终任务双重评价，再逐步开放学习。当前应重新定义分区职责，而不是再次无控制地换主干。

## 参考文献

[1] Jampani et al., “Superpixel Sampling Networks,” ECCV, 2018.
[2] Yang et al., “Superpixel Segmentation With Fully Convolutional Networks,” CVPR, 2020.
[3] Mei et al., “SPFormer: Enhancing Vision Transformer with Superpixel Representation,” arXiv:2401.02931, 2024.
[4] Irving, “maskSLIC: Regional Superpixel Generation with Application to Local Pathology Characterisation in Medical Images,” arXiv:1606.09518, 2016.
[5] Achanta and Süsstrunk, “Superpixels and Polygons Using Simple Non-Iterative Clustering,” CVPR, 2017.
[6] Zheng et al., “Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters,” RSE, 2021.
[7] Yuan et al., “Object-Contextual Representations for Semantic Segmentation,” ECCV, 2020.
[8] Cheng et al., “Masked-attention Mask Transformer for Universal Image Segmentation,” CVPR, 2022.
[9] Lew et al., “Superpixel Tokenization for Vision Transformers: Preserving Semantic Integrity in Visual Tokens,” arXiv:2412.04680, 2024.
[10] Stutz, Hermans, and Leibe, “Superpixels: An Evaluation of the State-of-the-Art,” CVIU, 2018.
[11] Barcelos et al., “A Comprehensive Review and New Taxonomy on Superpixel Segmentation,” ACM Computing Surveys, 2024.

[12] Chen et al., “Bright: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response,” ESSD, 2025.
