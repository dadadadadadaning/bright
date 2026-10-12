# RGB SAR建筑损毁中错误建筑分组的证据与改进设计

分析日期：2026年10月5日。任务是BRIGHT灾前RGB与灾后SAR四类分割，Train2918、Val450，当前开发不访问Test。本说明结合完成的四组分区对照与原始论文、作者代码，提出一个优先实验，不修改现有训练逻辑。

最重要的结论：保留当前DamageFormer前端、Soft ROI与学习Q；优先检验完全隔离建筑与背景是否让错误定位位置失去纠错信息。软Q已经存在，换一个超像素生成器并不自动解除B的硬分组。Damaged改善主要来自误报下降，同时Intact与Destroyed类别召回略降，但三类IoU均升，不能简单说牺牲两类换取一类。

## 当前证据和适用范围

证据是用户提供的ROI_PARTITION_FACTORIAL_V1正式日志，最终5000步、Val450、seed3407。所有比较除特别注明外均使用5000步，不混用独立挑选的最佳结果。前端来自A1初始化后联合训练3000步的快照，不是未改变的A1。Q、B及前端冻结，只有独立同初始化的117382参数区域读出接受训练。

| 方法 | 前景mIoU | Damaged P | Damaged R | Damaged F1 |
|---|---:|---:|---:|---:|
| G0硬网格 | 50.2495% | 54.7192% | 50.3482% | 52.4428% |
| G1硬网格建筑分组 | 60.7229% | 63.2931% | 55.3984% | 59.0832% |
| S0原软Q | 61.1712% | 63.7552% | 55.7080% | 59.4606% |
| S2原软Q建筑分组 | 61.5216% | 64.2785% | 55.7634% | 59.7189% |

S2比S0高0.3505pp，比G1高0.7987pp。S2与G1不仅区域形状不同，还存在软硬归属、占用数量差异。S2相对G1在14个事件均为正，但极小差值不能解读为可靠优势；同一种子下的10次验证也不是10次独立重复。

| S2相对S0 | 正确类别像素TP变化 | 类别Recall变化 | IoU变化 |
|---|---:|---:|---:|
| Intact | -52817 | -0.0951pp | +0.6029pp |
| Damaged | +4008 | +0.0553pp | +0.2621pp |
| Destroyed | -2112 | -0.0325pp | +0.1864pp |

Recall差值由同一GT类别支持数量计算。Damaged P提高0.5233pp，F1提高0.2583pp。用报告中的高精度P/R与GT Damaged像素数7242593反推并取整，FP约从2293724降至2244431，减少49293；TP从4034705增至4038713。反推数字不是另一次实测，应以原混淆矩阵复核。

Intact和Destroyed的类别漏判增加，不等于它们全部被改判成Damaged。需要原始harm_gt_to_wrong矩阵区分Background、Damaged及其他类别。IoU提高与TP下降并不矛盾，因为FP减少可以使分母下降。背景误报减少651503，真实前景正确类别像素合计减少50921。

S2相对S0在B定位正确的两组净增2195180，在定位错误的两组净损1594598。背景侧GT边界净增566495，前景侧边界净损89998。证据支持分组对定位判断有强依赖，但两组头经过独立训练，不能把每个像素变化唯一归因于一次路由运算。

## 论文可迁移模块与代码

以下是针对当前瓶颈的机制检索，不是穷尽综述。官方来源可访问、具体内容已核实不等于代码已在本地运行；本次未安装任何外部仓库。

| 论文与来源 | 原任务与监督 | 已核实代码与模块 | 对本项目的迁移与限制 |
|---|---|---|---|
| [SSN，ECCV2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.pdf) | RGB超像素及下游任务，支持任务相关监督与重建目标 | [作者create_net.py](https://github.com/NVlabs/ssn_superpixels/blob/master/create_net.py)，compute_assignments、exec_iter、decode_features | 借鉴软关联、中心更新和重建；当前Q已有类似机制，不应当作全新模块。官方Caffe/CUDA，不是官方PyTorch即插即用实现。迁移中等，直接针对错误B的能力有限。 |
| [SpixelFCN，CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.pdf) | RGB超像素与立体匹配，分区重建及下游任务监督 | [官方train_util.py](https://github.com/fuy34/superpixel_fcn/blob/master/train_util.py)，poolfeat、upfeat | 软归属加权汇总与回投最接近当前接口。换它并不能解除B硬隔离；PyTorch0.4时代代码有CUDA设备写死等适配风险。只迁移数学操作的成本低。 |
| [OCR，ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf) | 自然RGB语义分割，GT语义监督 | [官方seg_hrnet_ocr.py](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet_ocr.py)，SpatialGather_Module、_ObjectAttentionBlock、SpatialOCR_Module | 先软聚合类别区域，再按像素与区域关系读回上下文。启发是写入先验不必锁死读取关系。它不是逐栋建筑实例，也不是局部超像素；完整迁移改变头和参数，属于后续独立实验，成本中等。 |
| [Mask Scoring R-CNN，CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html) | 自然RGB实例分割，真实实例mask及IoU监督 | [官方maskiou_head.py](https://github.com/zjhuang22/maskscoring_rcnn/blob/master/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/maskiou_head.py)，ROIMaskIoUHead；[inference.py](https://github.com/zjhuang22/maskscoring_rcnn/blob/master/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/inference.py) | 借鉴分类置信与区域质量分开检验。不能把B高置信当作区域正确；不建议当前引入新质量头，GT连通域不是实例。直接迁移风险高，诊断思想成本低。 |
| [On Calibration of Modern Neural Networks，ICML2017](https://proceedings.mlr.press/v70/guo17a.html) | 图像与文档分类，标签用于校准评价及温度拟合 | [作者temperature_scaling](https://github.com/gpleiss/temperature_scaling)，ModelWithTemperature | 借鉴可靠性分箱，不把B熵当错误检测器。作者代码已声明停止维护、原为PyTorch0.3演示。正温度缩放不改变二分类0.5阈值的判决，不能修复当前硬组成员；软概率权重才可能改变。只读诊断成本低。 |
| [QuickQuakeBuildings，GRSL2024作者全文](https://elib.dlr.de/204614/1/QuickQuakeBuildings_Post-earthquake_SAR-Optical_Dataset_for_Quick_Damaged-building_Detection.pdf) | 灾后SAR和灾后光学、footprint，建筑级受损或完好二分类 | [官方model.py](https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage/blob/main/model.py)，MODEL_SAR分别编码SAR与footprint后融合；[dataset.py](https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage/blob/main/dataset.py) | 论文II-C考虑叠掩与footprint外证据，提示不要硬清零光学轮廓外SAR。不同于BRIGHT时相与四类协议；方向缓冲依赖观测几何，dataset.py读取已裁好的数据，不等于已提供裁剪实现。物理原则匹配，直接移植中高成本。 |

近期方向核实：[SPFormer原文](https://arxiv.org/abs/2401.02931)以及[作者实验室的TMLR2025记录](https://ucsc-vlaa.github.io/publications.html)支持Transformer可以结合自适应区域表示；本次未核实对应官方仓库，不借用其他同名3D或注视轨迹模型。它不是本项目更换Swin的理由。

## 优先方案是有限跨组重叠

这是从实验与软关联机制推导的项目假设，不是以上任一论文已经验证的BRIGHT方法，也不据此宣称原创。

保留硬决定g_i=1[B_i>=0.5]与原Q，只让每个位置向另一子组保留小权重。令u_i=epsilon+(1-2epsilon)g_i；建筑子组r_(i,k,1)=q_(i,k)u_i，背景子组r_(i,k,0)=q_(i,k)(1-u_i)。两类子组之和仍是原q，每个像素对所有支持的权重和仍为1。

- epsilon=0：恢复S2硬分组。
- 单个预先固定小epsilon，例如0.05：检验完全隔离是否有害，不称最优超参数。
- epsilon=0.5：两个组得到相同内容；在共享读出、无组嵌入、无随机组特异操作、质量归一化有效时应退化到重复的S0，可作为单元测试。

第一轮只训练epsilon=0和一个小epsilon，两组同18支持实现、相同初始化、同batch顺序、同5000步预算及同损失。不要同时引入连续B权重、新损失或新的解码器。旧S2结果保留为复核线；epsilon=0必须先复现旧接口的前向和读出梯度。

最重要的风险：每位置5%的交叉权重不等于每区域只有5%的污染。例如一个区域10个建筑位置、90个背景位置，忽略其他权重时建筑子组收9.5建筑质量和4.5背景质量，背景已占32.1%。所以需统计每类质量、微小质量组、实际占用与小对象代理结果。正epsilon会激活更多槽位，不能称有效区域预算完全匹配。

### 接口与数学一致性

原特征为[n,256,h,w]，B为[n,1,h,w]，Q为[n,9,h*w]，原区域数K=ceil(h/8)*ceil(w/8)。新关联和ID为[n,18,h*w]，Token为[n,2K,128]。不是增加主干或读出参数，增加的是关联运算和中间激活；区域部分9变18，整模型显存不能简单说翻倍。

```python
# 设计伪代码；未作为训练实现写入工程
# q, ids: [batch, 9, pixels]; building: [batch, 1, pixels]
g = (building >= 0.5).float()
u = epsilon + (1.0 - 2.0 * epsilon) * g
r = torch.cat([q * (1.0 - u), q * u], dim=1)
rid = torch.cat([2 * ids, 2 * ids + 1], dim=1)
# pool 与 unpool 均使用同一 r 和 rid；区域总数为 2*K
```

注意力主路应按r*attention求和，并除以相同权重质量；mean/std使用r归一化；回投用r混合四类概率。不能改Pooling却仍用旧硬ID回投。所有遍历支持的循环须从range(9)改为读取候选维度，仅改一个循环会漏掉一半支持。

空区域不参与回投；质量归一化需防零且显式标记无效组。如对微小质量组作屏蔽，必须两组统一并重新归一化剩余关联，不可悄悄截断。注意力可能导致主路质量极小，需单独检查有限值。

保留当前原生B与Q的同分辨率约束、输出双线性上采样align_corners=False及离线B最近邻映射，不在这轮改坐标或加入连通性修复。GT仅进入Train损失与离线诊断，不输入r、rid或推理路由。前端/Q/B依旧detach，梯度只更新读出；不得称作端到端分区学习。

本地检查位置：[原pool与Readout](../../bda_benchmark/experiments/damageformer_roi_partition_control_v1/core.py)和[四组partition](../../bda_benchmark/experiments/damageformer_roi_partition_factorial_v1/core.py)。它们当前存在多个固定9支持循环，因此新方案必须隔离适配，不直接覆盖旧实验。

## 两项后续方案而非同时叠加

1. 概率软二分：把u_i改成B_i，检验置信幅度是否比单纯保留跨组通道更有价值。与固定epsilon方案分开训练。真建筑B=0.01时仍几乎属于背景，不能称为自动定位纠错。若比硬组或固定重叠更差，且错误仍是高置信，停止把B熵当可靠性门控。
2. 受先验约束但可重估的局部读回：参考OCR，将建筑先验用于区域汇总，回读时由相同像素内容与附近区域内容重新计算关系，保留一个像素或未分组参考。它改变读出机制和容量，必须另设同容量像素适配器及固定读回对照；不与第一轮同时实施。若只增加容量而没有ROI错误子集改善，就不能说先验纠错有效。

QuickQuakeBuildings提供的是保留SAR支持域的物理提醒。当前代码没有直接以光学轮廓裁剪SAR，也已编码空间上下文；本轮不能把性能下降直接归因于SAR证据被裁掉。若未来实测支持域限制有害，才在知道几何信息的条件下单独比较邻域支持，不把RGB减SAR当真实变化。

## 检查顺序与停止条件

1. 先读已有audit_final.json中的ROI漏检和误检组转移矩阵，不新增训练。检查Intact/Destroyed被改成Background还是Damaged。
2. 固定最终S0/S2权重，按B区间统计ROI错误类型、harm/rescue及基线正确分母；为高置信错误单列结果。Val可做离线诊断，不用GT决定推理分组。校准拟合如以后需要，另设与报告评价分开的数据划分或交叉拟合，不能在同一批Val拟合并当独立验证。
3. 若要先解释Q的0.7987pp优势，补训练已有H1：与G1比较硬结构，与S2比较软硬机制。H1须训练自己的读出；推理argmax干预不是同等训练对照。
4. 方法改进先只跑有限跨组重叠，检查固定5000步终点和完整曲线；读出18支持epsilon=0回归通过后再启动。全图mIoU、三类P/R/F1、背景误报、ROI错误组、GT前景侧边界必须一起报告。
5. 若ROI错误组损失减少但背景误报大幅反弹且总体不改善，应承认权衡，停止宣称该改动有效，不立即接Memory掩盖负结果。阈值容忍范围须在看新结果前确定。
6. 若出现正向信号，再做至少三个配对训练种子。验证统计不确定性用每图/事件混淆矩阵进行成对重采样后重算mIoU，明确事件固定还是推广新事件；不把像素作为独立样本。种子间波动和Val采样不确定性分开。

## 研究题目候选与创新边界

这三个题目是递进备选，不是要求同时实现三个模块。

1. BRIGHT RGB SAR建筑损毁中受限跨组信息交换的区域读出：借鉴SSN和SpixelFCN软关联，保留固定Q与B，比较epsilon=0与固定小epsilon；预期贡献是检验抑制混合和容忍错误先验之间的边界。
2. BRIGHT预测建筑先验的置信幅度与区域分类可靠性：借鉴校准评价和Mask Scoring的质量区分，比较固定交叉质量和B概率分配；预期贡献是区分高置信错误与阈值不确定，非承诺新模型涨分。
3. RGB SAR损毁分割中结构先验写入与内容驱动读回的解耦：借鉴OCR的区域关系重估，局部区域Token作为上下文并匹配像素适配器容量；预期贡献是检验从错误先验中恢复读回自由度，迁移成本及混杂风险最高。

软归属、加权汇总、区域上下文、温度缩放均为已有机制。RGB SAR结构先验适配、错误ROI诊断与受限交换是本项目的应用设计；是否有充分文献新颖性仍需专门查新。任何公式、拼接或加入ROI本身不构成充分创新。没有依据保证达到75%，也没有证据说明区域只有加入Memory才有用。

本次仅新增此说明文档，未改旧模型、未启动训练、未删除结果。文献与代码核实不等于服务器环境运行验证。
