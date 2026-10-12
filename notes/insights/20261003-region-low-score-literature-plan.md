# 区域低分：证据分级与文献指导的实验计划

日期：2026-10-03。研究方向：BRIGHT灾前RGB、灾后SAR建筑损毁评估；DamageFormer/Swin区域分支。

## 已有证据，不等于唯一根因

固定3000步前端：pixel60.8998%、normal59.1447%、off60.3895%。关闭新增光学调制恢复1.2449pp，但背景IoU下降，Damaged IoU提高4.9064pp，属于类别权衡和推理干预，不是重训结论。相同调制特征和同容量新头的3000步结果：pixel58.8175%、region59.2195%；支持当前Q均值汇总的小幅价值，未超过旧像素。老头与新头之间还混有结构及训练历史差异。BN校准先前未恢复区域表现，不能继续当成已确认主要根因。混合区域伤害高是关联；Q概率投影近似无损不证明特征Pooling无损。

## 文献机制、收录与代码

| 来源 | 机制与可迁移点 | 限制与代码情况 |
|---|---|---|
| SSN，ECCV2018，https://arxiv.org/abs/1807.10174 | 任务可微的区域归属；区域应服务任务，不仅外观 | 原文支持；本次未核验官方环境可运行性；当前不是严格复现 |
| Superpixel Transformers，IROS2023，https://arxiv.org/html/2309.16889v2 | 局部Cross-Attention生成Token、区域Self-Attention后分类回投 | 自然图像语义分割，非BRIGHT；本次未核实官方训练代码 |
| OCR，ECCV2020，https://www.microsoft.com/en-us/research/publication/object-contextual-representations-for-semantic-segmentation/ | 聚合语义区域，再按像素与区域关系返回上下文 | 提供代码链接；不是建筑实例；旧T3残差失败意味着不可只拼接后再承诺有效 |
| ChangeOS，RSE2021，https://github.com/Z-Zheng/ChangeOS | 建筑定位与对象级语义损毁判断 | 作者官方实现存在；RGB-RGB，移植RGB-SAR存在模态风险；不能把GT实例输入推理 |
| Exploring Cross-Image Pixel Contrast，ICCV2021，https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.html | 训练时跨图像同类聚合异类分离，推理无需额外检索 | 作者论文提供https://github.com/tfzhou/ContrastiveSeg；需可训练表示，混合区域伪硬标签有风险 |

BRIGHT正式数据论文：https://essd.copernicus.org/articles/17/6217/2025/ 。比较外部指标必须先核对划分、类别口径及评估路径。

检索找到SRAC会议条目，但未获得可核验完整消融与训练代码；不能推断其Memory独立贡献或建筑实例硬限制。本次按相关机制选文，不宣称全部是最新趋势。

## 推荐实验次序

1. 完成5000步现有头探针，记录各类IoU/Precision/Recall、事件、背景误报；目前已见记录仅3060步。
2. 固定同一前端和Q，用正常调制与关闭新增调制两种内容，各训练同初始化同容量pixel/region新头。不得把关闭调制同时变成关闭A1 Soft ROI。四组相同训练步数、LR、样本；这是冻结特征可读性探针，不能取代端到端重训消融。
3. 若off更好，做normal/off完整前端匹配微调。否则根据错误定位选择表示或分区方向。保存独立目录，不改运行中源文件。
4. 对背景/建筑误报、Damaged内部到Intact伤害、全图三类mIoU、小尺寸、事件分别评估。像素净纠错不是可加mIoU分解。
5. Memory可以在同一冻结前端作同容量独立实验，不必等区域超过A1；Train-only库、同事件筛除与随机邻居/支持标签干预。不将其当作必然恢复损失的机制。

## 三个研究方案（候选，不是已证实创新）

### A. BRIGHT中保留原始辐射信息的区域损毁读出
任务：RGB-SAR损毁分割。方法：关闭新增RGB压暗/增亮，保持分区与汇总；借鉴任务适配原则。最小实验为上述四头探针与normal/off重训。难度低到中，最符合当前证据；贡献只能在公平评估后成立，去模块本身创新弱。风险：背景误报增加。

### B. BRIGHT区域Token的同图关系建模
借鉴Superpixel Transformers分类前区域交互；固定可靠特征和Q，在一个小关系模块与逐Token FFN间比较。难度中等，需报告参数、计算和重复种子；上游已有Swin上下文，不能说原模型完全没有上下文。风险：传播错误、背景占优、K平方成本。

### C. BRIGHT受损/完好类别的跨图像判别学习
借鉴Cross-Image Pixel Contrast；先确认Damaged/Intact在可靠前端仍难分，再训练轻量投影或前端，Train类别/事件均衡采样。控制相同参数的无对比损失组，推理不必保存完整Memory。难度中等；不得将混合区域简单标为多数类后无审计训练，不应强制跨事件所有同类压成单点。创新属于任务适配待验证。

研究主线：先分清特征内容、区域读出、归属、跨图证据四种因素；不同时新增多种损失和模块。
