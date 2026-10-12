# 区域路径退化：RGB–RGB文献与DamageFormer RGB–SAR迁移

日期：2026-10-02。任务：解释V2区域与像素共同下降，评估区域机制迁移风险；不新增模型。

## 当前证据

用户Val450记录：3000步region59.1445%、pixel60.8998%；5000步57.2771%、59.0854%。两分支共同下降，不能仅归因于区域汇总。
代码事实：原RGB/调制RGB复用RGB编码器、fusion1和fusion3；融合层有BN；batch2累积8不等于BN batch16；定位+区域+0.25像素监督；无Memory、无硬建筑分组；分区使用RGB嵌入与坐标，局部三轮迭代；没有SSN原论文显式重建与紧致度损失，但有空间距离偏置。
共享统计漂移、梯度干扰、区域混合是候选解释，不是已确认根因。

## 论文可迁移模块及依据

1. ChangeOS，Remote Sensing of Environment 2021：学习建筑对象、对象先验、对象后处理；其摘要与引言明确区分对象一致性和表征能力，并以学习对象替代传统超像素。官方代码：https://github.com/Z-Zheng/ChangeOS 。论文：https://www.sciencedirect.com/science/article/pii/S0034425721003564 。任务高度匹配，但光学双时相不等于RGB–SAR；迁移难度中等。不能把区域规整直接当作建筑语义正确。
2. Technical Solution Discussion for Key Challenges of Operational CNN-Based Building-Damage Assessment，Remote Sensing 2020，12(22)3808：第3节讨论像素/建筑级评价转换、轻中度损毁识别、类别不平衡与分辨率限制；检索到原文摘录，直接网页打开失败，未核查完整数值表，不报告量化增益。https://www.mdpi.com/2072-4292/12/22/3808 。说明类似权衡存在，不证明V2特定BN故障。
3. SSN，ECCV2018：可微超像素，重建与紧致度损失；第5.2节语义分割使用语义标签作为重建属性。https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.pdf 。模块相邻任务可迁移，监督需适配；本次未核验官方代码地址。不能宣称只做三轮聚类就是完整SSN。
4. Domain-Specific Batch Normalization，CVPR2019：允许共享其他参数、分开域特定BN。https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html 。是归一化设计依据，不是建筑损毁实验证据；代码本次未核验。诊断成本低，训练迁移中等。
5. Microsoft/NLRC xBD官方实现：先训练定位，再冻结定位任务参数训练损毁分类；支持分阶段训练是一种已使用策略，不证明端到端联合训练必然更差。https://github.com/microsoft/building-damage-assessment-cnn-siamese 。光学双时相；代码公开。

## 近期相关来源与边界

SRAC团队2026-04-19报道确认其四阶段方向及75.65%报告值：https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026 。本次未获得全文与官方实现，无法核实BN共享、损失系数、训练日程及完整评估协议；不据此反推用户模型应该达到相同分数。不是完整领域趋势综述。

## 落地顺序与创新评价

先执行已有隔离审计：original/shared/split BN统计；pixel/resize_only/projection/region四路径；再依证据做A1单独微调或joint_frozen_bn对照。Train校准，Val评估，禁止Test调参。冻结主干或BN、建筑分组、像素残差都不是仅凭组合即可成立的新颖性主张。

## 三个候选研究问题（备选，不同时启动）

1. BRIGHT跨模态建筑损毁的双路径归一化诊断：借鉴DSBN，比较固定权重统计与同预算冻结BN对照；预期贡献是验证调制与共享归一化的兼容条件，而非预设涨分。
2. BRIGHT建筑语义约束区域汇总：借鉴ChangeOS对象语义，比较预测建筑约束网格与学习区域，匹配预算；风险是定位漏检和对象合并，贡献是区分结构约束与区域数量作用。
3. BRIGHT损毁语义驱动可微分区：借鉴SSN重建目标，先确认标签支持，再单独验证辅助语义重建；保持主干、预算、解码一致；风险是建筑级标签无法监督内部局部损毁。只有实证增益与文献对比后才能主张创新。
