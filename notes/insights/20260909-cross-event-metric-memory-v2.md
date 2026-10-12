# Cross-Event Metric Memory V2 研究迁移记录

## 研究问题

PI-HSM v1在跨事件伪实例语义分类中使macro-F1、Damaged recall和Destroyed recall下降。本轮研究要区分标签污染、稀有类覆盖、语义距离、相似度置信和融合门控五类原因，并验证专门训练的跨事件Metric Key是否能恢复Memory价值。

## 可迁移机制

- 类别Memory：每类独立容量，避免Intact数量优势。
- 跨图像/跨区域对比：正对被限制为相同语义、不同灾害事件。
- 多原型思想：事件内类别原型向跨事件类别中心对齐，不把每类压成唯一固定中心。
- 分层语义：Occurrence与Severity分别读取Memory，匹配P5概率因子分解。
- 保守检索：相似度margin、熵和邻居可用性形成查询相关门，不使用全局标量。

## 代码与论文状态

本实现基于公开Memory Bank、跨图像对比、多原型和SRAC公开框架思想进行独立迁移，不是SRAC官方复现。它只复用本项目既有D1缓存，所有GT派生标签仅在meta_train训练库和meta_calibration机制筛选中使用。

## 任务匹配与新颖性

任务匹配高。潜在创新来自“跨事件正对约束 + 伪建筑父子关系 + Occurrence/Severity双库 + provenance排除”的联合设计，而不是Memory或对比学习本身。

## 难度、风险与停止条件

难度中高。最大风险是修改簇并非纯语义区域、Destroyed跨事件模式不足，以及区域分类增益不能转化为像素mIoU。若M1仍不能相对CE和无Memory基线提高至少1个百分点macro-F1，应停止，不增加容量、Top-K或Transformer深度。

## 落地路径

D0先审计数据和裸检索；M1以控制变量方式比较三种Metric目标；M2固定最佳Key后拆分Occurrence、Severity和双库。只有M2全部满足稀有类召回、事件稳定和非平凡门控，才接入P5并进行P3/P5/Memory/fixed-S2完整消融。

## 可执行题目

1. Cross-Event Metric Memory for Hierarchical Optical-SAR Building Damage Assessment
2. Provenance-Excluded Retrieval for Rare Damage Recognition in Multimodal Disaster Imagery
3. Occurrence-Severity Dual Memory with Pseudo-Instance Relational Keys
4. Event-Invariant Parent-Child Retrieval for Building Damage Segmentation

## 实验证据趋势

已有Memory研究共同支持类别均衡、跨图像关系和多个类内表示；但本项目此前结果说明，未经过跨事件目标训练的Stage2/3/4或手工拼接特征不能直接充当可靠检索Key。因此本轮首先验证表示空间，而不是把检索模块直接并入主网络。
