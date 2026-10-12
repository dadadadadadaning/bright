# P5层级主网络执行方案

## 用户方向与问题

在不替换为Mamba主干、不重新拆分训练集的前提下，把多尺度RGB–SAR变化表征放入DamageFormer主网络，并使用原始Train/Val/Test协议验证。

## 可迁移模块与证据

- P5原生SAR层级编码和多尺度投影：保留，负责模态非对称编码。
- P5局部双向跨注意力与共有—特有分解：保留，负责配准容忍的跨模态交互。
- 新增软层级解码：将建筑损毁分成Intact/Affected和Damaged/Destroyed两个条件决策。
- ChangeMamba（TGRS 2024，MambaCD官方代码）支持“多尺度变化建模应进入网络”的方向，但本阶段不迁移其主干，以免同时改变编码器、融合器和解码器。

## 任务匹配、创新性与难度

该方案直接匹配灾前RGB/灾后SAR异构输入。创新不是伪实例阈值，而是三段相互依赖的网络结构：非对称编码、局部配准交互、层级损毁解码。实现难度中等；P5已有前两段，主要风险在层级头梯度过弱或发生/严重度目标失衡。

## 风险与落地建议

1. 先做C2-flat与C2-hierarchical单变量对照；不要同时启用C3 DeCUR。
2. 层级转移参数零初始化，验证第0步输出严格等价。
3. 使用原始train_set训练、val_set选模；test_set只在模型冻结后导出预测，不能调参。
4. Pilot不通过时先检查两个transition、受损召回和严重度混淆，不直接换Mamba。

## 候选题目

1. Registration-Aware Hierarchical RGB–SAR Network for Building Damage Assessment
2. Asymmetric Cross-Modal Hierarchy for Optical–SAR Building Damage Assessment
3. Decoupled Multi-Scale Change Interaction with Hierarchical Damage Decoding
4. Sensor-Aware Local Correspondence Learning for Multimodal Disaster Assessment

## 趋势判断

2024–2026变化检测正从简单拼接/差分转向多尺度状态建模、局部对应和层级语义约束。本方案吸收该趋势，但通过受控P5消融避免“换主干带来的不可归因提升”。
