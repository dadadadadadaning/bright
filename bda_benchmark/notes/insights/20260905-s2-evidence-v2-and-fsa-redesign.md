# S2 Evidence V2 与 FSA 后续修正方案

## 用户方向与问题

目标是在BRIGHT灾前RGB/灾后SAR建筑损毁识别中，保留固定S2的净收益，同时识别并拒绝其corrected/harmed混合修改；并判断融合后FSA近似无效后是否仍有合理修正空间。

## 已有实证

- fixed S2相对P3提高0.7050pp mIoU，但内部校准同时包含349,822个corrected和280,522个harmed修改。
- 基础Utility Gate校准AUROC为0.40899，停止。
- 融合后SAR-FSA提高0.0001168pp，实质等于P3；SAR-RCM降低0.23051pp且95%区间完全小于零。

因此下一步不再直接叠加编码器模块，而先审计冻结特征证据。本文结论基于本地代码和用户实验结果，未重新核验文件名标注的论文收录信息，也不提出外部趋势事实。

## 可迁移模块与匹配程度

### Relation Difference

分别计算RGB和SAR的8邻域模态内余弦关系，再比较关系差。它避免假设异构编码器同一通道具有物理对应，任务匹配较高、迁移难度低。风险是融合后256通道已混合多尺度语义，空间关系可能被平滑。

### IM4CD式局部匹配

复用现有离散局部硬峰、峰值增益、margin、entropy和去除主平移后的残差。它只是局部平移可靠性探针，不等同完整IM4CD或投影变换估计。任务匹配中等，显存风险通过`MATCHING_MAX_SIDE`控制。

### 固定RGB-SLIC与Grid

使用灾前RGB生成一次SLIC标签，RGB/SAR关系证据共享该分区；Grid使用同一请求节点预算。它只测试分区能否改善证据聚合，不把超像素当作建筑实例。任务匹配中等，依赖scikit-image，且实际SLIC节点数需记录。

## 落地结构

```text
E0 P3 + fixed S2锁定
 ├─ E1 Relation vs Direct
 ├─ E2 Matching vs Direct
 └─ E3 SLIC vs equal-budget Grid
             ↓
   Train选择通道和方向
   291内部校准检查AUROC/事件
             ↓
  只允许一个通过组训练Verifier
             ↓
  Verifier必须超过fixed S2并减少harm
```

缓存只保存S2 changed像素的35通道基础特征、证据、类别变化、目标类别和图像索引。校准缓存额外保存每图P3/fixed-S2混淆矩阵，使Verifier接受修改后能精确重建完整分割指标，不需要再次前向P3。

## FSA修正方案

1. **Stage3频率探针**：冻结Stage3特征，只审计水平/垂直高低频能量与P3 gate-miss Damaged的AUROC；不通过则不训练Stage3-FSA。
2. **FSA加可适配读出**：FSA保持融合后位置，但只额外解冻`fusion_layer_3/clf_2`并使用适配器0.1倍学习率；用于验证原冻结读出是否使FSA变化无法进入决策边界。
3. **建筑不确定区FSA**：仅在detached building prior与高熵区域应用有界FSA，避免背景纹理主导梯度。

三者是新协议的竞争假设，不能同时修改位置、读出和空间支持。由于当前FSA实质等于P3，它们的优先级低于Evidence V2。

## 风险与修正

- 旧291张已反复用于探索：本轮只能作为development；候选确定后需从Train2918冻结新的未查看确认划分。
- SLIC依赖：preflight强制检查scikit-image；缺失时停止，不把软超像素冒充SLIC。
- 匹配显存：默认在最大边80的特征图上匹配，再上采样证据；必须在论文中披露。
- 通道选择泄漏：只在Train缓存选择通道和正负方向，Calibration只评价。
- 组合归因：Verifier构造函数和运行脚本只接受一个通过组；不提供多证据训练入口。

## 建议题目方向

1. 面向异构RGB-SAR建筑损毁识别的模态内关系可靠性审计：以BRIGHT、冻结P3、fixed-S2 corrected/harmed为设置，贡献是错误条件化证据协议。
2. 基于局部硬匹配置信度的灾后SAR修改拒绝机制：比较同位差与局部匹配，贡献是配准扰动下的安全选择。
3. 共享灾前RGB-SLIC拓扑的跨模态关系聚合：以等预算Grid为强对照，贡献是分区本身的可归因验证。
4. 单证据选择性上下文验证器：只接纳独立通过的证据，贡献是harm-aware且事件稳定的S2修正框架。

## 代码

实现位于`bda_benchmark/experiments/spi_s2_evidence_v2/`。运行顺序为`preflight → smoke → audit → 单个admitted group`，Val450、Test和官方实例均不在脚本能力范围内。

## 自动续跑边界

`watch_then_run.sh`只负责等待无参数审计成功结束并验证准入JSON。默认`best`策略仅训练AUROC与事件宏AUROC排名第一的单证据分支；`all_admitted`用于独立消融，逐个训练所有通过分支，但每个Verifier仍只接入一个证据组。自动流程不会启动多证据联合、Memory、RGID、可微超像素或Val450。若两个独立分支最终均通过，只输出`ELIGIBLE_FOR_MANUAL_CONTROLLED_S8_DESIGN`，确保S8仍需重新冻结组合和消融协议后手工启动。
