# S2 Evidence Ladder：有害修改拒绝的八阶段隔离路线

## 用户研究方向与问题

任务是灾前RGB—灾后SAR建筑损毁分割。当前固定LocCC-S2对P3产生正净收益，但同时引入
大量有害修改。研究问题不是继续生成更多变化，而是识别一次S2修改何时有用，并逐层验证
Memory、局部匹配、关系差和区域划分能否提供独立的修改效用证据。

## 可迁移模块与任务匹配

| 模块 | 本项目中的迁移角色 | 匹配度 | 难度 | 主要风险 |
|---|---|---:|---:|---|
| 固定S2 | 不读取标签的候选修改器 | 高 | 低 | 区域一致化伤害正确像素 |
| CMR local verifier | S2候选的跨事件Memory验证器 | 高 | 中 | 事件泄漏、类别不平衡 |
| IM4CD式局部峰值 | 配准可靠性/残差证据 | 中高 | 中 | 位移不等于损毁 |
| BIOS/RIEM式关系差 | 模态内组织变化证据 | 高 | 中 | 小建筑token不足 |
| Selective RGID | 只替换有风险LocCC的区域生成 | 中 | 中高 | 拆分本身而非语义带来伪增益 |
| H13软超像素 | 灾前RGB共享分区与等预算Grid对照 | 中 | 高 | 低分辨率、边界学习不稳定 |

对应本地依据：

- `experiments/spi_s2_utility_gate_v1/`：固定S2、方向审计和基础门控。
- `model/cmr/local_verifier.py`与`model/cmr/pooling.py`：局部相关和冻结P3区域表示。
- `script/standard_ML/im4cd_matching_runtime.py`：平移残差、匹配margin和entropy。
- `script/standard_ML/bios_e0_runtime.py`及BIOS审计：模态内关系差。
- `experiments/spi_rgid_semantic_v1/method.py`：选择性拆分与随机匹配控制。
- `model/DamageFormer_RGBSAR_H13_SPER.py`：RGB软超像素、Grid和图传播。

本轮没有重新联网核验2026年趋势；路线只依据用户提供的结果、既有论文迁移结论和仓库中
已存在的受控实现，因此不新增论文收录、性能或代码可用性主张。

## 创新主线

主线应表述为：在固定区域候选产生之后，通过跨模态可靠性和跨事件历史证据估计每次修改
的条件效用，保留纠正并拒绝伤害。创新不在“用了Memory/图/超像素”，而在候选—验证的
因果分工、跨事件防泄漏以及修改方向与证据的层级安全路由。

## 落地顺序

1. S1统计硬修改方向和argmax不变的软NLL伤害。
2. S2建立仅由训练集确定的方向白名单。
3. S3验证基础35通道Utility Gate。
4. S4只增加跨事件Memory分数。
5. S5a与S5b分别增加IM4CD和关系差，禁止合并选择。
6. S6只替换LocCC区域，做等数量随机拆分控制。
7. S7先做固定RGB软超像素与等节点Grid，再决定是否可微。
8. S8只组合至少两个独立通过模块，并做leave-one-module-out。

统一准入：相对父模型mIoU至少+0.10pp、伤害至少减少10%、selector AUROC至少0.60、
事件非负比例至少60%、任一前景IoU下降不超过0.50pp。区域分支再增加各自的结构控制。

## 可执行题目方案

1. **Cross-Event Memory-Verified Selective Context for RGB–SAR Building Damage Assessment**：
   固定S2，使用排除同事件的类别平衡Memory验证候选修改。
2. **Match-Reliability-Guided Correction Routing for Heterogeneous Damage Mapping**：
   以局部峰值、残差和熵判断S2修改是否建立在可靠跨模态对应上。
3. **Relation-Aware Utility Estimation for Selective Building Damage Correction**：
   比较建筑内部模态内关系，作为修改效用而不是直接损毁预测。
4. **Support-Conserving Region Refinement for Safe Contextual Damage Recognition**：
   选择性RGID仅改变区域分解，并用支持守恒及随机拆分控制验证作用。
5. **Evidence-Gated Candidate Verification for Optical–SAR Damage Segmentation**：
   组合独立通过的Memory与匹配/关系证据，以leave-one-module-out证明各模块贡献。

## 代码落点

统一实现位于`experiments/spi_s2_evidence_ladder_v1/`。协议文件限制父子关系；证据适配器
输出统一空间张量；单证据路由默认禁止堆叠；结果登记器验证父结果与权重指纹；advance
控制器只暴露当前科学协议允许开展的下一阶段。

