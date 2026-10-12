# 面向模型主体的 RGB–SAR 建筑损毁创新路线（2026-09-09）

## 1. 当前问题与结论

当前有效父模型为锁定的 DamageFormer/P3，辅以 LocCC、A6 与 fixed S2。伪实例路线已经证明建筑上下文和修改簇具有价值，但大量选择器实验也表明：仅依赖伪实例统计、整栋池化或 Stage3/4 粗尺度证据，难以稳定预测跨事件效用。

因此，下一篇方法论文不应继续把三个创新点都写成伪实例后处理。建议把伪实例降为辅助约束或最终可选后处理，把模型主体收敛为三个互补创新：

1. 模态专用浅层编码；
2. 配准容错的非对称跨模态交互；
3. 损毁发生—严重度层级解码。

## 2. DamageFormer 的结构缺口

本地 `bda_benchmark/model/DamageFormer.py` 使用两个非共享 Swin-T 编码器。四级特征分别上采样并拼接，经 RGB/SAR 两个 1440→256 融合层压缩；随后 RGB 与 SAR 仅在同一空间位置按通道拼接，经 512→256 融合层后，由一个四类 1×1 分类器输出损毁类别。

这带来四个可独立验证的模型缺口：

- 两个模态虽然权重不共享，但结构完全同构，没有显式建模 RGB 屋顶边缘与 SAR 散射/斑点的不同统计；
- RGB–SAR 交互仅为同位拼接，对配准残差敏感；
- 四类平面分类器没有利用 `Building → Affected → Severity` 的天然层级；
- 多尺度特征只是统一缩放后拼接，没有按结构、语义和可靠性自适应融合。

## 3. 建议的三个主创新

### 3.1 HMSE：异构模态专用编码

位置：两个编码器的 Stage1/Stage2 输出之后、原多尺度拼接之前。

RGB 分支使用轻量结构适配器，提取水平/垂直边缘、屋顶连续性和长条建筑结构；SAR 分支使用固定 Haar 高频、log-amplitude、局部中位数/MAD 与方向梯度构造多个散射专家，再由轻量门控选择专家。所有适配器采用并联残差和零初始化尺度：

`F'_m = F_m + gamma_m * A_m(F_m), gamma_m(0)=0`。

可迁移来源：本地 2025 RHDWT 的 Haar 高频分解、EdgeGaussian 的边缘平滑思想、StripModule 的方向结构建模，以及 M²CD/DMoE 的模态专用专家思想。不能直接复制原文件：RHDWT 有额外依赖，EdgeGaussian 缺少完整 Scharr/不确定性分支，部分模块来自非遥感或非异构任务。

### 3.2 RACI：配准容错的非对称跨模态交互

位置：RGB/SAR Stage2 或 Stage3 特征完成各自适配之后、原 `fusion_layer_3` 之前。

以灾前 RGB 为结构锚点和 Query；对每个 RGB 位置只在 SAR 的局部 K×K 邻域搜索 Key/Value，输出软匹配特征、匹配熵和置信度。只有高置信区域允许 SAR 证据以零初始化残差写入 affected feature：

`F_change = F_rgb + beta * confidence * LocalMatch(F_rgb, F_sar)`。

该模块不对 RGB 与 SAR 做全局 L1 对齐，也不要求真实变化区保持一致；它只解决“小范围配准误差下，应从附近哪个 SAR 响应读取证据”。应加入 ±1/±2 像素偏移、SAR 空间打乱和同位拼接三个因果对照。

可迁移来源：HeteCD 的非共享异构编码与差异挖掘、CGSL 的结构关系比较、2026 STSF-Net 的模态专用/公共特征分解。低照度增强中的 Lightweight Cross Attention 只假设分支对齐，不能直接作为 RGB–SAR 局部匹配。

### 3.3 HOSD：损毁发生—严重度层级解码

位置：替换或并联原 `clf_2` 四类平面头。

建立三个条件头：

- `p_B`：建筑发生概率；
- `p_A`：建筑条件下的受损概率；
- `p_X`：受损条件下的 Destroyed 概率。

重构四类概率：

- Background：`1-p_B`；
- Intact：`p_B(1-p_A)`；
- Damaged：`p_B p_A (1-p_X)`；
- Destroyed：`p_B p_A p_X`。

AO-E0 已经给出强因果依据：在 fixed S2 上，Affected Occurrence Oracle 仍有约 +6.00pp mIoU 和 +8.03pp Damaged F1 空间，且 14/14 事件非负。因此层级解码是当前最有数据支撑、也最独立于伪实例质量的模型创新。

## 4. 伪实例应保留的角色

LocCC/A6/fixed S2 不删除，但不再承担全部创新：

- 训练时可提供建筑内部一致性掩码、核心/边界采样和难例定义；
- 推理时 fixed S2 作为可开关后处理；
- 最终必须比较 `新模型` 与 `新模型 + fixed S2`，只有后者进一步提升才能证明两者互补。

伪实例不应控制 RGB–SAR 基础特征的全部传播，否则 LocCC 错误会传入主干。

## 5. 2024–2026 模块筛选

### A 级：进入隔离实验

- RHDWT/Haar：只用于 SAR Stage1/2 高频并联专家；固定滤波器、自实现、零初始化。
- EdgeGaussian/Scharr：只用于 RGB Stage2 结构分支；去除 MMCV 依赖并补全边缘对照。
- StripModule：只用于 RGB 浅层方向结构或小型边界头，不再次用于失败的 RGID 分裂路线。
- 局部匹配：作为 RACI 主体；必须有匹配置信度和偏移鲁棒性控制。
- 层级条件头：作为 HOSD 主体，优先级最高。

### B 级：通过基础实验后再用

- M²CD/DMoE 的 SAR 专家路由：仅当固定 Haar/MAD 专家显示信号后再学习路由。
- HeteCD/FreqHete 的一致性或频域交互：只能在 unchanged/high-confidence 区域使用，避免抹掉真实损毁差异。
- 自适应跨尺度融合：Stage2 结构、Stage3 语义、Stage4 上下文分别门控；它是增强组件，不单独冒充主创新。

### C 级：当前不建议

- 再次把 FSA 或 RCM 插入原 SAR 融合位置：既有实验已分别表现为门几乎不打开和明显负增益；
- 全量 Mamba/VMamba 替换 Swin：改变因素过多，难以归因且训练成本高；
- SAR→RGB 扩散翻译：成本高，并有生成伪变化风险；
- 通用注意力模块堆叠：不能自然解决异构统计和配准误差；
- Memory/GNN：现阶段已有跨事件检索和关系证据为负，暂缓。

## 6. 严格实验序列

### H0：锁定基线

- P3；
- P3 + fixed S2；
- 检查 checkpoint 指纹与复现误差。

### H1：只换层级解码器

- 原四类平面头；
- HOSD 三条件头；
- 等参数量平面 MLP 控制。

主要看 Affected recall、Intact→Affected 假阳性、Damaged F1 和事件稳定性。

### H2：只测 RGB 结构适配器

- identity；
- Scharr/Gaussian；
- Strip；
- Scharr + Strip（只有两者单独通过后）。

### H3：只测 SAR 散射频率专家

- identity；
- Haar 高频；
- log-MAD/方向梯度；
- 固定专家平均；
- 学习路由（只有固定专家有效后）。

### H4：只测跨模态交互

- 原同位 concat；
- 局部 K×K 匹配；
- 局部匹配 + confidence；
- SAR 空间打乱；
- RGB/SAR ±1/±2 token 偏移。

### H5：组合

至少两个独立模块通过后，才依次组合 `HOSD+HMSE`、`HOSD+RACI` 和完整模型。最后单独比较是否添加 fixed S2。

首轮使用短 pilot 只能判断模块是否打开、梯度和方向；正式结论必须使用完整训练、14-event LOEO/独立验证、bootstrap 置信区间，并报告参数量、FLOPs、推理时间和边界指标。

## 7. 建议的论文定位与题目

建议定位：模型方法论文，而不是伪实例后处理论文。

候选题目：

1. Heterogeneity-Aware Hierarchical DamageFormer for Optical–SAR Building Damage Assessment
2. Registration-Tolerant Asymmetric Interaction for Multimodal Building Damage Assessment
3. Occurrence-Severity Decoupled Learning for Optical–SAR Disaster Building Damage Assessment
4. Modality-Specialized and Hierarchically Decoded Damage Assessment from Pre-event Optical and Post-event SAR Imagery
5. Structure-Anchored Local Interaction for Registration-Robust Multimodal Building Damage Assessment

## 8. 风险与终止条件

- 如果 HOSD 不能独立优于等参数平面头，层级形式不能作为贡献；
- 如果 RACI 在偏移测试中不优于同位 concat，不能声称配准鲁棒；
- 如果 SAR 专家只在单一事件提升，必须停止学习路由；
- 若新模型只有添加 fixed S2 后才提升，需要把模型贡献和后处理贡献分开表述；
- 不得根据 Val450/Test 反复调模块，内部准入完成后再冻结访问。

## 9. 外部依据

- M²CD（2025）：https://arxiv.org/abs/2503.19406 ，代码 https://github.com/circleLZY/M2CD
- HeteCD（2025）：https://www.sciencedirect.com/science/article/abs/pii/S0924271625001066
- CGSL（2025）：https://www.sciencedirect.com/science/article/pii/S0924271625003223 ，代码 https://github.com/TongfeiLiu/CGSL-for-MCD
- STSF-Net（2026）：https://arxiv.org/abs/2604.05527
- FSTS-Net（2026）：https://www.sciencedirect.com/science/article/pii/S1569843226004504
- FreqHete（2026）：https://www.sciencedirect.com/science/article/pii/S0031320326013580
- M-CD（2025）：https://openaccess.thecvf.com/content/WACV2025/html/Paranjape_A_Mamba-Based_Siamese_Network_for_Remote_Sensing_Change_Detection_WACV_2025_paper.html
- DMoE-AttU-Net（2026）：https://github.com/aj1365/DMoE-AttU-Net

