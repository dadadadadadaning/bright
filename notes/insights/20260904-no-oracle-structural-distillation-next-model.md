# 无 Oracle 的下一阶段模型：伪区域选择性上下文蒸馏

- 用户研究方向：灾前 RGB 与灾后 SAR 建筑损毁语义分割；不使用官方实例训练，以预测建筑区域改善像素语义结果。
- 当前核心问题：S2 Selective Context 已在 Standard-ML Val450 将 P3 从 60.5020% 提升到 61.5716%，但学习式 Adapter、Risk Gate、局部扩散和多原型均未超过 S2。下一步应把 S2 的有效行为变成可训练模型，而不是继续添加无约束注意力。
- 分析日期：2026-09-04
- 本地证据：三个即插即用代码包、P3/S2、Core-Ring、Risk-Gate、Structural-Diffusion、Multi-Prototype 实现与结果。

## 1. 已解决与未解决的问题

### 已有可靠证据

1. P3 在 Standard-ML Val450 可精确复现：valid-mIoU=60.5020%。
2. LocCC + Selective Context S2 不用官方实例、不训练、不访问 Test，可达 61.5716%，说明预测建筑区域的长距离上下文有用。
3. 局部扩散最好仅提高 0.0810 pp；多原型最好仍低于 S2 0.2245 pp；所以 S2 的收益不是普通平滑或类别原型匹配。
4. Core-Ring Adapter、学习 Gate 和组件均衡训练没有超过 P3；当前问题不是模型容量不足，而是学习目标没有明确教会模型复现 S2 的安全纠错行为。

### 仍未解决

1. S2 是推理规则，不是端到端学习模块；其收益尚未被模型吸收。
2. S2 在部分事件下降，固定 purity/margin/blend 参数不具备域自适应能力。
3. LocCC 可能粘连多栋建筑，但多原型 V1 根据 P3 自身类别建原型，形成自确认错误。
4. RGB 屋顶纹理、SAR 强散射和边界均可能成为伪损毁证据；现有余弦相似度和风险特征不能可靠识别这些错误。
5. 现有 Adapter 直接用语义 CE/Lovasz 学习，并没有把“S2 相对 P3 做了什么”作为显式训练目标。

## 2. 三个代码包的迁移判断

| 模块 | 是否进入下一轮 | 理由 |
|---|---|---|
| GBC | 作为轻量残差块候选 | 深度可分瓶颈、显存和速度友好；但模块本身不是创新，且压缩包实现的非 GN 分支错误使用 InstanceNorm3d，需重写最小版本。 |
| RCM | 暂不进入 | 与已失败的区域空间 Adapter 因果功能重合；需要先证明训练目标有效，再比较空间块。 |
| CAA | 暂不进入 | 轴向大核适合遥感长目标，但建筑方向任意，且无法解决 S2 学习目标缺失。 |
| FreqFusion | 条件式后续实验 | 官方方法针对多尺度融合的类内不一致和边界位移；P3需暴露相邻尺度。只有边界分桶审计证明错误主要集中在上采样边界时才测试。压缩包版依赖 mmcv，应改用官方无 mmcv 版本。 |
| MDAF/LCA | 排除第一阶段 | 与当前 RGB/SAR 融合功能重复，无法单独归因伪区域贡献。 |
| TFFA/FAM | 暂停 | 高频分支可能同时强化 SAR 散斑和屋顶纹理；压缩包 TFFA 中所谓 DoG/墨西哥帽只是可学习普通卷积，不是固定小波算子。 |
| Strip/SSA | 只保留 RGID 几何支线 | 有利于细长/方向结构，但不能解决语义反馈；当前 RGID 未超过 LocCC。 |
| ContrastDrivenFeatureAggregation | 只迁移前景/背景对比思想 | 原块 unfold/fold 注意力较重，4090 上没有必要；Core-Ring Adapter 已显示单纯对比特征不足。 |

## 3. 推荐新模型：S2-guided Structure-Conditioned Distillation（SCD）

### 3.1 总体流程

```text
冻结 P3（教师）
  ├─ RGB/SAR 编码与融合特征 F
  ├─ 定位概率 -> LocCC 区域 R_j
  └─ 损毁概率 p0
          │
          ├─ 固定 S2 规则 -> 软结构教师分布 qS2
          │
          └─ 区域统计广播 T(R_j)
                         │
              轻量结构残差学生（Local 或 GBC）
                         │
                  z = z0 + beta * delta_z
                         │
                       p_student
```

训练仍只使用训练集语义 TIFF；S2 教师不读取官方实例，也不读取验证/测试 GT。验证只使用 Standard-ML Val450 语义 TIFF。

### 3.2 S2 软教师

对通过固定 A6 门控的 LocCC 区域：

\[
q_j(c)=\frac{1}{|R_j|}\sum_{x\in R_j}p_0(c\mid x),\qquad c\in\{I,D,X\}
\]

\[
q_{S2}(x)=(1-\alpha_j(x))p_0(x)+\alpha_j(x)q_j
\]

与旧多原型不同，保留 S2 已被验证的单区域分布，不再根据 P3 伪类别拆分原型。

### 3.3 学生残差

区域统计包括均值与离散度：

\[
\mu_j=\frac{1}{|R_j|}\sum_{x\in R_j}F(x),\qquad
\sigma_j^2=\frac{1}{|R_j|}\sum_{x\in R_j}(F(x)-\mu_j)^2
\]

学生输入：

\[
h(x)=[F(x),\mu_j,F(x)-\mu_j,\sigma_j,p_0(x),q_j]
\]

\[
\Delta z(x)=f_\theta(h(x)),\qquad
z_s(x)=z_0(x)+\beta\tanh(\Delta z(x))
\]

第一版 `f_theta` 用参数匹配的 depthwise 3x3 残差块；第二版只把该块换成修正后的 GBC。输出层零初始化，保证训练开始时严格等于 P3。

### 3.4 训练目标

\[
L=L_{sem}+\lambda_{s2}L_{S2}+\lambda_{anchor}L_{anchor}
\]

原始监督保持不变：

\[
L_{sem}=L_{CE}(p_s,y)+L_{Lovasz}(p_s,y)
\]

结构蒸馏只在 S2 接受区域内进行：

\[
L_{S2}=\frac{\sum_x w_x KL(q_{S2}(x)\Vert p_s(x))}
{\sum_x w_x+\epsilon}
\]

其中 `w_x` 只由部署时可获得的信息构成：LocCC 接受标记、区域 purity、margin、前景概率和边界深度。它不使用 GT 判断修改是否正确，因此不是 Oracle。

区域外保持 P3：

\[
L_{anchor}=\frac{1}{|\Omega_{out}|}\sum_{x\in\Omega_{out}}
KL(p_0(x)\Vert p_s(x))
\]

语义 GT 会纠正 S2 的错误建议，S2 蒸馏则显式告诉学生需要学习的结构行为；这正是旧 M3/M4 Adapter 缺少的监督。

## 4. 无 Oracle 的受控实验

| 编号 | 唯一变化 | 目的 |
|---|---|---|
| K0 | Frozen P3 | 锁定基线 |
| K1 | 固定 S2 | 当前可部署强基线，不训练 |
| K2 | 旧 Local Adapter + 原语义损失 | 参数与额外训练控制，可复用已有失败结果 |
| K3 | Local Adapter + S2 结构蒸馏 | 验证增益是否来自明确训练目标 |
| K4 | K3 的 Local 块换成修正 GBC | 检查轻量门控空间建模的额外价值 |
| K5 | K4 + 区域均值/方差条件化 | 检查区域异质性是否比类别多原型更安全 |

禁止加入像素 Oracle、区域 Oracle、GT Selector。内部 Calibration291 只用于自动准入；通过后才访问一次 Val450。

关键因果差：

\[
K3-K2:\text{S2 蒸馏目标价值}
\]

\[
K4-K3:\text{GBC 空间块价值}
\]

\[
K5-K4:\text{区域离散度条件价值}
\]

### 自动准入

Calibration291 上：

1. K3 相对 K2 至少 +0.20 pp；
2. K3 不低于固定 S2 超过 0.20 pp；
3. Damaged F1 相对 P3 至少 +0.5 pp；
4. 任一前景 IoU 下降不超过 0.5 pp；
5. 至少 10/14 个事件不下降；
6. 不通过则停止 K4/K5 和 Val450。

Val450 最终准入仍为：相对 P3 mIoU >= +0.50 pp、Damaged F1 >= +1.0 pp、无明显前景类牺牲，并与固定 S2 做配对事件 Bootstrap。

## 5. 4090 训练策略

1. 冻结 P3，单次前向同时生成 F、p0、LocCC 和 qS2；K2/K3/K4 分支共享这次前向。
2. 不生成全量特征缓存；此前 FP32 缓存预计 208 GiB，已证明不可行。
3. 在 P3 低分辨率特征网格上矢量化区域均值/方差，使用 `scatter_add/index_add`，禁止逐区域整图循环。
4. BF16 autocast、TF32、batch=4、accumulation=4、workers=4；先 20-step speed smoke，再 500-step 单 seed pilot。
5. 500步不通过自动停止；通过才运行 3000步和三 seed。
6. 每20步记录 forward、region target、backward、step 秒数和峰值显存，避免再次出现 CPU 区域循环瓶颈。

## 6. 论文定位

建议主线不再写“学习式实例拆分提高语义分割”，而写：

> 无实例标注的预测建筑区域能够产生有效的选择性长程上下文；通过结构教师蒸馏，模型在保留像素监督的同时学习该纠错行为，并避免硬广播、类别多原型和无约束区域 Adapter 的错误传播。

候选题目：

1. **Distilling Pseudo-Region Context for Optical-SAR Building Damage Segmentation**
2. **Structure-Conditioned Residual Distillation without Instance Annotations for Building Damage Mapping**
3. **From Post-hoc Region Refinement to Learnable Structural Context in Multimodal Damage Segmentation**

## 7. 外部证据

| 来源 | 可迁移内容 | 本项目限制 |
|---|---|---|
| FreqFusion, TPAMI 2024, official repository | 自适应低通、高通和偏移重采样用于类内一致性与边界对齐 | 需要相邻尺度特征；SAR 高频可能是散斑；仅在边界诊断后测试 |
| CGRSeg/RCM, ECCV 2024 | 轻量矩形自校准和金字塔上下文 | 与已失败区域 Adapter 功能重合，不能先于训练目标验证 |
| PKINet/CAA, CVPR 2024 | 遥感大核轴向上下文 | 原任务为旋转目标检测；不是 RGB-SAR 损毁可靠性模块 |
| SCSegamba/GBC, 2025 preprint/code | 轻量门控瓶颈残差 | 可作学生残差块，但需要独立参数匹配对照，不能单独声称创新 |

## 8. 最终行动结论

停止 Multi-Prototype V1、Structural Diffusion V1、Risk Gate V1 和继续堆叠 RCM/CAA。下一轮只验证一个问题：**显式蒸馏已经成功的 S2 结构行为，能否让轻量学生在语义监督下超过 P3并接近或超过固定S2。**

若 K3 失败，说明 S2 的收益依赖推理时显式区域统计，停止“把S2学进网络”，保留S2作为最终方法；若 K3 成功，再用 K4/K5判断 GBC与区域方差是否带来额外贡献。
