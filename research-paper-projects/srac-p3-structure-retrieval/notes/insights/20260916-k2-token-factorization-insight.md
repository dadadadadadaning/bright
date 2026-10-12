# K2跨模态证据Token的分解式设计

## 研究方向与问题

- 任务：灾前RGB与灾后SAR的建筑损毁评估。
- 基础模型：冻结P3/DamageFormer。
- 问题：如何把P3状态、RGB结构、SAR证据和跨模态关系组织成可检索Token，同时避免Memory复制P3最终判断。

## 核心结论

不能把所有统计量直接拼成一个检索向量。应把信息拆成四类：

1. `content_key`：RGB、SAR和跨模态关系的内容表示，用于近邻检索；
2. `geometry_condition`：面积、紧凑度、核心/外围尺度，用于轻量重排序；
3. `quality_gate`：对齐熵、预计偏移、建筑重叠率、区域稳定性，用于判断是否信任检索；
4. `p3_state`：P3概率、熵、margin、碎片度，只进入最终Verifier，不进入K2 Key。

Memory value保存Train标签分布、事件/图像来源以及后续可扩展的修改效用，不与Key混在一起。

## 推荐Token层级

### 1. 像素证据

使用冻结P3分类前的Stage-2高分辨率特征和Stage-3语义特征，经独立RGB/SAR投影器映射到共同维度。先做Stage-3粗对应，再在Stage-2小邻域内细化SAR对应。

像素关系包括RGB、对齐SAR、绝对差、带符号差、逐通道乘积、余弦相似度与方向结构。对齐熵和offset不作为语义内容，而作为质量输出。

### 2. 局部区域Token

灾前RGB的建筑条件SLICO只提供区域掩码。每个区域保留Core、Ring-8、Ring-16和边界四种视图。对关系特征采用mean/std/max/attention pooling；SAR额外保留分位数和局部极值；RGB额外保留边缘方向直方图。

建议第一版维度：RGB内容32维、SAR内容32维、关系内容48维，组成112维内容描述，经两层MLP得到128维L2归一化Key。几何16维、质量8维和P3状态约15维均独立保存。

### 3. 建筑级Bag

一栋建筑包含数量不固定的局部Token。第一版用Deep Sets：共享MLP编码每个Token，再做mean+max聚合；只有它明显优于简单聚合时，才运行Set Transformer。这样建筑级Key与建筑标签单位对齐，同时保留局部小损毁证据。

## 字段路由

| 字段 | 进入检索Key | 进入重排序 | 进入质量门控 | 进入Verifier |
|---|---:|---:|---:|---:|
| P3四类概率mean/std | 否 | 否 | 否 | 是 |
| P3 entropy/margin/碎片度 | 否 | 否 | 否 | 是 |
| RGB分类前特征统计 | 是 | 否 | 否 | 可选 |
| RGB边缘方向/核心外围 | 是 | 是 | 否 | 可选 |
| 面积/紧凑度 | 否 | 是 | 否 | 是 |
| SAR分类前特征统计 | 是 | 否 | 否 | 可选 |
| SAR分位数/局部异常 | 是 | 否 | 否 | 可选 |
| Core/Ring-8/Ring-16差异 | 是 | 是 | 否 | 是 |
| 对齐后RGB–SAR关系 | 是 | 否 | 否 | 可选 |
| alignment entropy/offset | 否 | 否 | 是 | 是 |
| 最大相似度/Top-k agreement/来源熵 | 否 | 否 | 是 | 是 |

## 训练顺序

1. 冻结P3，完整Train提取分类前多尺度特征。
2. 使用人工SAR位移训练粗到细对齐，监督offset并约束移动前后Key一致。
3. 用事件/类别均衡采样训练Key；跨事件监督对比损失为主，弱三分类或序关系损失为辅。
4. 用Train可靠区域建立Memory；Val/Test只读。
5. 先做裸检索与P3条件互补性审计，再训练安全Verifier；禁止直接硬覆盖P3。

## 推荐损失

```text
L_key = L_cross_event_supcon
      + 0.20 * L_soft_class
      + 0.10 * L_shift_consistency
      + 0.10 * L_offset
```

区域标签使用Intact/Damaged/Destroyed三类软分布；Occurrence只作为Verifier中的安全分解，不替代最终三分类。

## 消融

- T0：所有手工统计直接拼接。
- T1：分支编码与字段分流。
- T2：T1+局部容错对齐。
- T3：T2+Core/Ring-8/Ring-16。
- T4：T3+Deep Sets建筑Bag。
- T5：T4+安全Verifier。
- T2-shuffle：Memory标签打乱。
- T2-noalign：取消局部对齐。

评价包括裸检索Macro-F1、P3错误子集修正率、P3正确子集伤害率、Oracle recovery、Direct+Memory额外恢复、事件胜率、x/y方向位移稳定性和跨事件评估。

## 论文模块迁移

- XoFTR：迁移粗到细局部对应和匹配可靠性，不直接移植完整Transformer。
- MINIMA：迁移人工模态/位移数据产生准确对应监督的训练思想。
- HOMO-Feature：迁移跨模态稳定的边缘主方向，不依赖原始强度一一对应。
- SSN：固定SLICO通过后再考虑任务相关可学习超像素；官方实现基于Caffe，迁移成本较高。
- Deep Sets：第一版建筑Token集合聚合器，低容量且顺序不敏感。
- Set Transformer：只在Deep Sets表达不足时作为后续模型。
- NetVLAD/Patch-NetVLAD：迁移局部残差聚合和多尺度局部—全局描述思想，不直接照搬地点识别损失。

## 创新与风险

创新不应表述为“拼接更多统计量”，而应是：建筑条件、多尺度、局部容错的跨模态证据Token；检索内容、质量与P3决策状态解耦；跨事件Memory通过安全Verifier选择性回写。

风险包括P3分类前特征仍携带父模型偏差、SAR显示值存在事件域偏移、固定SLICO边界未必对应灾后损毁、手工统计维度过多导致距离被尺度支配，以及建筑Bag把小损毁平均掉。应通过分支归一化、attention/max保留极值、事件均衡采样和逐级消融控制。

## 建议题目

1. Building-Conditioned Cross-Modal Evidence Tokens for RGB-SAR Damage Retrieval
2. Quality-Gated Cross-Event Memory for All-Weather Building Damage Assessment
3. Coarse-to-Fine Relation Key Learning for Misregistered Optical-SAR Damage Mapping
4. Local-to-Building Set Retrieval for Instance-Consistent Damage Segmentation

