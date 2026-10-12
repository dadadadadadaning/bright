# DamageFormer伪实例主线与2024–2026即插即用模块重新定位

- 用户研究方向：灾前RGB—灾后SAR建筑损毁语义分割
- 用户研究问题：总结现有模型改动，并判断2024–2026模块代码应放到主结构的什么位置
- 分析日期：2026-09-06
- 本地依据：P3/S2全部实验汇总、BU-E0、PI-M0、Core–Ring E0、Selective RGID-E0代码、2024–2026模块文件
- 外部核验：AAAI 2026 Strip R-CNN、ECCV 2024 CGRSeg/RCM、CVPR 2025 FDConv官方页面

## 1. 当前模型实际上形成了什么

当前可部署父模型不是一个被大量注意力模块重写的DamageFormer，而是：

```text
Pre RGB -> RGB encoder -> rgb_feature -> localization head -> P3 building probability
                         |                                    |
Post SAR -> SAR encoder -> sar_feature -> fusion -> ROI mask -> pixel damage logits
                                                              |
                                                              v
                                               LocCC伪建筑 + A6可靠性
                                                              |
                                                       fixed S2 soft context
                                                              |
                                             Intact/Damaged/Destroyed prediction
```

已被实验证明的有效机制是P3预测建筑区域内的长距离语义共享。Val450上S1 Hard A6相对P3提高约1.0379pp，fixed S2提高约1.0696pp；内部校准协议上fixed S2也提高约0.7050pp。Soft相对Hard仅高约0.0317pp，所以核心贡献是“建筑范围上下文”，不是Soft公式本身。

BU-E0进一步证明，若能知道每个建筑应该使用P3还是S2，Building Oracle相对fixed S2仍约有0.9868pp空间，并覆盖Pixel Oracle空间的约73.6%。这说明修改效用在建筑内聚集。但是像素Utility Gate、整栋池化Memory、直接差、局部匹配、简单关系统计和Core/Ring二分都没有可靠预测这种效用。

因此现阶段瓶颈可分成三类：

1. LocCC是否真对应一栋建筑；
2. RGB与SAR证据如何在配准误差下比较；
3. 得到候选变化后，如何区分损毁发生与损毁等级。

## 2. 已验证模块的状态

| 模块/机制 | 结果 | 正确结论 |
|---|---|---|
| 锁定P3 | 稳定基线 | 保留编码、定位和像素损毁头 |
| LocCC+A6 | 明显正增益 | 建筑范围信息共享有效 |
| fixed S2 | 当前最佳 | 保留为父结果和候选生成器 |
| Direction Policy | 伤害更少但覆盖不足 | 只作安全控制组 |
| S2蒸馏 | 最好仅约+0.0395pp | 当前学生没有学到结构传播 |
| Pixel Utility Gate | AUROC约0.409，接受率0 | 当前输入定义失败 |
| FSA后融合SAR适配 | 约+0.0001pp | 当前接口近似恒等，不是普遍否定FSA |
| RCM后融合SAR适配 | 约-0.2305pp | 当前接口稳定有害，停止 |
| Direct/Relation/Matching/Grid/SLIC统计 | 均未准入 | 当前证据定义不能预测S2效用 |
| PI-M0跨事件Memory | AUC约0.5176 | 整栋平均表示不足，Memory检索本身未坏 |
| Core/Ring E0 | Oracle仅多约0.004pp | 简单二分没有新选择空间 |
| Building Oracle | 约+0.9868pp vs S2 | 建筑级选择空间很大但尚不可预测 |
| Selective RGID-E0 | 代码已完成，结果待运行 | 下一项分区因果实验 |

## 3. 不能把“即插即用”理解成随便插

模块是否适用至少取决于五个条件：输入语义、空间尺度、模态角色、残差初始化和它改变的因果路径。一个在检测主干有效的模块，直接放在`fusion_layer_3`后面，可能只是在扰动已稳定的损毁特征。

本项目应采用四类插入位置：

```text
A. 模态专用编码：只改善RGB结构或SAR纹理
B. 跨模态比较：处理局部错位与可靠性交互
C. 伪实例生成：改善LocCC粘连、碎裂和边界
D. 区域级判别：在少量建筑/子区域token上判断发生与等级
```

## 4. 第一优先级：把模块用于伪实例生成，而不是损毁头

### 4.1 StripModule / SSA：用于RGID Center–Boundary Head

StripModule通过水平和垂直大条带深度卷积扩大方向感受野。它与建筑长边、屋顶轮廓和成排建筑的几何模式匹配。最合理的位置不是SAR融合后，而是：

```text
fused_roi Stage3/Stage4
       -> 1x1 projection
       -> StripModule(1xK + Kx1)
       -> center head / boundary head
       -> Watershed + rollback
```

这样它只改变“区域怎么切”，损毁类别仍由P3+fixed S2决定。对照必须为原RGID头、3x3等参数卷积、Strip-RGID和随机等数量分裂。只有Strip-RGID提高Split Oracle与实际R2，才能说明方向结构有用。

### 4.2 EdgeGaussianAggregation / CAFR：用于高低层边界细化

当前RGID只读取较低分辨率`fused_roi`，小建筑和细边界容易丢失。CAFR类高低层融合更适合把Stage3细节注入Stage4语义；EdgeGaussian类模块更适合形成边界权重。推荐：

```text
Stage4 semantic center feature
Stage3 RGB edge feature -> projection/downsample
                    -> gated high/low fusion
                    -> boundary head
```

RGB只提供灾前建筑轮廓，SAR/融合特征提供灾后响应。不能让边界分支直接改damage logits。

### 4.3 RCM：不再用于SAR damage adapter，可作为几何对照

RCM当前“融合后SAR适配”已经稳定负效应，不应重跑或与FSA组合。但RCM的轴向矩形上下文与CGRSeg原本服务空间重建，因此若需要，它只能作为Strip-RGID的几何竞争对照，而不是恢复为损毁主干模块。

## 5. 第二优先级：把跨模态模块改造成容错比较

### 5.1 Lightweight Cross Attention / DeformableInteractiveAttention

它们不应在全图同位RGB/SAR token之间直接做注意力，而应在P3建筑ROI内执行局部候选匹配：

```text
RGB query token x
    -> SAR neighbourhood {x+delta}, delta in local window
    -> attention/matching confidence
    -> matched SAR token + residual + entropy
```

输出首先作为Evidence，而不是直接改类别。只有offset、匹配残差或confidence在gate-miss/utility任务上通过无参数审计，才训练轻量读出。这个迁移吸收IM4CD思想，但避免再次退化成同位余弦距离。

### 5.2 RelationAwareAttention / GeometrySelf-Attention

这类模块需要明确的关系权重或几何先验。适合在少量区域节点上工作：

```text
一栋建筑的SLIC/RGID子区域token
       + 相对位置、共享边界、建筑覆盖率
       -> relation-aware attention
       -> building occurrence evidence
```

当前简单Relation AUROC失败，所以不能直接上多层Transformer。先比较DeepSets、一层普通attention和一层relation attention；只有relation版本额外获益，才可归因于关系编码。

### 5.3 FCM/CMA：只做可靠未变化区的跨模态校正

全图互相校正可能抹掉真正损毁差异。若使用，必须加冻结可靠性掩膜：

\[
W_{safe}=P_{building}\,P_{intact}\,C_{match}
\]

只有`W_safe`高的位置执行RGB↔SAR校正，疑似损毁位置保留差异。FCM/CMA只能在局部匹配证据先通过后进入，不能作为下一项直接训练。

## 6. 第三优先级：重新研究SAR频率，但改变问题定义

### 6.1 FSA为什么仍有一个有限的重启机会

FSA的当前结果接近零，是因为最终门值很小、预测几乎没改变。它只否定融合后256通道接口。若重启，先做Stage3冻结探针：

1. 从SAR Stage3提取水平/垂直高低频能量；
2. 只评估P3漏判Damaged、建筑内部和边界污染子集；
3. 与同参数普通7x7深度卷积、随机频带和RGB-FSA对照；
4. AUROC/event稳定性不过，不训练。

若探针通过，使用一次可学习残差缩放，不能再形成内部零门乘外部零门。可以采用小非零残差起点或先只训练缩放参数的短warm-up，并监控feature drift和gate大小。

### 6.2 RHDWT与FDConv

RHDWT显式分离Haar低频和三个高频方向，适合作为“频率分解是否比普通下采样好”的Stage3 SAR对照，但增加`pytorch_wavelets`依赖，且会改变下采样路径。

FDConv是更强的空间可变频率卷积，官方工作验证了检测和分割适配能力。但本地实现很长、动态分支多，变量远多于FSA。它只有在轻量频率探针和FSA/RHDWT至少一项通过后才合理；首轮直接替换主干卷积会失去归因。

TFFA不建议使用：本地代码把可学习卷积称为小波分支，FFT分支也没有明确的逆变换/空间对应恢复，三类机制同时改变，难以形成可信论文解释。

## 7. 第四优先级：区域分类从“整栋平均”升级为层级token

PI-M0失败说明整栋平均池化抹掉了修改位置，而不是所有Memory思想都失败。若Selective RGID证明子区域有价值，可以采用：

```text
pixel feature
 -> RGID/SLIC child tokens
 -> building token
 -> neighbouring-building token
 -> cross-event memory token
```

分类必须分两步：

1. occurrence：是否应从Intact推向Affected；
2. severity：在Affected内区分Damaged/Destroyed。

Memory不得成为自由的第二分类器。第一版只让Memory输出置信度或校准残差；检索必须排除同图和同事件。只有区域表征本身先通过跨事件审计，才重启Memory。

## 8. 模块优先级清单

| 优先级 | 本地模块 | 推荐用途 | 当前意见 |
|---|---|---|---|
| P0 | Selective RGID现有代码 | 检查LocCC分区是否值得修 | 立即运行 |
| P1 | StripModule/SSA | RGID中心与边界几何头 | 最值得新建实验 |
| P1 | CAFR/EdgeGaussianAggregation | Stage3→Stage4边界细化 | 与Strip-RGID分开验证 |
| P1 | Lightweight Cross Attention | ROI内局部RGB→SAR匹配 | 先做无参数/冻结探针 |
| P2 | DeformableInteractiveAttention | 学习局部偏移匹配 | LCA通过后升级 |
| P2 | RelationAware/GeometrySelf-Attention | 子区域图的一层读出 | 分区与关系证据通过后 |
| P2 | FSA | SAR Stage3频率探针 | 允许有限重启，不直接训练 |
| P2 | RHDWT | SAR频率对照 | 注意依赖和下采样变化 |
| P3 | FDConv | 已证明频率信号后的主干升级 | 暂缓 |
| 控制 | PCA/PSA/DualPool/SCSA | 参数量匹配注意力控制 | 可作对照，不作主创新 |
| 停止 | RCM后融合SAR | 已有稳定负效应 | 不重跑当前接口 |
| 暂停 | TFFA、MHCB、Area Attention、PAT_sf | 变量多、依赖重或任务不匹配 | 不进入近期主线 |

## 9. 推荐的开放式研究路线

```text
P3 + fixed S2（锁定父结果）
        |
        +-- G0 Selective RGID-E0：分区是否有实际价值？
        |       |
        |       +-- 若有：Strip-RGID vs CAFR-Boundary
        |       +-- 若无：停止实例分裂，保留LocCC
        |
        +-- M0 ROI局部匹配探针：是否有配准容错证据？
        |       |
        |       +-- 若有：LCA -> deformable matching
        |
        +-- F0 SAR Stage3频率探针
                |
                +-- 若有：FSA vs RHDWT；之后才考虑FDConv

至少两个独立机制通过
        -> 层级child/building token
        -> occurrence residual
        -> severity MICB/DSMIL
        -> 最后才加入cross-event memory calibration
```

这里不把思路限制为“换卷积/加注意力”。模块还可以迁移为：分区生成器、边界置信度、匹配核、关系偏置、Memory键值构造、损失权重或失败回退规则。对本项目而言，这些功能迁移通常比直接替换主干更有创新性。

## 10. 最小实验矩阵

| 实验 | 唯一变化 | 必要对照 | 晋级条件 |
|---|---|---|---|
| G0 | LocCC→Selective RGID | 全分裂、随机等量、Split Oracle | R2优于R0/R1/R3且CI>0 |
| G1 | RGID head加入Strip | 等参数3x3 head | Split Oracle和实际mIoU都提高 |
| G2 | Stage3边界注入 | 无高层注入、普通concat | 小建筑分裂与语义结果同时提高 |
| M0 | ROI局部匹配 | 同位距离、随机offset | gate-miss AUROC≥0.60且跨事件稳定 |
| F0 | SAR频率统计 | RGB频率、普通卷积、随机频带 | 目标错误子集有独立信号 |
| T0 | child token读出 | building mean、DeepSets | child token跨事件优于整栋平均 |

任何模块成为新父模型后，都必须重新生成fixed S2的corrected/harmed审计；旧Utility、Memory、Relation结果不能直接沿用。

## 11. 题目方向

1. **Geometry-Selective Pseudo-Instance Correction for RGB–SAR Building Damage Assessment**
   - 用选择性RGID与方向几何头修正LocCC，只在分裂可靠时改变S2区域。
2. **Misregistration-Tolerant Region Matching for Multimodal Building Damage Mapping**
   - 在建筑ROI内做局部可变形匹配，并把匹配置信度用于Affected发生判断。
3. **Hierarchical Occurrence–Severity Reasoning with Pseudo-Building Tokens**
   - 子区域判断变化发生，MICB/DSMIL判断严重程度，避免一个分类器同时解决两个问题。
4. **Frequency-Probed SAR Adaptation for Cross-Event Damage Assessment**
   - 先证明频率统计对P3错误子集有效，再使用轻量FSA/RHDWT适配。
5. **Retrieval-Calibrated Pseudo-Instance Correction under Unseen Disasters**
   - 在有效child/building表征形成后，用严格跨事件Memory只校准纠错置信度。

## 12. 证据边界

本地模块文件是整理版原型，文件名中的年份和会议不等同于官方实现保证。已核验的代表性工作包括：Strip R-CNN/大条带卷积、ECCV 2024 CGRSeg/RCM、CVPR 2025 FDConv。其余模块在迁入前仍需核对论文、官方仓库、许可证和张量接口。原论文在其他任务上的提升也不能直接证明BRIGHT会提升。

