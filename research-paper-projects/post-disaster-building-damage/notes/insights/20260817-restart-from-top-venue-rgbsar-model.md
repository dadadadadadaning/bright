# 抛弃 DamageFormer 后的顶会 RGB–SAR 新基础模型选择

- 用户研究方向：灾前高分辨率 RGB + 灾后 SAR 建筑损毁评估
- 用户研究问题：是否有可直接替换当前 P3/A2/H6/H7/H8 基础的顶会模型，以及如何形成新论文主线
- 分析日期：2026-08-17

## 结论

若要真正从顶会完整代码重新开始，首选 CVPR 2026 的 MM-OVSeg 代码框架，而不是官方 BRIGHT Mask R-CNN 早融合基线。MM-OVSeg提供独立RGB/SAR DINO编码器、SAR跨模态统一预训练权重、多层双编码器融合和完整训练代码。其任务不是损毁变化检测，因此创新不应是简单替换分类头，而应改成“变化条件的跨模态统一”：只在背景/真实完好建筑上对齐传感器表征，在受损/摧毁区域保留时相差异。

若用户愿意切换到实例级任务，则架构上最直接的是 BRIGHT Challenge 第二名 Damage-Aware SAR-Optical Query Learning：独立 ResNet50-FPN、光学直通初始化、FPN晚融合、模态差异 Transformer 和可变形 Proposal Query。遗憾是论文链接的公开仓库当前为空，只能在本地 `cvprw26` Mask R-CNN 基线中重建，不能直接复现。

## 候选比较

| 候选 | 任务匹配 | 代码成熟度 | 改造难度 | 推荐用途 |
|---|---:|---:|---:|---|
| MM-OVSeg, CVPR 2026 | 中高：同为VHR RGB-SAR分割，但非时相损毁 | 高 | 高 | 新主模型工程底座 |
| BRIGHT Challenge第二名 | 最高：完全同数据同实例任务 | 低：仓库为空 | 高 | 按论文重建的实例级主线 |
| Building-Guided Pseudo-Label, IGARSS 2025 | 最高：同像素级任务 | 低/未证实 | 中 | 训练策略与对照，不宜作为顶会新底座 |
| CROMA, NeurIPS 2023 | 中：同模态但分辨率/任务不同 | 高 | 中高 | 编码器或预训练初始化 |
| SARDet-100K, NeurIPS 2024 | 中：SAR检测而非跨模态损毁 | 高 | 中 | SAR迁移策略与权重 |
| STSF-Net, arXiv 2026 | 高：在BRIGHT上做像素级MMCD | 低：仅README | 高 | 结构对照，不能直接依赖其代码 |

## 推荐新主线：CCU-BDA

建议命名：**CCU-BDA: Change-Conditioned Cross-Modal Unification for RGB–SAR Building Damage Assessment**。

```text
灾前RGB ── 冻结/轻调RGB DINO ─────┬─ RGB建筑定位头
                                  └─ 多层RGB语义

灾后SAR ── CMU预训练SAR DINO ─────── 多层SAR结构
                                         ↓
      背景/完好区域：跨模态统一，消除传感器差异
      受损/摧毁区域：禁止强制对齐，保留变化特有证据
                                         ↓
                Stage8/12深层变化条件DEF融合
                                         ↓
          background/intact/damaged/destroyed语义头
```

这不是将MM-OVSeg的开放词汇头直接换成四分类，而是修正其最不适合BRIGHT的假设：MM-OVSeg输入通常描述同一时刻/同一语义场景，而BRIGHT的RGB与SAR跨越灾害事件。如果对全部对应位置做跨模态统一，会把真实损毁也当作模态差异消除。

## CCU-BDA 与“独立编码 + 光学定位”的层级区别

“模态独立编码”和“光学主导定位”是基础架构原则，只分别回答两个问题：RGB/SAR怎样提取特征，以及建筑在哪里。它们没有回答RGB与SAR在建筑区域出现差异时，差异究竟来自成像机制、局部错位还是真实损毁。

CCU-BDA完整包含上述两项原则，并新增变化条件的跨模态统一：

1. RGB与SAR独立编码，避免共享浅层参数；
2. RGB分支独立输出建筑定位，SAR不反向破坏边界；
3. 根据建筑先验与变化状态，将空间划分为稳定区域与疑似变化区域；
4. 稳定区域执行跨模态统一，学习并消除正常传感器差异；
5. 疑似变化区域停止强制对齐，保留SAR与RGB之间的差异作为损毁证据；
6. 深层融合器同时接收统一特征、保留的变化残差和RGB建筑先验，输出损毁等级。

因此其区别可概括为：

```text
独立编码：解决“RGB和SAR不能共用同一种特征提取方式”
光学定位：解决“建筑在哪里、边界由谁负责”
CCU-BDA：解决“同一建筑上的RGB-SAR差异，哪些应消除，哪些应保留为损毁”
```

如果模型只有前两项而没有CCU，则通常只能做拼接、相减或注意力融合，所有区域使用同一种规则；CCU的研究贡献是让融合规则随真实变化状态而改变。

## 最小可行实验

1. `N0-MMOV-4C`：MM-OVSeg双编码器 + 普通四类语义头，不做变化条件限制。
2. `N1-CCU-unchanged`：仅在背景与完好建筑上执行CMU式对齐；变化区域停止对齐梯度。
3. `N2-CCU-late`：在N1上仅保留深层DEF融合，去掉浅层交互。
4. `N3-CCU-building`：使用RGB建筑概率限制融合区域，但仍保持像素级评价。

关键对照必须包含P3、A2、官方DamageFormer和原始MM-OVSeg式全区域对齐。第一轮只跑20步smoke和1000步pilot；若N1在验证集的Damaged IoU、Event mean/worst均优于N0，再进入3000步。

## 风险

- MM-OVSeg完整模型较重，官方补充材料报告全模型约331M参数；RTX 5090应优先使用ViT-B/16和冻结编码器，而不是ViT-L。
- 其Detectron2工程与当前训练框架不同，应新建独立实验目录，禁止把代码拼进DamageFormer类。
- CMU的原始SAR权重来自其他SAR传感器/分辨率，必须先做输入统计和线性探针审计。
- 不能利用测试集逐轮选择模型；跨事件泛化应以留一事件验证选择，最终测试一次。
- 如果切换为BRIGHT实例标注与mAP，论文任务、指标和数据协议会改变，应单独立项，不能与旧mIoU结果混成一张主表。

## 建议题目方案

1. **Change-Conditioned Cross-Modal Unification for All-Weather Building Damage Assessment**
   - 以MM-OVSeg双DINO为底座，提出变化条件CMU和深层融合。
2. **Optical-Anchored Radar Evidence Learning for Cross-Event Building Damage Mapping**
   - 光学定位固定，SAR结构只在建筑区域以残差形式注入，突出跨事件稳健性。
3. **Damage-Preserving Radar–Optical Representation Alignment for Heterogeneous Change Detection**
   - 对齐未变化区域、保护变化区域，将传感器域差与灾害变化显式分开。

## 证据

- MM-OVSeg（CVPR 2026）及官方代码：https://github.com/Jimmyxichen/MM-OVSeg
- BRIGHT Challenge outcome与官方资源：https://arxiv.org/abs/2607.22746
- CROMA（NeurIPS 2023）：https://proceedings.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html
- SARDet-100K（NeurIPS 2024）：https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html
