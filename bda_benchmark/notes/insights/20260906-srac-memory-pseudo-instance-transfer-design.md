# SRAC Memory向P3伪实例体系的迁移设计

- 用户研究方向：灾前RGB与灾后SAR的多模态建筑损毁评估
- 用户研究问题：如何将SRAC的超像素检索增强思想迁移到P3、fixed S2与LocCC伪实例体系
- 分析日期：2026-09-06
- 论文证据：用户提供的SRAC架构图与实验室录用介绍；IGARSS 2026 accepted papers页面
- 证据限制：目前未获得论文全文或官方实现，因此Memory容量、损失、检索距离、Top-R、更新策略等不能作为已知论文事实

## 1. 从现有证据可以确认的SRAC结构

1. 灾前光学图像经过VMamba与Differentiable SSN生成可学习超像素。
2. 建筑掩膜以背景抑制和超像素边界突出等方式注入像素表征。
3. 超像素嵌入采用空间注意力加权池化：`mu_k = E[w_u * x_u]`，并带有均值/标准差Raw Skip Path与投影主路径。
4. 超像素特征被预计算进Memory Bank，并建立Semantic Neighbors Table。
5. MMAP Gatherer查找Top-R邻居，同时注入标签、灾害类型和区域几何元数据。
6. 查询超像素与检索邻居通过Transformer式Cross-Attention和FFN完成分类。

这说明SRAC的Memory本质是“带结构元数据的局部区域样本库”，不是简单的类别均值原型。

## 2. 不应直接照搬SRAC的原因

- 当前方法没有真值实例，而是P3建筑定位形成的LocCC伪实例，可能合并相邻建筑或切碎单栋建筑。
- fixed S2已经稳定提升，但逐像素Utility Gate失败；当前主要矛盾是选择S2修改，而不是重新训练一个完整分类器。
- 现有固定SLIC、Relation、Matching像素级证据均未通过准入；直接增加可微超像素与Cross-Attention会同时改变候选、表征和分类器，无法归因。
- 灾害类型可能成为事件捷径。正式跨事件实验必须排除同事件邻居，并将灾害类型注入作为单独消融。

## 3. 推荐主线：Pseudo-Instance Utility Memory（PIUM）

### 3.1 Memory条目

每个meta_train伪实例保存：

```text
event_id / image_id / pseudo_instance_id
P3 checkpoint SHA / split SHA
detached LocCC mask geometry
normalized frozen embedding
P3/S2 class histogram and confidence
GT-derived semantic histogram（仅用于建库监督）
utility = (corrected - harmed) / changed
utility label: positive / negative / ambiguous
validity flags: area, purity, changed pixels
```

GT只允许构造训练库标签，不进入查询特征。`ambiguous`条目不作为有益/有害邻居监督。

### 3.2 查询嵌入

第一版不训练新编码器，冻结P3/H6，从同一伪实例掩膜池化：

```text
RGB Stage3 mean/std
SAR Stage3 mean/std
P3 semantic probability histogram
core/boundary confidence statistics
shape and S2 behavior features
```

所有向量L2归一化。Stage3为主，Stage4只作竞争实验。

### 3.3 严格检索

对查询建筑B，只允许从meta_train检索，并强制：

- 排除相同图像；
- 正式结果排除相同事件；
- positive/negative效用邻居采用等预算Top-k；
- 没有足够邻居时输出availability=0并回退基础方案；
- 保存邻居ID用于泄漏审计。

效用分数：

`r_B = sum_i exp(sim(q_B,k_i)/tau) * utility_i / sum_i exp(sim(q_B,k_i)/tau)`

同时输出：Top-1相似度、Top-k均值、正负相似度差、邻居效用均值/方差、邻居标签熵、检索可用性和距离margin。

### 3.4 安全输出

Memory不直接生成新类别，只验证fixed S2：

```text
Memory认为S2有益且置信度足够 -> 整栋伪实例采用S2
否则                         -> 整栋伪实例回退P3
```

因此最终输出仍严格属于`{P3, fixed S2}`，不会改变建筑支持范围。

## 4. 实验阶梯

### BU-E0：建筑级可行性（当前必须先完成）

- Building Oracle相对fixed S2至少`+0.30pp`；
- 至少`10/14`事件非负；
- 至少一个建筑证据组达到预注册准入。

Oracle不足时，Memory选择器没有足够上限，路线停止。

### PI-M0：无参数Memory审计

不训练Cross-Attention，只测检索效用分数区分positive/negative建筑的能力。

对照：

```text
M0 geometry-only nearest neighbors
M1 frozen semantic embedding
M2 RGB-only embedding
M3 SAR-only embedding
M4 RGB+SAR embedding
M5 M4 + same-event neighbors（只作泄漏诊断）
M6 M4 + shuffled utility labels
M7 random neighbors with identical k
```

准入建议：AUROC>=0.60、event-macro>=0.58、至少10个事件方向正确，并超过geometry-only至少0.02。

### PI-M1：低容量建筑Memory Verifier

只有PI-M0通过才训练Logistic/两层MLP，输入`BU基础特征 + Memory统计`。输出整栋建筑接受S2的概率。禁止训练P3、S2和Memory编码器。

### PI-M2：伪实例内层级检索

只有PI-M1通过，才把一个LocCC伪实例拆成core/boundary或固定RGB-SLIC子节点：

```text
建筑级检索筛选候选邻居
      ↓
只在候选建筑内部进行超像素Top-R检索
      ↓
聚合为建筑级utility证据
```

这种“伪实例粗检索 + 超像素细检索”是相对SRAC更适合当前体系的改造，也避免全库超像素检索被背景与事件风格主导。

### PI-M3：Selective RGID

只有审计证明低效用主要集中在疑似合并LocCC时，才运行RGID拆分。必须保证children并集等于parent，并以随机拆分和面积规则为控制组。

## 5. 关键消融

```text
P3
P3 + fixed S2
Building Oracle
BU基础建筑选择器
BU + random Memory
BU + geometry-only Memory
BU + semantic Memory（跨事件）
BU + semantic Memory（不排同事件，泄漏诊断）
BU + shuffled labels
BU + building-to-superpixel hierarchical Memory
完整模型 - metadata
完整模型 - retrieval uncertainty
```

灾害类型元数据必须单独消融。若只在允许同事件检索时提升，则结论是事件记忆而非跨灾害泛化。

## 6. 研究创新定位

不建议以“我们也使用超像素Memory”为创新。建议定位为：

> 在无真实实例标注的多模态BDA中，以语义模型生成的保守伪实例作为决策边界，通过跨事件层级检索估计区域上下文修改的净效用，并以拒绝机制保证预测只在P3与fixed S2之间安全选择。

与SRAC的主要区别：

- SRAC检索用于超像素损毁分类；本方法检索用于伪实例级修改效用验证。
- SRAC具有可学习超像素；第一阶段使用detached LocCC，保持候选不变。
- 本方法显式排除同事件、输出不确定性并允许拒绝检索。
- 后续才引入伪实例内超像素细检索，形成层级Memory。

## 7. 风险与停止条件

- BU-E0 Oracle不足：停止全部Utility Memory。
- PI-M0低于准入：不训练Cross-Attention或Transformer。
- 同事件有效、跨事件无效：判定为事件泄漏，不进入论文主结果。
- Geometry-only与完整Memory相当：表明检索只利用形状捷径。
- 小建筑或检索不可用：强制回退fixed S2或P3，不产生第三种类别。
- 伪实例合并严重：先完成Selective RGID审计，不用GT实例修正部署掩膜。

## 8. 建议题目

1. **Cross-Event Pseudo-Instance Retrieval Verification for Multimodal Building Damage Assessment**
2. **Retrieve to Accept or Reject: Safe Building-Context Refinement for RGB-SAR Damage Mapping**
3. **Hierarchical Pseudo-Instance and Superpixel Memory for Rare Damage Recognition**
4. **Utility-Aware Retrieval under Noisy Building Pseudo-Instances for All-Weather Damage Assessment**

## 9. 证据

- IGARSS 2026 accepted papers页面列出了SRAC题目与作者，确认录用信息：<https://2026.ieeeigarss.org/papers/accepted_papers.php>
- 江海实验室介绍给出VMambaSSN、超像素检索增强、BRIGHT 75.65% mIoU与提升8.08pp等公开描述：<https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026>
- BRIGHT官方仓库确认2026实例级挑战使用pre-RGB、post-SAR和三类实例分割指标：<https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md>

