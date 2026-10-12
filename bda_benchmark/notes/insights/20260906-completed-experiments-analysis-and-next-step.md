# 全部已完成实验的机制分析与下一步决策

- 日期：2026-09-06
- 当前父模型：锁定P3
- 当前最佳候选：P3 + fixed S2
- 结论依据：Val450旧结果、Train2918内部291张开发校准、S2蒸馏、Utility Gate、SAR-FSA/RCM、Evidence V2

## 1. 总结论

已经完成的实验共同指向一个明确结论：

> fixed S2能够提出总体有益的建筑区域修改，但现有逐像素特征无法跨事件稳定判断单个修改应当接受还是拒绝。

因此当前瓶颈不是“缺少更多模块”，而是监督单位、候选单位和证据单位没有对齐：

```text
S2候选单位：建筑LocCC区域
现有Gate标签：changed pixel
现有Evidence：pixel/local neighbourhood
最终目标：建筑损毁语义
```

继续增加像素级MLP、训练步数、频域模块或跨模态注意力，不能自动解决这个层级错位。

## 2. 已经被数据支持的结论

### 2.1 建筑区域长距离上下文有效

Val450中S1 Hard和S2 Soft分别提高`+1.0379pp`和`+1.0696pp`，而普通Local Control只有`+0.0060pp`。这说明收益来自LocCC建筑区域内的信息共享，而不是一般局部平滑。

### 2.2 fixed S2有效但不安全

内部291张中fixed S2提高`+0.7050pp`，但699,348个修改中包含280,522个harmed。其决定性精度只有`55.50%`。S2不是随机后处理，但也远未达到可以无条件信任每次修改的程度。

### 2.3 简单方向规则能提高精度但损失覆盖

方向规则把决定性精度提高到`65.50%`，同时将修改数从699,348压缩到162,944。它证明修改方向存在先验，但粗方向无法区分同一方向中的不同建筑，只能作为控制组。

## 3. 已经被当前实现否定的假设

### 3.1 当前S2蒸馏不能替代fixed S2

最佳K3仅比P3高`0.0395pp`，比fixed S2低`0.6655pp`。延长训练没有依据。它说明当前学生输入/监督没有复现S2的关键区域传播机制，而不是说明所有蒸馏方法都无效。

### 3.2 35维逐像素Utility不足以判断效用

Utility Gate AUROC为`0.40899`，高分反而更偏向P3正确样本；取反也只有约`0.591`。这不是阈值0.5单独造成的问题，而是排序能力不足。

### 3.3 融合后SAR适配不是当前突破口

FSA实质恒等，RCM稳定负向且置信区间完全低于零。当前融合后256通道位置不再训练FSA/RCM；若研究Stage3，只允许先做冻结特征探针。

### 3.4 当前局部Relation和IM4CD统计不能验证S2

Relation AUROC为`0.5142`。Matching虽然达到`0.5777`，最佳通道却是同位余弦距离，匹配特有的offset/residual/confidence没有胜出，而且只有5/14事件方向正确。因此当前IM4CD实现不进入Verifier。

### 3.5 当前SLIC区域统计不足

SLIC比Grid高`0.0159` AUROC并有10/14事件方向正确，但绝对AUROC只有`0.5316`，事件宏只有`0.5087`。它只能说明RGB边界分区可能有轻微相对价值，不能支持训练读出。

## 4. 尚未被否定的核心假设

当前实验没有真正检验以下命题：

> 以“整栋预测建筑”为单位，能否判断fixed S2对该建筑的净作用为正还是为负？

现有pixel selector会遇到标签冲突：同一LocCC内可能同时存在corrected、harmed和neutral像素；同一份建筑证据被迫拟合互相矛盾的像素标签。S2本身又是区域上下文方法，因此建筑级效用比像素级效用更符合候选生成机制。

同样，当前SLIC只聚合局部relation map，没有实现：

```text
P3 building ROI ∩ pre-RGB SLIC
→ RGB/SAR共享节点
→ 分别建立模态内图边
→ 比较图关系
→ building-level utility score
```

所以原始SP-E0“建筑内部图关系差”仍未被当前结果完全否定。

## 5. 下一步实验：BU-E0建筑效用可学性审计

下一步只做无参数、建筑级审计，不训练GNN。

### 5.1 建筑标签

对每个P3 LocCC建筑区域B，定义：

```text
corrected_B = B内fixed S2纠正像素数
harmed_B    = B内fixed S2伤害像素数
neutral_B   = B内两者都错但类别变化像素数
net_B       = corrected_B - harmed_B
```

为了减少接近零的噪声区域，定义带间隔标签：

```text
positive: net_B / changed_B >= +0.10
negative: net_B / changed_B <= -0.10
ambiguous: 其他区域，只统计不参与AUROC
```

`0.10`必须预先固定，不能根据结果调整。另报告阈值0的敏感性分析，但不用于选择方法。

### 5.2 第一项必须计算的上界

构造Building Oracle：整栋建筑只能选择P3或fixed S2，选择标准使用GT，仅用于估计上限。必须比较：

```text
P3
fixed S2
direction policy
building oracle
pixel oracle
```

如果Building Oracle相对fixed S2提升不足`+0.30pp`，说明建筑级开关上限太低，应停止该路线。如果Building Oracle明显高于fixed S2且接近Pixel Oracle，才说明建筑是合理决策单位。

### 5.3 无参数特征

先使用低容量、可解释特征：

1. 建筑面积、周长、长宽比、边界比例；
2. P3建筑内entropy、margin及其均值/标准差/分位数；
3. fixed S2修改覆盖率和修改方向直方图；
4. Direct差异的建筑级均值/方差/分位数；
5. Matching真正的offset/residual/confidence统计；
6. SLIC有效节点数、节点覆盖率和Grid对照；
7. 建筑核心与边界分开统计。

每个特征方向只能在新的meta-train上确定，在未查看的内部confirm上报告。

### 5.4 准入门槛

BU-E0必须同时满足：

```text
Building Oracle vs fixed S2 >= +0.30pp mIoU
building decisive count足够，且正负类均非稀有
单组证据AUROC >= 0.60
event-macro AUROC >= 0.58
至少10/14事件AUROC >= 0.50
相对面积/方向简单控制提升 >= 0.02 AUROC
```

任何条件失败都不训练BU-E1。

## 6. BU-E0通过后的BU-E1

只训练一个小型建筑Verifier：

```text
building features
→ Linear/DeepSets小读出
→ g_B = P(fixed S2优于P3)
```

输出只能二选一：

```text
g_B >= threshold：整栋接受fixed S2
g_B < threshold ：整栋回退P3
```

不得生成第三套类别，不改变P3定位，不训练编码器。阈值只在meta-train选择。

相对fixed S2的准入门槛：

```text
mIoU >= +0.10pp
harmed减少 >= 10%
selector AUROC >= 0.60
至少60%事件不低于fixed S2
任一前景IoU下降不超过0.50pp
paired bootstrap下界 > 0
```

## 7. 真正SP-E0的顺序

只有BU-E0证明建筑级决策有上限但简单建筑特征不足时，才增加图关系：

```text
BG0：等预算Grid + 直接节点差
BG1：固定RGB-SLIC + 直接节点差
BG2：等预算Grid + 模态内图关系差
BG3：固定RGB-SLIC + 模态内图关系差
BG4：BG3空间打乱控制
BG5：BG3在SAR ±1/±2 token偏移下的稳定性
```

关键归因：

- `BG3 > BG2`才能说SLIC边界有效；
- `BG3 > BG1`才能说图关系优于直接节点差；
- `BG3 > BG4`才能说空间拓扑有效；
- 偏移下稳定才能主张配准鲁棒性。

第一版不训练GAT、不训练SLIC、不接Memory。

## 8. 数据划分修正

旧internal calibration 291张已经被多次查看，不能继续承担最终确认。下一项新方法开始前，从现有2627张gate_train中按14事件分层一次性冻结：

```text
meta_train：用于确定方向和训练Verifier
meta_calibration：用于Train阈值和早停
meta_confirm：方法完全冻结后只访问一次
```

旧291张只用于development和复现，不再用于新方法的最终准入。Val450继续锁定，Test不访问。

## 9. 其他路线的状态

| 路线 | 下一步状态 |
|---|---|
| fixed S2 | 保留为当前最佳父结果 |
| Pixel Utility Gate | 停止，不改阈值、不延长 |
| S2 Distillation | 停止当前实现 |
| IM4CD pixel verifier | 停止当前实现 |
| Local Relation verifier | 停止当前实现 |
| SLIC pooled statistics | 停止当前读出 |
| Building Utility | 下一项主实验 |
| Building SLIC Graph | BU-E0通过上限审计后再做 |
| Memory | 暂停；建筑Verifier独立通过后才可作为另一分支 |
| Selective RGID | 暂停；它改变候选区域而非验证证据 |
| 可微超像素 | 暂停；固定SLIC图先通过才有资格 |
| FSA Stage3 | 仅冻结频率探针，不直接训练 |
| RCM融合后 | 永久停止当前接口 |
| 多模块联合 | 至少两个独立分支通过后才设计 |

## 10. 论文方向修正

当前不适合把论文主线写成“融合多个2024–2026即插即用模块”。更符合现有实证的题目方向是：

> Building-Utility Verification for Selective Optical-Context Transfer in Cross-Modal Damage Assessment

核心贡献应围绕：

1. 将候选生成与候选验证分离；
2. 揭示像素级效用监督与建筑区域候选之间的层级冲突；
3. 以建筑级净效用和跨事件确认协议训练安全选择器；
4. 只有在无参数证据成立后，使用共享RGB-SLIC图关系作为可解释验证证据。

如果BU-E0的Building Oracle上限也不足，则方法创新路线应停止，fixed S2作为最终后处理结果，论文转向机制审计、错误分析或benchmark/evaluation贡献，而不是继续扩大网络。
