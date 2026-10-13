# SAR 辅助 Q × soft 路由：文献、源码与最小组合实验

核对日期：2026-10-13。范围为六项文献／官方代码。下面区分“实际核对到的实现”“可借鉴的设计”和“尚未证明的本项目结论”。所有 GitHub 源码链接固定到本次 `git ls-remote HEAD` 返回的提交，不使用随时变化的 main/master 链接作为实现证据。

## 1. SSN：让区域分配可学习

**Superpixel Sampling Networks，ECCV 2018。** 官方代码：[NVlabs/ssn_superpixels](https://github.com/NVlabs/ssn_superpixels/tree/082918950d768f29466148666b7f434c78663dc6)。

- 实际访问：[官方 README](https://github.com/NVlabs/ssn_superpixels/blob/082918950d768f29466148666b7f434c78663dc6/README.md) 返回成功。确认这是 ECCV 2018 论文配套代码，使用 Caffe，可设置超像素数量、迭代次数，并额外执行连通性后处理。
- 作者项目页由 README 提供：http://varunjampani.github.io/ssn 。本次尝试 NVIDIA 官方 PDF 地址与作者 PDF 地址，均返回 HTTP 403；尝试作者网页对应 GitHub raw 地址返回 404。因此**没有取得或阅读 SSN 正文 PDF**，不写页码或假称读过公式。
- 可借鉴：把 Q 视为像素到区域的信息组织算子，分别评估其区域质量和下游分类效果。
- 不能推出：本项目的软 Q 已保证空间连通、区域等于真实建筑、迭代三次最优，或者输入 SAR 就必然改善 Q。

## 2. SpixelFCN：语义重建约束区域，不只是 RGB 颜色相似

**Superpixel Segmentation with Fully Convolutional Networks，CVPR 2020。** [论文入口](https://arxiv.org/abs/2003.12929)，[官方代码](https://github.com/fuy34/superpixel_fcn/tree/ae81e171a64dc9ed26a039a1a52b87b5fe744cf1)。

- 实际读取：[loss.py](https://github.com/fuy34/superpixel_fcn/blob/ae81e171a64dc9ed26a039a1a52b87b5fe744cf1/loss.py)，`compute_semantic_pos_loss` 对语义 one-hot 与坐标执行 `poolfeat → upfeat`，从回投结果计算语义重建和位置误差。
- 可借鉴：Q 可受 Train 的四类损毁语义重建约束，使其学习对任务有帮助的信息共享；同时保留坐标约束，防止区域无限扩散。
- 实验边界：增加 SAR 输入与增加语义监督是两个因素。SAR 组若用了四类监督而 baseline 没用，涨分不能只归因于 SAR。Val 标签不能参与 Q 的生成，只能用于事后指标。
- 不能推出：自然图像的重建目标在 BRIGHT 已被验证，或者官方损失系数适合本项目。

## 3. STViT：区域汇总后回到像素，并保留像素残差

**Vision Transformer with Super Token Sampling。** 本次官方 README 的引文是 arXiv:2211.11167（2022），因此这里不额外推定会议版本。[论文入口](https://arxiv.org/abs/2211.11167)，[官方实现](https://github.com/hhb072/STViT/blob/3a499f0a633ad41ede49f58e98c473476018c9ba/models/stvit.py)。

实际读取的 `models/stvit.py`：

1. `StokenAttention.stoken_forward` 从特征生成 soft affinity（局部 9 个邻域 token）。
2. 加权汇总并按质量归一化，得到 super-token 特征。
3. `stoken_refine` 在 super-token 上做 attention／细化。
4. 用 affinity 转置将区域信息映射回原像素特征网格。
5. `StokenAttentionLayer.forward` 执行 `x = x + self.drop_path(self.attn(self.norm1(x)))`，随后还有 MLP 残差；原像素通路被保留。

- 可借鉴：区域负责提供上下文，像素通路保留细节，区域结果回投后再融合，避免每个区域只剩一份类别决定。此结构方向与本项目“像素＋区域＋Memory 共同判断”相容。
- **关键差异**：该实现的 affinity 迭代在 `torch.no_grad()` 内；不能原样复制并声称分类损失会反传到关联学习。要训练新增 SAR→Q，必须明确梯度在哪条路径流动，并做非零梯度检查。
- 不能推出：STViT 的 soft affinity 等于本项目按建筑概率进行的 soft 路由；它也没有本项目的 Train 案例 Memory 或 BRIGHT SAR 机制验证。

## 4. MCIBI：Memory 与当前像素、图内上下文一起分类

**Mining Contextual Information Beyond Image for Semantic Segmentation，ICCV 2021。** [论文入口](https://arxiv.org/abs/2108.11819)。作者维护工具箱中的[官方模型说明](https://github.com/SegmentationBLWX/sssegmentation/blob/ac52faea8f68f4a83d69d177cc74d99c98500081/docs/modelzoo/mcibi/README.md)明确标注 Official Repo，并提供论文与代码。

- 核对 [memory.py](https://github.com/SegmentationBLWX/sssegmentation/blob/ac52faea8f68f4a83d69d177cc74d99c98500081/ssseg/modules/models/segmentors/mcibi/memory.py)：将当前像素预测 softmax 与类别 Memory 相乘；当前像素特征对选择的 Memory 做 attention；通过 `bottleneck(torch.cat([feats, selected_memory], dim=1))` 融合；启用图内上下文时，再与 `feats_ms` 融合。
- 核对 [mcibi.py](https://github.com/SegmentationBLWX/sssegmentation/blob/ac52faea8f68f4a83d69d177cc74d99c98500081/ssseg/modules/models/segmentors/mcibi/mcibi.py)：只在 TRAIN／TRAIN_DEVELOP 路径通过训练标签更新 Memory，随后产生第二阶段类别预测；推理路径不更新标签库。
- 可借鉴：保留当前像素，用训练过的特征融合分类器读取图内和图外证据；不必先让历史标签投票替代当前预测，也不需要另设硬拒绝门控。
- **关键差异**：MCIBI 使用按类别组织的原型／特征 Memory，并以动量更新；不是本项目固定 3411 个区域案例的 Top64 检索。不能直接把原型库公式当成现有案例库的等价实现。
- 不能推出：历史标签投票分数低，训练融合也无效；反过来也不能用 MCIBI 在其他数据集的收益证明 BRIGHT 已有 Memory 增益。

## 5. BRIGHT：跨模态配准需要独立诊断

**BRIGHT，ESSD 2025。** [DOI](https://doi.org/10.5194/essd-17-6217-2025)，[固定版本官方配准说明](https://github.com/ChenHongruixuan/BRIGHT/blob/59269142f3a3550320513e362692732f46486985/umim_benchmark/README.md)。

- 实际读取说明：官方 UMIM 对原始光学／SAR 图像使用专家控制点评价配准，并明确这些点是代理参考，不是绝对真值；提供配准后空间偏移计算。
- 可借鉴：检验对齐 SAR cue 与人为错位 SAR cue 的差别，评价局部空间信息，而不只看有无额外输入通道。
- 控制要求：只对新增 Q cue 进行错位；原 A1 读取的灾后 SAR 保持原样。否则实验同时改变像素前端，无法把差异归给 Q。
- 数据要求：沿用当前 Train2918／Val450，不将实例级挑战榜单、跨事件协议、Test 四类 mIoU 与 Val 前景三类 mIoU 混用。

## 6. CFOG：用跨模态结构线索，而不是假设 RGB 与 SAR 亮度等价

**Fast and Robust Matching for Multimodal Remote Sensing Image Registration，TGRS 2019。** [固定版本官方说明](https://github.com/yeyuanxin110/CFOG/blob/bdae6ade539ad8de0951d384e08c97694756a5a0/README.md)。BRIGHT 的 UMIM 说明明确将其列为基础实现。

- 实际 README 描述可用梯度、边缘、相位等描述符建立结构特征，再进行跨模态模板匹配。
- 可借鉴：从局部 SAR 特征提取结构／纹理 cue，采用与既有输入预处理一致的尺度；不要把 SAR 高亮直接编码为“损毁严重”。需要检查 speckle、阴影、叠掩和配准残差的影响。
- 边界：官方工具主要是 MATLAB／Windows 配准代码，第一轮不必额外安装，也不是可直接塞进当前 Q 的 Torch 模块。上述 speckle 等注意事项属于 SAR 领域设计考虑，不是本次 README 的定量实验结论。

## 将这些启发映射到本项目

需要分清四种“soft”：Q 的像素—区域软归属；已有建筑／背景 soft 路由；Train 区域软标签；Reader 的 soft attention。它们所在位置不同，不应因为名称相同就视为同一个模块。

本项目的合理联合结构是：

```text
原 A1 的 RGB＋SAR 像素特征 ───────────────────┐
                                           │
RGB 结构＋新增 SAR cue → Q → 区域汇总／soft 路由 → 回投 ├→ 联合分类器
                                           │
同规则 Train 区域案例 → Memory Reader → 回投 ───────┘
```

这张图是待验证设计，不是文献已经验证的组合。第一阶段先测试 SAR-Q 与 soft，Memory 保持独立正在运行的实验；不要中途改动已有 R0／RT／RF 的 Q 或库。

## 最小组合验证：先归因，再组合

### A. 新增 SAR cue 的必要对照

同一初始化、同一 Train 顺序、同一训练步数和损失，训练以下 Q：

- `q_baseline`：匹配预算的原输入 Q。
- `q_rgb_capacity`：同容量新增编码器读取 RGB 派生线索。
- `q_sar`：同容量新增编码器读取对齐 SAR cue。
- `q_shifted_sar`：同容量新增编码器读取预先固定错位规则的 SAR cue。

只训练指定 Q／cue 参数，冻结原 A1 和其他部件；四类语义重建若启用，要么四组都启用，要么另建明确因素对照。对 cue 的标准化、错位量、循环平移／填充规则作协议记录。

比较 `q_sar - q_baseline` 看候选收益；`q_sar - q_rgb_capacity` 区分额外容量与新增模态；`q_sar - q_shifted_sar` 检验空间对齐的价值。后两项仍不是完整物理因果证明：错位可能引入边界伪影、仍保留事件级纹理，也可能使任务难度不同。

### B. 核心 2×2 组合

从相同训练预算的 `q_baseline` 和 `q_sar` 出发，冻结各自 Q，在每个 Q 下使用相同头初始化与训练预算，训练 `unsplit` 与 `soft` 四个组合：

| 组合 | Q | 路由 |
|---|---|---|
| B-U | 匹配 baseline Q | unsplit |
| B-S | 匹配 baseline Q | soft |
| S-U | 对齐 SAR Q | unsplit |
| S-S | 对齐 SAR Q | soft |

这里的 `soft` 必须是已核对的原 soft 路由公式，不是新添的任意门控；不把 GT 建筑掩膜当作推理输入。

交互效应（单位为百分点）：

```text
I = (S-S − S-U) − (B-S − B-U)
```

组合高于各单项才支持组合有实际价值；`I > 0` 只表示本设置中的正交互迹象，还要看种子波动。不要将旧原 Q 的 soft 涨幅直接与另一套 SAR baseline 的涨幅相加。

这个 2×2 是最小筛选，不足以单独证明 soft 机制：四个路由头的参数量相同；若要区分建筑分组与增加槽位/路由数的作用，还需在相同 Q 下加入旧 `capacity_control`；若要明确硬／软区别，再加入 `hard`。完整但较贵的方案是四种 Q 各跑 unsplit／capacity_control／hard／soft 共 16 个头，第一轮可先保留配对 baseline/SAR 下的八个头，Q 的 RGB／错位对照独立完成。

### C. 何时再接 Memory

当 SAR-Q 或 SAR-Q＋soft 有稳定候选结果后，再固定该 Q 和聚合规则，重新生成 Train 支持 key、value 与软标签，并对查询端使用相同表示规则。Q 改变区域身份，不能把新查询 Q 与旧 Q 的案例定义默认为一致。Memory 有／无对照必须匹配新的区域算子；沿用相同已选支持身份只有在有明确、可验证的身份映射时才合理。

训练目标是共同分类；attention 权重属于学习融合，不等于人为设置“接受／拒绝 Memory”的阈值门控。第一轮不引入新的拒绝门控。

## 必须记录的结果

- 每组正式末步 Val450 的前景三类 mIoU、四类 IoU、Damaged 精确率／召回率、Intact 误改及逐事件表现。
- Q 的前景混合、边界与小前景区域统计；这些属于机制诊断，不等于模型预测成绩或逐栋建筑统计。
- 所有组的固定输入、权重、Q checkpoint、池化／回投代码和 runtime 配置；不要混用同步调试与异步运行的未经匹配指标。
- 至少在候选组上完成重复种子；先验固定终点，不分别挑各组最佳中间步。

## 访问证据

本次通过 HTTPS 实际取得以下固定提交文件，未运行其大型训练：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| SSN README | 3435 | `8cf0caf211298f73a4276d8e39c526ea83b7838fa28026f23007c4ce3c74f516` |
| SpixelFCN loss.py | 1429 | `18c8a720c14559d84faec14839aef9baad8ef70790a64053bc84ed452e9325dd` |
| STViT models/stvit.py | 23746 | `cddfc7a779025df6325fd4c5eb6a8b5d910a25469f8223d72291ac00dcebf302` |
| MCIBI mcibi.py | 8586 | `21322cf47e5438e8e8a775b0202efd21fe01d01ff29eef39b28ffd20009b6f31` |
| MCIBI memory.py | 9948 | `07d7ee84964190ca2596ffe06262bfe765229a9f6edf26c7453cc81f19481dda` |
| MCIBI modelzoo README | 15546 | `f7ade611392d1b3dc525df15deb497ddfa05a1007d2211346c675fd58da15ce5` |
| BRIGHT UMIM README | 6500 | `e3571775a6e22dcdbae8c76f29421b0a952ea74f464c1c2986b9f58369ff7b34` |
| CFOG README | 2001 | `5899e58ee4bf43f22d6eff11d1a5a6c488594921eb07de1b8ed3715bf9ceaadd` |

论文链接用于溯源；本说明的具体实现结论来自以上已读取源码，而非未能访问的 PDF 正文。
