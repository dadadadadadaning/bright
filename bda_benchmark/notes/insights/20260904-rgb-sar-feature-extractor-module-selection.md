# RGB–SAR 特征提取优化：2024–2026 即插即用模块筛选与隔离实验

## 结论

可以优化，但当前阶段不建议替换完整 Swin 编码器，也不建议把多个注意力、频域和跨模态模块同时堆入主干。最稳妥的方案是在 P3/S8 已得到的 256 通道 `rgb_feature` 和 `sar_feature` 后增加零初始化残差适配器，并将单模态增强与跨模态校正分成两轮实验。

首轮优先级：

1. SAR 分支：2024 FSA 与 2024 RCM 二选一竞争；
2. RGB 分支：2024 RCM；
3. 跨模态：仅在可靠、疑似未损毁区域使用 2025 FCM；2025 CMA 作为廉价对照；
4. 2026 LWGA/CAFR 只进入第二轮；TFFA、MHCB 暂不进入主实验。

这些判断来自本地模块代码的接口、初始化、计算方式与当前 BRIGHT 结构；本文没有重新核验文件名中标注的会议、年份或论文结论。

## 一、当前模型的正确插入位置

当前 `DamageFormer_S8_MICB_DSMIL.forward` 的主路径为：

```text
Pre RGB -> encoder_1 -> multi-scale fusion -> rgb_feature [B,256,H,W]
Post SAR -> encoder_2 -> multi-scale fusion -> sar_feature [B,256,H,W]

rgb_feature -> clf_1 -> building prior
cat(rgb_feature, sar_feature) -> fusion_layer_3 -> damage head
```

推荐新增：

```text
rgb_base ---------------------------> localization head
    |
    +-> RGB adapter -> rgb_damage ---+
                                      +-> original fusion/damage head
sar_base -> SAR adapter -> sar_damage-+
```

也就是说，第一轮不得用增强后的 RGB 特征重算定位结果。建筑定位继续使用锁定的 P3 RGB 特征，模块只改变损毁分支。这样新增模块若带来收益，才能归因于损毁表征，而不是定位掩膜发生变化。

## 二、安全适配器

所有候选模块统一包装为：

\[
F'_m = F_m + \gamma_m\,A_m(F_m),\qquad \gamma_m(0)=0
\]

其中 `m` 为 RGB 或 SAR，`gamma` 为零初始化标量或逐通道参数。初始化时必须满足：

```text
enhanced model output == locked P3 output
```

建议不要直接采用模块原始输出，因为不同文件的初始化约束并不一致。即使模块内部已有残差，也应在外部增加统一的零初始化缩放，保证所有消融拥有相同起点。

## 三、候选模块筛选

### 1. SAR 分支首选

#### FSA（2024）

本地文件：`2.0 (NN 2024) Frequency Strip Attention (FSA).py`

优点：

- 输入输出同形；
- 内部 `gamma=0, beta=1`，初始化接近恒等映射；
- 用水平、垂直局部均值分解低频和高频，计算量较小；
- 与 SAR 的方向性纹理、屋顶边缘和散斑尺度差异有较自然的对应关系。

风险：源任务是图像恢复，并不自动等价于灾害损毁识别；因此只能作为 SAR 适配器候选，不能直接宣称是 SAR 专用模块。

#### RCM（2024）

本地文件：`6.0 (ECCV 2024) RCM.py`

优点：

- 形状保持；
- 内部 layer scale 默认为 `1e-6`，训练起点安全；
- 轴向、矩形上下文适合建筑物长边和块状结构。

风险：使用 BatchNorm，当前有效 batch 较小时应冻结 BN 统计或改为与基线一致的归一化；它应与 FSA 竞争，而不是首轮叠加。

首轮建议把 FSA 作为主假设，把 RCM 作为结构上下文对照。

### 2. RGB 分支首选

#### RCM（2024）

RGB 灾前影像边界清楚，RCM 的矩形上下文比频域大模块更贴合建筑结构。先放在 256 通道融合特征后，只服务损毁头。若有效，再考虑放到 Stage3；不要同时改 Stage3 和融合后特征。

#### CAFR（2026，第二轮）

本地文件：`10.0 (AAAI 2026) CAFR.py`

它更适合做同一模态内部的高低层特征校正。如果要验证，应该在 Stage3/Stage4 之间做 RGB-only 或 SAR-only 多尺度细化，而不是当作 RGB–SAR 融合模块。它会改变现有多尺度聚合路径，因而应排在融合后适配器验证成功之后。

### 3. 跨模态候选

#### 可靠性掩膜 FCM（2025）

本地文件：`48.0 (TGRS 2025) FCM.py`

FCM 会把一方的空间和通道信息加到另一方。在普通多源分割中这可能有用，但在变化检测中，全局互相校正可能把真正的损毁差异抹掉。因此不能直接全图使用，应改为：

\[
\tilde F_R=F_R+\gamma_x W_{safe}(F_R^{FCM}-F_R)
\]

\[
\tilde F_S=F_S+\gamma_x W_{safe}(F_S^{FCM}-F_S)
\]

其中：

\[
W_{safe}=\operatorname{stopgrad}(P_{building}\,P_{intact}\,C_{agreement})
\]

`W_safe` 只允许高置信建筑、倾向未损毁、且两模态可靠性较高的位置发生校正。在疑似损毁区域令其接近 0，保留真实变化。

#### CMA（2025，对照）

本地文件：`53.0 (AAAI 2025) CMA.py`

它在通道维做跨模态注意力，内部 `gamma=0`，成本比空间全注意力低。可作为“是否只需廉价通道交互”的对照，但原代码是单向输出，应分别做 `RGB<-SAR` 与 `SAR<-RGB`，且使用独立参数。

### 4. 暂不进入第一轮

- 2026 LWGA：四路局部/中程/全局处理有价值，但通道须满足拆分约束，全局分支在较大特征图上开销高，归因也比 FSA/RCM 困难。
- 2026 TFFA：空间、FFT 和所谓小波分支同时变化，显存与数值行为更复杂；代码中的“小波”实际为可学习卷积，不应当成严格小波变换解释。
- 2026 MHCB：是通用多尺度卷积候选，原实现对输入输出通道配置有隐含约束，且默认没有统一的恒等初始化。
- 2025 FFM：空间 token 交叉注意力开销较高，容易把未配准位置强制匹配；只有可靠性实验通过后才考虑在 Stage4 使用。
- 2025 MSC：实现仍构造稠密 token 相关矩阵并假定方形特征图，不适合作为当前首轮轻量验证。
- FDConv、复杂频域模块、Mamba 类模块：变量太多，不适合当前需要严格归因的阶段。

## 四、独立实验序列

### E0：锁定基线

- 固定 P3/S8 checkpoint、数据划分、seed 和评估代码；
- 新模型所有 `gamma=0`；
- 验证逐像素 logits 与原模型一致，记录最大绝对误差；
- 该步骤不训练。

### E1：SAR-FSA

- 仅启用 SAR 适配器；
- 冻结两个 Swin 编码器、定位头与原损毁头；
- 只训练 FSA 参数和外部 `gamma_sar`；
- RGB 路径完全不变。

### E2：SAR-RCM

- 与 E1 完全相同，只把 FSA 换成 RCM；
- 不允许和 FSA 叠加；
- 比较结构上下文与频率条带哪个真正有效。

### E3：RGB-RCM

- 原始 `rgb_base` 继续进入定位头；
- `rgb_damage` 只进入损毁融合头；
- 这一步专门判断灾前建筑结构增强是否改善严重度识别。

### E4：最佳单模态组合

只有 E1/E2 中至少一个通过，并且 E3 独立通过，才组合最佳 SAR 适配器与 RGB-RCM。组合模型分别从两个单模块权重加载，不从零重训全部网络。

### E5：可靠性掩膜 FCM

- 使用 E0 或 E4 作为父模型，但必须明确登记；
- FCM 只作用于 `W_safe` 区域；
- 设置三组对照：全图 FCM、建筑内 FCM、可靠未损毁区 FCM；
- 若全图更好但可靠版本无效，需要检查收益是否来自标签/场景偏差，不能直接解释为可靠跨模态校正。

### E6：CMA 对照

- 参数量和训练步数尽量与 E5 匹配；
- 判断收益来自复杂的空间/通道校正，还是廉价通道交互已经足够。

### E7：Stage3/Stage4 位置消融

仅对已经通过的模块比较：

```text
post-fusion 256-channel
Stage3 only
Stage4 only
Stage3 + Stage4
```

不要在模块筛选前做位置穷举。小建筑比例高时，Stage3 通常比 Stage4 更值得优先验证，但最终必须由有效建筑 token 数和跨事件结果决定。

### E8：接回 Utility Gate / S1–S8 证据链

编码器实验通过后，重新生成固定 S2 对新父模型的逐像素修改审计，不能沿用旧 P3 的 `corrected/harmed` 标签。然后才测试：

```text
new encoder parent -> fixed S2 -> Utility Gate -> Memory / IM4CD / Relation / RGID
```

编码器优化和 Utility Gate 必须各自先独立通过，最后的联合实验才有可解释性。

## 五、训练策略

第一阶段只训练适配器 500 optimizer steps，用于快速淘汰，不用于宣布最终结论。若曲线仍明显上升，可延长到 1500；若 500 步已稳定且低于基线，应停止。

通过 pilot 后再运行预注册完整训练：

- adapter 学习率：主学习率；
- 原 Stage4（可选解冻）：adapter 学习率的 0.1 倍；
- 其余编码器、定位头先冻结；
- 不使用 Val450 选模块或阈值；
- 所有候选使用同一训练预算与早停规则。

建议损失：

\[
L=L_{semantic}+\lambda_{anchor}L_{anchor}
\]

其中 `L_anchor` 约束适配后特征不要无界偏离基线；在 P3 高置信且预测稳定区域权重更大。跨模态一致性损失只能施加在 `W_safe` 区域，不能强迫疑似损毁区域对齐。

## 六、准入指标

除总体 mIoU 与 Damaged F1 外，必须记录：

- Intact、Damaged、Destroyed 分类别 F1/IoU；
- 14 事件宏平均、最差事件、正向事件数；
- P3 gate-miss Damaged 子集；
- fixed S2 的 corrected、harmed、net 重新审计；
- 建筑面积分层，特别是小建筑；
- 参数量、显存、吞吐；
- `gamma` 最终大小和特征漂移；
- bootstrap 置信区间。

建议 pilot 准入线：

- mIoU 相对父模型至少 `+0.30pp`；
- Damaged F1 至少 `+0.50pp`；
- 任一前景类别下降不超过 `0.50pp`；
- 至少 10/14 个事件方向非负；
- 相对父模型 bootstrap 下界大于 0；
- 推理成本增长与收益相称。

## 七、最终推荐

当前最值得立即执行的顺序为：

```text
E0 exact reproduction
  -> E1 SAR-FSA
  -> E2 SAR-RCM
  -> E3 RGB-RCM (damage path only)
  -> 独立通过后组合
  -> reliability-masked FCM
  -> CAFR / LWGA 的 Stage3/4 二轮实验
  -> 重做 S2 有害修改审计并接回 Utility Gate
```

论文层面的主张不应写成“堆叠最新即插即用模块”，而应收敛为：

> 在保持灾前建筑定位不变的前提下，使用模态专用的安全残差适配器改善 RGB 结构表征与 SAR 频率/方向表征，并仅在可靠未变化区域执行跨模态校正，从而避免将真实损毁差异错误对齐。

## 八、FSA/RCM 的非破坏式实现补充

实验实现位于 `experiments/spi_sar_adapter_v1/`，没有修改 P3/S8 源模型。父模型先严格加载原 checkpoint 并完全冻结，适配器 checkpoint 单独保存。

FSA 不应再乘一个零初始化的外层门控。原 FSA 在内部混合系数为零时已经输出恒等映射；如果外层门控也为零，会形成零乘零，使内层参数和外层门控都可能收不到有效的首步梯度。安全实现保留一次零初始化混合，并用 `tanh` 限制最大扰动。

RCM 原实现的 layer scale 为 `1e-6`，只是近似恒等而非严格恒等。实验实现将逐通道残差门控设为严格零初始化；门控第一步可以收到来自非零 RCM 分支的梯度，随后分支参数开始学习。

pilot 阶段仅增强 SAR 进入像素损毁头的路径，保持以下组件不变：

- RGB 定位头；
- building prior；
- MICB/DSMIL；
- Memory；
- fixed S2；
- Utility Gate。

模块通过后，它将形成新的父模型。此时必须重新产生 S2 的 corrected/harmed 标签，并重新校准 Utility Gate；旧审计结果不能直接沿用。IM4CD、Relation 和超像素如果直接读取 SAR 特征，也必须在新父模型下重新审计特征分布。最小化影响的方法不是强行保持所有下游数值不变，而是延迟耦合：先证明 SAR 适配器独立有效，再逐层重新校准下游模块。

## 九、500步内部校准实证结论（2026-09-05）

FSA的mIoU增益为`+0.0001168pp`，Damaged F1增益为`+0.0000806pp`，bootstrap 95%区间为`[-0.0000168,+0.0002663]pp`。该数量级实质上等于P3，事件非负并不代表存在有效改进，而是因为预测几乎没有变化。FSA在当前“冻结P3、融合后256通道SAR特征”接口上判定为无效，不进入完整训练。

RCM的mIoU增益为`-0.23051pp`，Damaged F1增益为`-0.32010pp`，bootstrap 95%区间为`[-0.40272,-0.06882]pp`，区间完全小于零；Damaged IoU下降`0.41434pp`。RCM在当前接口上形成稳定负效应，停止且不与FSA组合。

该结论只否定当前迁移接口，不证明原论文模块或所有Stage位置普遍无效。若未来考虑Stage3版本，必须先用冻结特征探针证明Stage3的频率或轴向统计对目标错误子集具有独立区分能力；不能仅因模块名称与SAR纹理看似匹配就再次训练。
