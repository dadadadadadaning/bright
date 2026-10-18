# soft＋Memory 退化：文献与代码复核

核查日期：2026-10-18。复用了已有 LwF、L2-SP、Lovász 和 ReCo 源码记录，另外读取了 PCGrad、ReCo、MCIBI 的作者代码。未运行 BRIGHT／CUDA 训练。部分论文全文网页返回 403，因此下文明确区分作者代码所支持的机制与本项目新适配，未声称复现论文成绩。

## 本项目证据的解释边界

1. Q、建筑概率、前端与案例库被冻结；同图的原始两个槽没有随训练更新。会更新的是槽编码、Reader 与联合分类头。
2. A／B／C／D 都退化，不能把共同退化唯一归因于 soft 或 Memory；四组都有两个槽，也不能排除双槽共同结构的影响。
3. D500→D5000，Train 和 Val 的原损失均改善，前景 mIoU 却下降。后期建筑二类 IoU 高于 grid_probq，但 Intact→Damaged 的净误判增加。仅保护定位或加大 Damaged 类权重都没有充分依据。
4. 0.2 倍学习率仅部分缓解；删除背景 Lovász 未显示实质改善。不能把同一实验再次作为未做过的建议。
5. 输出 KL 是新候选。它不知道哪些基线预测正确，可能同时抑制有益纠错，不能称为保证保留正确预测的方法。

## 已核查的六条研究路线

| 方法与原始资料 | 作者代码核查 | 对本项目有用的部分 | 不应混淆的地方与优先级 |
|---|---|---|---|
| Learning without Forgetting，ECCV 2016；[论文](https://arxiv.org/abs/1606.09282) | [作者仓库](https://github.com/lizhitwo/LearningWithoutForgetting/tree/795176d49017663ae76455e6536b4c2b7caf35fe)；`vl_nnsoftmaxdiff.m` 132–143 行用软目标交叉熵，并支持温度 | 以冻结 grid_probq 作为四类软目标，约束联合预测漂移，同时保留 GT 监督 | 原论文是学习新任务而保留旧任务；当前是同任务正则化适配，不是原 LwF 复现。建议首轮只检验这一项 |
| L2-SP，Explicit Inductive Bias for Transfer Learning with Convolutional Networks，ICML 2018；[论文](https://arxiv.org/abs/1802.01483) | [作者代码](https://github.com/holyseven/TransferLearningClassification/blob/1426356195d79613b12c94ad8d6d0db576b09261/model/network_base.py#L138-L156) 对已有权重惩罚与预训练权重之差，新层另做 L2 | 若输出 KL 锁住了基线错误，可单独检验将可训练后端约束在已学习的 500 步参数附近 | AdamW weight decay 拉向零，L2-SP 拉向参考参数；二者不同。需要参考权重，不能对冻结前端加此项并期待 Reader 获益。不要与 KL 同时新增 |
| PCGrad，Gradient Surgery for Multi-Task Learning，NeurIPS 2020；[论文](https://arxiv.org/abs/2001.06782) | [作者实现](https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/PCGrad_tf.py) 计算分任务梯度；内积为负才做投影 | 先借鉴梯度诊断：NLL／Lovász、原目标／KL 的范数、夹角及加权后相对量级 | 作者代码是 TensorFlow 多任务实现；把同一分割任务的损失项当作任务是新适配。负夹角不等于已找到退化根因。首轮记录诊断，不立即加入梯度投影 |
| Lovász-Softmax，CVPR 2018；[论文](https://arxiv.org/abs/1705.08790) | [作者 README](https://github.com/bermanmaxim/LovaszSoftmax/tree/7d48792d35a04d3167de488dd00daabbccd8334b) FAQ 明确 dataset mIoU 与 batch／类别组成相关；[损失代码](https://github.com/bermanmaxim/LovaszSoftmax/blob/7d48792d35a04d3167de488dd00daabbccd8334b/pytorch/lovasz_losses.py) 区分 `per_image` 与 batch，`classes='present'` | 解释为什么逐图平均 surrogate 改善，不保证全 split 混淆汇总的前景 mIoU 改善；需要同步报告按类损失、像素和事件分布 | 单纯梯度累积不等于跨图联合计算一次 Lovász。不能据此直接删除 Lovász；若以后验证 batch／采样，需独立实验 |
| ReCo，Bootstrapping Semantic Segmentation with Regional Contrast，ICLR 2022；[论文](https://arxiv.org/abs/2104.04465) | [作者实现](https://github.com/lorenmt/reco/blob/71c1b7324fd36049c6657b122ba3fef373877b72/module_list.py) `compute_reco_loss` 选低置信困难查询、计算类原型，并按负类原型相似度抽负例 | 对 Intact／Damaged 易混淆，可后续训练检索投影或判别表征，加入真正相似的 Intact 困难负例 | 这不是超像素或推理 Memory Reader。若父 T128 检索冻结，只对末端分类头加对比损失不会改变 Top64。新投影须用于查询与支持两端，重新匹配训练 |
| MCIBI，Mining Contextual Information Beyond Image for Semantic Segmentation，ICCV 2021；[论文页面](https://openaccess.thecvf.com/content/ICCV2021/html/Jin_Mining_Contextual_Information_Beyond_Image_for_Semantic_Segmentation_ICCV_2021_paper.html) | [作者框架实现](https://github.com/SegmentationBLWX/sssegmentation/blob/ac52faea8f68f4a83d69d177cc74d99c98500081/ssseg/modules/models/segmentors/mcibi/memory.py) 保存按类组织的非梯度 Memory，用预测类别概率读取，再与像素特征和图内上下文融合；训练时更新类记忆 | 若实例库大量重复或损毁类覆盖差，可将小型类别原型 Memory 作为独立对照，研究历史信息究竟来自类别原型还是具体案例 | 当前是固定父案例＋软标签 Top64，MCIBI 是训练中更新的类记忆；不能直接替换后仍称同库同预算消融。后于训练稳定性与候选质量诊断 |

以上方法均不能保证 BRIGHT 上得到相同收益，也不应将现成正则化包装为新的网络创新。

## 首轮推荐：一个共享训练器，两种目标

沿用同一个 D 模型、同一个被冻结的前端与案例库、相同数据和更新顺序。两份模型仅目标配置不同：

* control：原 NLL＋原 Lovász。
* grid_anchor：原 NLL＋原 Lovász＋λ KL(p_grid || p_joint)。

第一轮 T=1、λ=0.1 为预先指定的探索设置，不声称为最优。教师 `detach`；只排除 ignore 标签位置；有效像素均值再按图像平均；四类、双槽和 Memory 都保留。不得偷偷使用只保留教师正确像素、置信度门控、建筑硬掩膜等额外规则。

在 T=1 且无额外像素权重时，NLL＋λ KL 的 logits 梯度等价于由真值 one-hot 与教师概率混合产生的软监督（教师熵对学生是常数），再加原 Lovász 梯度。这是输出保持正则化，不是一个新的知识检索模块。

若从原 D500 续训，需恢复同一优化器、调度、图序和随机状态，两个分支共同用 0.2 倍原 LR，调度总长度仍是 5000，预定终点 1500。旧对照仅在起点、图序、后端配置完全匹配时复用。原权重缺失时，不用摘要伪造恢复；建立新实验，先训练共同 500 步起点再分叉。这个新轨迹不能冒充原实验重放，也不代表多个独立种子。

## 必须增加的诊断：不能只看 KL 和总 loss

1. **共同目标评分**：两组均打印原 NLL、原 Lovász、原总损失及 KL；新增项导致目标数值变化不算收益。
2. **基线错／对的配对变化**：按同一像素统计正确→错误、错误→正确、错误→另一错误、错误保持；按 GT 类别和事件分层。KL 不识别教师是否正确，因此还要观察教师错误是否被更难纠正。
3. **四类 KL 的准确分解**：令教师建筑总概率 b0=sum(p0[1:])，学生 b=sum(p[1:])；等级条件分布 r0=p0[1:]/b0、r=p[1:]/b，则 KL(p0||p)=KL(Bernoulli(b0)||Bernoulli(b))+b0 KL(r0||r)。作为日志分解，不引入额外 loss。这样可检查约束主要作用于定位还是等级。
4. **按类 KL／有效数量**：背景占多数，全像素均值可能由背景主导。记录前景、背景及各类贡献，而不在第一轮偷偷改成类别重权。
5. **梯度量级与冲突**：稀疏记录 g_original、λg_KL 范数及余弦，必要时进一步分 NLL／Lovász。负余弦表示局部冲突，不是因果结论。
6. **预测评价**：前景 mIoU、各类 IoU、Damaged P/R、Intact→Damaged／Destroyed、建筑二类指标和逐事件统计。新候选既要优于匹配 control，也要检验是否超过同轮 grid_probq，而不只是靠近教师。

诊断只保存聚合统计与少量预先指定案例，不默认写全图大张量，避免再次产生大量 debug packet 占用磁盘。

## 后续实验按证据选择，不同时叠加

* 若 KL 明显减少负翻转但把教师错误锁住、没有超基线：停止扩大 λ；在同预算下单独测试参数锚定或有证据支持的表征学习，而不是不断强化蒸馏。
* 若 KL 梯度几乎为零：先查 reduction／有效像素／实际权重；不能因无效强度下无收益就否定机制。
* 若 loss 项在混淆类持续冲突：先报告冲突与错误关系，再独立测试 PCGrad 或采样方案。不能与 KL 同轮一起加入后单独归因。
* 若混合区域槽内容明显不同、却在加权求和后难以利用：再做保持两槽独立到分类头的容量匹配对照。
* 若 Top64 缺有用案例：先固定案例数量检查排序；ReCo式训练要有实际可更新的检索投影，并同时重编码支持特征。
* 若仅因重复案例或类别覆盖差：类原型 Memory 可作为对照，不直接替换本轮固定案例库。

首轮候选通过后，把相同目标应用到 B，检查新 D−新 B。最后才用统一设置复核 A／B／C／D 和独立配对种子。Test 不参与 λ、停止步数或结构选择。

## 不重复代码的实现原则

模型、slot pooling、检索、数据加载、冻结来源验证保持单一实现。增加独立 `objective` 配置和聚合 `metrics` 模块，训练循环和评估循环共用；候选不是复制一个新的 `train_kl.py`。resume 和从共同起点分叉须在同一检查点协议中写清楚 source、step、optimizer、schedule、RNG、data order、code version。缺依赖时明确停止并列出缺失项，不能重新生成一个 `complete=true` 摘要绕过验证。
