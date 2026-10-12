# SRAC Memory：复现检查失败后的改进分析

分析日期：2026-10-10。研究任务：BRIGHT 灾前光学、灾后 SAR 建筑损毁四类分割。目标：在当前 SRAC 区域前端与 J1/J2/J3 框架内，判断 Memory 应如何改进，以及用什么实验决定改哪一层。

本报告依据当前代码、一手论文与作者仓库，采用 research-paper-insight 工作流。没有启动训练、完整 Val 推理或修改实验实现。此次“450 张处理完成、13 项超过 1e-5、最大差约 2.97e-5”的数字来自用户提供的运行结果；本地未找到对应的原始 audit.json / images.jsonl，不能声称已独立复算这些数字。旧 ROI 分支的成绩不作为当前 J123 的证据。

## 1. 结论与证据边界

本次首先暴露的是复现一致性检查失败；验证协议差异是优先排查的原因，尚不能据此选择或否定 Memory 结构。值得优先检验的研究方向是：**让检索学习真正困难的查询，并让历史标签证据可以被检查、校准和拒绝；将 Memory 的贡献限定为对固定基线的可测增量。**

当前代码已经有跨事件监督对比学习、像素特征与区域上下文联合分类。因此，不应把“加入对比学习”“保留像素分支”“加一个残差”重复当作新方法。真正需要验证的是：现有对比目标覆盖了哪些查询，Top-K 是否带来有效证据，以及读出如何处理弱证据与冲突证据。

代码确认的事实：

- [最终检查入口](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_evidence_followup_v1/memory_audit.py:226) 在图像循环之后运行，先保存报告，再因 gate 失败抛出异常。
- [复现检查](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/memory_core.py:205) 对 A1、direct_region、J1、J2、J3 各检查 17 项，共 85 项，并要求 450 张。门槛是逐项绝对差 1e-5。
- [审计执行设置](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_evidence_followup_v1/memory_audit.py:74) 额外启用确定性设置、关闭 matmul/cuDNN TF32；[种子函数](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/p3_damageformer_ablation_v1/core.py:23) 默认并不做这些操作。
- numerical_unchanged 仅比较本次审计前后，不能证明与历史训练验证一致。
- A1 计算在区域处理之前；A1 也漂移，使“仅由检索或 Reader 引起本次复现失败”的解释不成立。

浮点运算顺序、计算后端、TF32 均可能改变数值，PyTorch 官方说明支持这种可能性，但不证明本例的具体原因。[数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)

2.97e-5 × 100 = 0.00297 个百分点。这个量是复现误差，不是 Memory 的效果量，也不是统计显著性门槛。维持旧失败状态与 1e-5 门槛。

## 2. 当前 Memory 实际做了什么

| 环节 | 当前实现 | 分析含义 |
|---|---|---|
| 前端 | 冻结 A1、Q、区域 Embedding；区域 token 为 128 维 | 本轮检索不能凭空恢复前端已经无法表达的信息 |
| 建库 | Train 区域 token、GT 软类别分布、事件、灾种、几何等；按事件×主类别随机优先级限额抽样 | 已有事件和类别配额，并非完全不均衡的随机总库；限额不等于困难形态覆盖 |
| J1 | A1 像素特征 + 当前区域经 learned slots 得到的上下文 | J1 不是纯 A1，是判断历史 Memory 增量的主要匹配对照 |
| J2 | 原 token 余弦、跨事件/跨图排除、全局 Top-8 | 有邻居不等于有可信邻居 |
| J3 | 恒等初始化 128×128 检索投影；Train 跨事件 SupCon 后冻结；仅改变 Top-8 行号 | 已有度量学习；Reader 接收的仍是原始支持 token 和标签 |
| Reader | K 用原始支持 token；V 将 token、软标签嵌入、事件、灾种、几何相加；MHA+FFN | 检索分数和 Reader 注意力并不是同一个权重 |
| 分类 | 同 Q 将增强区域特征回投，z = Wp F + Wc C | 上下文列零初始化只保证初始身份；训练中 Wp 也更新 |

关键代码：[建库](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/build_memory.py:89)、[J123 语义](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint123_v1/joint_runtime.py:34)、[检索](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_qmetric_v1/metric.py:535)、[Reader](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint_context_v1/model.py:61)、[联合分类头](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint_context_v1/model.py:139)。

snapshot 建库脚本显式配置每事件每类最多 64、min_purity=0.0；投影训练默认 purity≥0.8。服务器本次实际参数仍需以其 manifest 为准。**建库纯度门槛与投影训练纯度门槛不能混写。**

## 3. 先把验证协议固定下来

最小执行顺序：

1. 固定权重、Val 名单及顺序、数据变换、GPU/软件版本、batch、dtype；分别用原验证入口与严格设置重算 A1。原配置不能通过今天的 PyTorch 默认值反推，需记录实际运行值。
2. 两种配置各做独立进程重复，先分清配置间偏差和同配置内部不稳定。记录逐图整型混淆矩阵、预测变化像素、logits 最大差及低 margin 像素的变化。
3. 若原设置可复现，再分别改变 matmul TF32、cuDNN TF32、确定性设置；组合切换能确认协议差异，不能单独归因某个开关。训练有 BF16 不代表历史验证使用 BF16，原验证函数没有 autocast。
4. A1 对齐后再查 direct_region 与 J1/J2/J3；native/grouped 区域归约作为另一项独立因素。检索同时记录 Top-8 行号变化、第 8/9 名分差，区分连续数值漂移和离散邻居切换。
5. 如果旧环境不可恢复，明确建立新的统一评估协议，对全部基线和候选共同重评。保留旧失败报告，不能用新参考替换旧参考后宣称旧实验复现成功。

本次不执行以上 GPU 实验，因此不将 TF32、确定性或 grouped 归约中的任何一项写成已确认根因。

## 4. 具体改进方向

### 4.1 让检索学习“困难查询到可靠支持”的关系

**已确认的缺口：**[投影训练](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_qmetric_v1/train_metric.py:54) 只从已抽样 bank 中选高纯度 token，训练时 query 与比较样本都来自这个子集；推理却要处理所有区域，包括混合区域与 J1 判错区域。它优化的是典型区域之间的类别距离，并不直接优化困难查询的检索。

建议新对照保持前端、库成员、Reader 不变，仍先用同等大小投影：

- query 从完整 Train shards 抽取；在普通查询之外加入固定比例的困难 Damaged/Intact 查询，按类别、事件、区域尺度记录覆盖。
- support 保持质量控制，并排除 query 所属图像和事件。难查询与高质量支持不必采用相同筛选标准。
- 保留跨事件同类正对，增加与当前误检相关的 Damaged↔Intact 难负例。若增加同事件异类难负例，把它作为明确的新训练因素；这不会要求推理放开同事件检索。
- 混合区域使用 Train 软类别分布的相容性或低权重目标，不能一律 argmax 成 Damaged 再强行拉近。损毁 GT 是建筑等级标签，区域软分布不能解释成建筑内部破坏程度。
- 困难标记在 Train 内生成；需要预测正确/错误监督时优先使用分折读出预测，避免拿 Val 的 J1 错误来训练。若前端已见过全 Train，要说明这里只对新读出阶段做分折，不能称全模型完全独立的 OOF 评估。

这借鉴 ReCo 的困难查询和混淆负类采样，而不是宣称 ReCo 已解决跨事件 RGB–SAR 检索。先不同时换投影深度、Q、容量和损失。[ReCo 论文](https://arxiv.org/abs/2104.04465)、[作者代码](https://github.com/lorenmt/reco)

**推进条件：**固定 J1 错误集合上，困难 Damaged 的标签支持及邻居质量改善，并能转成分割收益。只改善全部区域平均检索分数不够。当前名为 recall_at_k 的部分字段实质是至少一个同主类邻居的 Hit@K，不是真正召回率。

### 4.2 先用显式标签读出检验信息，再做可拒绝的修正

**已确认的缺口：**当前 Top-K 对有限分数直接取邻居，没有语义质量拒收；Reader 的 fallback 只处理缺失行。它没有显式判断“本次参考是否足以改变原预测”。

低成本探针：固定原 Top-8，用检索相似度对邻居软标签加权：

    a_ij = softmax_j(s_ij / tau)
    p_mem(i) = sum_j a_ij * y_j

无有效支持时标记 unsupported，不制造均匀后验。先报告这一标签证据，再与原 MHA 输出比较。它能测试“参考标签有无增量信号”，不能把标签投票直接当 Reader 注意力或其能力上界。[Tip-Adapter](https://arxiv.org/abs/2207.09519) 提供特征—标签缓存的直接先例，但可靠性门控是这里的待验证设计，非该论文原机制。

建议在冻结 J1 的基础上训练一个小修正器：

    z_new(x) = z_J1_fixed(x) + g(x) * bounded_delta(x)

g 使用推理时可得的信息：邻居绝对相似度、标签分歧、支持有效质量、不同支持事件数、基线概率和 margin；软区域回投质量也应作为可靠度输入。先做无门控/有门控对照，别用大量新因素掩盖原因。

- g=0 时，整个新修正项消失，准确退回固定 J1；若另设 A1 锚点，则整组都使用相同 A1，不能混比。
- 将 delta 末层零初始化，gate 保持可学习；不要把两个乘法分支都精确初始化为零而阻断初始梯度。
- 在 Train 上包含无支持、弱支持情形的训练，检验 gate 是否学会拒绝。低相似度阈值、温度与修正幅度需统一校准，不能用 Val GT 按图选择是否接纳。
- 保留查询正确样本作为保护对照；基线高熵不一定会错，高置信度也不保证正确，不能只依赖基线熵开关。
- 类别配额改变了支持分布；标签加权结果是证据分数，不应未经校准就当作真实后验。

仓库已有可复用机制：[SelectivePosteriorHead](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_roi_memory_selective_v1/memory.py:280) 和 [gate_features](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_roi_memory_selective_v1/memory.py:342)。但它属于 ROI_EXCHANGE/hard 来源，不是当前软 Q 联合分支；不能搬权重、成绩或该分支的复现例外。

特别注意旧公式 [memory.py:433](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_roi_memory_selective_v1/memory.py:433) 含独立 query_delta，所以仅令 gate=0 仍未退回 base。迁移时要门控整个新增 delta，或将 query-only 模型先固定为公共基线。

### 4.3 检查检索与读取的关系，减少元数据捷径

J3 按投影后的相似度取 Top-8，Reader 随后对原 token 再学习 Q/K 打分。这是合法设计，**不是已经证实的 bug**，但若检索有益而输出无益，应测试两者是否协同。

保持候选集合不变，逐一比较：原 MHA；直接 metric 权重标签读出；给 MHA 加检索分数偏置。另做“token+标签”对“token+标签+元数据”的等预算重训，观察 event/disaster/geometry 是否帮助还是造成跨事件偏置。先不把事件 ID 必然有害当结论。

开启 MHA attention 导出可能改变后端执行路径，须验证导出前后输出一致性；注意力大小仍不是因果解释。原模型 random/shuffle/off 干预回答依赖性，等预算重训才能回答方法增量。

如果训练时几乎一直能凑齐 Top-8，Memory 模式下的 learned fallback 可能缺少训练。因此 support-off 可能是训练外输入，不能把它等同于训练好的 J1，或直接当部署回退路径。

### 4.4 Bank 配额、多样性与区域表示按证据触发

当前已有每事件×类别限额，以及 [M0/M1 配额对照方案](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_evidence_followup_v1/README.md:219)。不要把再次做类别均衡作为新贡献。

- 若排除同事件后，困难 Damaged 缺少候选，再考虑配额。M0/M1 同类容量、相同优先级、相同训练预算；M0 不是历史 bank 的逐行复制，需要重新训练匹配对照。
- 若候选多但外观重复，在相同配额内比较随机抽样与多样性选样/多 medoid，保留真实样本和软标签；先不增加容量。
- 高纯度筛选可能去掉小建筑与边界困难区域，需同时报告覆盖、质量与最终收益，不能只追求平均纯度。
- ProtoSeg 的多原型提供方向，但其表征共同学习；冻结 token 后简单聚类不等价于原论文，也不保证提升。[论文](https://arxiv.org/abs/2203.15102)、[作者代码](https://github.com/tfzhou/ProtoSeg)
- 只有 A/B/C 特征对照提示区域路径明显退化，才优先比较匹配的像素/区域头适配，再考虑区域粒度或编码器。若改 Q/Embedding，需重新提取匹配库并重训投影、Reader。

## 5. 按结果决定下一项实验

| 在统一协议通过后观察到什么 | 下一项对照 | 不应直接下的结论 |
|---|---|---|
| 原配置复现、严格配置偏移 | 单因素数值设置对照，建立统一协议 | TF32 一定是唯一原因 |
| 困难 D 跨事件候选不足 | 已实现的 M0/M1 或同容量多样性建库 | 库数量大就一定有效 |
| 库有合适支持，Top-8 常偏向 I | 原 J3 vs 困难 query 训练的投影 | 普通对比学习尚未实现 |
| 标签证据有用，原输出未救回 | 原 Reader vs 显式标签读出；再验证门控 | 仅凭投票即可证明 Reader 是唯一瓶颈 |
| 救回 D 增加，非 D→D 误报也增加 | 有无可靠度门控及校准对照 | 召回提升等于总体方法成功 |
| 区域路径先破坏判别 | 固定 F/Q 的分类头适配对照 | 平均汇总已证明不可逆信息丢失 |

实际执行可压缩为：E0 协议；E1 固定证据读出；E2 困难查询投影；E3 在有效分支加可靠度门。Bank 配额、Q 和更深 Reader 都是条件分支。先取得机制信号，再做多种子完整实验。

所有对照使用同一固定 J1 判错 Damaged 集合，并同时报告：困难 D 救回率、J1 原本正确 D 的伤害率、非 D→D 假阳性变化、四类 IoU/P/R/F1、逐事件表现。软 Q 下统计支持质量与标签证据，不能把像素投影统计解释成像素独立检索。

最终指标必须由完整混淆矩阵计算；“救回数减伤害数”不能代替 IoU，因为假阳性也影响分母。统计重复按事件/图像或训练种子组织，不把大量相关像素当独立样本。只有通过完整有效审计，才使用干预结果选路线；Test 不参与选择。

停止条件：困难检索改善但同预算分割长期无改善时，不继续靠扩库解释；只在同事件支持下有效时，不能声称跨事件泛化；只有训练对比正则有收益、推理查库无增量时，可将 Memory 留在训练期，须另做有无推理库对照。

## 6. 一手文献与代码的可迁移范围

| 文献、收录与来源 | 可迁移模块及代码位置 | 匹配程度、成本与限制 |
|---|---|---|
| SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment，IGARSS 2026；[会议页](https://2026.ieeeigarss.org/view_paper.php?PaperNum=2851&SessionID=1356)、[作者机构介绍](https://emse.ucas.ac.cn/index.php/zh/xwzxe/535-igarss-2026) | 同任务超像素与外部标注库增强，是最接近的领域先行工作 | 任务高度匹配；本次核实到会议与机构材料，会议页无资源，未核实到完整官方源码。不能从宣传指标推断本实现严格复现成功 |
| Bootstrapping Semantic Segmentation with Regional Contrast（ReCo），ICLR 2022；[论文](https://arxiv.org/abs/2104.04465)、[代码](https://github.com/lorenmt/reco) | module_list.py / compute_reco_loss：困难 query、类原型正键、混淆负类采样 | 分割匹配较高，模态不同；迁移难度中。跨事件要求是本项目新增约束 |
| Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification，ECCV 2022；[论文](https://arxiv.org/abs/2207.09519)、[代码](https://github.com/gaopengcuhk/Tip-Adapter) | main.py / run_tip_adapter：相似度核×标签缓存，补充基线分类 | 直接标签读出匹配，CLIP/少样本协议不同；探针成本低。不得照搬作者代码中使用 test 选择 epoch 的流程 |
| Rethinking Semantic Segmentation: A Prototype View（ProtoSeg），CVPR 2022；[论文](https://arxiv.org/abs/2203.15102)、[代码](https://github.com/tfzhou/ProtoSeg) | lib/models/nets/hrnet.py / prototype_learning 与 lib/loss/loss_proto.py：多类内原型、分配和表征学习 | 分割匹配较高；难度中高。冻结聚类只是简化实验，正确样本更新策略也可能遗漏困难 D |
| Mining Contextual Information Beyond Image for Semantic Segmentation（MCIBI），ICCV 2021；[论文](https://arxiv.org/abs/2108.11819)、[原作者库](https://github.com/CharlesPikachu/mcibi)、[维护库](https://github.com/SegmentationBLWX/sssegmentation) | ssseg/modules/models/segmentors/mcibi/memory.py：动态类别记忆与上下文 | 可作类别上下文对照，难度中高；不是静态案例 Top-K，也不直接证明本任务检索有效 |
| MCIBI++: Soft Mining Contextual Information Beyond Image for Semantic Segmentation，TPAMI；[论文](https://arxiv.org/abs/2209.04471)、[代码](https://github.com/SegmentationBLWX/sssegmentation) | mcibiplusplus/memoryv2.py：类别分布统计与软上下文 | 可研究类内分布，不是支持质量门控，也不是多个真实案例检索；不建议与当前所有改动同时加入 |

截至本次定向检索，2026 年 SRAC 说明同任务已有“超像素+检索增强”先行工作；这不是穷尽综述。上述论文提供机制依据，不提供当前改进的性能保证。本文设计中的拒绝门控、困难查询跨事件监督及相应组合均为待验证提案。

## 7. 三个可执行的题目方向

1. **面向困难区域查询的跨事件检索适配用于光学–SAR建筑损毁评估。** BRIGHT 原 Train/Val 下，固定同一前端、bank 和 Reader，比较现有 bank-only 高纯度 J3 与全 Train 困难 query 投影；预期贡献是明确困难查询训练分布与语义参考命中的关系。创新首先是任务化训练设计，不能将 SupCon 本身称原创。成本中；混合标签噪声是主要风险。
2. **可拒绝标签证据驱动的建筑损毁区域 Memory。** 在固定 J1 上比较显式标签读出、无门控修正、可靠度门控，报告救回、伤害、假阳性和逐事件结果；预期贡献是可检查的证据使用及增量收益。仓库已有相关旧分支，必须说明迁移与新增之处。成本低中；类别采样偏置和过度拒绝是主要风险。
3. **跨事件支持多样性约束的区域记忆构建用于建筑损毁识别。** 固定类别容量与 Reader，比较事件配额、随机代表与多 medoid；预期贡献是参考覆盖与困难类别收益之间的可验证联系。仅在候选覆盖证据支持时开展。成本中；新颖性偏应用组合，且类内多样性不等于损毁可判别性。

优先做第一、第二方向的最小对照。第三方向等待有效审计确认库覆盖问题，避免一次改变过多因素而无法判断收益来源。
