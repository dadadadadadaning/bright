# DamageFormer：证据驱动的区域与调制研究路线

日期：2026-10-03。任务：BRIGHT RGB/SAR 建筑损毁，既有 Swin/DamageFormer。问题：图像调制、区域汇总、分类器和历史信息分别有何增量价值？无 SRAC 全文/官方实现，以下为自主研究方案，不是严格复现。按 research-paper-insight 的证据迁移框架组织。

## 现有证据与边界

3000步冻结前端：pixel60.8998、normal59.1447、off60.3895；关闭输入调制的推理干预恢复1.2448pp，但不是重新训练的因果结论。旧像素60.8998和新像素58.8414不共享相同输入/分类器/训练历史，不能只怪新头。相同调制特征上新区域59.2255高于新像素58.8414约0.3842pp，说明不能宣布区域无用。低纯度区域高伤害是关联；不能用GT纯度进行推理筛选。

## 文献、公开代码与迁移范围

| 文献 | 来源/代码 | 可迁移内容 | 匹配、难度、风险 |
|---|---|---|---|
| FiLM, AAAI2018 | https://arxiv.org/abs/1709.07871 ; https://github.com/ethanjperez/film （vr/models/filmed_net.py） | 条件控制特征缩放/平移，不涂改原RGB | 原任务视觉问答；机制匹配中等，工程低；空间化、有界化为本项目改编，并非原文已验证BDA |
| OCR, ECCV2020 | https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf ; https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet_ocr.py | SpatialGather_Module、SpatialOCR_Module：汇总对象上下文再返回像素 | 分割匹配高、模态不同；中等难度；语义对象区域不是建筑实例 |
| SSN, ECCV2018 | https://arxiv.org/abs/1807.10174 ; https://github.com/NVlabs/ssn_superpixels | 可微局部聚类、任务相关分区、连通性处理 | 机制匹配高；官方Caffe且有自定义层，不建议直接装进当前环境；迁移中高 |
| Superpixel Transformers, 2023 | https://arxiv.org/abs/2309.16889 | 像素到超像素token与区域交互 | 自然图分割不是BRIGHT；本次未核实官方可运行代码，不提供猜测链接 |
| ChangeOS, RSE2021 | https://github.com/Z-Zheng/ChangeOS | 建筑对象约束与损毁判别分开考察 | BDA任务匹配高但原RGB/RGB，不代表对SAR或本协议必然有效；预测连通域不等于真实实例 |
| Cross-Image Pixel Contrast, ICCV2021 | https://github.com/tfzhou/ContrastiveSeg | 跨图像判别特征的训练约束 | 中等迁移；训练期对比不是推理期检索，不是冻结特征加损失就能学习 |
| BRIGHT, ESSD2025 | https://essd.copernicus.org/articles/17/6217/2025/ | 固定数据、模态与指标协议 | 最高任务匹配；不能用不同类数/实例协议榜单替代当前前景三类验证 |

这些是可核验的机制基座，不按发表年份堆新模块，也不借自然图结果预言BRIGHT涨点。

## 阶段0：可信接口，停止条件明确

固定同一snapshot、Val450、预处理、分类ID/ignore255。重放 exact clf_2 输入与 logits，上采样顺序不变。保持old_pixel/old_region复现、前端eval、Q复用；所有控制先过单元测试与真实数据smoke。上一轮发现的 inference-mode 缓存问题不能说已修复；本轮不编辑旧缓存实现，也不在新入口调用 inference_mode。以后全模型训练前须独立处理该缺陷。

## 阶段1：输入来源×读出方式（已实现）

目录：bda_benchmark/experiments/damageformer_modulation_controls_v1。
raw/normal/roi三种输入，各自pixel/region同结构MLP，共六个头，同初值/批次/预算。raw和normal都未施加像素Soft ROI，避免把ROI差异归因于涂改图像。roi是原像素头实际输入，做分类器诊断。一个前向共用Q，不缓存全部数据；只保存小分类头。

先20步smoke，再5000步/头，固定前端，不再默认跑50k。主比较同一步结果，best/last辅助。若raw明显优于normal，支持调制后特征对这种新头更难读；若两者一样，原关闭调制收益可能涉及旧头适配。若roi新像素仍低，下一项仅比较原架构新头/复制权重头，保持输入和输出解码一致，不直接声称区域失败。

## 阶段2：从图像涂改转为可选特征调制（只提供原型）

F' = F * [1 + a*tanh(gamma(B))] + a*scale(F)*tanh(beta(B))，a=0.1为待验证初值。B来自预测、detach；scale为每图每通道空间RMS，detach；末层零初始化。不要直接将异模态RGB-SAR相减当作损毁证据。

core.py的BoundedFeatureModulation实现上述式子，但train.py未启用。它是FiLM启发的空间改编，不是新颖性证明。先B单条件，不输入Q硬边界；只有成功后再单独测试可微边界/上下文条件。对照：无调制、相同模块常数条件、真实B条件。固定其他因素/同分类器，先前端冻结筛选，再统一初始化联合训练。范围有限不代表绝对无伤害，残差结构之前失败也不能忽略。

## 阶段3：只在区域确有增量时优化信息汇总

采用OCR的原则：区域提供参照，保留每个位置独立判别；不是马上用区域类别覆盖所有像素。对照必须包含同容量像素适配器、网格上下文、学习Q上下文，不能只比旧弱区域头。先固定Q/尺度，后单独改变建筑约束。已有T3残差低于T0，故仅换名字重跑不算新实验，须落实阶段1诊断后不同的特征接口/控制变量。

若证据集中在混合区域，比较未分组、预测建筑/背景组、预算匹配空间拆分。记录实际非空组数；同一建筑允许多个区域保存局部损毁。预测对象ID可作后续约束，但误合并/漏检会固化错误。不用GT实例生成推理分区。分区连通外观不是准确率的替代目标。

## 阶段4：历史信息分支，非必选主线

固定选定前端/分区/checkpoint哈希，用Train生成Token库。对照先同容量nomemory，再真实邻居；随机邻居、打乱支持标签、无元信息是同权重干预，不替代重训练对照。报告有效支持覆盖、邻居标签分布、跨事件结果及前景IoU。Memory的增量目标是memory-nomemory，不要求前端先超过A1才可研究，也不能期待Memory救回已丢掉的局部证据。

已有V2文件：build_memory.py、retrieval.py、train_retrieval.py；保留其Train-only和同事件/同图像排除策略。现有代码已有元信息/邻居表，不重新声称缺少这些。学习度量空间时参考ContrastiveSeg单独做训练期对比；至少更新投影/编码器，再重建bank。教师/学生或EMA字典为后续方案，当前不实现。

## 判断增益与有限预算

Val选择，Test不参与迭代；主报完整前景三类mIoU，配套Damaged P/R/F1、背景假阳性、类别×边界、像素aligned rescue/harm、每事件支持数。三配对种子报告均值/差值；如bootstrap应按图像或事件而非像素采样。提前设研究预算，例如单机制5000步筛选后只对有信号项做联合训练；没有信号可停，不为凑创新强留模块。不得承诺涨几个点。

## 3个可执行候选题目（不是同时开展）

1. BRIGHT跨模态建筑损毁评估中结构引导调制的信息保真研究：FiLM式特征调制对比输入压暗/绘边，主要验证Damaged与事件稳定性；新颖性在任务机制与证据，不在FiLM本身。
2. 面向局部损毁保持的像素—区域条件上下文解码：OCR原则配合建筑约束，同容量像素/网格/学习Q对照；难度中等，风险是再次证明无增量。
3. 跨事件损毁证据检索的有效性与可靠性控制：冻结前端、真实支持与参数匹配无库对照，进一步才学度量；贡献须体现少数类和跨事件净收益，难度中高。

当前优先1的诊断阶段。不把原型、检索到的代码和已验证的有效方法混为一谈。
