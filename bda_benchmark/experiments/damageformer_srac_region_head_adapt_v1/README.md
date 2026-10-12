# 固定 F/Q 的区域分类头适配

本实验对应已观察到的 `region_mean → region` 退化：保持当前 A1 特征 F 和软区域分配 Q，单独训练分类头，检查简单均值区域表示是否能保住 Damaged。它使用已通过的 **legacy/native、all-path、Val450 E0** 绑定数据、权重、旧库和运行条件。所有新增文件位于本目录；原实验与原结果保持只读。

## 三个新头与公平比较

| 输出 | 输入 | 初始权重 | 可训练部分 |
|---|---|---|---|
| `p` | 原 256 维像素 F | 原 A1 `clf_2` 的独立副本 | 256→4 仿射头 |
| `r` | 同一 F 按当前 Q 精确均值汇总 | **同一个** A1 `clf_2` 的另一独立副本 | 同规格 256→4 仿射头 |
| `t` | 当前已有 128 维区域 token | 原 `region_classifier` 的副本 | 128→4 区域头 |

P：像素头→网格 softmax→概率上采样。R：区域头→区域 softmax→原 Q 回投概率→概率上采样。T 使用原区域 token 和同 Q 概率回投。P/R 同初始化、同参数量、同图像顺序、同预算、同损失；T 是不同表示与参数量的辅助诊断，不是第三个完全匹配的表示对照。

初始 P/R/T 应分别对应 `pixel_grid`、`region_mean`、`direct_region`（原表中的 `region`）。初始参数完全相等，概率采用既有 atol=rtol=2e-5 检查并记录 argmax 差异；这与旧五臂**绝对 1e-5 分类指标复现门槛**是不同检查，不能替代或放宽后者。

原生 A1 的检查只读取同次前向中唯一一次 `clf_2` 调用的输出。当前 SRAC 包装的是原生 `DamageFormer_S8_MICB_DSMIL`，不是调用分类头两次的其他插件分支；捕获为零次或多次仍会报错。额外探针调用分类头得到的 logits 不用于原生 A1 一致性门槛，避免独立重放问题。四条冻结 Q 探针仍沿用原公式，方便解释本次对照。

## 训练协议

- 正式：Train2918，全图、batch=1、无数据增强、FP32，每头恰好 **5000 次更新**。
- 同一个前端前向为三个头提供固定输入；每步三个头使用同一张图。A1、Q、adapter、embedding、Memory、检索投影、J1/J2/J3 Reader 均不更新。
- 每头独立 AdamW，默认 lr=1e-4、weight_decay=5e-3，相同 warmup/cosine 与梯度裁剪。
- 监督沿用回投后的像素 NLL + Lovasz。未新增类别加权、区域软标签辅助损失、门控或实例约束。
- 优化前先做完整初始 Val450；固定预算训练后做最终 Val450。只保存等步数 `heads_last.pth`，不按 Test 或每头不同最佳步数挑权重。
- Smoke：20 次更新、Val 前两图，明确不是正式实验。正式评估不能接受 smoke 权重。
- 新代码、旧文件、原模型参数/缓冲区与运行条件检查均保留。正式阶段若复现失败，保留失败记录及未完成权重，不把结果标为正式。

这不是原 Stage A 的完全重训，也不是50000步原流程的复刻；它是三个分类头在同一冻结表示上的5000步适配实验。性能结论只适用于这个预算和初始化。

## 文件与阶段

| 入口 | 作用 |
|---|---|
| `analyze.py` / `run.sh analyze` | 只读已完成的旧 Val450 九路径结果；校验 SHA、图像顺序、混淆矩阵，拆解 Damaged 救回/误改、假阳性来源、逐事件和 GT 类别边界统计 |
| `train.py` / `run.sh train` | P/R/T 分类头训练，自动执行 initial_val 与 final_val |
| `evaluate.py` / `run.sh evaluate` | 使用完整 fit 目录中的等步数权重，独立重算 Val450 |
| `run.sh probe` | 初始化头、不训练；在相同 legacy/native 协议运行四种区域组件敏感性检查 |
| `report.py` / `run.sh report` | 读取已通过的最终 Val 报告，不重新推理 |
| `locate.py` / `run.sh locate` | 在 E0 所属运行目录寻找唯一已完成旧 Val 区域结果；多个候选时要求显式指定 |

离线分析读取九路径的 `summary.json`、`result.json`、`images.jsonl` 和 NPZ 预测，拒绝把 Test878 或未完成 Val 当正式来源。GT 边界定义为原标注图有效四类标签间的4邻域边界，ignore=255 不参与；这是像素类别边界，不是建筑实例。旧文件没有保存 Q，因此该步骤**不生成混合 Q 分层结论**。

可选 `probe`：`uniform_attention`、`no_structure`、`no_geometry`、`no_stats` 各自只改一次读出，Q 固定。均匀注意力把 score 置0，对应 sigmoid=0.5；不是把区域质量清零。它们是已训练模型的输入/组件干预，不能代替同预算重训消融。

## 输出与判读

训练输出含 `manifest.json`、`summary.json`、`training.jsonl`、`heads_last.pth`、`initial_val/`、`final_val/`。每次 Val 保存各类 IoU、像素转移、逐事件结果、SHA绑定的逐图预测和新 Top-8。训练日志记录共同图像流，防止误把不同数据顺序视为匹配实验。

正式成绩必须同时满足：完整450张、`complete=true`、`full_validation_evidence=true`、旧五臂 `reproduction.passed=true`、完整性与同次原生 A1 检查通过。训练根目录 `summary.json` 完成还需核对其 `final` 字段；一小段 smoke 或一条训练loss不能作为提分结论。

优先比较 R−P、R−未训练region_mean、P−未训练pixel_grid、T−原region，兼看 A1、grid_probq、四类IoU和Damaged救回/误改/FP变化。R若更好，支持在本预算下采用均值区域表示；T若恢复，支持进一步做头部适配。任何不提升都不证明信息永久丢失。原J1/J3分别训练，差值仍不是纯Memory因果增益。

只适配输出头不改变历史 Memory token。若以后改变 token 或 Q，必须同步重新提取查询与库、检查投影和 Reader，不能直接混接旧空间。后续结构与参数选择用 Train/Val；本包不提供 Test 训练或选择入口。

完整可复制的服务器命令见仓库根目录 `deliverables/srac_region_head_adapt_v1_RUN.md`。ZIP 内从 `bda_benchmark/experiments/...` 开始，**解压目标是仓库根目录**。

## 概率归一检查失败时的诊断

如果出现 `Probabilities are not normalized`，现在保留原检查并报告图名和分支，在新的失败输出目录保存 `probability_failure.json` 与 `.pt`。JSON包含每条输出路径的值域、四类和的最大偏差、超限像素数，以及原设备求和和同一张量CPU双精度求和的对照；PT保存失败输出及相关冻结前向输入，不含GT，不作自动修正。

`run.sh recheck_probability` 从已保存的、训练前初始Val失败记录选择准确图名，用E0绑定的源模型重新初始化头，仅检查该图。它不加载smoke训练头，不训练，也不重跑前面的图；必须新建输出目录。即使单图通过，`full_validation_evidence` 仍为false，不能宣称原失败消失或完整Val已通过。单图重算有新的进程及分配历史，可能无法重现依赖前序图像的数值问题；若未重现，需要完整前缀或Val的失败记录来继续定位。

这是诊断增强，不是已确认的概率数值修复。概率检查 `atol=2e-5, rtol=0` 和最终历史分类指标 `1e-5` 两个门槛均保持原样。命令见 `deliverables/srac_region_head_adapt_probdiag_RUN.md`。
