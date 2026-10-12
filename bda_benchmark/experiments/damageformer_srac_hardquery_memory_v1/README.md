# SRAC Hard-query Memory V1

这是新的独立实验包，完成 E0 数值协议核对、E1 显式标签证据诊断、E2 困难查询检索训练、E3 可拒绝修正对照。旧模型、旧库、旧代码和失败报告保持原样；不读取 Test。

本地验证使用 CPU 合成数据，不能代表服务器 CUDA 或 BRIGHT Val450 已通过。实际效果以服务器完整验证为准。所有输出目录必须是新的，程序拒绝覆盖。

## 改了什么

1. **统一执行协议。** `protocol.py` 分别运行原入口的 seed-only 策略（legacy）和严格策略（strict），保存逐图预测、GT、整型混淆矩阵、Top-8 行号和 SHA256。A1-only 直接调用原 A1，避开区域计算；all 检查 A1/direct_region/J1/J2/J3 的全部 85 项指标，逐项容差仍为 `1e-5`。
2. **同一邻居的标签读出。** `evidence.py` 固定历史余弦/J3 的 Top-8，比较原 Reader 与相似度加权软标签。无支持的 Q 质量退回固定 J1。它是诊断标签证据，不是 Reader 注意力或校准后验。
3. **困难查询训练。** `prepare_queries.py` 只对 Train 图像重新计算固定前端的 FP32 token、软标签及固定 J1 概率，并核验原 Train shard。`train_metric.py` 比较旧 bank 高纯度 SupCon 与全 Train 困难查询训练，两组均从恒等投影开始，各 2000 次更新；旧库成员不变。
4. **固定 J1 上的有界修正。** `train_reader.py` 训练六个相同初值的小头，前端/Q/原 J1/J2/J3/新旧投影/Memory 全部冻结。输出为 `J1_logits + SameQ(g * bounded_delta)`。关闭全部修正或没有支持时严格回到 J1；不会有独立的未门控 query residual。

| 组 | 输入历史证据 | 门控 | 用途 |
|---|---|---|---|
| query_ungated | 无；posterior/quality 置零，只看当前 query/J1；共享 bank 组支持有效掩码 | 无 | 同容量新分类头对照 |
| query_gated | 同上 | 有 | 同容量带门控对照 |
| bank_ungated | 本轮匹配训练的 bank-only 检索投影 | 无 | E2 检索对照 |
| bank_gated | 同上 | 有 | E3 门控对照 |
| hard_ungated | 困难查询检索投影 | 无 | E2 新检索 |
| hard_gated | 同上 | 有 | E3 新检索＋门控 |

四个 Memory 组必须与同门控的 query-only 组比较，才能区分历史证据增量和额外分类器容量。历史 J2/J3 分数用于背景参考，不是六个新头的等预算对照。

## 训练配置与解释边界

- 默认 Top-8 沿用历史源，未增大库，也未更改区域 Q。
- 支持侧训练筛选 `purity>=0.8`；推理仍使用完整原库。两组投影均有相同的高质量支持、参数规模、2000 步预算。
- 新查询按主类别分层，仅用于抽样；每类困难度最高 25% 构成困难池，每批该类 50% 查询来自困难池，其余 50% 来自普通池。损失保留完整软类别分布，并在排除同事件/同图之后归一化支持标签相容性。
- 困难度为 `1 - dot(Train软标签, 固定J1同Q平均概率)`。它也包含区域混合带来的不确定性，不能等同“建筑真实损毁程度”或仅仅分类错误。J1/前端训练时见过 Train，这里不是 OOF 泛化误差估计。没有用 Val 错误标签训练。
- E2 同时改变查询来源、困难抽样和软目标，属于一组明确的训练策略改进；归因某一个因素仍需后续消融。首版没有额外加入同事件负例、深层投影、配额变化或多原型。
- 六个修正头每组正式 5000 步，每 500 步完整 Val；同一训练图、同初值、同 loss、同预算。单步一张完整 1024 图，FP32，无随机裁剪/增强/梯度累积；损失是原 NLL＋Lovasz。
- 门控输入为归一化 query、J1 概率/熵/间隔、标签证据、相似度/支持标签分歧/纯度/事件多样性。修正限制为 `2*tanh(...)`，末层零初始化；初始 gate=.5，使修正先学习而不形成乘法死梯度。
- 默认每槽位 0.1 dropout，整查询 0.1 dropout；各组共享随机抽取模式。空支持测试严格退回，硬掩码下本身不产生 gate 梯度，不能宣称空支持训练已学会拒绝错误支持。
- 六个头都具有同样可训练结构。sigmoid 门控通常为接近 0 而非精确 0；严格相等保证适用于 `force_off` 或无支持时。门控不保证训练后必然提分。
- 记录各类 IoU/P/R/F1、完整混淆矩阵、逐事件结果、同一固定 J1 困难集合的救回/伤害/假阳性变化和检索证据。Hit@K 不是严格意义的检索召回率。

## 文件与依赖

上传本文件夹到服务器原工程同名路径。它调用工程中已经存在的 transformer、joint_context、qmetric、joint123/snapshot_joint123、stage_diagnostics 和 evidence_followup 模块，不是脱离原工程的独立网络。

`load_bundle` 原有源码、权重、split、bank、快照和完成状态校验全部保留。不要为通过检查修改旧 source hash。只加载你本人生成且可信的 PyTorch 权重。

需要匹配、已完成 5000 步的 `j1_last.pth / j2_last.pth / j3_last.pth`，以及其 `manifest.json / summary.json / validation.jsonl`。路径搬迁可设置 `FRONT_CKPT / CACHE / OLD_METRIC_CKPT`，但 SHA256 必须一致。不会拿不匹配的历史 best 权重替代。

## 1. 环境与本地合约测试

以下命令在 Linux 服务器执行。路径示例来自当前上下文；若目录位置不同，填写真实值。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export EXP=bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export JOINT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint
export DEVICE=cuda:0 WORKERS=4 SEED=3407
export DATASET=/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset
export TRAIN_LIST=$DATASET/splitname/standard_ML/train_set.txt
export VAL_LIST=$DATASET/splitname/standard_ML/val_set.txt
bash "$EXP/run.sh" preflight
```

测试覆盖合成两组检索优化、六组修正训练、软标签/事件排除、Same-Q 梯度、关闭修正的逐位身份、冻结检查、保存重载、两图评估全路径以及来源/协议/失败结果拒绝。不会启动真实 450 张推理。

## 2. E0：先确认数值协议

legacy 仅表示不修改进程启动默认、沿用旧 seed-only 入口，不证明当前默认与过去完全相同。程序记录 PyTorch/CUDA/cuDNN、硬件、TF32、确定性、线程及相关环境变量；同一 GPU 机器的 CPU/CUDA 运行不会互相冒用协议。若原环境无法恢复，不能将新参考覆盖旧失败参考。

首次按真实旧运行环境设置 shell 环境。不要在两次独立进程间悄悄改 TF32 环境、设备或数据目录。strict 会显式配置确定性与关闭 TF32；每项对照必须新 Python 进程。

```bash
export RUN_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/e0_$(date +%Y%m%d_%H%M%S)"
export POOL_MODE=native
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC
bash "$EXP/run.sh" e0_pair
```

它依次运行 legacy/strict 的 A1-only，各独立重复两次。一项复现失败仍保存失败，并继续另一配置用于排错；总体退出码保留失败。这四次不产生正式训练许可。

比较同设置重复，再比较设置间变化：

```bash
export LEFT_RUN="$RUN_ROOT/e0_legacy_a1_1"
export RIGHT_RUN="$RUN_ROOT/e0_legacy_a1_2"
export OUTPUT="$RUN_ROOT/compare_legacy_repeat"
bash "$EXP/run.sh" compare

export RIGHT_RUN="$RUN_ROOT/e0_strict_a1_1"
export OUTPUT="$RUN_ROOT/compare_legacy_strict"
bash "$EXP/run.sh" compare
unset OUTPUT
```

`comparison.json` 报告逐像素翻转、混淆矩阵差与支持行号变化（A1-only 无支持）。两次即使最终复现失败，只要推理及输入不变性检查完成，仍可比较；比较成功不把失败协议改成通过。

单因素检查使用 `protocol`，设置 `MATMUL_TF32=0/1`、`CUDNN_TF32=0/1` 或 `DETERMINISTIC=0/1` 中的一项；确定性是一组已明确记录的算法/cuDNN/cuBLAS策略。不要一次改三个开关后声称确认某一个原因。

## 3. 两图 smoke，贯通全部阶段

选择有依据的设置，下列以 legacy/native 为例。若原生 Pool 问题未排除，可显式测试 grouped；两种模式要分别建立协议，不能隐式 fallback。grouped 只用于冻结前端和诊断均值，训练梯度在新修正头及 Same-Q 回投上。

```bash
export PROFILE=legacy POOL_MODE=native
export RUN_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/smoke_$(date +%Y%m%d_%H%M%S)"
unset OUTPUT PROTOCOL_RUN QUERY_RUN METRIC_RUN SCOPE
bash "$EXP/run.sh" smoke
```

顺序：Val2 协议 → Val2 标签证据 → 四个不同事件的 Train 查询 → 两投影各20步 → 六头各20步 → Val2。全部明确标记 smoke，不可用于挑模型或作为正式结果。

## 4. 正式运行

只有统一设置的完整 450 图 all 检查通过，才会进入正式训练。原失败目录不作为 `PROTOCOL_RUN`，不能改失败标记或门槛。

推荐先分阶段检查：

```bash
export RUN_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_$(date +%Y%m%d_%H%M%S)"
unset OUTPUT PROTOCOL_RUN QUERY_RUN METRIC_RUN SCOPE
bash "$EXP/run.sh" protocol
bash "$EXP/run.sh" evidence
```

核对 `protocol/protocol.json` 的 `complete=true / full_validation_evidence=true / reproduction.passed=true / image_count=450`；查看 `evidence/evidence.json`。若仍有13项或其他项失败，到此停止，不会进入后续拟合。

继续已通过协议的同一 RUN_ROOT：

```bash
bash "$EXP/run.sh" queries
bash "$EXP/run.sh" metric
bash "$EXP/run.sh" reader
bash "$EXP/run.sh" report
```

或者在全新 RUN_ROOT 下执行 `bash "$EXP/run.sh" pipeline`，串行完成上述所有阶段；任何阶段失败，后续停止。该命令不会调节阈值或修改旧结果，也不会自动为你选择数值配置。

需要后台运行时，可在已确认路径和配置后执行：

```bash
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_HARDQUERY_MEMORY_V1
export LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_HARDQUERY_MEMORY_V1/$(basename "$RUN_ROOT").log"
nohup bash "$EXP/run.sh" pipeline > "$LOG" 2>&1 &
echo "$!"
tail -F "$LOG"
```

这里只能对全新 RUN_ROOT 用 pipeline。如果已完成 protocol/evidence，应后台执行尚未完成的单阶段命令，不能用 pipeline 覆盖重跑。没有完整 RNG/采样状态恢复入口；中断后产物保持不完整，换新阶段目录重新运行，勿声称精确续训。

## 结果应怎样判断

- `hard_ungated - bank_ungated`：困难查询训练策略的增量。
- `bank_* / hard_* - query_*`（相同门控）：历史参考相对同容量查询分类器的增量。
- `*_gated - *_ungated`：门控是否改善救回与误改的权衡。
- 所有组相对固定 J1：最终是否值得增加这条分支。

优先同一步 final 结果。best 权重由各组各自 Val 选择，不能把各组独立最优时刻误称同一步比较。单种子小幅上升不足以建立可靠效果；多种子重复及事件层面的稳定性尚需真实实验。

报告保留 `full_validation_evidence=false` 的 smoke/失败状态。完成 Train-only metric 拟合不等于已经完成新的 Val450 评估。代码检查权重、列表、cache/shard等已列明输入及数据根路径；未对所有原始 Train/Val 图像做全量内容哈希，不宣称排除了未纳入清单的原始影像变化。

## 输出位置

```text
RUN_ROOT/
  protocol/protocol.json, images.jsonl, predictions/*.npz
  evidence/evidence.json, images.jsonl
  queries/query_manifest.json, shards/*.pt
  metric/metric_bank_last.pth, metric_hard_last.pth, training_*.jsonl
  reader/initial_identity.json, training.jsonl, validation.jsonl
  reader/{query,bank,hard}_{ungated,gated}_{best,last}.pth
  reader/summary.json
```

查询缓存会重新提取完整 Train，消耗时间和磁盘；只存区域 token、标签和概率，不额外保存所有像素特征。训练映射使用内存映射 shard，按采样行加载特征。`preflight` 不验证服务器依赖版本、显存容量或真实数据完整性。
