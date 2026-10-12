# DamageFormer：Q 生成学习与 SAR 支持的三套独立对照

本目录新增实验，不覆盖旧实验、来源权重或结果。每次只运行一套对照，不自动串行启动全部正式训练。
三套共 8 个分支；所有分支的读出从同一个已训练 `hard_last.pth` 完整复制，参数各自独立。
原前端权重及运行统计、建筑概率 B、分类内容特征冻结；允许训练的 Q 模块是新分支中的副本。
不增加 Memory，不对 RGB/SAR 做直接相减，不改 attention/meanstd 读出结构。

## 对照与能回答的问题

| suite | 固定分支 | 唯一需要检验的变化 |
|---|---|---|
| `structure` | `fixed_q`、`learn_q`、`learn_q_structure` | 冻结 Q；任务梯度更新 Q；再加入二值建筑结构重建损失 |
| `multimodal` | `rgb_rgb`、`rgb_sar` | 同等容量、相同来源初始化的双 Q decoder；RGB+RGB 控制额外容量，RGB+SAR 检验灾后信息 |
| `support` | `hard`、`uniform`、`sar` | 固定 Q 和 B；完全隔离、均匀有效异组支持、SAR 相似度支持 |

结构项 U2 用训练 GT 的建筑/背景二值图，经**未被 B 拆组的 Q**聚合和重建，再计算损失。
GT 只进入训练损失与评估，不作为推理分区输入。该监督不指定超像素 ID，也不标记楼内破损位置。
建筑等级 GT 不能证明某块屋顶局部损坏；相邻同类别建筑也不因此自动变成不同实例。

`learn_q` 通过已有损毁任务梯度学习分区，`learn_q_structure` 再增加显式结构约束。
两者仍从相同的灾前 RGB Q 生成器开始；不能把仅看 RGBpre 的生成器称为直接感知灾后 SAR。
是否需要 SAR 输入由第二套对照单独检验，不在第一套中同时换输入和损失。
`rgb_rgb` 与 `rgb_sar` 使用同等容量的复制 decoder，避免把增加参数量误当作 SAR 增益。

第三套只改变支持分配：SAR 支持依据冻结 SAR 融合特征及其区域中心，中心不引入可训练额外编码器。
`uniform` 和 `sar` 使用相同候选有效性规则、相同异组预算；预算 ε 读取来源实验记录。
它们仅改变 pooling 的支持分配，回投仍保留原硬 B 路由。`uniform` 在有效候选中按原 Q 比例分配，
不表示对每个 ID 等权；SAR 在相同 Q 基础上按余弦相似度重加权。
没有有效异组支持时，两者都回退为原硬聚合，不丢失归属质量。**新 `uniform` 不等同于旧 exchange**：它按有效支持分配预算，
不能把旧 exchange 的结果直接作为本轮 uniform 控制。主要比较 `sar - uniform`，同时保留 `hard`。
候选数相同不代表有效槽位数相同；须结合 routing、Q drift 和最终诊断理解结果。

## 来源和上传

将本目录的**全部文件**上传到：

```text
/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_partition_learning_v1
```

依赖服务器已有的 DamageFormer 前端及旧 region/content、ROI partition、ROI exchange 等实验模块；
不要只上传 `run.sh`，也不要为绕过来源检查而覆盖旧目录或改 manifest 哈希。
仅接受已完成的正式 `ROI_EXCHANGE_V1` 来源，要求 `hard_last.pth`、`manifest.json`、`summary.json`
及匹配的不可变前端快照。`hard_best.pth`、原 A1、未完成/smoke 来源都不能替代它。
来源头在 CLI 中固定为 `--source_head hard`；不提供 `--pool_mode` / `--read_mode` 模式选择。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
# 使用本轮已完成且与 FRONT_CKPT 匹配的 exchange 正式训练目录。
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_20261005_133348
export DEVICE=cuda:0
export BATCH=2
export ACCUM=8
export WORKERS=4
export SEED=3407
export LR=1e-4
```

显式指定绝对解释器路径，不依赖 shell 是否激活 conda。默认数据及 Train/Val 列表位于仓库
`bda_benchmark/dataset` 和 `splitname/standard_ML`；需要更改时设置 `DATASET`、`TRAIN_LIST`、`VAL_LIST`。
程序不访问 Test。前端快照必须与真实 SOURCE_RUN 记录匹配，上面的快照路径也应据来源核实。

## 先做 CPU 检查，再做单套 smoke

直接同步本目录源文件后，先执行版本检查（不需要压缩包，也不用在终端手写 Python）：

```bash
bash bda_benchmark/experiments/damageformer_partition_learning_v1/run.sh version
```

应打印实际导入的 `train.py`、`reproduction.py` 绝对路径和
`VERSION CHECK PASSED: history_bounded_v1`。这只验证修复入口安装情况，不等于完整源码校验，
也不等于 Val 指标已经通过复现。版本检查不加载权重、不读取数据、不创建结果目录。
本脚本的所有训练入口和 preflight 也会先执行这项检查；旧版或不完整安装会在 Val450 之前停止。
`report` 仍只读 JSON，不执行模型导入检查。
已运行的旧 Python 进程不会因上传而更新；直接执行 `python -m ...train` 不经过这个启动检查。
请勿在该目录仍有训练时覆盖源文件，`run.sh` 也参与训练期间的代码完整性检查。

```bash
bash bda_benchmark/experiments/damageformer_partition_learning_v1/run.sh preflight
```

preflight 从仓库顶层发现本目录全部 `test_*.py` 并检查训练帮助，只要求 `PYTHON`。
CPU 测试验证数学/梯度与报告契约，不能替代服务器真实数据、CUDA 算子和显存检查。
本地已通过 66 项 CPU 测试（PyTorch 2.12.1），包含三套实际 20 步优化器循环；
这些循环使用合成输入和临时输出，不是 BRIGHT 数据训练。启动脚本已通过 Bash 语法检查。
本次启动版本保护另增 5 项 CPU 测试并通过，覆盖旧入口、错误导入路径和漏传校验文件等情形。
目标服务器 PyTorch 2.2.1/CUDA 的实际兼容性以服务器 smoke 为准，尚未代你运行。

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/PARTITION_LEARNING_V1/smoke_structure_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_partition_learning_v1/run.sh smoke_structure
```

smoke 每个分支更新 20 步，在第 20 步评估，仅验证执行和梯度；小样本 Val 结果不能作为正式性能。
第二、第三套分别用 `smoke_multimodal`、`smoke_support`，每次都必须设置全新 `OUTPUT_ROOT`。
训练入口要求 `PYTHON`、`FRONT_CKPT`、`SOURCE_RUN`、`OUTPUT_ROOT`；路径或来源检查失败就停止。
例如（在前一套结束后顺序运行）：

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/PARTITION_LEARNING_V1/smoke_multimodal_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_partition_learning_v1/run.sh smoke_multimodal

export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/PARTITION_LEARNING_V1/smoke_support_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_partition_learning_v1/run.sh smoke_support
```

## 正式训练：每次明确选一套

对应 smoke 通过、前一 GPU 任务结束后，选择需要的 suite。例如先做结构对照：

```bash
export OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/PARTITION_LEARNING_V1/structure5000_seed3407_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/PARTITION_LEARNING_V1
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/PARTITION_LEARNING_V1/$(basename "$OUTPUT_ROOT").log"
nohup bash bda_benchmark/experiments/damageformer_partition_learning_v1/run.sh structure > "$TRAIN_LOG" 2>&1 &
echo "PID=$!"
echo "结果=$OUTPUT_ROOT"
echo "日志=$TRAIN_LOG"
tail -f "$TRAIN_LOG"
```

需要另两套时，单独设置新输出和日志路径，再把最后入口选为 `multimodal` 或 `support`。
不提供 `all` / `train_all`；不要复制后台命令后在同一 GPU 上同时启动三套。
退出 `tail` 只停止看日志，不停止后台训练。
每套正式训练开始前都会先做完整 Val450 来源复现；此时只打印 `evaluate` 是正常的，
不是训练停住。`source_hard`、`old_pixel`、`old_region` 必须与来源记录复现一致才更新新模块。

正式协议固定每分支 5000 次更新，每 500 步评估。batch=2、accum=8 时有效 batch=16，
即每分支 80000 个裁剪；5000 步不是在分支之间分摊。同一套分支看到相同训练批次。
默认 `STRUCTURE_WEIGHT=0.1`、`SAR_WEIGHT=0.2`、`SUPPORT_TEMPERATURE=0.2`。
旧环境变量 `VARIANTS`、`EXP`、`STEPS`、`EVAL_EVERY` 不会改变本启动脚本的分组或步数。
Python CLI 同样限制正式训练为 5000 步、每 500 步评估；自定义短步数必须显式使用 smoke，不能作为正式结果。

## 结果和判断

```bash
bash bda_benchmark/experiments/damageformer_partition_learning_v1/run.sh report
```

report 只要求 `PYTHON` 与现有结果 `OUTPUT_ROOT`，仅读 JSON，不加载 GPU、不访问数据集。
训练会写 manifest、逐步训练/验证记录、summary、`audit_final.json`，以及每分支 `*_best.pth` / `*_last.pth`。
只保存分支权重，不复制前端、不保存优化器，因此不支持中断后的精确恢复；输出目录存在时拒绝重用。
branch 权重须与原前端及来源记录配合使用，不能当成完整前端模型文件。
实验中的 `hard` 指预测建筑/背景分组的硬隔离，区域内仍保留 9 候选的软 Q，
并不是把 Q 取 argmax。三套都没有真实建筑实例隔离或空间连通性保证。

报告把同一步终点与各自 best 分开；未完成、smoke、步数或验证记录不一致时，只显示临时验证结果。
主要对比是 `learn_q - fixed_q`、`learn_q_structure - learn_q`、`rgb_sar - rgb_rgb`、
`uniform - hard`、`sar - uniform` 和 `sar - hard`。关注全图前景 mIoU、三个建筑等级 IoU、
Damaged 的 P/R/F1，以及 audit 中漏检/误检分组、伤害/救回和事件差异。
只提高 Damaged 精度却降低召回，不能概括成全面改善；不同 best 步数也不是主要对照。

summary 保留分支权重是否变化、非零梯度步数、前端不可变验证、时间和峰值 CUDA 显存；
验证记录包含 routing 和 Q drift。资源是同进程前端加本套所有分支的总量，不能直接当单方法耗时。
最终验证还分别记录未拆组 Q 和建筑/背景拆组后的分区质量，逐类报告同类支持、前景/背景污染
与各自分母，不把背景占比造成的全局高纯度当作前景表现。support 的回投 Q 固定不变，
所以这些分区质量不是其跨组 Pooling 内容质量；应联合看交换量与实际分类错误转移。
`audit_final.json` 使用已有 DiagnosticAccumulator；条件子集的 mIoU 不可相加，GT 连通域是对象代理，
不是真实建筑实例。单种子结果仅用于检验假设，不能据此宣称稳定提升或原创性。

## 复现门槛修复：history_bounded_v1

2026-10-06 的实验一在首次完整验证后报 `old_region/precision`，尚未执行任何参数更新。
来源的 10 次冻结验证表明，旧区域分支相对最终记录已有最高约 0.003673 pp 的单项变化，
超过旧版统一的 0.001 pp 门槛。这证明校验需要考虑历史重复性，但没有证明失败那次的具体差值，
也不代表已经确定是哪一个 CUDA 算子导致波动。

本次只修复新实验的校验和诊断记录，模型、损失、学习率、旧实验源文件均不改：

- `old_pixel` 与 `source_hard`：所有指标继续使用 `1e-5`，即 0.001 pp。
- `old_region`：只读取来源已完成实验的 `validation.jsonl`。至少三条完整 Val450、无 Test 的
  历史记录中，某项相对来源终点的最大绝对差为 `d`；若 `d <= 1e-5`，仍用 `1e-5`；否则
  使用 `min(5e-5, d + 1e-5)`。阈值用于 0～1 的原始指标尺度；上限 `5e-5` 对应 **0.005 个百分点**。
- 校准在新评估前确定并写入 manifest，**不从当前模型结果调整门槛**。该经验边界不是统计置信区间。
- 历史缺失或损坏时退回严格门槛并记录警告；有效历史已超过硬上限时标记异常，不因一次偶然通过而放行。
- 权重、运行统计、代码、数据列表的完整性检查不变；历史验证文件也加入哈希和不可变检查。
- `initial.json` 先保存，再校验；`reproduction.json` 保存所有指标的原值、当前值、差值和门槛，
  失败也保留。`summary.json` 标记 `phase=baseline_check_failed` 与 `optimizer_steps=0`。
- 正式校验通过后日志出现 `BASELINE REPRODUCTION PASS`，随后才创建优化器并开始第 1 步。

请更新本目录全部文件（包含新增 `reproduction.py`），但不要覆盖来源 ROI_EXCHANGE/FOURSTAGE 等目录。
如果本目录另有活跃训练，不应在它运行过程中上传代码；本次已退出的失败运行可以更新后从新目录启动。
保留原失败结果，使用全新 `OUTPUT_ROOT`。重跑预检查后直接使用原 `structure` 正式入口即可；
不需要额外启动一次诊断推理，也不要为了通过而编辑来源 summary 或跳过校验。
