# 已训练 Memory 读出审计 V1

这个独立目录回答一个具体问题：已训练的 Memory 头实际怎样读取支持信息，这些读取是否改变预测，以及正确修正与错误修正分别发生在哪里。它复用已有正式权重和 Train 支持库，不训练新模型、不重建库、不改旧实验文件，也不访问 Test。

主要对照是 `normal - nomemory`，用于判断已训练 Memory 是否提供超出同构残差容量的收益。同时检查 `normal - source_hard`，以及打乱支持标签、关闭外部支持、限制区域修正后的变化。注意力较大、余弦检索分数较高或标签更相似，本身不能证明支持信息改善了最终预测。

## 九种输出

所有输出使用同一批 Val 图像配对评价。

| 名称 | 含义 |
|---|---|
| `old_pixel` | 已冻结旧像素分支，作为来源与结果复现对照。 |
| `old_region` | 已冻结旧区域分支，作为来源与结果复现对照。 |
| `source_hard` | 当前 hard 汇总和 hard 回投下的原区域分类器。 |
| `nomemory` | 同次正式训练的无外部记忆残差头。 |
| `normal` | 正式训练的 `balanced_hard_last.pth` Memory 头，从 Train hard 库按类均衡读取支持（原训练 Top-8，每类 2 个）。 |
| `shuffle_labels` | 固定查询、支持特征和检索条件，打乱允许支持的标签信息。 |
| `off` | 屏蔽全部外部支持，保留已训练头的 query fallback。 |
| `hard_only` | 仅对 `hard_count > 0` 的查询槽保留正常 Memory 修正。 |
| `tail_only` | 仅对 `hard_count == 0 且 mass > 0` 的查询槽保留正常 Memory 修正。 |

`off` 仍有已训练的 query-only 残差，不等于 `source_hard`。`nomemory` 是独立训练的匹配对照，也不等于把 Memory 的支持关闭。

两种已训练头固定使用同一正式训练最终步数的 `balanced_hard_last.pth` 和 `nomemory_last.pth`。`normal` 不使用 `global_all`；均衡检索读取 Train 支持的类别信息，不读取查询 GT。

`hard_count` 按原生特征网格逐位置取候选轴 `q.argmax`，再将对应全局槽 ID 累计计数。它表示一个槽是否至少赢得一个位置，不表示槽纯度、建筑实例或真实类别。`mass > 0` 但 `hard_count == 0` 的槽称为 soft-tail；很小的正质量也保留，不加质量阈值。

`hard_only` 和 `tail_only` 只拆分残差修正：不删除槽、不改 Q、不停用原 hard 分类器，也不重跑分区。两者都保留全部 `source_hard` 基础 logits，严格零质量槽保持基础输出。区域残差满足：

```text
(hard_only - source_hard) + (tail_only - source_hard)
    = normal - source_hard
```

这是区域 logits 的恒等式；经过 softmax、回投和 argmax 后，像素准确率和 mIoU 不具有可加性。

## 读取证据怎样解释

预测使用原 `ResidualSupportHead` 的真实 `need_weights=False` 路径。另一次观测请求每个注意力头的权重，并核对 context、logits 和 argmax 一致性；观测输出不替换原预测。权重在“全部支持 + 最后的 query fallback”上归一化，屏蔽支持的权重为零。

报告分开呈现真实注意力、只依赖余弦相似度的标签证据、残差幅度与概率变化，并按全部正质量槽、hard 槽、soft-tail 槽和类别统计。按槽平均与按汇总质量加权分别显示，不能把区域准确率当作像素准确率。注意力加权标签只是一种解释性统计，不是 reader 的最终后验，也不是因果贡献分解。零初始化末层即使有注意力，仍可能完全没有有效残差。

像素证据使用配对统计，区分“原本错误而被修正”和“原本正确而被改错”，并查看事件、边界、区域内部和 ROI 等子集。类别收益须同时看 Precision、Recall、FP/FN 及其他类别代价。查询 GT 只用于预测完成后的评价和分组，不进入检索、注意力、干预或分类器。

## 运行

把整个新目录放在服务器仓库对应位置。下面使用已有正式源、正式 Memory 训练和正式 Train 支持库；运行器会继续核验来源、训练配置与权重身份。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_20261005_133348
export MEMORY_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/formal5000_seed3407_20261007_143225
export BANK_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_BALANCED_V2/bank_seed3407_20261007_140746
unset TOPK QUERY_CHUNK SEED

bash bda_benchmark/experiments/damageformer_memory_readout_audit_v1/run.sh preflight
```

`preflight` 运行 `test_core`、`test_report`、`test_provenance`、`test_audit` 和命令行入口检查，不需要 GPU 或正式文件。通过只表示 CPU 合成契约成立，不代表真实 CUDA、正式权重或 Val450 已验证。

先用两张 Val 图检查完整流程：

```bash
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/MEMORY_READOUT_AUDIT_V1/smoke_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_memory_readout_audit_v1/run.sh smoke
```

`smoke` 仍读取上述正式权重和正式库，只把查询限制为两张 Val 图；它不训练，也不是 smoke 权重或 smoke 库实验。其结果不能用于性能或有效性结论。

流程检查通过后，运行完整 Val450：

```bash
export AUDIT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/MEMORY_READOUT_AUDIT_V1/run_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT="$AUDIT_ROOT"
bash bda_benchmark/experiments/damageformer_memory_readout_audit_v1/run.sh audit
```

不要预先创建 `OUTPUT_ROOT`。脚本拒绝已有输出和已有同名 `.lock`，避免覆盖历史证据。五个源/输出路径都要求服务器绝对路径。`PYTHON` 未设时使用 `python`；脚本强制无缓冲输出。可设置 `DEVICE`、`WORKERS`、`DATASET`、`TRAIN_LIST`、`VAL_LIST`；默认设备为 `cuda:0`，workers 为 4。`topk`、`query_chunk` 和 `seed` 从保存的训练 manifest 自动读取，不能在审计中调参；清除旧 shell 中对应的大写环境变量。

仅生成已有审计目录的文字报告：

```bash
export REPORT_ROOT="$AUDIT_ROOT"
bash bda_benchmark/experiments/damageformer_memory_readout_audit_v1/run.sh report
```

`REPORT_ROOT` 必须明确设置并已存在，报告动作不读取 `OUTPUT_ROOT`，因此不会误用旧 shell 中遗留的输出目录。它直接运行报告模块，不检查 GPU、源权重或库，也不启动评估。默认 `BOOTSTRAP=0`；需要时可以设置正整数，例如 `BOOTSTRAP=2000`。置信区间只在完整、来源核验和复现均通过的非 smoke Val450 配对证据满足要求时计算；缺失或失败项保留为缺失，不补造结果。

脚本仅支持 `preflight`、`smoke`、`audit`、`report` 四个动作。真实结果以本次产物和来源核验为准；这一目录中的 CPU 测试不能替代服务器 smoke 和完整审计。
