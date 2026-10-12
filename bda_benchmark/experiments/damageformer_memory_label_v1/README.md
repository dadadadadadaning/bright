# 固定区域上的显式标签检索验证

本实验只检查既有 Train Memory 是否提供原 hard 分类器缺少的损毁证据。不是严格复现 Tip-Adapter，不加入新主干、Q 学习、PaCo、类别损失或新库采样，也不承诺涨分。旧实验、权重和库均不覆盖。

## 每组在验证什么

| 输出 | 内容 | 问题 |
| --- | --- | --- |
| `source_hard` | 原冻结区域头 | 当前真实基线，不是历史 A1 |
| `knn_only` | 邻居 Train 软标签按余弦温度加权 | 历史标签本身是否有等级信息，原模型错误处有没有互补 |
| `knn_mix` | 固定 0.1 比例融合，固定温度 0.1 | 不训练时，少量历史标签是否能净纠错 |
| `nomemory` | 只训练自身概率温度、前景 logit 偏置 | 不查库的简单校准能改善多少 |
| `memory` | 只训练邻居温度、融合比例 | 显式历史标签是否超过自身校准 |
| `*_random` | 相同预算随机合法邻居，仍按真实余弦加权 | 找到特定邻居是否重要 |
| `*_shuffle_labels` | 特征与排名不变，只对合法库标签作每图固定置换 | 支持标签与特征对应关系是否重要 |
| `*_off` | 直接返回 `source_hard` | 关闭标签读取的参照 |

`knn_only` 不是 GT oracle，不用查询标签选邻居。没有合法邻居时精确退回基线。随机、标签打乱和 off 都是同一参数下的推理干预，不是独立训练组。此处 off 等于 source_hard，与旧 residual Memory 的 off 定义不同。

两组可训练模型均只有 **2 个标量**，初值均为 0；不是以前的 83844 参数修正头。相同参数量不等于相同函数或计算量。冻结 Swin、Q、B、Soft ROI、Embedding、原区域分类器和 bank。

## 概率和训练规则

对合法 Top-8 支持，令余弦分数为 s，四类 Train 分布为 d：

```text
w = softmax(s / T)
p_knn = sum(w * d)
p_memory = p_base + lambda * (p_knn - p_base)

p_nomemory = softmax(log(p_base) / T_self + bias * [0, 1, 1, 1])
```

Memory 的 lambda 在 [0,0.5]，温度在 [0.02,4]；无 Memory 的前景偏置在 [-0.5,0.5]，温度范围相同。Memory 初始 lambda=0，精确复现原概率。无 Memory 初始温度1、偏置0，在 FP32 舍入误差内复现。训练后投影回预设范围，边界固定，不根据本次 Val 改动。

采用直接概率回投，**不把概率再次传入 softmax**。同图和同事件始终排除，包含随机与打乱对照；查询 GT 只用于 Train 损失或预测完成后的 Val 诊断。

默认训练1000次更新，每250步完整 Val450，batch2×累积8，实际裁剪数16000；Adam、标量学习率0.01、余弦衰减、梯度裁剪1、FP32。两组同批次独立训练，沿用原像素 NLL+Lovasz，不加类别权重。0.01只用于这两个标量，不能套给主干。可在首次正式运行前固定为5000步，但不要根据最高Val点反复延长预算。

lambda 学到0、梯度暂为0或最终负收益，均允许正常保存，不作为运行错误。无 Memory 使用直接温度／偏置校准，避免“自身概率混合＋lambda0＋T1”导致两个参数初始梯度同时为0的退化对照。

## 不复制旧库 先上传整个新目录

本地目录：

`D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_memory_label_v1`

上传至服务器：

`/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_memory_label_v1`

依赖服务器已有 roi_followup、roi_exchange、region_memory_v2、partition_learning 的来源校验及其依赖；不要只传 run.sh。无需改旧文件或安装新 CUDA 扩展。只加载自己产生且可信的权重文件。

## 明确设置所有来源

以下路径来自当前对话已有运行。若检查不存在，请定位实际完成目录，不要用 A1、其他框架缓存或 smoke 库替代。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_20261005_133348
export BANK_ROOT=/root/shared-nvme/BRIGHT-checkpoints/REGION_MEMORY_V2/all_seed3407_20261007_084008/bank
export BATCH=2
export ACCUM=8
export SEED=3407
export DEVICE=cuda:0
export WORKERS=4
export LABEL_TOPK=8
export LABEL_LR=0.01
export LABEL_TEMPERATURE=0.1
export LABEL_MIX_WEIGHT=0.1
export LABEL_ROOT="/root/shared-nvme/BRIGHT-checkpoints/LABEL_MEMORY_V1/run_$(date +%Y%m%d_%H%M%S)"

ls -lh "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth" "$BANK_ROOT/bank.pt" "$BANK_ROOT/manifest.json"
bash bda_benchmark/experiments/damageformer_memory_label_v1/run.sh preflight
```

`BANK_ROOT` 是目录不是 bank.pt 文件。正式训练需要全部 Train2918 建库成功；新的 smoke 可以读取已经验证的完整正式库，仅减少查询图数／步数，不必额外建库。来源哈希不一致时应查清版本，不要放宽检验。

## 先做两项 smoke

```bash
export OUTPUT_ROOT="$LABEL_ROOT/smoke_audit"
bash bda_benchmark/experiments/damageformer_memory_label_v1/run.sh smoke_audit

export OUTPUT_ROOT="$LABEL_ROOT/smoke_train"
bash bda_benchmark/experiments/damageformer_memory_label_v1/run.sh smoke_train
```

前者是 Val 前2张只读检查，后者是20步训练并验证2张图的真实数据冒烟，不是本地单元测试的合成数据。smoke 不判断性能，无论分数高低都不能证明有效。目录存在就换一个新子目录，不删旧结果。

## 完整只读审计

```bash
export OUTPUT_ROOT="$LABEL_ROOT/audit"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/LABEL_MEMORY_V1
export AUDIT_LOG="/root/shared-nvme/BRIGHT-train-logs/LABEL_MEMORY_V1/$(basename "$LABEL_ROOT")_audit.log"
nohup bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_memory_label_v1/run.sh audit > "$AUDIT_LOG" 2>&1 &
echo "审计PID=$!"
echo "结果目录=$OUTPUT_ROOT"
tail -n 80 -F "$AUDIT_LOG"
```

完整审计完成后：

```bash
export REPORT_ROOT="$LABEL_ROOT/audit"
bash bda_benchmark/experiments/damageformer_memory_label_v1/run.sh report
```

`report` 只读已有结果，不用GPU。必须使用已完成目录，**不要重新生成时间戳**。尾随日志的 Ctrl+C 仅退出 tail，不终止后台审计。请避免同时在同一张卡启动现有大训练和本次任务导致OOM；本脚本不杀其他进程。

## 标量训练对照

两项 smoke 通过，并读完完整审计后，再决定是否执行。严重负向邻居证据可以直接停止，不强求训练。

```bash
export STEPS=1000
export EVAL_EVERY=250
export OUTPUT_ROOT="$LABEL_ROOT/train1000"
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/LABEL_MEMORY_V1/$(basename "$LABEL_ROOT")_train1000.log"
nohup bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_memory_label_v1/run.sh train > "$TRAIN_LOG" 2>&1 &
echo "训练PID=$!"
echo "结果目录=$OUTPUT_ROOT"
tail -n 80 -F "$TRAIN_LOG"
```

完成后：

```bash
export REPORT_ROOT="$LABEL_ROOT/train1000"
bash bda_benchmark/experiments/damageformer_memory_label_v1/run.sh report
```

这不是继续训练旧Memory头。旧库／前端只读，两个标量读出从身份映射开始。每组仅保存best和last的小型参数文件，不复制前端、bank和optimizer，不能精确断点续训。

## 怎么判定检索提供了原模型缺少的证据

正式运行先完整验证源模型再训练，使用已有历史限定复现门槛，保存失败证据，不根据新结果扩大容差。结束再验证源文件、参数、buffers和bank哈希未变。

重点产物：

- `summary.json`：是否完整、最终同一步指标、两个标量、冻结校验。
- `audit_final.json`：全图、GT前景、各类、ROI错误、边界两侧、小对象代理、每事件／每图的混淆和救回／伤害转移。
- `retrieval_quality.json`：按等级及原hard区域正确／错误分组，记录邻居真类质量、投票和合法库先验。
- `label_evidence.json`：`supported_rescue` 为“原hard错、纯邻居标签对、融合也对”的像素交集；`aligned_harm`为“原hard对、邻居错、融合也错”；同时记录未利用的邻居机会。这是关联证据，不是因果证明。
- `training.jsonl/validation.jsonl`：配对训练和验证曲线。
- `initial.json/reproduction.json/manifest.json`：身份复现、配置和来源。

报告采用保守的单次Val信号检查：完整450张、非smoke、源复现和冻结通过，融合同时优于实际比较基线及source_hard的前景mIoU，全图和真实前景救回多于伤害，Damaged净纠错与F1不下降，正常检索优于随机和标签打乱。未通过会明确输出，不自动挑另一个参数／最佳点冒充通过。

这是一组预设诊断条件，不是统计显著性证明或普适必要条件。可能出现前景mIoU上升但像素净纠错为负；报告会分别展示，不能混为一谈。事件退化单独列出，不能被总体掩盖。有信号再重复种子，不能把多次同运行验证当作独立重复。

GT连通域不是建筑实例；条件分组分数不是全图mIoU；区域GT比例不是楼内损坏面积；查询GT不参与检索或融合。当前代码不训练检索key，不改变超像素尺度，也不声称复现SRAC。
