# 固定 hard 区域表示的 Memory 对照

本实验只回答：冻结当前已核验的区域表示后，Train 历史区域能否提供超出同容量分类适配器的帮助？不再训练Q，不加SAR分区、跨组交换、新主干、对比损失或类别重加权，不宣称严格复现SRAC。

来源是3000步区域联合训练前端快照，以及ROI_EXCHANGE已经训练5000步的`hard_last.pth`。它不是历史61.6866%的原始A1。这里hard指预测建筑/背景硬分组，组内仍为软Q，不是Q的argmax。

## 三个结果分别说明什么

| 名称 | 更新内容 | 用途 |
|---|---|---|
| source_hard | 无 | 冻结来源，约61.5216%的参考；不用重新训练 |
| nomemory | 新修正头 | 没有真实历史样本时，增加分类容量本身能带来什么 |
| memory | 相同结构的新修正头 | 相同容量下，检索Train历史区域是否进一步有效 |

核心输出是区域logits残差，不是像素logits残差：

```text
原图RGB与SAR → 冻结DamageFormer及Soft ROI → 固定Q和建筑分组
                                           ↓
                                  冻结的区域Embedding
                                    ├─ 冻结原区域分类器 → base_logits ─┐
                                    └─ 新修正头                     ＋
                                          ↑                         ↓
                        无Memory学习槽位 或 Train检索邻居      区域softmax
                                                                    ↓
                                                             同一hard路径回投
```

`final_logits = frozen_region_logits + correction`。修正头最后一层零初始化，两组step0严格等于source_hard，避免随机新分类器改变初始预测。零初始化不保证训练后不伤害原预测；无邻居时回退到query自身注意力，而不是自动零修正。

每组83,844个训练参数（dim128）。两组初值相同、参数独立、读入同一批裁剪，使用相同优化器预算与损失。nomemory使用4个可学习类别槽位加query，memory使用Top-8历史支持加query；参数相同不代表支持长度、KV构造或FLOPs相同。nomemory不是原像素头，也不是将训练后的Memory关掉。

## 库和检索

- 遍历Train2918；Val450和Test都不进入库。
- 复用冻结hard Embedding，维度128。存真实区域Token，不平均成单个类原型。
- 每事件×多数GT类别随机蓄水池最多64个，包括Background。14事件时上限3584条；实际桶可能不足。
- value为Train四类GT的软分布。它表示区域覆盖的建筑等级标签组成，不表示一栋建筑内部局部损坏比例。
- 全部正标注质量区域都可参与抽样，不新增纯度/尺寸过滤；manifest记录小质量候选数量和槽位纯度，不能用全局纯度掩盖背景占比。
- query仅凭冻结特征的余弦相似度检索，不接受query GT，不按预测B或真实类别过滤邻居。
- 默认排除同图和同事件；Train、Val规则相同。事件ID只用于排除，不进入分类器。
- Train支持以完整1024图像提取，Train查询为随机512裁剪，Val查询1024。冻结模型不能消除上下文差异，此限制保留在manifest。
- 复用旧库格式是因为Embedding/路由不变；只有身份、文件hash、分割列表、完整性与smoke标记均匹配的库才接受。不能使用旧ChangeMamba或旧四阶段Token缓存。

本轮不是强制建筑区域从本图背景再接收5%信息，因此不继续使用那种发送端预算设计。Memory的标签注意力仍可能伤害预测，需要审计，不能说彻底消除了错误上下文。

## 文件与隔离

- `core.py`：新残差分类器，配对头、检索调用和修正幅度统计。
- `build_bank.py`：冻结hard的Train库，复用经过测试的蓄水池和标签汇总。
- `train.py`：配对训练、初始复现、最终干预和错误分层。
- `diagnostics.py`：仅最终Val使用GT进行近邻质量诊断，不参与分类/训练。
- `report.py`：只读取结果，可做按事件分层的配对图像bootstrap。
- `run.sh`：preflight、smoke_all、bank、train、all、report。

仅新增本目录及研究记录；不修改旧实验文件、权重或结果。不下载论文仓库作为运行依赖，不需要新的CUDA扩展。模型加载仍依赖项目现有实验模块。若源代码或源权重hash不一致，停止而不是关闭检查。

## 上传和预检

把本地整个目录上传到服务器对应位置，不要只复制run.sh：

```text
本地 D:\software\dowmload\BRIGHT-master\bda_benchmark\experiments\damageformer_region_memory_v2
服务器 /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_region_memory_v2
```

以下使用真实已有路径，不依赖conda activate，不需要SWIN_PRETRAIN：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_20261005_133348
export BATCH=2
export ACCUM=8
export LR=1e-4
export SEED=3407
export TOPK=8
export PER_BUCKET=64
export QUERY_CHUNK=256
export DEVICE=cuda:0
export WORKERS=4
export ALLOW_SAME_EVENT=0
unset OUTPUT_ROOT BANK_ROOT

ls -lh "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth"
bash bda_benchmark/experiments/damageformer_region_memory_v2/run.sh preflight
```

旧`VARIANTS`、`POOL_MODE`、`READ_MODE`不会改变本轮固定hard规则。不要为了通过加载而覆盖旧文件。

## 冒烟检查

```bash
export MEMORY_ROOT="/root/shared-nvme/BRIGHT-checkpoints/REGION_MEMORY_V2/smoke_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_region_memory_v2/run.sh smoke_all
```

使用两张不同事件Train图建微型库，两组各20次更新，Val只看2张。只验证流程与梯度；不能用于比较精度或替代完整初始复现。已有输出会拒绝覆盖，换新目录即可，不要删除旧结果。使用同一目标的并发任务会被锁阻止；脚本不自动终止其他GPU任务。

## 正式建库和训练

确认smoke通过、GPU空闲后，以下all先完整建库，再训练两组各5000次更新，不是两组共用5000次更新：

```bash
export STEPS=5000
export EVAL_EVERY=500
export MEMORY_ROOT="/root/shared-nvme/BRIGHT-checkpoints/REGION_MEMORY_V2/formal5000_seed3407_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/REGION_MEMORY_V2
export TRAIN_LOG="/root/shared-nvme/BRIGHT-train-logs/REGION_MEMORY_V2/$(basename "$MEMORY_ROOT").log"
nohup bash /root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_region_memory_v2/run.sh all > "$TRAIN_LOG" 2>&1 &
echo "训练PID=$!"
echo "结果目录=$MEMORY_ROOT"
echo "日志=$TRAIN_LOG"
tail -n 80 -F "$TRAIN_LOG"
```

不要在重定向前另插空行。`Ctrl+C`只终止上述tail，不终止nohup训练。先出现bank 1/2918等建库进度，然后evaluate进行step0完整复现，最后才是step1训练；不能把初始化评估当作卡住。

如建库已经成功但训练失败，保留bank，用新训练目录重试，不重复建库：

```bash
export BANK_ROOT="$MEMORY_ROOT/bank"
export OUTPUT_ROOT="${MEMORY_ROOT}/train_retry_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_region_memory_v2/run.sh train
```

这不是断点恢复：新修正头从零初始化开始，AdamW状态不恢复。不得把失败后的两段累计成同一次完整训练。

## 训练协议与输出

有效batch16，AdamW lr1e-4、weight decay5e-3，100步warmup与余弦衰减、梯度裁剪1、FP32。5000次更新对应每组80000裁剪。损失仍是区域概率回投后的NLL+Lovasz，没有新增类别/边界/对比损失。源模型始终eval，无参数和运行统计更新。

每500步完整Val450。主要比较固定5000步的memory−nomemory及memory−source_hard。最佳checkpoint仅作为次要记录，不用Test挑方案。训练第一步attention和label embedding可能没有梯度，因为末层为零；之后逐模块梯度计数应非零。

输出包括：

- `initial.json`、`reproduction.json`、`baseline_policy.json`：初始恒等与来源复现；历史容差只由旧日志提前确定，不用当前误差调阈值。
- `training.jsonl`、`validation.jsonl`、`summary.json`：步数、损失、覆盖率、残差幅度、改类比例、权重和梯度检查。
- `audit_final.json`：全图与各类、ROI TP/FN/FP/TN、边界/内部、GT连通域尺寸代理、事件、每图混淆与rescue/harm。
- `retrieval_quality.json`：近邻标签与离线query GT组成的匹配、允许库先验参照。不是全图mIoU，也不把像素当独立样本。
- `nomemory_best.pth`、`nomemory_last.pth`、`memory_best.pth`、`memory_last.pth`：只存新修正头。没有新的761MB前端副本、优化器或密集特征缓存。

库纯向量上限约1.75MiB（14×4×64×128×4字节），加标签、索引和元数据略大。四个FP32修正头纯参数合计约1.28MiB。实际磁盘占用还包括JSON诊断，不能把这两个数当作总文件大小上限。缺少优化器/RNG/数据顺序状态，不能精确续训。

最终干预`random`、`shuffle_labels`、`off`都使用同一个训练后Memory头，仅用于依赖性检查，不是独立训练对照。标签打乱在每张查询图内固定一次，不随query_chunk重复洗牌。`off`仍保留训练后的query修正；真正关闭全部修正对应source_hard。

```bash
export OUTPUT_ROOT="$MEMORY_ROOT/train"
export BOOTSTRAP=2000
bash bda_benchmark/experiments/damageformer_region_memory_v2/run.sh report
```

bootstrap按事件内的配对图像采样，不把像素当独立样本；它不等于训练种子波动，也不能消除相邻影像块相关性。有正向信号再运行预先约定的独立种子。

## 结果后的决定

1. memory优于nomemory且优于source_hard，同时关键事件/类别无明显代价：重复种子，再考虑正式方法。
2. 两组都提升但差距很小：收益主要可能来自分类适配；不能证明检索有价值。
3. memory下降且近邻标签匹配不佳：先查表示距离、事件差异、混合标签和裁剪上下文；不立即解冻Q。
4. 近邻匹配尚可，但Memory伤害高：查修正幅度与off/random/shuffle变化，再单独研究读出；不同时加新损失。
5. 无明确增量：保留hard参照，停止这版检索，而不是无限增加库或训练步数。

本轮没有实现接收端5%约束修复；那是独立的本图支持域实验，不与外部Memory同时改变。
