# SRAC Evidence Follow-up V1：先定位问题，再验证事件配额

这是新的独立实验包。没有改写历史模型、历史建库程序或旧结果。
本地只有 CPU 合成测试；不能据此宣称服务器 CUDA 故障已解决或模型已提升。

## 1. 这次分别验证什么

| 阶段 | 更新参数吗 | 回答的问题 |
| --- | --- | --- |
| 原生 Pool 单图捕获与独立重放 | 否 | 区域 ID/尺寸/有限值是否异常？同一份输入的 CPU、原生 CUDA、分组归约结果是否一致？ |
| A/B/C 完整特征审计 | 否 | 固定 F、Q、原 A1 头，特征汇总与分类/回投顺序怎样影响 Damaged？ |
| 完整 Memory 审计 | 否 | 同一批 J1 判错的 Damaged，有没有参考、Top-K 是否取到、取到后是否救回？ |
| M0/M1 建库 | 否 | 在相同类别容量下，用部分事件均衡替代当前事件配额，改变了哪些参考？ |
| 新 J1/M0/M1 分类器训练 | 只训练三个新读出头 | 配额改变能否转化为真实的检索、救回和最终预测增益？ |

先运行第 2–5 节。第 7 节必须在完整证据支持配额假设后手动启动。
不要把纯度降低的 95.6% 来自事件配比，解释成性能下降的 95.6% 来自事件配比。

### A/B/C 的准确含义

- A=`pixel_grid`：F 经原 A1 的 1×1 分类头，在特征网格 softmax，再上采样。
- B=`feature_mean`：按 Q 平均 F，再按同一 Q 回投 F，原头分类，softmax，再上采样。
- C=`region_mean`：按 Q 平均 F，原头逐区域分类并 softmax，按 Q 回投概率，再上采样。
- `grid_probq`：先在特征网格算类别概率，再按 Q 汇总/回投概率。
- 原生 A1 单独报告。其 logits 上采样/softmax 顺序可能不同，不将这个差异算成区域汇总损失。

B/C 的区别主要包含 softmax 与回投的顺序。A→B 变差说明原头对平均后的特征敏感，
不能仅凭该项断言信息不可恢复。这里的诊断均值采用严格正质量归一化；
父模型注意力 Pool 的原有 `clamp_min(1e-6)` 则原样保留，两者没有混为同一算子。

## 2. 上传与统一配置

上传整个 `damageformer_srac_evidence_followup_v1/` 文件夹到服务器对应位置。
它依赖工程中已有的 `damageformer_srac_stage_diagnostics_v1`（包括 `grouped_pool.py`）、
`damageformer_srac_joint123_v1` 及父模型模块。缺少依赖时先同步匹配版本，
不要为了通过 hash 检查临时修改旧模型源文件。

以下都在服务器执行；不要复制终端提示符或 Markdown 的三反引号。
`JOINT_RUN` 是包含三个权重的目录，不是 `.pth` 文件，也不是 smoke/probe 目录。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export EXP=/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_evidence_followup_v1
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export JOINT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint
export FAILED_MEMORY=/root/shared-nvme/BRIGHT-checkpoints/SRAC_STAGE_DIAGNOSTICS_V1/full_memory_20261010_135837_2958464
export FOLLOWUP_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_EVIDENCE_FOLLOWUP_V1/run_$(date +%Y%m%d_%H%M%S)_$$"
export DEVICE=cuda:0
export WORKERS=4
export SEED=3407
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export FOLLOWUP_START_INDEX=0
export FOLLOWUP_MAX_IMAGES=0

ls -lh "$A1_CKPT" "$JOINT_RUN/j1_last.pth" "$JOINT_RUN/j2_last.pth" "$JOINT_RUN/j3_last.pth"
bash "$EXP/run.sh" preflight
```

新特征审计自动从该 J 实验解析并核对前端快照，不再默认使用另一份诊断快照。
所有输出目录必须全新，不要提前 `mkdir FOLLOWUP_OUTPUT`。失败目录保留，下次换名字。
新代码同步完整后再运行；运行过程中不要修改源代码或输入权重。

已有 408 张的旧 Memory 结果可以先只读汇总，不用 GPU：

```bash
export FOLLOWUP_MEMORY_RUN="$FAILED_MEMORY"
unset FOLLOWUP_FEATURE_RUN
bash "$EXP/run.sh" report
```

报告明确标为不完整，只用于排错，不能选模型或启动正式配额实验。

## 3. 先把原生区域 Pool 的异常单独检查清楚

### 3.1 捕获原失败图片

从旧 manifest/summary 读取准确失败图片与来源，先检查所有输入文件哈希。
原日志写入过 2918 个缓存分片的哈希时，这一步也会核对它们，可能需要等待。

```bash
export POOL_MODE=native
export POOL_CAPTURE=all
export POOL_CHECK_REFERENCE=1
export RECHECK_PREFIX=0
export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/recheck_native"
bash "$EXP/run.sh" recheck_memory
```

拦截器在原生汇总调用前保存 CPU 输入，检查 ID 范围、形状、dtype、有限值、非负质量，
记录 layout、调用位置和图片。若仍出现非法访问，该 Python 进程会退出，不在损坏的 CUDA
上下文中尝试恢复。异常可能发生在更早的算子，此时不能保证得到当前调用的完整输入。

`POOL_CAPTURE=all` 仅允许一/两张图，会占用较多磁盘；完整审计只用 `failure`。

### 3.2 同一份输入分别启动新进程重放

前一步成功或已经保存 `pool_debug/input_*.pth` 后运行：

```bash
export POOL_CAPTURE_RUN="$FOLLOWUP_ROOT/recheck_native"
export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/pool_replay"
bash "$EXP/run.sh" replay_captured
```

每份输入的 CPU、native、grouped 各开新 Python 进程，默认每种重复 3 次，
保留输入 strides/storage offset，按原容差比较 CPU 参考及重复结果。
即使 native 子进程失败，也会继续独立检查 grouped，失败不会被改成通过。
检查 `$FOLLOWUP_ROOT/pool_replay/summary.json` 中的 `backend_passed` 和各子日志。

- ID/尺寸/有限值失败：先定位上游数据或分区，不能继续训练。
- CPU 参考也失败：先定位公式或输入问题。
- native 失败、grouped 全通过：支持进一步验证另一种归约实现，不证明已查明 CUDA 根因。
- 单图都通过：仍需前缀/完整验证，不能宣称故障已消失。

`grouped` 将散点累加替换成按目标区域组织的分组归约；不改 Q、不改注意力权重，
保留重复 ID、空区域、微小质量和原有分母处理。它是显式数值执行对照，不是新网络模块，
也不自动 fallback。只有 CPU 对照及真实图 smoke 通过后再用于完整审计。

## 4. 同源 A/B/C 与 Memory 两图 smoke

下面显式选择 grouped。如果决定继续检查原生实现，请显式设置 native，
不要让两次实验不知不觉使用不同执行模式。

```bash
export POOL_MODE=grouped
export POOL_CAPTURE=failure
export POOL_CHECK_REFERENCE=1
export INVENTORY_SHARDS=0
export FOLLOWUP_START_INDEX=0
export FOLLOWUP_MAX_IMAGES=0

export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/smoke_feature"
bash "$EXP/run.sh" smoke_feature

export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/smoke_memory"
bash "$EXP/run.sh" smoke_memory
```

两图均通过后，可先以相同验证顺序运行到原失败图片，验证连续执行条件：

```bash
export POOL_CHECK_REFERENCE=0
export RECHECK_PREFIX=1
export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/prefix_memory"
bash "$EXP/run.sh" recheck_memory
```

前缀仍不是完整 Val，也不能拿其分数与完整 A1 对比。
关闭逐调用 CPU 参考仅用于降低完整运行开销；ID/输入检查、来源检查、冻结检查及
最终复现检查仍保留。前述 CPU 参考未通过时，不应通过关闭它来继续。

## 5. 串行完成两个完整审计

确认第 3–4 节通过后执行。只运行审计，不建新库、不训练、不访问 Test。
每张图会记录原始混淆矩阵与成对错误转移；中断后报告仍明确为不完整。

```bash
export POOL_MODE=grouped
export POOL_CAPTURE=failure
export POOL_CHECK_REFERENCE=0
export INVENTORY_SHARDS=1
export MEMORY_TEMPERATURE=0.1
export FOLLOWUP_START_INDEX=0
export FOLLOWUP_MAX_IMAGES=0
export FOLLOWUP_FEATURE_RUN="$FOLLOWUP_ROOT/feature"
export FOLLOWUP_MEMORY_RUN="$FOLLOWUP_ROOT/memory"

mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_EVIDENCE_FOLLOWUP_V1
export FOLLOWUP_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_EVIDENCE_FOLLOWUP_V1/$(basename "$FOLLOWUP_ROOT")_audits.log"

nohup bash -c '
set -euo pipefail
cd /root/shared-nvme/BRIGHT-master
export FOLLOWUP_OUTPUT="$FOLLOWUP_FEATURE_RUN"
bash "$EXP/run.sh" feature
export FOLLOWUP_OUTPUT="$FOLLOWUP_MEMORY_RUN"
bash "$EXP/run.sh" memory
bash "$EXP/run.sh" report
' > "$FOLLOWUP_LOG" 2>&1 &

echo "审计PID=$!"
echo "日志=$FOLLOWUP_LOG"
tail -F "$FOLLOWUP_LOG"
```

`Ctrl+C` 只退出日志查看。完成后可再次 `bash "$EXP/run.sh" report`。
完整有效证据必须为 `complete=true`、`gates_passed=true`、
`full_validation_evidence=true`，并且 450 张、原预测复现、输入文件及冻结状态检查通过。
不能因新执行方式得到更高分就放宽复现容差。

## 6. 用哪些结果决定下一步

Memory 固定使用「GT=Damaged 且同一个 J1 判错」的样本，另外统计 J1 原本正确的
Damaged、Intact、Destroyed，避免只看救回不看伤害。GT 只在预测完成后做诊断。

新增困难样本字段 `difficult_retrieval` 包含：有支持的 Q 质量、有至少一个 D 主类邻居
的 Q 质量，以及 D/I 邻居比例。软 Q 与插值可能混合多个区域，这些是投影加权统计，
不是每个像素独立检索，更不是 Reader 注意力。

| 有效完整诊断结果 | 更合理的下一步 |
| --- | --- |
| A→B 明显退化，D-minus-I margin 下降 | 固定同特征/Q，比较像素头与区域头重新适配；先不扩大 Memory |
| A/B/C 能保留信息，而实际 direct_region 差 | 查结构 adapter、attention、mean/std 与新分类头；不先否定超像素 |
| 跨事件排除后合适 D 参考不足，或容量明显偏向低信息事件 | 才考虑第 7 节的 M0/M1 配额假设 |
| 库里有参考，Top-K 经常选到 Intact | 优先检索表示/度量；单纯改配额未必有效 |
| 检索标签证据支持 D，但联合输出没救回 | 进一步查 Reader、上下文回投与联合分类；不能仅凭标签投票归责 Reader |
| D 救回增长，但 I/X 伤害与误报更大 | 检查证据使用强度与偏置，不只报告 D 分数 |

关闭支持、随机邻居、打乱标签采用同一个已训练 Reader。关闭支持的 fallback 不是
重新训练的 J1。输入干预只能说明模型对参考的依赖，不等于重新训练后的增益。
库存纯度与配比只说明库的组成；不构成配额导致性能瓶颈的单独证明。

## 7. 只有配额假设值得检验时，才启动 M0/M1

### 7.1 控制条件

- M0：重现历史「事件×类别」数量，使用新的固定随机优先级选区域。
  **M0 不是历史 bank 的逐行复制，不能直接沿用历史 J2 分数当对照。**
- M1：每个有该类候选的事件先给可行保底（默认 16），剩余按候选数量平方根分配。
  若总量无法容纳保底则统一降低；不超过各事件实际容量；整数配额严格守恒。
- M0/M1 都使用同一个 row-specific 优先级，单事件内部选样嵌套，只改配额。
- 各类别总量保持旧库：这份数据为 Background 896 / Intact 896 / Damaged 853 / Destroyed 766。
- 不新增纯度阈值，保留旧缓存的候选资格规则；不改软标签，不换 Q/Embedding，
  不重提取 Train 特征，不用 Val GT 选参考。
- 不改 J2 式余弦检索；J3 的旧 metric 只用于训练前基线复现，不用于新的 M0/M1。
- 新训三个独立、相同初始化的读出：J1（无 Train Memory）、M0、M1。
  Pixel 和 context 共同分类；新增 context 列零初始化，使初始预测可与 A1 校验，
  并不是把 Memory logits 加到 A1 logits 上。

按已提供的 D 数量，默认 M1 的 14 事件配额依次为
`34,48,23,26,45,30,88,24,155,29,44,26,120,161`，总计 853。
这只是待验证的分配规则，不是已证明更好的配额。

### 7.2 接口 smoke（仍不作为方案证据）

只在准备做配额实验时执行。建库 smoke 仍核对并扫描完整 Train 缓存，可能较慢；
它不重提取图像，但产物标为 smoke，不能用于正式训练。

```bash
export POOL_MODE=grouped
export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/smoke_banks"
bash "$EXP/run.sh" smoke_banks
export QUOTA_BANKS="$FOLLOWUP_OUTPUT"

export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/smoke_quota_train"
bash "$EXP/run.sh" smoke_train
```

smoke_train 强制 20 步，不能把前景 mIoU 低解读为方案失败。

### 7.3 正式建库与训练

先根据真实完整报告设置 `QUOTA_EVIDENCE_NOTE`，写出具体事件/困难样本/检索证据，
不要填「纯度低所以一定是配额问题」。代码检查证据完整性，但不能代替你的因果判断。

```bash
# 先设置 QUOTA_EVIDENCE_NOTE 为你从有效完整报告读到的具体观察。
: "${QUOTA_EVIDENCE_NOTE:?请先填写完整审计中支持配额对照的实际证据}"
export QUOTA_EVIDENCE_NOTE
export CONFIRM_QUOTA_TEST=1
export FOLLOWUP_FEATURE_RUN="$FOLLOWUP_ROOT/feature"
export FOLLOWUP_MEMORY_RUN="$FOLLOWUP_ROOT/memory"
export QUOTA_FLOOR=16
export POOL_MODE=grouped

export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/quota_banks"
bash "$EXP/run.sh" banks
export QUOTA_BANKS="$FOLLOWUP_OUTPUT"

export STEPS=5000
export EVAL_EVERY=500
export LR=1e-4
export FOLLOWUP_OUTPUT="$FOLLOWUP_ROOT/quota_train"
export QUOTA_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_EVIDENCE_FOLLOWUP_V1/$(basename "$FOLLOWUP_ROOT")_quota.log"

nohup bash "$EXP/run.sh" train > "$QUOTA_LOG" 2>&1 &
echo "配额对照PID=$!"
echo "日志=$QUOTA_LOG"
tail -F "$QUOTA_LOG"
```

一个进程内依次更新三组，每组 5000 步；不要重复启动三个相同进程。
本实验采用**每步一张完整 Train 图，batch=1、无梯度累积、无数据增强**，
三组看到完全相同的图和更新预算。不是旧 `batch=2, accum=8` 的 80000 裁剪协议，
不要与历史最佳成绩当作等预算比较。显存不足则先停下检查，不自行改成不同组不同预算。

首次更新前复现原 J1/J2/J3 完整 Val，并检查三组新头初始预测与 A1 一致。
训练只更新新读出，冻结 front/Q/Embedding 和库；同一损失、优化器、学习率计划，
每 500 步完整 Val，保存独立头、优化器及结果。不提供完整 RNG/采样状态续训入口，
因此停止不等于可以精确原地续训。

同一步主要比较 M1−M0：是否提高困难 D 的参考命中、标签证据和实际救回，
同时检查 I/X 伤害、FP、每事件表现。再比较各 Memory 组与本轮新 J1。
这些统计保存在 `validation.jsonl` 的 `difficult_retrieval`、`difficult_cohorts`、
`eligible_bank_by_query_event` 及相应逐事件字段；最终摘要另保存 `final_` 前缀的统计。
诊断标签证据温度固定为 0.1，不是 Reader 注意力，也不影响预测。
同一步两组共享同一批 J1 错误；不同步的 J1 会变化，不能把跨步比例当成固定样本对比。
单次最好值不等于可靠提升；有正向信号后才安排不同随机种子重复，Test 不参与选择。
若命中改善但输出未改善，停止继续调配额，回到 Reader/联合读出的证据使用问题。

## 8. 文件与局限

- `native_pool.py` / `replay_pool.py` / `replay_captured.py`：仅冻结计算的输入捕获、CPU 对照及独立进程重放。
- `feature_audit.py`：固定来源的 A/B/C 审计入口。
- `memory_audit.py` / `cohorts.py`：同一困难样本上的检索与救回统计。
- `report.py`：只读汇总；失败日志只输出有明确标记的部分诊断。
- `quota.py` / `build_banks.py` / `train_quota.py`：显式证据门槛、Train-only 配额对照及匹配训练。

所有结果保存新目录；没有删除、覆盖旧库，没有修改训练好的模型。
诊断的同步、CPU 拷贝、分组归约会增加时间开销，不把这种运行时间作为模型速度比较。
本工具不声称确定了非法内存访问根因，也不承诺配额对照提分。
