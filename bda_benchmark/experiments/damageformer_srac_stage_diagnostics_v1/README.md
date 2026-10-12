# 固定证据的分阶段诊断：F汇总 / Memory困难样本

这不是新训练方法，而是两个独立的只读审计入口。目的是区分：

1. 相同F与Q，均值汇总和分类次序改变了哪些判断？
2. 对J1已经判错的Damaged，历史库是否有候选、是否取回证据、最后是否救回？

本目录不修改旧模型，不重训Q，不重建Memory，不修改旧结果，不使用Test。
`run.sh`没有`train`命令，也不会在审计之后自动启动其他阶段。
只读取你本人生成、可信的PyTorch权重与缓存。

## 1. 文件与范围

| 文件 | 用途 |
|---|---|
| `feature_core.py` | 256维F的精确归一化均值、同Q回投、A/B/C路径、线性与常数测试 |
| `feature_audit.py` | 加载固定快照，完整Val与错误转移、定位/边界/纯度分组 |
| `memory_core.py` | 固定J1困难集合、支持标签证据、检索与干预统计 |
| `memory_audit.py` | 加载已完成且匹配的J1/J2/J3、读取Train缓存、完整Val审计 |
| `report.py` | 仅读取已有JSON；不用GPU、不重新推理 |
| `test_feature.py`、`test_memory.py`、`test_entry.py` | 合成张量及入口合约，不替代真实BRIGHT/CUDA检查 |

这次未实现P/R两头重新训练。只有只读结果指向分类器适配问题，才应另开该训练对照；
不要把原头对汇总特征不适应，直接写成“信息永久丢失”。

## 2. 特征审计：每条路径在验证什么

每张图只调用一次父模型，捕获原A1的`fused_roi`特征F、建筑概率B、原生Q及区域ID。
不重新生成另一份Q替换它，不向分区器输入GT。

典型形状为F `[1,256,h,w]`、Q `[1,9,h*w]`、ID `[9,h*w]`、区域特征 `[1,K,256]`。
实际h/w/K以快照运行结果为准，不能把stride8解释成原图8像素。
原A1头是`base.clf_2`的256→4线性1×1卷积，不是父模型的128维新区域头。

| 输出名 | 路径 | 与谁比较、回答什么 |
|---|---|---|
| `a1` | 原生A1：F→旧头→插值logits→Softmax | 复现原A1，不能用不同快照的旧分数代替 |
| `pixel_grid`（A） | F→旧头→网格Softmax→插值概率 | A/B/C共同的像素对照；与a1比较插值/Softmax次序影响 |
| `feature_mean`（B） | F→按Q均值汇总→同Q回投F→旧头→网格Softmax→插值概率 | A对B：固定旧头下，均值重建影响什么 |
| `region_mean`（C） | F→按Q均值汇总→旧头→区域Softmax→同Q回投概率→插值概率 | B对C：Softmax与回投的次序影响 |
| `grid_probq` | F→旧头→网格Softmax→均值汇总概率→同Q回投 | 在统一网格上对照“概率平滑” |
| `a1_resize` | 原生A1概率→area降到Q网格→插值概率 | 对应你此前probq的分辨率对照 |
| `native_probq` | 上一行的网格概率→原有概率汇总/回投 | 对接既有probq审计；不替代B/C |
| `region` | 原生adapter＋注意力/统计/几何Embedding＋区域头 | 原生区域对照；C与它同时存在多处差异 |

**C优于实际region，只能把嫌疑缩到完整表示/分类路径，不能单独证明区域头有错。**
Q的形状、权重和ID不变；A/B/C统一在特征网格完成非线性后再插值概率。
原生A1的插值logits路径单列，避免悄悄改变基线。

### 均值与回投定义

设ID把每个位置的9个候选槽映射到全局区域k；同一个全局ID在候选中重复时累加权重：

```text
Q_global[i,k] = 所有ID指向k的候选槽权重之和
mass[k]      = Σ_i Q_global[i,k]
T[k]         = Σ_i Q_global[i,k] F[i] / mass[k]
R_Q(F)[i]    = Σ_k Q_global[i,k] T[k]
```

正质量区域使用真实质量做分母；质量恰为0的Token置0。
不把极小正质量偷偷截到1e-6，另外报告`tiny_positive_regions`。
这是数学均值诊断，**不是宣称与父模型的注意力/epsilon汇总相同**。
常数保持和`H(R_Q F) = R_Q(H(F))`只在logits线性层面检查，不要求Softmax可交换。

### 记录哪些错误

- 四类IoU、Precision、Recall、F1及完整混淆矩阵。
- 配对的`GT × 旧预测 × 新预测`，逐类伤害/救回、TP变化和FP变化。
- B判断正确、漏检、误检；B置信度分箱。
- GT四类变化边界及两侧、内部。该边界不只是建筑外轮廓。
- GT前景8连通域大小代理，不称为真实建筑实例。
- 事件、逐图配对计数；GT区域纯度桶及按GT类别统计的混合程度。
- GT-Damaged面积加权的特征网格`z_D-z_I`变化；不是楼内局部损毁定位。

高纯度桶包含背景，不能只看它的总数；必须同时看Damaged行与对应错误转移。
纯度分桶是关联证据，不是因果证明，也不是前景mIoU上限。

### 可选Embedding敏感性诊断

设置`DIAG_EMBEDDING_INTERVENTIONS=1`才额外运行四条：

- `no_structure`：保留adapter的128维content_only，去掉结构残差。
- `uniform_attention`：注意力score设0，经sigmoid为0.5，不将总权重清零。
- `no_stats`：仅将mean/std旁路输出置0。
- `no_geometry`：仅将几何嵌入输出置0。

Q、分类头及其他组件不变。临时hook用完恢复。
这检查已训练模型的依赖，不能代替重新训练的消融；默认先不启用。

## 3. Memory审计：从“没有证据”到“不会用证据”

必须加载同一已完成训练目录的`j1_last.pth`、`j2_last.pth`、`j3_last.pth`，统一5000步。
不能混用三个不同best步数，不能把当前特征审计快照强行配到另一轮Memory。

支持两种已经存在的协议：

- `SRAC_JOINT123_V1`：原已完成父流程的J1/J2/J3。
- `SRAC_SNAPSHOT_JOINT123_V1`：显式快照协议的J1/J2/J3。

快照协议不会被冒充为父实验完成。新脚本仍使用各协议原有加载器、SHA256、数据划分与源代码检查。
前端、Memory和metric路径默认从该J训练的manifest读取；`SNAPSHOT_CKPT`不用于Memory入口。

### M1：库里有没有、筛选后还有没有

正式审计默认读取全部2918个Train缓存分片，无需重新提特征：

```text
全部有效Train区域 → 纯度筛选后的区域 → 最终bank → 排除同事件/同图像后的候选
```

报告多数类别、四类软标签总量、非多数Damaged含量、按事件/纯度/区域质量分布，
并核对bank Token/软标签与声明的源分片一致。
Q质量以特征网格单位计，不是实际建筑面积；软标签比例之和也不是像素数或建筑数。
即使候选很多，也不能据此证明它们与困难查询相关。

`DIAG_INVENTORY_SHARDS=0`可仅审计bank与推理，但会明确标记未检查筛选前覆盖。
两图smoke默认不扫描Train分片，完整Memory审计默认扫描。

### M2：困难Damaged有没有取到有用的标签证据

困难集合固定为`GT=Damaged 且 J1预测不等于Damaged`。
只在检索和最终预测完成以后使用Val GT划分该集合。

J2按原余弦检索，J3按原metric检索；不改变Top-K、排除规则、Reader或已训练权重。
另外计算一个**辅助诊断**：

```text
α = softmax(选中邻居相似度 / 固定温度)
区域标签证据 = Σ α × 邻居四类软标签
```

温度默认0.1，记录在manifest中，不用Val扫温度。
该证据不是Reader实际attention，也不是校准后的概率承诺。
按同Q回投证据及“有支持质量”，无支持保持显式缺失（预测标记-1），不伪装为背景或均匀概率。
位置只有0.1的Q质量连接到有效支持时，不能因为有非零支持就声称已充分覆盖。

困难集合中报告：

1. 有任意支持的位置比例，以及平均支持Q质量。
2. 有支持时，标签证据判断Damaged的位置数。
3. 最终J2/J3救回数。
4. 证据已判断Damaged而最终仍没救回的数量，可由前两项联合计数相减得到。

同时检查J1原本正确的Damaged、Intact、Destroyed；全四类转移含背景。
另报每事件、各查询多数类Hit@K、同类槽位比例原始计数，以及包含在非多数Damaged查询中的GT-Damaged软质量。
这里Hit@K是“至少一个同多数类别邻居”，**不是检索Recall，也不能与像素mIoU比较**。

### M3：Reader是否真的依赖这些历史支持

每个已训练J2/J3各做四种推理：

| 条件 | 改变什么 |
|---|---|
| 正常 | 原选择器、原支持 |
| `random` | 相同有效槽位数及排除规则，每个query内不放回抽随机候选 |
| `shuffle_labels` | Token/身份/元信息/有效位不变，仅打乱有效支持标签 |
| `off` | 将历史支持有效位置false，调用该Reader原有learned fallback |

`off`不是重新训练的J1，也不是随意输入全零特征。
随机干预不保持原相似度或类别分布，故仅用于检查相似检索的依赖，不宣称纯粹的训练因果效应。
完整报告既比较正常J2/J3与J1，也比较**同一个Reader**正常与干预后的变化。

Memory入口保存每事件/每图/四类转移和困难集合；它本次不额外计算ROI/边界/大小分组。
后者在特征入口中提供，避免把两类审计的输出范围混为一谈。

## 4. 数值与结果保护

- 每次使用新输出目录，存在则退出；不要提前`mkdir DIAG_OUTPUT`。
- 模型处于eval且冻结；入口禁梯度，没有optimizer。
- 严格确定性、FP32，TF32关闭；CUBLAS配置必须在新的Python进程初始化CUDA之前设置。
- 原生预测重算、Q归一化、合法ID、常数/仿射合约、概率合法性均检查。
- F/Q/ID/Token等捕获证据、参数及buffer比较；输入权重、数据划分、相关代码版本记录并检查。
- 特征完整Val检查A1与region对应快照记录；Memory完整Val检查原A1、region及J1/2/3对应末步记录。
- 重算数值测试使用明示`atol=2e-5, rtol=2e-5`；完整指标复现使用原有固定1e-5门槛。二者不同，不按失败结果放宽。
- 两图smoke或显式子集即使通过，也不会标记为完整Val证据。
- 失败保留JSON/逐图记录并退出；不跳过失败图片，不用CPU或随机重跑掩盖失败。

这不是对历史非法访问/非确定性故障的修复声明。
若新进程仍出现illegal memory access，结束该失败进程后再定位；不能在损坏CUDA上下文里继续比较分数。
运行期间不要同步修改相关Python文件、替换输入权重或继续写同一J训练目录。

## 5. 服务器命令：先特征审计

先把**整个新目录**同步到服务器对应的`bda_benchmark/experiments/`下。
不要只传run.sh，不要覆盖旧实验包。依赖当前已有的cause/selective/partition-probe/J123代码。

### 5.1 本地合约与真实数据两图smoke

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export SNAPSHOT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_CAUSE_AUDIT_V1/diag_20261009_100438/snapshot/front_snapshot.pth
export DEVICE=cuda:0
export WORKERS=4
export SEED=3407
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1
export DIAG_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_STAGE_DIAGNOSTICS_V1/run_$(date +%Y%m%d_%H%M%S)_$$"
export DIAG_EMBEDDING_INTERVENTIONS=0
export DIAG_START_INDEX=0
export DIAG_MAX_IMAGES=0

bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh preflight

export DIAG_OUTPUT="$DIAG_ROOT/smoke_feature"
bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh smoke_feature
bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh report
```

### 5.2 smoke通过后完整Val450（不训练）

```bash
export DIAG_OUTPUT="$DIAG_ROOT/feature"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_STAGE_DIAGNOSTICS_V1
export DIAG_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_STAGE_DIAGNOSTICS_V1/$(basename "$DIAG_ROOT")_feature.log"

nohup bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh feature > "$DIAG_LOG" 2>&1 &
echo "特征审计PID=$!"
echo "结果=$DIAG_OUTPUT"
echo "日志=$DIAG_LOG"
tail -F "$DIAG_LOG"
```

完成后Ctrl+C退出tail，仅停止查看日志；然后：

```bash
bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh report
```

先看`summary.json`：完整审计应`complete=true`、`gates_passed=true`、`full_validation_evidence=true`。
有任何false，先看failure/reproduction，不解读干预涨跌。
若需可选Embedding检查，换新DIAG_OUTPUT后设置`DIAG_EMBEDDING_INTERVENTIONS=1`，重复smoke→完整；不要覆盖本次结果。

## 6. 服务器命令：再Memory审计

### 6.1 确认你实际完成的J1/J2/J3目录

```bash
find /root/shared-nvme/BRIGHT-checkpoints -type f -name j1_last.pth -print
```

使用包含同轮三个last权重、manifest、summary、validation.jsonl的目录。
**这里不猜测你的最新J目录。** 在同一shell用下面命令输入实际路径：

```bash
read -r -p '请输入已完成J1/J2/J3的结果目录：' DIAG_JOINT_RUN
export DIAG_JOINT_RUN
ls -lh "$DIAG_JOINT_RUN/j1_last.pth" "$DIAG_JOINT_RUN/j2_last.pth" "$DIAG_JOINT_RUN/j3_last.pth"

export DIAG_INTERVENTION_ARMS='j2 j3'
export DIAG_MEMORY_TEMPERATURE=0.1
export DIAG_INVENTORY_SHARDS=0
export DIAG_OUTPUT="$DIAG_ROOT/smoke_memory"
bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh smoke_memory
bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh report
```

上面沿用第5节设置的PYTHON/A1/数据划分/设备。Memory从J manifest读取自己的前端，不要求等于第5节快照。
如果原缓存位置搬迁，可指定`DIAG_MEMORY_FRONT`、`DIAG_MEMORY_CACHE`、`DIAG_MEMORY_METRIC`；
仅允许字节哈希完全相同的文件/缓存，不允许改用相近分数的权重。
如果文件已经丢失，先找备份，不能伪造manifest或绕过检查。

### 6.2 完整Memory审计

```bash
export DIAG_OUTPUT="$DIAG_ROOT/memory"
export DIAG_INVENTORY_SHARDS=1
export DIAG_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_STAGE_DIAGNOSTICS_V1/$(basename "$DIAG_ROOT")_memory.log"

nohup bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh memory > "$DIAG_LOG" 2>&1 &
echo "Memory审计PID=$!"
echo "结果=$DIAG_OUTPUT"
echo "日志=$DIAG_LOG"
tail -F "$DIAG_LOG"
```

此入口需全Train缓存扫描和450张多分支推理，时间比普通单模型验证长。
建议特征审计结束后再启动，不要同时开几份争用4090。
完成后用同一report命令读取结果。

## 7. 拿到结果后如何决定下一步

| 观察 | 可以支持的方向 | 不能直接下的结论 |
|---|---|---|
| A→B显著增加Damaged→Intact伤害 | 原头决策对F均值压缩敏感；再做相同P/R头适配对照 | “超像素生成彻底失败” |
| B正常、C下降 | 分类非线性与概率回投值得优先检查 | “区域Token一定没有信息” |
| B/C正常、原生region差 | adapter/完整Embedding/新分类路径是重点 | “已唯一定位为分类头” |
| Train有D，但筛选后/事件排除后很少 | 候选供给有损失，调整库构建值得单独验证 | “增加bank必然涨分” |
| 候选有D，困难查询Top-K证据仍偏I | 相关性/检索判别应优先检查 | “Reader一定有问题” |
| 证据支持D，最终仍不救回 | Reader/联合分类值得检查 | “标签证据等于Reader真实注意力” |
| 正常与随机/打乱相近 | 尚未显示相似支持或标签的额外作用 | “所有Memory都无效” |
| D救回增加但I/X被改成D更多 | 检查类别偏置与使用强度，并报告FP变化 | “D召回涨就代表整体改善” |

保存的逐图/事件配对计数可用于后续图像或事件级bootstrap。本脚本没有自动做多种子显著性结论，
更不能把数百万像素当成独立重复。训练种子波动与验证抽样不确定性是两件事。

最先提交分析的文件：特征`summary.json`、`audit.json`；Memory还加`inventory.json`。
失败时先提供`summary.json`与日志末尾，而不是继续训练或放宽门槛。

## 8. 本地验证记录

2026-10-10，在`D:/anaconda3/envs/bright/python.exe`、PyTorch `2.12.1+cpu`下：

- 31项合成/入口测试通过。
- 通过Git Bash实际执行`run.sh preflight`，三个CLI帮助入口全部通过。
- `bash -n run.sh`通过。

没有服务器权重、BRIGHT图片或CUDA可用，因此未执行真实数据smoke、完整Val或任何训练。
目标服务器PyTorch 2.2.1+cu121仍须按第5、6节验证；本地通过不能替代服务器结果。

## 9. 2026-10-10：feature均值汇总CUDA异常的隔离修订

服务器在 `mean_pool` 的三维 `scatter_add_` 报非法内存访问。这能定位到报错路径，
但不能单凭异步CUDA回溯认定是该内核、零步长索引、某张图的Q或显存不足。
已有ID范围/形状/Q归一化检查保留；没有发现就不能假设ID越界。

本修订仅改变**新增审计的数学均值算子**：

```text
原实现：像素贡献 → scatter_add_累加到区域
新实现：按区域整理所有(candidate,pixel)整数索引 → gather贡献 → 分块sum

token[k] = sum_i Q[i,k]*F[i] / sum_i Q[i,k]
```

重复候选ID全部计入，微小正质量不截断，真正空区域输出零。CPU只做整数索引规划；
F/Q加权和归约留在原设备，不是失败后自动切CPU。工作块有大小上限，不生成密集N×K矩阵。
浮点求和顺序改变，不能承诺逐位等同于原scatter；原有仿射交换、常量重建和复现容差均不放宽。

这**没有修改**A1、Soft ROI、Q生成、原Embedding、Memory或任何训练文件。
父模型及旧概率诊断内仍有原算子，所以不能宣称所有CUDA错误已经根除。

新增故障记录：

- `progress.json`：图片、Val位置、当前阶段、上一个同步完成阶段。
- 在进入新均值路径前建立独立CPU输入副本，只有失败时写 `failed_pool_inputs.pt`。
  包含F/Q/区域ID/A1仿射头，不含GT；典型256通道、256×256网格约70MiB。
  每图只暂存一份，会增加CPU搬运和审计耗时，不应用这轮耗时代表原模型推理速度。
- 失败时保留 `complete=false`，不在已出错CUDA进程内重试，也不绕过失败图。
- `recheck_feature` 从旧失败记录定位图片，不从“60/450”猜测是第61张。
  校验同一权重、snapshot元信息和Train/Val列表哈希，再在新目录运行。

### 9.1 同步代码后：合约测试 → 原失败图片

先把本地这个实验目录的最新文件同步到服务器同名目录；不要仅更新 `feature_core.py`，
它还依赖新增 `grouped_pool.py`。无需同步/覆盖旧模型目录或结果目录。
复制下面代码块里的命令，不复制Markdown的三个反引号。

```bash
cd /root/shared-nvme/BRIGHT-master

export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1

export DIAG_FAILED_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_STAGE_DIAGNOSTICS_V1/run_20261010_122955_2934045/feature
export DIAG_RECHECK_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_STAGE_DIAGNOSTICS_V1/poolfix_$(date +%Y%m%d_%H%M%S)_$$"
export DIAG_OUTPUT="$DIAG_RECHECK_ROOT/single"
export DIAG_RECHECK_PREFIX=0

bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh preflight &&
bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh recheck_feature

bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh report
```

服务器有CUDA时，preflight包含真实尺寸的GPU合成均值测试；没有CUDA时会明确skip该项。
它不加载训练权重。随后单图使用原manifest中的device/workers/seed/快照，
不需要重新手工拼接图片路径。单图通过只说明该条件下未复现，不是完整Val证据。

### 9.2 单图通过后：原顺序前缀

在同一个shell运行；不要重用已经存在的输出目录：

```bash
: "${DIAG_RECHECK_ROOT:?请先运行上一段设置}"
export DIAG_RECHECK_PREFIX=1
export DIAG_OUTPUT="$DIAG_RECHECK_ROOT/prefix"

bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh recheck_feature &&
bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh report
```

从原Val第1张连续运行到原失败图片（含它）。这不是续跑，不合并旧失败报告，也不跳过图片。

### 9.3 前缀通过后，才完整450张

```bash
cd /root/shared-nvme/BRIGHT-master
: "${DIAG_RECHECK_ROOT:?请先完成单图和前缀重查}"
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export SNAPSHOT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_CAUSE_AUDIT_V1/diag_20261009_100438/snapshot/front_snapshot.pth
export DIAG_START_INDEX=0
export DIAG_MAX_IMAGES=0
export DIAG_EMBEDDING_INTERVENTIONS=0
export DEVICE=cuda:0
export WORKERS=4
export SEED=3407
export DIAG_OUTPUT="$DIAG_RECHECK_ROOT/full_feature"

mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_STAGE_DIAGNOSTICS_V1
export DIAG_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_STAGE_DIAGNOSTICS_V1/$(basename "$DIAG_RECHECK_ROOT")_full_feature.log"
nohup env CUDA_LAUNCH_BLOCKING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/run.sh feature \
  > "$DIAG_LOG" 2>&1 &
echo "完整审计PID=$!"
echo "结果=$DIAG_OUTPUT"
tail -F "$DIAG_LOG"
```

若再失败，提供新 `summary.json`、`progress.json` 和日志最后60行。
`failed_pool_inputs.pt` 可供后续独立重现，但本修订不自动运行额外试验。
不要改原门槛、删除旧结果或将部分图的分数当完整Val结论。

本修订本地验证：52项测试，51通过、1项CUDA测试因本机 `2.12.1+cpu` 跳过；
另通过Git Bash入口/语法检查。服务器PyTorch 2.2.1+cu121与真实BRIGHT尚待上述验证。
