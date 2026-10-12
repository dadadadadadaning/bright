# 选择性区域读取：先检查，再做单因素训练

这是新的隔离实验，读取 `damageformer_srac_transformer_v1` 的固定快照。不会编辑父模型、接管正在训练的进程、覆盖结果、修改父流程/J1J2J3完成门槛或访问Test。

本轮分成两个**互不自动串联**的任务：

1. `boundary`：固定权重，只更换边界输入，检查结构量的含义与影响。不训练。
2. `train`：同时训练P、Q-read、Content-read三个独立读出，冻结前端、Q、B、adapter和原区域Embedding。所有组保留原边界，不混入边界定义改动，不用Memory。

只有可运行接口不等于方法已有效；本地CPU合约/合成数据检查不能代替服务器真实BRIGHT/CUDA smoke与完整Val450。

## 一、区域信息怎样被选择？

保持原图、DamageFormer、Soft ROI及已有分区器不变：

```text
RGB + SAR -> 冻结A1 + 原Soft ROI -> 像素特征 F -------------------+
                         |                                      |
                         +-> 冻结Q/区域Embedding -> 区域Token T   |
                                                   |            |
                                      相同结构的无Memory Reader |
                                                   |            |
                                               区域上下文 U     |
                                                   |            |
                       Q-read: Q读取 / Content-read: R读取 -> C -+
                                                                |
                                                    联合分类器(F,C)
```

`Reader`是每组独立训练的现有无Memory模块，以可学习支持槽位处理区域Token。没有历史Train库、检索、支持GT输入。

Q继续负责原区域的生成与汇总。新增R只负责把U读取到像素，不反向更改Q：

`R_ik = Q_ik * exp(score(F_i,U_k)) / sum_j Q_ij * exp(score(F_i,U_j))`

其中`score=phi(F_i)·psi(U_k)/sqrt(d)`，默认d=32，只有原先合法的9个候选；Q=0或空区域保持不可访问。数值实现对有效路线的`log(Q)+score`减去最大值后再取指数，不给Q加epsilon来打开非法路线。没有有效候选时上下文为0，保留像素分支。

`phi`零初始化，`psi`普通随机初始化，使最初R等于有效Q，又避免两个投影同时零初始化造成无法学习。联合分类器的像素块复制A1，区域输入列为0，三组第0步都复现A1。因为零初始化的上下文列先学习，Reader/打分器第一步梯度可能为0；需看后续梯度、`route_l1_mean`及结果，不能据第一步宣称分支失效。

| 组别 | 实际训练 | 要回答的问题 |
|---|---|---|
| `p` | A1初始化的新像素分类头 | 仅继续训练分类头能得到多少改善？ |
| `q_read` | 同结构无Memory Reader＋联合分类头 | 现有区域上下文是否有增量？ |
| `content_read` | 同结构Reader＋联合分类头＋像素/区域打分器 | 根据RGB–SAR内容选择区域信息是否更好？ |

三组初始权重相同但参数存储独立；冻结前端每个裁剪只运行一次，三组看到同一F/Q/B/T和GT，同一优化器步数和有效batch。P的Reader/上下文列不训练，Q-read的打分器不训练，因此**不能声称有效容量完全相同**。manifest分别记录分配参数数和可训练参数数，若Content-read有效，还需等容量像素分支等对照。

本轮是新的同预算读出对照，**不是继续训练某个既有J1权重**。三组用512训练裁剪、有效batch16；不能把它和采用完整图像/缓存的旧J1/J2/J3日志直接当成严格配对结果。

冻结已有T时，新R只能在现有候选Token中挑选；不能找回T已经抹去的信息，也不能访问9候选之外的位置，所以始终保留F直达分类器。

张量：`F[B,256,h,w]`，`Q/R[B,9,h*w]`，`T/U[B,K,128]`（维度从父配置读取），`C[B,128,h,w]`。像素分块读取，默认4096点，不建立`N×K`全局注意力；反向传播仍会保留必要的中间量。原分区、Soft ROI、分辨率和全局种子ID不变。

## 二、边界检查具体改变什么？

原边界量的单条邻接边公式为`E_old=1-dot(global_Q_i,global_Q_j)`。它是独立抽取种子的不一致概率，既受空间变化影响，也受Q自身不确定性影响；相同的9候选均匀分布也会得到8/9。

新检查量`E_change=0.5*||global_Q_i-global_Q_j||²`仅比较归属分布变化。实现按全局种子ID对齐，并正确合并重复ID，再沿用原先相邻边最大值汇总。**这仍不是建筑GT边界或真实实例边界。**

| 输出 | 改变 | 目的 |
|---|---|---|
| `a1` | 原A1输出 | 固定参考 |
| `region` | 原区域输出；另验证原边界重算能复现 | 基线复现 |
| `edge_zero` | 边界输入置零 | 原边界通道是否在影响预测？ |
| `edge_mean` | 每张图用其原边界均值填充 | 去掉空间变化、保留图像级均值的干预 |
| `edge_change` | 输入归属分布变化量 | 边界/不确定性混合是否值得进一步训练验证？ |

固定权重、F/Q/B/H，但**重新计算adapter、区域Embedding、分类、回投**。若连Token也固定，边界变化不会到达输出。其他分支和GT都不参与预测干预。

报告边界数值分布及GT类别/边界条件统计。`edge_mean`不是完整幅度匹配：change还改变了分布与尺度。如果有改善只能作为线索，不能直接宣布原定义造成全部下降。本轮**未实现/未启动E0/E1/E2训练**；读出训练也不会自动采用change。

## 三、上传、环境及固定快照

上传整个本目录到服务器相同相对路径，不能只上传run.sh。依赖服务器既有BRIGHT项目与旧实验公共模块，不需要新增CUDA扩展。

以下是服务器命令，不是本地Windows命令：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export SNAPSHOT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_CAUSE_AUDIT_V1/diag_20261009_100438/snapshot/front_snapshot.pth
export SELECTIVE_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_SELECTIVE_READOUT_V1/run_$(date +%Y%m%d_%H%M%S)"
export DEVICE=cuda:0
export WORKERS=4
export SEED=3407
unset SELECTIVE_OUTPUT

ls -lh "$A1_CKPT" "$SNAPSHOT_CKPT" "$(dirname "$SNAPSHOT_CKPT")/snapshot.json"
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh preflight
```

上面选择的是已审计的13000步快照路径；如文件不存在，不要填旧V2的761MiB前端。可从当前SRAC_TRANSFORMER_V1父流程另建稳定副本：

```bash
export PARENT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/formal50000_seed3407_20261008_173616
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh snapshot
export SNAPSHOT_CKPT="$SELECTIVE_ROOT/snapshot/front_snapshot.pth"
```

新副本的真实步数以snapshot.json为准，它不一定还是13000步。所有比较固定同一副本。只复制小adapter checkpoint，原A1通过SHA256引用。中途快照允许诊断/新读出适配，但不能冒充完成父流程，不能替代正式J1/J2/J3入口要求。

## 四、先运行边界小检查

```bash
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh smoke_boundary
```

两张Val图片仅检查程序。通过后独立运行完整Val450：

```bash
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_SELECTIVE_READOUT_V1
export BOUNDARY_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_SELECTIVE_READOUT_V1/$(basename "$SELECTIVE_ROOT")_boundary.log"
nohup bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh boundary > "$BOUNDARY_LOG" 2>&1 &
echo "边界检查PID=$!"
tail -F "$BOUNDARY_LOG"
```

审计使用GPU，但不优化参数/BN；不会自动停止其他训练。只有资源允许才并行。Ctrl+C退出tail，不是停止后台任务。

```bash
export SELECTIVE_REPORT="$SELECTIVE_ROOT/boundary"
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh report
```

先看`complete`和复现门槛，然后比较edge_*对region的Damaged伤害/救回。完整统计在`audit.json`。没有明显信号时停止边界支线，不再反复搜索Val上的阈值。

## 五、再运行P/Q-read/Content-read训练

边界结论不会被自动应用到这轮训练；三组都使用原生边界定义。

```bash
export SELECTIVE_BATCH=2
export SELECTIVE_ACCUM=8
export SELECTIVE_AMP=none
export SELECTIVE_STEPS=5000
export SELECTIVE_EVAL_EVERY=500
export SELECTIVE_LR=1e-4

bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh smoke_readout
```

smoke为20步，验证前2张Val，输出`readout_smoke`；训练数据仍只从Train取裁剪，绝不使用Val更新权重。正式训练验证完整Val450，第0步复现A1与原region，再每500步验证。

```bash
export READOUT_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_SELECTIVE_READOUT_V1/$(basename "$SELECTIVE_ROOT")_readout.log"
nohup bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh train > "$READOUT_LOG" 2>&1 &
echo "读出训练PID=$!"
echo "输出目录=$SELECTIVE_ROOT/readout"
tail -F "$READOUT_LOG"
```

三个读出在同一作业内共享冻结前端前向，分别优化；每组5000次更新、80000训练裁剪。不是三组平分5000步，也不是三份Swin参与反向传播。AdamW、weight decay=5e-3、相同warmup+cosine、NLL+Lovasz，不加类别重加权/边界损失/Memory。

默认FP32以匹配复现与诊断。若使用`SELECTIVE_AMP=bf16`，应是新目录的独立设置，源前端仍FP32。已有输出拒绝覆盖，不要删除原结果；可用`SELECTIVE_OUTPUT`指定该任务新的空路径，完成后注意`unset SELECTIVE_OUTPUT`避免后续命令误用。

```bash
export SELECTIVE_REPORT="$SELECTIVE_ROOT/readout"
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh report
```

## 六、如何解释结果以及存多少？

- `q_read - p`：区域上下文相对同预算像素头的增量。
- `content_read - q_read`：内容选择相对固定Q读取的增量；仍包含新增打分器容量。
- 若R明显变化而分数不改善，不要把注意力图变化当效果；若R始终等于Q，检查后续打分器梯度及上下文是否真正被使用。
- 同时看Damaged P/R/F1、Damaged→Intact与反向救回、Intact/Destroyed、定位错误、前景/背景侧边界、内部、小对象代理、每事件。GT连通域不是建筑实例；条件分组指标不等于全图mIoU，背景高纯度不能作为成功依据。
- 日志保存实际读取变化、熵（这里是未除log9的nats，不能直接与父日志归一化熵混比）、非法读取质量和空上下文比例。耗时/显存为整个配对作业，不是单模型部署速度。
- `validation.jsonl`存每次验证总指标和事件指标；最终`audit_final.json`包含配对和条件错误、每图混淆矩阵，支持之后图像/事件级配对重采样。不把像素当独立重复，也不把一个种子的10次验证当10个种子。
- 只保存三个读出的`best_*.pth`和`last_*.pth`共6份小权重；不复制761MiB A1，不保存原图预测、逐步大权重或Memory库。边界审计额外保留前2张Val图像的小尺寸特征网格E/B和GT面积比例，用于离线查看。具体大小取决于Token维度，manifest记录参数数。
- 这些权重用于分析/推理，不含优化器和随机数状态，**不是精确断点续训文件**；完成标记只表示本隔离实验完成，不能作为正式父流程完成材料。
- 任何新的Embedding训练成功后，若接Memory需重建匹配的Train库；本轮Q和Embedding始终冻结，未实施该步骤。

机制参考：OCR（ECCV2020，像素—区域内容关系）、SPFormer（像素/区域交互）、现有J模型像素证据保留。此实现是任务适配假设，不是严格复现，也不宣称通用注意力或L2边界公式本身原创。

## 七、`Boundary intervention is not normalized: edge_zero` 定点排查

这表示某张图的`edge_zero`四类输出之和未通过原有检查，不是指“边界全零不能使用”，也不是训练权重损坏。原回投函数本身已经除以类别总和，因此在没有偏差数值和运行时信息前，**不能直接断言是普通浮点误差，更不能删除检查或放宽阈值**。

新增版本保留原预测代码及`atol=rtol=2e-5`；不重新归一化观测输出、不跳过图片，也不使用CPU结果替换GPU结果。失败时保存：

- `failure.json`：具体图像、Val序号、dtype、范围、类别总和偏差、异常位置、分阶段回投与CPU对照诊断；
- `runtime.json`：实际PyTorch/CUDA版本、回投函数路径/源码/哈希；
- `partial_audit.json`：本次已完成图像的统计，明确`complete=false`，不能当完整Val结果。

旧版本在330后报错时，最后输出的`330/450`仅是进度，故先检查第331—340张（零基索引330）。先同步整个新目录、重新preflight，然后：

```bash
export BOUNDARY_START_INDEX=330
export BOUNDARY_MAX_IMAGES=10
export SELECTIVE_OUTPUT="$SELECTIVE_ROOT/boundary_probe_$(date +%Y%m%d_%H%M%S)"

bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh boundary_probe

export SELECTIVE_REPORT="$SELECTIVE_OUTPUT"
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh report
```

`boundary_probe`只读取选中的Val图，不是断点续算。`audit_scope.full_validation=false`，即使该探针`complete=true`，也不代表完整450张审计通过。范围探针不适用完整Val复现门槛，但每图的原输出重算、Q/特征不变及概率检查仍保留。

如果失败，先查看/提供`failure.json`，根据实际阶段定位错误；不要开始新的正式训练来“修复”这类审计异常。探针通过后才能另起全量复查；若问题只在长时间运行后出现，仍需全量运行定位，探针不能排除它：

```bash
export SELECTIVE_OUTPUT="$SELECTIVE_ROOT/boundary_retry_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh boundary

export SELECTIVE_REPORT="$SELECTIVE_OUTPUT"
bash bda_benchmark/experiments/damageformer_srac_selective_readout_v1/run.sh report
unset SELECTIVE_OUTPUT
```

不覆盖旧失败目录，也不拼接旧目录前330张统计来冒充续算。此补丁增强审计可定位性，**在服务器新诊断返回前，不宣称已确认或修复GPU归一化异常根因**。
