# Partition probe V1：建筑约束、迭代与边界的隔离诊断

这是**固定权重的测试包，不是训练包**。新增目录，不改旧模型、旧实验、已有 checkpoint 或 Memory；不访问 Test。需要服务器现有 BRIGHT 工程，以及可信的 `front_snapshot.pth` 和同目录 `snapshot.json`。

## 1. 三个问题分开验证

| suite / 组别 | 只改变什么 | 验证的问题 |
|---|---|---|
| 每组都有 native | 原 checkpoint 的 stride8、3轮、9候选、建筑权重0.5等原配置 | 新实现能否复现原分区和原区域预测？ |
| prior / b_strong | 原先验公式，权重默认改为2.0 | 单纯增强建筑引导是否有效、有何定位错误代价？ |
| prior / b_bounded | 权重同为2.0，加入置信度调节和有限相容性下限 | 较强引导、同时保留跨组通路是否比单纯增强好？ |
| iterations / iter1、native、iter5 | 1/3/5轮，均9候选、相同种子数 | 三轮是否尚未稳定？增加轮数是否有用？ |
| iterations / radius2_iter3 | 3轮不变，9候选改为25候选，种子数不变 | 受限的是更新次数，还是候选支持范围？ |
| boundary / edge_zero、edge_mean、edge_change | 固定Q/B/F/H/权重，仅改变结构适配器的边界输入 | 当前边界的空间变化信息是否真的有用？ |
| scale / stride4（可选） | 同分辨率、9候选、3轮，种子间距8改4 | 小建筑损失是否主要来自区域粒度？**区域槽位约4倍，不是同区域预算。** |

`all`包含以上所有组；推荐先用`smoke_all`检查接口，完整审计按suite串行执行。全部都是推理干预，不能代替分别训练后的公平消融，也不是可部署的新训练checkpoint。

当前父实验必须是feature stride8、3轮。特征分辨率约为原图1/4；stride8种子间距约32原图像素。迭代次数、候选半径、种子间距是三个不同变量。

## 2. “介于软与硬之间”的建筑约束

预测建筑概率为 `b_i`，区域中心概率为 `b_k`：

```text
c = b_i*b_k + (1-b_i)*(1-b_k)
confidence = abs(2*b_i-1) * abs(2*b_k-1)
原先验： strength * log(max(c, 1e-4))
有限先验： strength * confidence * log(leakage + (1-leakage)*c)
新归属：softmax(原特征相似度/temperature - 原空间代价 + 先验)
```

默认 `strength=2.0, leakage=0.1`，只是预先指定的诊断值，不是已验证最优参数。建筑判断明确时加强相容性；不确定时放松；相容性先验保持有限，不因建筑分组将候选硬置零。`leakage`是**相容性函数的下限，不是跨建筑概率必须达到10%**。最终跨组质量仍取决于特征和空间项。极端总分差异也可能出现浮点softmax下溢，不能保证每条有效路线有数值下限。

高置信度的定位错误仍可能受伤害；须查看`roi_false_negative`、`roi_false_positive`和B分箱，而不是只看平均mIoU。这是软偏好，不是建筑实例隔离，不强制某组输出某类，不裁剪SAR光学轮廓外的内容，保留原Soft ROI。

`b_bounded`同时更改了先验的置信度调节和截断形状，属于一个组合候选。如果有效，后续应再拆开这两项；本轮不把收益归给其中任一项。

## 3. 迭代问题怎么检查

逐图保存每轮：`q_l1_previous`（归属改变量）、`center_shift`（以种子间距为单位）、软/硬占用、空区域、小质量区域、概率行和误差、实际可用候选数和归一化熵。

逐轮trace的熵除以“空间有效且上一轮中心非空”的候选数对数；汇总表及按类别统计统一除以当前`Q>0`的候选数对数，另记空间有效候选数量。两种计数在softmax下溢时可能不同。原适配器H输入不改。native逐轮trace来自兼容性重放，而所有native预测/边界干预严格使用第一次捕获的Q，避免将GPU重算误差混入边界对照。

- 3到5轮变化已很小：不支持“继续堆轮数”是主要答案。
- 5轮不如3轮：原头可能只适应3轮，不能证明训练5轮也无效。
- 25候选改善：支持扩大局部支持值得进一步训练，但扩大支持也可能增加背景混合；它并不自动更好。
- stride4改善：可能只是区域更细、预算更大，不能宣称分区算法更优。

旧`grouped_centers`和`soft_pool_unpool`都写死9候选。本包实现通用候选数版本，并测试9候选完全复现、25候选全部参与质量汇总；没有直接修改旧工具或用全局monkeypatch。

## 4. 边界和不确定性分开

将两个位置的稀疏归属按**全局seed ID**对齐：

```text
old(i,j) = 1 - dot(Q_i, Q_j)
change(i,j) = 0.5 * ||Q_i-Q_j||^2
gini(i) = 1 - ||Q_i||^2
old(i,j) = change(i,j) + 0.5*(gini(i)+gini(j))
```

相同的 `[0.5,0.5]`：old=0.5，但change=0。不同的确定one-hot：两者都是1。这是**区域归属的变化边界**，不是已经证明对应真实建筑轮廓。通过相邻方向取max后，上面的逐对分解不能被误写成`max(old)=max(change)+max(uncertainty)`。

仅替换E输入，不更改原H熵输入，也不重新生成Q。零/均值控制用于检查是否依赖空间分布或整体输入幅度。新的公式不能修复旧审计的数值/状态异常；原检查失败结果仍然失败。

## 5. 干预边界与输出

**Q相关组：** 固定原A1内容F、B、原结构适配器输出（因此原E/H的影响固定），用干预Q重做区域汇总、几何、分类、同Q回投。这样只研究分区路由；没有同时换结构调制输入。它不是“新Q+所有模块重新适应”的完整新模型。

**边界组：** 固定原Q，用替换E重算适配器→Embedding→原分类头→回投。固定H和原图特征。

此外，每个Q组都有`*_probq`：**A1概率缩小到同一特征图→按Q汇总→同Q回投**。`a1_resize`是只缩小/放大的控制，用于分离分辨率损失与区域压缩损失。概率Token不会再次softmax。

结果文件：

- `manifest.json`：源快照/代码/数据哈希、干预参数、计算环境、选取的Val列表。
- `summary.json`：完成状态、复现检查、模型不变检查、峰值显存/总耗时。
- `audit.json`：完整mIoU、每类P/R/F1、逐事件/逐图、Damaged→Intact伤害与反向救回、ROI定位错误/B分箱、边界/内部、小对象代理、Q实际占用和按GT类别加权纯度。
- `per_image.jsonl`：每完成一张图即写入；逐轮收敛、逐图概率检查、Q统计和分组时间（包含诊断开销，不是纯模型推理基准）。
- 失败时`summary.json`保持`complete=false`，并保留部分证据/错误图片/错误阶段；固定张量变化时记录CPU字节比较与变化坐标。没有跳图、静默归一化修复或放宽原容差。

GT仅在得到预测后用于统计；不参与分区/先验/推理预测。GT前景连通域只是对象大小代理，不是真实建筑实例。纯度须分前景类别读，不能以全图背景高纯度代替效果。没有把GT重构/ASA当成前景mIoU上界。

本包不复制A1/快照，不保存新权重或逐图特征，不建Memory；主要保存JSON证据。

### 5.1 数值重复性故障：先检查执行条件，不修改分区公式

曾出现`Native Q replay failed`，后来单图检查发现**原生分区自己重复也会产生数值差异**，而该次检查的输入CPU字节、模型参数和buffer不变。这支持优先排查数值执行的重复性，但尚不能把根因唯一归于某个scatter算子，也不能认定这些差异对最终类别无影响。

本包现对实际审计和单图重复性检查统一使用严格确定性运行条件：

- `torch.use_deterministic_algorithms(True, warn_only=False)`；
- `torch.backends.cudnn.deterministic=True`，`torch.backends.cudnn.benchmark=False`；
- `run.sh`在启动Python前设置`CUBLAS_WORKSPACE_CONFIG`，未设置时默认`:4096:8`；
- 不更改TF32、AMP、模型结构、checkpoint、分类公式或检查容差；
- 不支持确定性执行的算子直接报错，**不切成warning，不自动退回非确定性实现**。

固定随机种子并不等于固定CUDA归约执行顺序。`load_fixed`中的默认`seed_everything`不会把上述确定性配置重置掉。配置只作用于本次新进程，不会修改已有训练进程或历史checkpoint；确定性计算也可能更慢。

新增`repeatability`模式：对指定Val图片仅提取一次固定输入，然后重复原生与重放分区，比较原生自身、重放自身、原生与重放，以及最初capture的原生结果。默认检查`ukraine-conflict_00000089`，默认10次，同时保留输入与模型不变检查。**原生与重放的Q最大绝对差门槛仍为`2e-5`，不靠放宽容差过关。**此外，严格确定性诊断要求同一条路径重复结果逐元素完全相同，单独报告并检查`self_repeat_exact`；这不是把微小但仍存在的自重复漂移当作成功。

单图模式保存`manifest.json`、`summary.json`和逐次记录`repeats.jsonl`。摘要包括`mode=repeatability`、`complete`、`gates_passed`、`self_repeat_exact`、`weights_buffers_unchanged`、`repeats_completed`、`checks_max_abs`和执行配置。它的`complete=true`仅表示单图重复性流程完成，**不是450张验证完成**。

单图通过只说明该图在当前执行条件下通过重复性检查；不代表完整Val通过，也不能将过去失败的`summary.json`改成成功。任一次重复失败就应保留全部结果排查，不能仅选出某次通过的结果。

## 6. 上传和运行（服务器）

将更新后的整个`damageformer_srac_partition_probe_v1`上传到服务器同名`bda_benchmark/experiments/`目录，包括新的数值运行控制和测试文件；不能只上传run.sh。依赖服务器现有工程和旧诊断公共工具。

先配置路径并运行本地契约测试：

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export SNAPSHOT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_CAUSE_AUDIT_V1/diag_20261009_100438/snapshot/front_snapshot.pth
export DEVICE=cuda:0
export WORKERS=4
export SEED=3407
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PROBE_IMAGE=ukraine-conflict_00000089
export PROBE_REPEATS=10

# 防止前一次子集诊断留下环境变量，导致本次不是完整Val。
unset PROBE_START_INDEX PROBE_MAX_IMAGES

ls -lh "$A1_CKPT" "$SNAPSHOT_CKPT" "$(dirname "$SNAPSHOT_CKPT")/snapshot.json"

bash bda_benchmark/experiments/damageformer_srac_partition_probe_v1/run.sh preflight
```

这是已有稳定快照路径；文件不存在就先定位原快照及其manifest，不要用A1本身或live best.pth替代，也不要伪造snapshot.json。这里不用`conda activate`。

### 6.1 先只验证失败图片

本地契约测试通过后，先运行单图重复性检查。它不训练、不建库，不将一张图当作完整Val：

```bash
export PROBE_RUN_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_PARTITION_PROBE_V1/deterministic_$(date +%Y%m%d_%H%M%S)"
export PROBE_OUTPUT="$PROBE_RUN_ROOT/repeatability"
bash bda_benchmark/experiments/damageformer_srac_partition_probe_v1/run.sh repeatability
bash bda_benchmark/experiments/damageformer_srac_partition_probe_v1/run.sh report
```

需要换另一张Val图片时设置`PROBE_IMAGE`，并为`PROBE_OUTPUT`换一个新目录。已经存在的目录不会被覆盖。若有不支持确定性的报错，先保留完整报错，不继续完整审计，也不设置`warn_only=True`绕过。

重复性检查通过后，再做两张图的全接口smoke：

```bash
export PROBE_OUTPUT="$PROBE_RUN_ROOT/smoke_all"
bash bda_benchmark/experiments/damageformer_srac_partition_probe_v1/run.sh smoke_all
```

### 6.2 两项检查通过后，再串行完整审计

确认上面`repeatability`与`smoke_all`都成功，才执行下面三项完整Val。每项使用新子目录；任何一项失败，后续停止。**这是审计，不是正式训练。**

```bash
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_PARTITION_PROBE_V1
export PROBE_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_PARTITION_PROBE_V1/$(basename "$PROBE_RUN_ROOT").log"
nohup bash -c '
set -euo pipefail
cd /root/shared-nvme/BRIGHT-master
for suite in prior iterations boundary; do
  export PROBE_OUTPUT="$PROBE_RUN_ROOT/$suite"
  bash bda_benchmark/experiments/damageformer_srac_partition_probe_v1/run.sh "$suite"
done
' > "$PROBE_LOG" 2>&1 &
echo "诊断PID=$!"
echo "日志=$PROBE_LOG"
tail -F "$PROBE_LOG"
```

查看某一项完整审计的报告，例如先验：

```bash
export PROBE_OUTPUT="$PROBE_RUN_ROOT/prior"
bash bda_benchmark/experiments/damageformer_srac_partition_probe_v1/run.sh report
```

可选粒度检查：`PROBE_OUTPUT="$PROBE_RUN_ROOT/scale"`后运行`run.sh scale`。它改变实际区域预算，应与前三项分开解释。

不要复用已经运行的同名子目录再次启动序列；新目录重跑也不等于旧失败已修复。只有新的完整450张审计通过复现和不变检查，才可以解释该新报告中的干预分数。不要将单图通过、smoke通过、GPU占用或`evaluate 450`单独当作审计成功。

定位晚出现的运行异常可设置 `PROBE_START_INDEX=0, PROBE_MAX_IMAGES=340`；这仍是子集诊断，不是完整Val。保持原顺序从0执行可保留前序运行条件；从330直接开始只回答单图附近能否复现。smoke不能同时指定子集参数。

## 7. 测试后怎么进入训练

本次没有训练入口，这是刻意的。先确认完整复现/不变检查通过，再选一个因素做独立训练：

1. 基线Q与有限建筑先验：同初始化、种子数/候选数/迭代数、内容、读出和训练预算。
2. 若问题指向支持范围：9候选与25候选，迭代数和种子数不变。
3. 若问题指向边界输入：原边界与change边界，Q固定，独立重新训练相同适配器/读出。

不要把三个候选同时合并再训练。任何会改变Q/Embedding的训练，后续都需从Train重新建库；不能继续使用绑定旧快照的Memory。先不引入Memory或新分类损失，也不解冻A1。

这些机制是任务适配与可证伪测试，不承诺涨分、实例分离或原创性。CPU契约测试不代替服务器真实BRIGHT/CUDA结果。
