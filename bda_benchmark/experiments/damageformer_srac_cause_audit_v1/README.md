# 当前父实验的原因诊断：一次只回答一个问题

适用父实验：`damageformer_srac_transformer_v1`，原始 A1 冻结、可训练软 Q 与完整区域读出、Stage A 无 Memory。

本目录是独立诊断代码。旧父实验的源文件、完成门槛、权重和输出目录保持原样。这里可以读取**正在训练的父实验的一份中途权重副本**；这不代表父实验已完成，也不能用它提前通过 J1/J2/J3 的父实验门槛。

你现在先做 A 和 B。C 的训练命令已经提供，但不会由其他命令自动启动。

| 阶段 | 做什么 | 回答的问题 | 是否训练 |
|---|---|---|---|
| A：logs | 读已有完整 Val450 记录 | 哪个类别退步、Damaged 漏报还是误报、什么时候出现 | 否，无GPU |
| B：snapshot + audit | 同一权重，改变指定汇总/回投操作 | Q 平滑、区域读出、建筑分组分别有什么信号 | 否，需推理 |
| C：freeze_probe | 相同固定 Q/B，两组配对训练 | 允许融合层适应区域分类是否有帮助 | 是，可选3000步 |

不能由 B 的结果直接宣称“某个设计导致全部下降”，也不能由 C 宣称“完整解冻 Swin 一定更好”。

## 上传和设置

把**本目录全部文件**上传到服务器：

```text
/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_cause_audit_v1
```

进入服务器后完整执行这一段。使用明确的 Python 路径，不依赖是否激活 conda。新的 `CAUSE_ROOT` 避免此前 `OUTPUT_ROOT`、`FRONT_CKPT` 遗留变量指向旧 V2 实验。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export PARENT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/formal50000_seed3407_20261008_173616
export CAUSE_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_CAUSE_AUDIT_V1/diag_$(date +%Y%m%d_%H%M%S)"
export DEVICE=cuda:0
export WORKERS=4
export SEED=3407
unset CAUSE_OUTPUT

ls -lh "$A1_CKPT" "$PARENT_RUN/front/best.pth"
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh preflight
echo "本次诊断目录=$CAUSE_ROOT"
```

`preflight` 不访问真实数据、不启动训练。本地已通过26项CPU测试，包括真实模型小尺寸前向、两张合成图的审计全流程、失败时保留诊断结果、配对训练梯度检查。真实BRIGHT/CUDA仍需服务器 smoke。

## A：先看已经有的分类结果

```bash
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh logs
```

输出全部验证步数的区域/A1前景mIoU和Damaged F1，以及最佳/最近验证的四类IoU、Precision、Recall、F1、混淆矩阵、GT Damaged被预测成Intact的数量。

这里的两个混淆矩阵是各自的统计：数量差**不是**“A1正确→区域错误”的逐像素伤害。B阶段才会把相同像素对应起来统计。

该命令只读`front/manifest.json`与`front/validation.jsonl`。正在追加但尚未写完的最后一行会明确提示；完整行损坏会报错，不静默丢弃。可以在父实验训练时运行。

如需机器可读的完整报告：

```bash
"$PYTHON" -m bda_benchmark.experiments.damageformer_srac_cause_audit_v1.logs \
  --front_run "$PARENT_RUN/front" --json
```

## B：固定权重的原因审计

先复制当前最佳的 **adapter checkpoint**。复制过程固定一个文件句柄并检查SHA256，不额外复制761MiB的A1文件。

```bash
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh snapshot
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh smoke_audit
```

副本放在 `$CAUSE_ROOT/snapshot/front_snapshot.pth`，`snapshot.json`记录其真实步数。它可能是13000步，也可能已经出现更好的新checkpoint，不按聊天里旧步数猜测。后面的 smoke、full audit 和可选训练全部用这同一个副本。

两张图 smoke 只证明接口能运行，不能用其分数选择方案。看到 `CAUSE AUDIT COMPLETE` 后才运行完整 Val450。此任务会用GPU；不会自行终止正在运行的父实验。GPU有空余可同机运行，但会争用计算/显存，资源不足时在父实验结束或空闲GPU上执行。

```bash
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_CAUSE_AUDIT_V1
export CAUSE_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_CAUSE_AUDIT_V1/$(basename "$CAUSE_ROOT").log"

nohup bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh audit \
  > "$CAUSE_LOG" 2>&1 &
echo "审计PID=$!"
echo "结果目录=$CAUSE_ROOT/audit"
tail -F "$CAUSE_LOG"
```

完成后，Ctrl+C只退出日志查看。读取报告：

```bash
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh report
```

输出目录已存在时会拒绝覆盖。重新运行审计可只设置新的输出目录，并保留同一快照：

```bash
export CAUSE_OUTPUT="$CAUSE_ROOT/audit_repeat_$(date +%Y%m%d_%H%M%S)"
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh audit
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh report
unset CAUSE_OUTPUT
```

### 十个输出不是十次训练

它们共享一次原生前端前向，没有优化器，也不更新BN统计。

| 输出 | 改了什么 | 主要比较 |
|---|---|---|
| `a1` | 原A1像素输出 | 原始参考 |
| `a1_resize` | A1概率只缩到Q特征图，再放回原图 | 对a1：分辨率处理损失 |
| `a1_q` | A1概率按原Q汇总并回投 | 对a1_resize：Q平滑的额外损失 |
| `a1_q_sharp` | 用Q平方后归一化，汇总并回投A1概率 | 对a1_q：分配变集中是否保留更多判断 |
| `a1_q_split` | 按预测B>=0.5将每个区域分成建筑/背景两组 | 对a1_q：隔离混合的概率路径信号 |
| `region` | 原生区域预测 | 必须复现快照记录 |
| `region_read_sharp` | 区域Token与logits不动，只将回投Q变尖锐 | 对region：回投平滑的影响 |
| `region_pool_sharp` | 用变尖锐Q重建Token，按原Q回投 | 对region：Token汇总/统计/几何路径的影响 |
| `region_both_sharp` | 同时改变Token构建和回投Q | 对region：二者联合的干预信号 |
| `region_split` | 预测建筑/背景拆成独立Token后分类回投 | 对region：分组收益和定位错误代价 |

默认 Q_POWER=2、B_THRESHOLD=0.5，作为事先规定的诊断，不自动搜索Val最优超参数。区域Token重建包含注意力Pooling、mean/std旁路和几何编码；不能把该项差异全部称为“均值汇总损失”。

所有区域干预都固定原始内容特征和结构adapter输出（其边界/熵仍来自原Q）。因此不会同时改变原图、Soft ROI、主干或结构调制。

预测拆分增加槽位上限 K→2K，实际占用另记。不会把背景组强制分类成Background，也不是建筑实例分割。GT仅在输出完成后做分组诊断，绝不作为分区输入。

### 读数与下一步

| 看到什么 | 支持的解释 | 下一步最小动作 |
|---|---|---|
| a1_resize就明显下降 | 分辨率路径贡献了损失 | 先记录该项，别全部归给Q |
| a1_q比a1_resize明显差，变尖锐后恢复 | Q概率平滑有不利信号 | 单独做同预算尖锐度训练对照 |
| a1_q接近原像素，但region明显差 | 当前区域表示/分类值得优先查 | 不据此证明高维特征汇总无损；再做区域表示对照 |
| read_sharp恢复，pool_sharp无改善 | 回投的平滑可能比Token汇总更突出 | 下一轮只改回投 |
| region_split提高，GT混合降低且ROI漏检处不恶化 | 预测分组有积极信号 | 匹配区域预算后重新训练分组对照 |
| split提高总分但ROI漏检、前景边界受伤 | 定位错误被放大 | 不直接加强硬建筑限制 |
| 干预下降 | 模型可能依赖训练时的表示分布 | 不能证明该设计重新训练也无效 |

结果保存完整全图指标、Damaged P/R/F1、逐像素伤害/救回、GT Damaged→Intact伤害、B分箱、定位TP/FN/FP/TN、边界/内部、小对象代理、每事件与每图混淆矩阵、实际区域数和显存/时间。

GT纯度按Background/Intact/Damaged/Destroyed分别加权报告，避免用占多数的背景掩盖问题。GT连通域仅是对象代理；条件分组统计不等于全图mIoU。报告不产生像素独立假设下的置信区间，不把一次训练多次验证当作多种子。

### 张量和坐标约定

- Val原图1024×1024时，当前默认结构特征网格约256×256，Q为`[1,9,65536]`；stride=8是在此特征图上的种子间隔，K=32×32=1024。
- 训练裁剪512×512时，通常对应128×128特征、K=256。代码从实际张量尺寸推导，不写死上述数字。
- 灾前RGB结构决定Q；RGB–SAR融合后的Soft ROI特征提供区域内容。两种输入模态的特征差不被解释成真实损毁变化。
- 令`Q_ik`为像素i到区域k的归属，`sum_k Q_ik=1`。概率汇总为`T_k=sum_i Q_ik P_i / max(sum_i Q_ik, eps)`，回投为`P'_i=sum_k Q_ik T_k`，不对概率再做softmax。
- 变尖锐为`Q'_ik=Q_ik^2 / sum_j Q_ij^2`，原先零权重仍为零。空区域均值为零、无有效归属的区域不参与读取。
- 预测硬二分只改区域ID：`k'=k+K*1[B_i>=0.5]`，每像素仍9条候选支持，槽位上限变成2K，不扩成18条软支持。
- 原图概率与GT类别面积比例缩小到特征图使用area；概率回原图使用bilinear、align_corners=False；条件分组图用nearest保持该特征位置实际使用的分组。
- 审计所有参数冻结、eval、无梯度；完整模型参数/缓冲区前后哈希一致才算完成。

原生区域和A1必须在完整Val450复现，绝对mIoU误差上限事先固定为1e-5。失败时仍保存昂贵的`audit.json`，但`complete=false`，不得解释成方法增益。不要为了通过而手改期望指标或放宽容差。

## C：可选，验证冻结融合层是否限制适配

**先看B的结果，不需要现在启动C。** C不是所有参数解冻，也不改Swin。

| 条件 | F | U |
|---|---|---|
| 源A1、Swin、定位B、Q生成器 | 全冻结 | 同一份，全冻结 |
| 原始RGB/SAR特征与同批裁剪 | 相同 | 相同 |
| 拷贝的区域adapter/Embedding/分类器 | 训练 | 同样初始权重，独立训练 |
| 拷贝的跨模态fusion_layer_3 | 冻结 | **训练** |
| 融合层BN运行统计 | 固定eval | 固定eval；仅其可训练权重可更新 |
| Memory、新类别损失、新分区规则 | 无 | 无 |

两组使用同一个源模型前向产生Q/B，区域读出参数对象和存储互不共享。初始两组输出必须等于源区域输出。训练仍用原NLL+Lovasz，读出学习率1e-4，U融合层1e-5；每组有效batch16、3000次更新。读出梯度分别以相同阈值裁剪，U融合层单独裁剪。

这里的“固定Q”指生成器冻结且两组对同一张裁剪使用同一个Q，不是把一张图的Q套给所有图。

先GPU/真实数据smoke：

```bash
unset CAUSE_OUTPUT
export PROBE_BATCH=2
export PROBE_ACCUM=8
export PROBE_AMP=none
bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh smoke_freeze
```

smoke输出为`$CAUSE_ROOT/freeze_probe_smoke`，不会占用正式目录。通过后，在GPU空闲时启动：

```bash
export PROBE_STEPS=3000
export PROBE_EVAL_EVERY=500
export PROBE_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_CAUSE_AUDIT_V1/$(basename "$CAUSE_ROOT")_freeze.log"
nohup bash bda_benchmark/experiments/damageformer_srac_cause_audit_v1/run.sh freeze_probe \
  > "$PROBE_LOG" 2>&1 &
echo "配对训练PID=$!"
tail -F "$PROBE_LOG"
```

默认FP32的源前端与原验证一致；`PROBE_AMP=bf16`只影响两个可训练副本，需作为另一份记录，不能运行中改。若显存不足可batch1、accum16，但应明确该配置并重新使用新目录。

U稳定优于F，支持“融合特征适配有价值”；U无改善，只能说明本次融合层解冻未显示收益。两者都不能确定完整解冻Swin的结果。需要比较同一步结果及Damaged P/R/F1、Intact/Destroyed、各事件，不能只挑U最好的步与F较差的步。

只保存`best_F/last_F/best_U/last_U.pth`的小模块权重，A1与源快照按哈希引用；不新增整份A1副本。训练不会接管或覆盖当前父实验，不产生供父流水线/J123冒充已完成的材料。

## 推荐交回分析的文件

现在先提供 A 的输出，以及完整B结束后的 `$CAUSE_ROOT/audit/audit.json` 或 `report` 输出即可。

可选C完成后再提供`freeze_probe/summary.json`和`validation.jsonl`。这些小文件足够检查原因，不需要传761MiB的A1。
