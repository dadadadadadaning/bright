# soft 双槽与 Memory 内容诊断

这个工具检查两个问题：**两个槽是否提供不同证据，以及历史案例是否提供图内参考之外的信息。** 它读取已有 B／D 权重和完整案例库做前向统计，**没有训练、没有优化器更新、不访问 Test**。因此无须为了检查这两点再训练一遍，也不会把诊断结果称为新的训练收益。

默认读取原四组实验的500步检查点。B 是 soft＋图内参考，D 是 soft＋Train Memory。两者均保留像素、父区域、soft 两槽和 `grid_probq`。可以另选1500或5000步，各次输出放入独立的新目录。

## 依赖与输出位置

原训练目录：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D
```

原运行代码：

```
/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2
```

工具复用该运行的前端来源协议、Q、父/子案例库、B/D头部检查点和Train/Val划分。缺失文件会停止并显示实际路径，不重建来源记录，不改写旧报告。保留这些依赖，不要为释放空间删除。

新结果目录：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_AUDIT_V1/run_...
```

产物为汇总JSON、CSV和有限数量的案例记录；不复制权重或整个数据集，不保存全套预测图。这减少磁盘占用，但仍需要原数据和CUDA前向计算。此工具没有安装新的Python环境。

流水线下分别保存 `smoke/` 和 `diagnose/`。主要产物为 `result.json`（最终结果）、`summary.json`（进度与完成状态）、`REPORT.txt`（可阅读汇总）、`regions_summary.csv`、`images.jsonl`（逐图简表）、`cases.jsonl`（每图最多4条案例）、`per_event.json` 及来源文件校验记录。`report --root` 优先打印完整诊断报告，尚无完整报告时显示小样本结果。

本轮只对Val图像前向；Train案例库存量直接读取已保存的Train标签和元数据，不重新运行Train图像，也不把Val标签写入案例库。

## 解压与运行

将交付ZIP放在：

```
/root/shared-nvme/BRIGHT-master/New/bright_slot_memory_audit_v1.zip
```

按交付消息的SHA256校验后解压：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -m zipfile -e /root/shared-nvme/BRIGHT-master/New/bright_slot_memory_audit_v1.zip /root/shared-nvme/BRIGHT-master/New
```

先做来源和文件检查：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_v1/audit.py plan
```

完整诊断：

```bash
bash /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_v1/start_server.sh full
```

`full` 依次执行CPU合约测试、两个不同事件的真实Val图像检查和完整Val450。小样本也使用完整案例库，不重现旧smoke中支持案例全部被同事件排除的设置。通过小样本检查不等于完整Val450结果。

只检查运行链路：

```bash
bash /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_v1/start_server.sh check
```

检查另一个已保存步数（不继续训练）：

```bash
BRIGHT_AUDIT_STEP=1500 bash /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_v1/start_server.sh full
```

把1500换成5000即可诊断末步。先完成默认500步、查看结论，再决定是否补其他步数。每次运行使用同一个选定检查点，不将不同步数混在同一张对照表内。

查看最新日志：

```bash
tail -n 60 -f "$(cat /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_latest.txt).log"
```

读取最新报告：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_v1/audit.py report --root "$(cat /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_latest.txt)"
```

退出 `tail` 按Ctrl+C，只退出日志查看。后台诊断有自己的运行锁；它不停止、不等待其他训练任务。多个CUDA任务会争用同一张GPU和CPU资源。启动器打印PID只是创建进程，完成情况须看最终报告。

可以通过环境变量覆盖Python、来源和原工具路径：`BRIGHT_PY`、`BRIGHT_TRAIN_ROOT`、`BRIGHT_ORIGINAL_TOOL`。默认值即以上服务器路径。

## 双槽统计如何解释

沿用原代码的槽顺序：**槽0为偏背景槽 `Q×(1−B)`，槽1为偏建筑槽 `Q×B`**。B 是模型预测的建筑概率。两种权重相加为Q，每槽单独归一化后汇总，因此不能将槽的名称当作真值类别。

统计两槽特征相似度、加权标签分布差异、槽质量及有效样本量等。有效样本量定义为：

```
ESS = (sum(w))² / sum(w²)
```

这里的单位是**特征网格位置**，不是原图像素，更不是建筑实例数量。恒定权重不会减少ESS；单个大权重支配时ESS接近1。空槽无有效比较，不能用零填充后当作“相似度低”或“标签差异大”。

如果一个父区域的B几乎不变，两槽归一化后可能非常相似。这是可解释的情形，不自动说明实现错误；应同时看混合区域、边缘和等级混淆中的结果。

建筑边缘、内部、低B真实建筑等分层使用真值**事后诊断**。建筑边缘是建筑前景/背景边界，不是逐栋实例边界；同类相邻建筑不一定被区分。部分分层相互重叠，不能把分层查询数直接相加作为总数。低B由预测概率定义，真实建筑由标签定义，二者不能混为一类字段。

## Memory统计如何解释

沿用原父区域T128、Top64、同图/同事件排除和原案例身份。64个父案例最多展开128个槽，不是128个独立案例。

报告关注：Damaged查询的候选标签质量、独立来源图像/事件、候选重复性以及被检索的Intact困难负例。事后真值只用于评估标签吻合和错误分层，**不参与检索、候选排序、模型输入或正式预测**。库中的标签仍仅来自Train。

图内参考的标签字段是模型预测概率，历史案例使用Train真值软标签。因此两者标签质量比较同时包含“参考来源”和“标签来源”的差异，不能全部归因于检索排序。若报告图内GT理想参考，它是单独标记的事后oracle，不进入Reader，也不是可部署模型成绩。

候选及Reader的重点统计针对偏建筑槽查询；`role1_gt_damaged`、`role1_gt_intact`分别按该槽内受损／完好的真值质量加权。原父区域的`gt_damaged`等空间分层仍保留，但两者不是同一口径：父区域含受损像素，不一定意味着偏建筑槽主要描述受损内容。

固定余弦权重投票只用于可解释的候选诊断，**不等于已训练Reader的注意力**。实际Reader及B/D预测的诊断与这种固定投票分别记录，不能互相代替。

实际B/D预测可计算同一位置的救回与误伤，以及相对共同 `grid_probq` 的变化。这提供当前检查点的效用快照；它不是新增训练，也不是独立重复种子。发现Memory拥有某类案例、特征不同、注意力非零或个别救回案例，都不足以单独证明稳定的Memory收益。

## 拿到报告以后怎么决定

| 观察 | 优先验证的下一项修改 |
|---|---|
| 混合区域的两槽仍近乎相同 | 检查B变化及汇总信号，再决定是否改分槽/表示 |
| 极小ESS的槽占用较多注意力且误伤集中 | 验证证据量编码或质量权重，保留匹配对照 |
| 有合适历史案例，但不在Top64 | 固定64父案例预算，比较父区域与建筑槽参与的检索 |
| 候选合理，但D仍明显误伤Intact | 优先查Reader读取和输出，而非盲目增大Damaged权重 |
| D仅少数事件受益 | 分析案例来源与事件差异，随后做配对种子验证 |

这些是后续决策，不是工具自动修改模型的动作。先完成这次诊断，再选一项有证据支持的训练改动。

## 验证范围

云端开发环境仅能进行CPU合约测试；测试数量以交付消息和运行日志为准。**交付时没有在你的BRIGHT数据上完成真实CUDA诊断。** 真实数据的小样本和Val450检查在服务器执行，不能将CPU测试通过当作正式研究结果。

完整诊断将B/D和共同`grid_probq`的四类IoU及前景mIoU与原选定检查点的Val450记录比较，最大允许偏差为**0.001个百分点**。通过重放且来源、前端、库、头部权重终检通过后，才声明完整诊断完成。它不会把运行结束或子进程返回0单独当作有效证据。
