# 指定检查点的 A/B/C/D Test878 评估 V2

本包对应上传的 `bright_soft_memory_q_rebased_v1_fix2.zip`。原训练包和原 Test V1 保持不变。

V1 只接受5000步训练、最终Val450及完整终检均通过的末步权重。V2是用户要求的**独立探索性Test**：默认明确选择2500步，完整核验这个检查点后测试，不将它描述为完成5000步正式训练。若未来根据Test调整设计，本次成绩仍应写为探索性分析。

## 五条输出

| 输出 | 含义 |
|---|---|
| grid_probq | 同次前向的冻结基础预测 |
| A | 固定空间双槽＋图内参考 |
| B | 建筑概率soft双槽＋图内参考 |
| C | 固定空间双槽＋跨事件Train案例 |
| D | 建筑概率soft双槽＋跨事件Train案例 |

固定现有Q、A1、父库、双槽库、Top64和检查点内保存的检索设置。直接加载四个头，不重训、不建新库、不用Test标签选择支持。Test名单保持独立，不修改Train或Val名单。继续按同图／同事件规则排除Train支持。

## 推荐：单文件启动，无需手工解压

下载配套 `RUN_SOFT_MEMORY_TEST_V2.py`，上传到服务器的 `/root/shared-nvme/BRIGHT-master/New/`。

默认运行2500步（执行一次）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/RUN_SOFT_MEMORY_TEST_V2.py --step 2500
```

启动器在New内解出自己的独立目录，然后后台核验并推理。无网络下载，不覆盖原工具。打印PID、结果目录和日志。PID只表示启动，结果以新目录的 `summary.json complete=true` 与 `full_test_evidence=true` 为准。

若先只检查源文件和检查点，不创建CUDA上下文或读取Test数据：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/RUN_SOFT_MEMORY_TEST_V2.py --plan --step 2500
```

`--plan`不是推理通过证明。正常运行本身也包含上述检查，无需先运行plan。若2500文件缺失，不自动选择另一个步数。需要比较哪个检查点，应在看Test分数之前明确指定；例如 `--step 1500`。5000使用 `checkpoint_last.pt`，即便选择5000，本包仍按探索性评价报告，不替代V1正式工作流终检。

默认路径：

```text
训练工具 /root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2
来源运行 /root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D
检查点   train/checkpoints/checkpoint_002500.pt 和 checkpoint_002500.pt.json
新输出根 /root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_TEST_V2
```

可覆盖 `--train-root`、`--tool`、`--output-parent`、`--step`。不应将新测试脚本加入原fix2目录，否则会改变训练源码哈希。

## 磁盘和运行资源

默认 `--cpu-threads 2 --workers 1`，与原训练的4个DataLoader worker不同，仅改变读取并行度，记录在Test报告中。数值profile、TF32和CUDA_LAUNCH_BLOCKING按检查点恢复；不为了速度改变原pooling检查。本次会直接使用来源协议的GPU，不停止或排队其他实验。若训练仍在运行，可能竞争资源。

默认只保存每图混淆矩阵、救回/误改转移统计和来源信息，不保存878张的GT/五组预测大数组。结束后重新读取统计journal核对聚合；`saved_predictions_reopened=false`，不将统计回读说成预测图复算。不能仅凭这些统计还原预测图。

保留来源协议、A1/Q权重、原V3成员目录、parent、values、Train/Val文件和所选检查点。检查点记录的源文件必须存在且匹配哈希。只看当前GPU占用无法证明这些来源可用。

## 结果在哪里、怎么看

`result.json`：五条路径前景三类mIoU、含背景四类mIoU、IoU/precision/recall、逐事件结果、救回/误改、匹配对比；单位原始0–1，报告命令显示百分数，差值为百分点。

`metrics.csv`：可直接用Excel查看百分数表。

`summary.json`：Test执行状态、所选步数、`training_complete_at_checkpoint`、错误。

`images.jsonl`：逐图统计。`progress.json`：处理张数。`input_fingerprints.json`：本次来源和Test输入哈希。`checkpoint_lock.json`：读Test前固定的检查点与四头哈希。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/RUN_SOFT_MEMORY_TEST_V2.py --report /实际Test结果目录
```

重点看D−同次Test的grid_probq、D−B、D−C，不能拿Test分数减Val的61.9027%。`complete=true`仅表示本次Test878与独立完整性检查通过，不证明Memory涨分，也不证明原训练5000步已完成。原2500步Val的`final_integrity_pending`不会被本程序改成已通过。

## 本地验证限制

配套CPU测试验证真实fix2接口、检查点读取、计数和失败路径；无法代替服务器4090、真实权重与878张图的验证。没有修改原pooling算子、容差或源码。真实CUDA若再次发生pooling错误，会保留失败报告，不能将部分Test当成完整结果。
