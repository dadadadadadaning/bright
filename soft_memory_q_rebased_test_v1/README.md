# 冻结 Q-rebased A/B/C/D 的一次性 Test878 评估

这套程序只评估已经完成的 `soft_memory_q_rebased_v1_fix2` 四组实验。它不训练、不恢复训练、不修改 fix2、前端、父库、双槽案例库或源运行目录。当前只有后台 PID、20 步 smoke、中间 Val 或 450 张处理进度时，程序会停止，**不会读取 Test**。

## 必须先具备的证据

原 `full` 结果目录须有通过的 `workflow_state.json`，完整父库与双槽案例构建、各组 5000 步训练、`train/summary.json` 和 `train/final_val/result.json`。检查点必须是同一运行的 `train/checkpoints/checkpoint_last.pt`，四组都恰好 5000 步。检查包括来源代码和发布文件 SHA256、检查点与头权重哈希、完整 Val450 的 450 份保存预测回读、Train/Val 输入文件哈希及运行时完整性。任一检查失败就保留失败原因，不会把结果标为正式 Test。

Test 模型选择只根据**末步 Val450 的前景三类 mIoU**。并列时按 `grid_probq、A、B、C、D` 顺序。选择结果先写入 `val_locked_selection.json`，然后才读取 Test 列表和图像。一次 Test 前向同时产生 `grid_probq` 和 A/B/C/D 的描述性结果；不能用 Test 分数再挑选、调阈值或重训。本项目以往已查看过其他模型的 Test878，因此本轮不声称这是整个项目从未接触过的盲测。

## 服务器启动

把发布 ZIP 解压到 `/root/shared-nvme/BRIGHT-master/New/` 后，设置**实际已完成**的四组运行目录，再运行启动器：

```bash
export BRIGHT_TRAIN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_实际完成目录
bash /root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_test_v1/start_server.sh
```

当前用户报告的 `run_20261014_202057_full_fEYBZv7D` 当时只证明启动，尚未证明通过 smoke 和正式训练。若它未完成，脚本应拒绝；不要手动把 `summary.json` 的 `complete` 改为 true。

脚本立即打印 PID、日志和新结果目录。后台仍先做完整只读来源校验；只有全部通过才读取 `standard_ML/test_set.txt`。也可只做来源核查，**不读 Test**：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_test_v1/locked_test.py plan --train-root "$BRIGHT_TRAIN_ROOT"
```

程序固定需要 Test878 的灾前 RGB、灾后 SAR 和四类标注。Train2918、Val450、Test878 名称必须互斥。Test 数据集缺标注时不能报告 mIoU。Test 输出目录必须全新，且位于原训练运行、数据集和工具目录之外。若只得到部分 Test 进度，`full_test_evidence` 保持 false。

## 报告

正式结果在新目录的 `result.json` 与 `summary.json`。`images.jsonl` 记录每张图的哈希和混淆矩阵，`predictions/*.npz` 保存真实标签及五条路径的预测并在结束时重新读取校验，`input_fingerprints.json` 记录所用文件。运行时前端、模型、案例库、代码和 PoolGuard 仍需全部通过终检。

报告**分别**提供前景三类 mIoU（Intact/Damaged/Destroyed）和 Test 四类 mIoU（Background 也计入），以及每类 IoU、逐事件结果。Val 前景三类数字不能直接与 Test 四类数字比较；应在同一 Test878、同一指标内比较。Test 报告的 A/B/C/D 是描述性比较；预先选出的主路径写在 `selected_on_val`。

本包的 CPU 单元测试只验证协议与合成数据，不代表服务器 CUDA、BRIGHT Test 或最终效果已通过。
