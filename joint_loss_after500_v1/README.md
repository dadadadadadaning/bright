# D 第500步后的训练目标对照

这个工具实际运行的是“原目标 / 删除背景 Lovász 独立项”的两组对照。它与 `joint_lr_after500_v1` 的学习率对照是两个独立实验。

| 分支 | 看什么 | 怎么训练 |
|---|---|---|
| `d_control` | 原 D 的像素、父区域、soft 两槽、Top64 Memory、grid_probq | 直接调用原代码的 NLL + present 类别 Lovász |
| `d_no_bg_lovasz` | 完全相同 | 保留原 NLL；Lovász 只删除背景分子项，保留原 present 类别数作为分母 |

两个槽、背景输出类别、背景像素的监督均保留。没有 Memory 接受/拒绝门控。Q、A1、案例库、检索规则和前端继续冻结。最终仍预测背景/完好/受损/毁坏。

## 训练目标

令 S 为当前图像有效标签中实际出现的类别，L_c 为类别 c 的 Lovász 项：

```
原目标 = 原 NLL + sum(L_c for c in S) / len(S)
候选   = 原 NLL + sum(L_c for c in S if c != 0) / len(S)
```

- 原 NLL 对所有有效像素求均值；ignore=255。
- 前景 Lovász 也包含全部有效背景像素作为负例。
- 不使用 `classes=[1,2,3]`，避免改变缺失类别的处理。
- 不改为前景类别数作分母，避免把剩余项额外放大。
- 全背景图：候选 Lovász 为零，NLL 保留；全 ignore 图：两目标都是可微零。
- `loss` 始终表示共同原目标；`candidate_loss` 表示共同候选目标；`training_loss` 才是该分支反传的目标。
- 去掉一个非负项会自然降低目标数值；这不代表预测更好。

## 实验设置

读取下列已经完成的四组训练，不修改其任何文件：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D
```

使用原运行 `train/checkpoints/checkpoint_000500.pt` 的 D 模型、AdamW 动量和随机状态，建立两个独立副本。保留原数据顺序，重新读取前500个 CPU batch 验证位置，此过程没有旧500步的前端计算或梯度更新。

- 正式新更新：501～1500，共1000次；不是从头训练1500次。
- 学习率：两支都沿用原最大1e-4、warmup500、总长度5000的余弦调度；没有乘0.2。
- AdamW、weight decay=0.005、梯度裁剪1.0保持原设置。
- 同一张 Train 图像只计算一次冻结前端，两支分别训练，输入、支持与 RNG 对齐。
- 第500步先做完整 Val450 重放；允许的原指标复现差异仍是继承的0.02个百分点，原阈值未放宽。报告逐类偏差，不声称逐像素位级相同。
- 在500/1000/1500步做完整 Val450；1500步为预先规定的主终点。
- 固定 Train 子集在500/1500步都评估：使用原训练最先见过的450张图像，记录并检查同一名单和顺序。这不代表完整Train2918。
- 不访问 Test；不按 Test 分数选目标或步数；这是从单个训练种子分叉的探索实验，不是独立种子重复。

正式训练前先做20步真实 CUDA smoke（501～520）；smoke通过后，正式阶段重新从500步恢复，绝不接着smoke的520步训练。小样本检查不是正式成绩。

## 默认依赖

原工具必须仍在：

```
/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2
```

依赖原四组完整报告、500步模型/优化器/RNG、原 source 协议、相应前端模型、父/子案例库、Train/Val数据。它不要求已删除的旧 E0 快照重新作为来源，但会验证这次 Q-rebased 运行实际记录的依赖。

工具不修改源代码，不替换来源文件，不重建案例库，不修改旧完成字段。缺失输入会明确报路径并停止。不要清理上述来源运行、模型、案例库或正在运行的工具目录。

默认Python：`/root/shared-nvme/conda-envs/bright4090/bin/python`。复用已有PyTorch/BRIGHT依赖，不重新安装环境。云端仅有CPU环境；真实4090检查在服务器smoke进行。

## 解压和启动

把 ZIP 放到：

```
/root/shared-nvme/BRIGHT-master/New/bright_joint_loss_after500_v1.zip
```

按交付消息提供的 SHA256 校验 ZIP 后解压：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -m zipfile -e /root/shared-nvme/BRIGHT-master/New/bright_joint_loss_after500_v1.zip /root/shared-nvme/BRIGHT-master/New
```

只做文件/协议预检（不启动 CUDA）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_loss_after500_v1/study.py plan
```

完整启动（CPU测试 → 真实CUDA20步smoke → 500重放 → 1000次正式新更新及评估）：

```bash
bash /root/shared-nvme/BRIGHT-master/New/joint_loss_after500_v1/start_server.sh full
```

若仅要检查可用 `check` 代替 `full`；它只做smoke。后续 `full` 会建立新目录、重新smoke，不修改check结果。

每次输出到新的：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_LOSS_AFTER500_V1/run_...
```

同一输出基目录有文件锁，阻止重复启动这个损失实验。它不会停止或等待你已有的其他实验。若已有学习率实验在跑，两者同时运行会竞争CPU/GPU；只需启动一次。

查看日志：

```bash
tail -n 60 -f "$(cat /root/shared-nvme/BRIGHT-master/New/joint_loss_after500_latest.txt).log"
```

读取最新报告：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_loss_after500_v1/study.py report
```

按指定结果读取：在上述命令末尾加 `--root /完整结果目录`。退出tail按Ctrl+C，只退出查看日志。

## 报告怎样看

每条预测路径都同时记录原目标与候选目标，并保存：

- 前景mIoU、四类IoU/精确率/召回率、混淆矩阵、建筑二类指标。
- 原NLL、原Lovász、候选Lovász、背景项贡献。
- 各类对总NLL/Lovász的实际贡献（先逐图计算，再所有图均值）；贡献相加还原对应总损失。不是各类别内部的独立平均。
- 逐事件指标、有效像素数、各类出现图像数。
- 候选与对照之间、每条路径与grid_probq之间的逐像素配对救回/误伤。1500步保存压缩预测并重读核验。
- 每步原loss、候选loss、实际training_loss、学习率、裁剪前梯度、裁剪次数、Reader梯度、参数实际更新范数、AdamW真实步数。
- 前端/案例库/原报告/代码的终检；主终点与中间监测明确区分。

重点比较1500步 `d_no_bg_lovasz - d_control`，同时看候选是否超过自己的500步和共同grid_probq。只改善召回却损失精确率和IoU，不算成功；仅比退化的对照好、仍低于500步，是减轻退化的信号，不是新的最好模型。

如果有效，下一步再给B加同样目标，检查Memory在新目标下是否仍有增量，并使用配对独立训练种子复验。不要用这次单种子分叉声称稳定收益或已证明背景项是唯一原因。删除一项也会改变梯度大小；应结合已有低学习率对照和本次实际更新范数解释。

## 产物与磁盘

工具复用现有案例和模型，不复制整个旧实验、不生成新库。新产物主要是1000/1500步两支头部与优化器检查点、JSON/JSONL和末步压缩预测。预检会显示目标分区剩余空间，实际占用取决于检查点及预测压缩率。

完成条件：`workflow_state.json` 与 `train/summary.json` 正式完成，1500步Val完整且终检通过。看到PID、CPU测试通过或smoke完成，均不表示正式训练完成。
