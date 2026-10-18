# 新训练 → 双槽／Memory 内容诊断：连续运行

本包是顺序执行与检查点接线，不再复制模型、损失、检索、槽统计或训练循环。新建一个实验目录，前一步真实完成并通过校验后才开始下一步；任一步失败，后续停止。不是 Test878 评估。

## 默认流程

1. 安装或复用固定版本的 `joint_anchor_study_v1`、`slot_memory_audit_v1_fix1` 与共享状态工具。已有不同版本不会被覆盖。
2. 检查冻结 A1、Q、来源协议、Q 正式完成证据、V3 Train 成员清单和数据。
3. 原父案例库、双槽库完整且一致时直接复用；派生库缺失或未完成时，在本次永久结果目录中依次重新建库。完整但哈希错误／来源矛盾的库会停止，避免悄悄换来源。
4. CPU 测试，真实训练 smoke；随后新初始化 D（soft双槽＋Memory），原目标共同训练500步。
5. 复制这个新500步模型、AdamW及随机状态，分别用原目标／原目标＋0.1 KL继续到1500步。两支均使用原调度×0.2。期间和末步完成 Val450；固定Train450用于诊断。
6. 只有训练、Val450与终检全部完成，才做后续诊断 smoke 和完整 Val450 内容诊断：新共同500步一次，新1500步 control/anchor 两支一次。

新500步模型不是旧D500的恢复。没有重新训练A/B/C/D四组5000步，没有改双槽结构，没有Memory拒绝门控，没有用Test挑模型。

## 权重缺失的边界

**本流程能重新训练丢失的联合分类头。冻结A1和Q是必要起点，不能靠运行脚本恢复已经删除且没有备份的权重。** V3原始成员清单及其绑定文件也需存在；报告不能代替模型或成员身份。

来源缺失时会把文件列入本次 `source_plan.json` 和主日志，标记 `training_complete=false`，停止且不伪造完成报告。配置沿用已经安装的anchor包中的示例；路径改变时，复制示例到包外，修改后传 `--config /绝对路径/配置.json`。不要将其他实验的last.pth改名冒充原Q来源。

## 一次启动

解压到 `/root/shared-nvme/BRIGHT-master/New` 后：

```bash
bash /root/shared-nvme/BRIGHT-master/New/joint_anchor_chain_v1/start_server.sh
```

无需另外启动训练器或旧诊断器。脚本首先创建真实存在的日志文件，再创建后台任务，打印PID与输出位置。它不等待GPU空闲、不终止其他GPU任务，也不修改旧实验。

默认用已安装的原模型包：

```text
/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2
```

如需指定已确认的新来源配置：

```bash
bash /root/shared-nvme/BRIGHT-master/New/joint_anchor_chain_v1/start_server.sh --config /root/shared-nvme/BRIGHT-master/New/joint_anchor_source.json
```

默认同时诊断500与1500步。只诊断末步可明确追加 `--audit-steps 1500`；不要根据Test成绩挑步数。

## 看日志与完成状态

```bash
tail -n 80 -f "$(cat /root/shared-nvme/BRIGHT-master/New/joint_anchor_chain_log_latest.txt)"
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_anchor_chain_v1/chain.py report
```

主流程锁防止同一结果根目录下重复启动链。若失败，读取当前目录的日志和traceback，不应反复启动制造多份实验。没有自动续跑失败目录的行为。

## 输出位置

```text
SRAC_JOINT_ANCHOR_CHAIN_V1/run_.../
  workflow_state.json             总流程状态，区分训练完成与诊断完成
  source_plan.json                依赖、来源哈希、是否需要重新建库
  fresh_source_config.json        本轮实际使用的来源和库路径
  parent/、values/                仅需要重建时生成，必须保留
  study/train/                    新训练与Val450
    checkpoints/checkpoint_000500.pt
    checkpoints/checkpoint_001000.pt
    checkpoints/checkpoint_001500.pt
  audit_smoke/                   两图诊断接线检查
  audit_000500/REPORT.txt         新共同500步内容诊断
  audit_001500/REPORT.txt         新1500步两分支内容诊断
  logs/                          每阶段日志
```

诊断报告输出双槽特征相似度、标签差异、有效样本量、建筑/背景边缘、建筑内部、低建筑概率的真值建筑分层；同时分析候选类别质量、独立来源图像与事件、相似Intact负例、Reader加权标签证据及配对救回/误伤。GT只参与事后统计，不进入候选选择和预测。

`common500`只代表新共同起点，`control/anchor`都是D结构。图内候选质量仅作为参考统计，**没有伪装成一个训练过的B分支，也不将control误标为B**。旧统计器内部的b/d计数键在输出前转换为真实新分支名。

两次诊断重放对应的新Val450，不能冒充历史ABCD成绩。两槽内容和候选冻结时应基本一致，学习到的Reader权重和最终预测可以不同。高相似度、好邻居或注意力变化本身不证明因果收益。

默认只保存聚合JSON、逐图计数和CSV，不保存全图大张量或debug packets。父库/双槽库含绝对来源引用，不能训练完就随意删除或移动；后续诊断仍依赖这些文件。

## 代码分工与验证范围

- `chain_io.py`：安装器、流程与诊断共用的JSON、哈希和代码包校验。
- `chain.py`：顺序执行、进程锁、真实完成校验、失败停止。
- `source_plan.py`：复用原配置和来源检查，决定派生库复用／重建。
- `post_audit.py`：将新检查点接入原槽、邻居、Reader与像素统计函数。
- `install_deps.py`：只安装固定版本代码，不下载权重或数据。

所有旧发布包保持不变。CPU模拟检查只验证接线、哈希和流程；不等于真实BRIGHT/CUDA训练已执行。真实成功须以服务器新日志和终检为准。
