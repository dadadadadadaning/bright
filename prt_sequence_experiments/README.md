# BRIGHT：完整顺序检查与区域读出实验队列

## 证据与问题

旧初始 Val450 中 P 有 5 张、R 有 4 张概率一致性失败，训练更新为 0；历史五路径复现、E0 对比、原生 A1、输入和源码完整性均通过。随后只选两张图片的独立诊断中，原设置下所有差异为 0。局部关闭卷积 TF32 改变了 P 的参考输出，而 R 不变。

因此当前只能说：故障没有在两图独立诊断中复现。不能宣布修复，也不能断定 TF32 是旧故障原因。本包先检查原验证序列中的行为，再有条件执行已准备的区域研究。

## 固定实验预算

1. 第一个新进程按原训练初始化顺序创建私有头和三个优化器，复用原 `evaluate_heads`，完整原顺序 Val450；不安装诊断 hook、不改精度设置、不更新参数。
2. 若第一个完整检查通过，第二个新进程做同条件完整确认。两次通过只标记 `history_not_reproduced`，不声称根因解决。
3. 若第一个完整检查只在 P/R 初始概率门失败，第二个进程变为现场捕获：按原 Val 顺序执行，首次观察到失配后保存实际特征、区域均值、分类分数、概率、权重、布局与来源到 `scene.pt`，然后停止。捕获过程安装 Python 观察器并保留真实张量，可能改变分配和时序；未复现不代表修复。CUDA 或其他门失败则直接停止。
4. 最多两个诊断进程，不反复运行直到通过。任一原完整检查失败都会阻断训练，捕获过程通过或未复现都不能绕过阻断。

局部现场捕获是数值排查，不产生新的正式损毁分割成绩。原历史指标门 `1e-5` 和初始化张量 `atol=rtol=2e-5` 均保留。

## 研究训练

默认只检查。显式设置 `CONTINUE_TO_TRAINING=1` 后，必须两个未插桩完整检查都通过，才调用原夜间队列一次；它自己的完整初始化检查仍须通过才更新参数。本包使用已通过的原 fresh E0，不生成替代协议，不改变旧运行时身份。

随后串行训练：

| 组 | 预算与用途 |
| --- | --- |
| P/R/T | 各 5000 次头更新，共享 5000 张次冻结主干前向；P/R 为同容量对照，T 为原复杂摘要读出适配 |
| local_delta/region_delta | 各 5000 次更新，共享另一段 5000 张次前向；同架构、参数数与图像顺序，检验额外区域均值上下文 |

两段训练均包含完整初始、末期和独立 Val450。基础 A1、Q、区域 embedding、旧 J、Memory 和投影均冻结；新候选不读取 Memory 生成新预测。后续只比较各自匹配条件，不相加历史 SAR 或 soft 路由的涨分。

重点结果：R−P；T−同次原 region；region_delta−local_delta；region_delta−同次 grid_probq。同时看 Damaged IoU、Intact 误报、救回/伤害与事件差异。T 改善不证明区域压缩无损；Memory 尚需独立审计。

## 服务器运行

将整个本目录放在原工程外。默认原工程、Python、失败来源和已有 `region_night/` 工具位置已按聊天记录设置；不需要覆盖旧工程或重新下载旧权重。

```bash
bash /实际解压目录/prt_sequence_experiments/run.sh
```

默认仅做两个以内的检查。若希望两次检查通过后自动执行上述研究：

```bash
CONTINUE_TO_TRAINING=1 bash /实际解压目录/prt_sequence_experiments/run.sh
```

脚本等待 GPU 空闲，不终止其他任务；不要同时在目标 GPU 启动另一套训练。日志由脚本先创建，后台任务断开 SSH 后继续；日志查看用 `tail -F`，Ctrl+C 只退出查看。不会自动续训失败任务。

默认旧来源：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34
```

结果父目录：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_SEQUENCE_EXPERIMENTS_V1
```

日志实际 `OUTPUT_ROOT=` 下包含 `workflow_state.json`、`check_01/check.json`、`check_01/initial_val/result.json`、`check_02/check.json`。第二次捕获到失配时另有 `check_02/scene.pt`。正式夜间子队列的具体路径记录在日志，位于该根目录的研究输出父目录下。

`training_entry_started=true` 只表示启动了旧训练入口；实际更新步数看原 `prt_fit/summary.json` 和 `residual_fit/summary.json`。`root_cause_resolved` 始终为 false；两次通过是规定预算内未复现。

把新 `workflow_state.json`、两个 `check.json` 和失败时的对应日志末尾提供给后续分析。完整正式成绩另看原独立验证结果，不能用诊断或训练中间值代替。
