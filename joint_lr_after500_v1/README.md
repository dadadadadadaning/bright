# 第 500 步后的学习率匹配实验

本工具接入已完成的 `soft_memory_q_rebased_v1_fix2` 真实训练运行。
它建立新实验，不修改旧脚本、旧检查点、来源协议、案例库、旧完成标记。
不重建库，不使用 Test，不改变损失、Q、soft 槽、检索 key 或邻居数量。

## 研究问题和四个分支

原 B 是 soft＋图内参考，原 D 是 soft＋Train Memory。
**本轮四个分支不是原 A/B/C/D 消融。** 每个 B/D 从自己已保存的第 500 步状态分出两份：

| 分支 | 来源权重与 AdamW 状态 | 第 501 步起的学习率 |
|---|---|---|
| b_control | 原 B500 | 原 5000 步调度 |
| b_lr02 | 同一 B500 | 原调度 × 0.2 |
| d_control | 原 D500 | 原 5000 步调度 |
| d_lr02 | 同一 D500 | 原调度 × 0.2 |

四个分支共享一次冻结前端前向、相同样本与支持案例，各自有独立权重及优化器。
沿用梯度裁剪 1.0、NLL＋Lovász、原 AdamW 配置和原 batch=1 全图训练。
原最大 LR=1e-4；新的低 LR 分支从约 2e-5 开始。
原 warmup 为 1–500 步，cosine 时间轴仍到 5000 步。本轮只在 1500 步停止，
不把调度总步数改成 1500，不再 warmup。

主比较预先规定为 **1500 步同次 low−control**，分别对 B、D 计算；
同时比较 D−B 和每组相对同次 grid_probq。500/1000 的成绩作为轨迹与候选，
不以各组不同最佳步数取代主比较。属于单种子、基于既有观察设计的探索实验。

## 为什么能从 500 步建立分支

已核对原包代码，检查点保存了权重、AdamW moment/step、Python/NumPy/Torch CPU/CUDA RNG。
原 loader 是固定私有 Generator 的全图无增强 loader，train.jsonl 记录了实际图片顺序。
工具重建 loader，逐条核对并跳过前 500 个 CPU batch，然后核对 501–1500 步样本顺序。
跳过时不做前端推理、不更新参数。前 1500 步在同一 epoch，没有迭代器换轮问题。

旧训练器没有原生 resume 入口；本工具是另一个有独立协议的显式分支实现，
不绕过旧训练器的 5000 步完成检查，也不宣称逐位复现历史训练轨迹。
因此增加同次原学习率对照，避免只把新结果与历史分数相减。
若完整状态、数据顺序、来源或冻结校验不符，停止并保留日志；不自动退化成只加载权重。

## 服务器准备

代码放在 `/root/shared-nvme/BRIGHT-master/New/joint_lr_after500_v1`。
默认 Python：`/root/shared-nvme/conda-envs/bright4090/bin/python`。
原工具目录：`/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2`。
原结果目录：
`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D`。

所需旧资料：原来源协议与其模型文件、原完整 parent/values 库及其成员归档、
`train/checkpoints/checkpoint_000500.pt` 和 `.pt.json`、
`train/manifest.json`、`train/summary.json`、`train/train.jsonl`、
`train/val_step_000500/result.json`、`workflow_state.json`，以及原 repo 的运行依赖。
**本工具不要求第 5000 步固定 Train/Val 诊断已经结束，不读取 Test。**

不要删除这些仍被协议引用的资料。文件缺失时报告原路径，不生成替代 manifest。

## 运行

解压后只需：

```bash
bash /root/shared-nvme/BRIGHT-master/New/joint_lr_after500_v1/start_server.sh full
```

`full` 自动顺序执行：CPU 合约 → 真实 CUDA smoke → 正式 501–1500 对照。
smoke 从 500 分支后各更新 20 步，确实覆盖 LR 改动，使用完整来源库、两张 Val。
smoke 权重丢弃；正式阶段重新从原 500 步检查点建立四个分支。
不等待 GPU 空闲，也不会停止其他任务。原 GPU/CPU 忙时本轮可能变慢或显存不足。

若只想检查：

```bash
bash /root/shared-nvme/BRIGHT-master/New/joint_lr_after500_v1/start_server.sh check
```

查看最近一次启动的报告（无需记住环境变量）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_lr_after500_v1/study.py report
```

查看日志：

```bash
tail -n 60 -f "$(cat /root/shared-nvme/BRIGHT-master/New/joint_lr_after500_latest.txt).log"
```

输出在新的 `BRIGHT-checkpoints/SRAC_JOINT_LR_AFTER500_V1/run_*`。
重复启动会产生独立运行；请不要因日志短暂不更新而重复启动。

## 会得到什么

- 初始 500 步：完整 Val450，并对原 B500/D500/grid_probq 的类别 IoU 重放。
- 1000、1500 步：同次四分支 Val450，逐事件混淆、NLL、Lovász、前景 mIoU。
- 1500 步：额外评价原训练最早见过的固定 450 张 Train 图像；该子集不代表全 Train2918。
- Damaged 精确率/召回率、Damaged→背景计数、建筑二类指标。
- GT×原LR预测×低LR预测的联合计数，用于真实配对的救回/误伤；不是两张混淆矩阵的净变化。
- 末步保存预测并重新读取重算指标；保存四分支完整状态、输入哈希、冻结检查和错误现场。
- 初始各继承500次更新，新训练各1000次；合计看过1500张不同 Train 图像，不声称访问完整Train2918。

## 怎样解释

1. low1500 超过 control1500：支持此 LR 改动在本次分支实验中缓解退化。
2. low1500 仍不如同次500：可能只是延缓下降，尚未超过已有候选。
3. D_low 与 B_low 相当：不能把训练策略收益归给 Memory。
4. D_low 超过 B_low，但仍低于 grid_probq：局部对照正向，系统整体尚未改善。
5. 候选有效后，从共同初始模型以预先声明的规则重新训练多个种子，并补 A/C，
   才能讨论稳定性和 soft 的贡献。克隆 B/D500 不等于新的独立种子。

本轮不额外加 Damaged 类权重。若低 LR 不足，再独立测试损失设置，避免混淆原因。

## 验证范围

本地 CPU 测试验证真实 AdamW 500 步状态复制、学习率切换、参数隔离、随机流匹配、
无效状态拒绝及评估报告逻辑。真实 BRIGHT 权重/数据未进入本地工作区，
因此本地测试不构成 CUDA 或 BRIGHT 收益证据；服务器真实 smoke 和 Val450 才能验证接线。

开发测试：

```bash
python -m unittest discover -s joint_lr_after500_v1 -p 'test_*.py' -v
```
