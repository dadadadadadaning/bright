# 第151–158步普通执行对照

适用于已完成观察回放的 `backward158_btuleshk`。该回放375次Gather/Sort节点检查通过，
151–157步与父日志最大损失差为1.7881393432617188e-7。此工具只执行一次同窗口普通对照。

## 比较什么

相同checkpoint150、模型与AdamW状态、RNG、数据顺序、支持库、原始NLL+Lovasz损失、
学习率、PoolGuard、CUDA_LAUNCH_BLOCKING=1。普通对照直接调用已固定版本的`experiment.update()`。

关闭的是新增的反向图索引观察、CPU张量快照、节点prehook与anomaly detection。
原训练函数本来就有有限性/梯度检查及引发CUDA同步的统计转换，因此不能将这个版本称为“没有同步”
或“没有任何hooks”。不改变workers、chunk_size、精度或allocator，不加empty_cache/GC，不修复索引。

程序先只读核验已完成观察回放的程序SHA、检查点来源、原始文件指纹、完整8步journal、
Gather/Sort覆盖与恢复记录；从原检查点150恢复后仅执行151–158，记录与观察回放逐步损失和梯度范数差。
差值是描述性证据，不作为任意“误差很小就算修好了”的阈值。

## 运行

发布zipapp位于本目录，下载并核验`.sha256`后执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_backward158_plain_v1.pyz start
```

自动新建`SRAC_SEVERITY_MEMORY_V1/backward158_plain_*`，后台一次运行，最多1800秒。
GPU已有计算任务或原night队列租约被占用时拒绝启动，不终止其他任务。
原实验和已完成诊断保持不变，固定原150检查点只读，无新正式检查点，无Val/Test，不续跑2500。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_backward158_plain_v1.pyz status
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_backward158_plain_v1.pyz logs
```

启动输出也给出当前目录的`runner.pyz`命令；该路径始终指向本次运行。
`check`仅stdlib预检，不加载Torch或使用GPU。状态同时展示`comparison.json`，未生成时显示提示。

## 结果怎么用

- `plain_window_complete`表示普通回放八步完整完成，原故障本轮没有复现。
  它不证明硬件健康、代码根因已修复，也不等于2500步正式实验完成。
- 普通回放失败而观察版通过，提供执行方式敏感的线索。一次对照不足以证明具体同步操作是原因；
  继续依据首个异常位置定位，不更改学习率/类别权重来规避索引错误。
- 两边通过就结束这轮八步短窗口重复测试，保留未复现结论；后续正式单分支恢复另行记录执行协议。
- 损失和梯度范数接近不能证明全部参数、梯度逐位一致。这里不对未保存的观察版终点参数作此声明。

新增程序、JSON和日志占用少量磁盘，不复制数据集/Memory/检查点，不新增张量快照。
监督进程和计算进程都以追加模式写日志，避免独立文件偏移互相覆盖。

## 验证范围

CPU单元测试与原始更新函数对照检查见`VALIDATION.md`。本地没有用户数据和4090，
不将CPU通过描述为GPU根因修复。这里保留`exact_historical_replay=false`和
`historical_failure_resolved=false`，因为进程历史及外围日志仍然不同。
