# 第158步反向索引诊断

适用于 `SRAC_SEVERITY_MEMORY_V1/night_3407_zi_sjcjz/train_original`，正式训练在
`mexico-hurricane_00000077`、第158步 `loss.backward()` 因 CUDA ScatterGather
索引越界退出；最后完整更新为157，预期最近持久检查点为150。

这是有界诊断。只恢复并检查151–158，最多1800秒，不自动重试或续跑2500。
不访问验证集/Test，不覆盖原运行，不修改损失、模型、路由或索引，不输出可用于正式续训的新检查点。
观察会改变同步和内存布局，因此通过不等于历史故障已消失。

## 运行

使用发布的 `.pyz`，在原服务器环境运行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_backward158_diagnostic_v1.pyz start
```

`start` 核验已安装severity代码的三个固定SHA、原运行计划、Train2918/Val450名单、
157条完整训练记录、检查点150及其sidecar与summary中的SHA，然后创建独立 `backward158_*`
目录并后台启动。检查点SHA来自已有sidecar和summary，不声称它是外部独立认证。
原始来源、Bank和输入指纹在子进程继续由原Runtime核验。

无GPU任务时才启动；共享原night队列的锁，每30秒有后台心跳。超过1800秒，仅终止自己
创建的worker进程组。不会结束其他用户进程。SSH断开后仍运行。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_backward158_diagnostic_v1.pyz status
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_backward158_diagnostic_v1.pyz logs
```

启动时还会给出本次目录 `runner.pyz` 的同等命令；优先用本次固定路径，避免读取其他任务日志。
`check` 只做stdlib预检，无Torch加载、GPU分配或训练更新。

## 观察什么

1. 恢复模型及AdamW完整状态；重建1–150原始batch顺序后恢复Python/NumPy/Torch随机状态。
2. 151–157与原journal逐图对齐；158必须是目标图片。
3. 每步前后监测父区域Q/IDs、子槽Q/IDs、支持IDs、标签的不可变性。
4. 前向、原始NLL+Lovasz及反向均位于`detect_anomaly`上下文。
5. 遍历实际计算图，记录GatherBackward/SortBackward的forward调用栈、保存索引、源尺寸及界限。
   索引节点执行前用prehook检查范围和逐位变化。不使用saved_tensors_hooks，不替换任何Torch函数，
   不禁用原生版本检查，不修改梯度。
6. 失败时只使用已在CPU上的证据写报告。第一异常独立保存，清理阶段的CUDA报错不覆盖它。

Torch2.1的Gather保存`_saved_index`与`_saved_self`；Sort保存`_saved_indices`及
`_saved_self_sym_sizes`。负维度的无符号表示会规范化。扩展索引的零步长维先去重，避免把
`[1,65536,128]`索引展开为巨大的CPU/GPU副本。

持久CPU快照上限128MiB，单次当前CPU副本另限128MiB；失败证据文件上限32MiB。
这些是观察器预算，模型自身仍有显存/内存需求。磁盘只增加程序副本、JSON、日志及最多一份32MiB失败张量；
不复制数据集、Memory或原检查点。

## 如何判读

- `graph_failure_000158.json` 的 `first_observed_corruption` 若非空，表示实际捕获了索引/路由变化
  或越界。查看节点类型、forward_traceback、before/after、范围；它不是硬件故障的证明。
- CUDA直接assert且未提前捕获时，`latest_node`只是最后观察到的节点，不一定就是致错节点；
  同时看anomaly输出及`first_failure.json`，不要把候选当定论。
- `diagnostic_window_complete`、`diagnostic_complete=true`表示本次八步完成。
  `complete=false`、`training_complete=false`刻意保留：这不是2500步实验结果。
- `failed`或预算/超时失败不自动重试；保留记录。未重现的有限回放不能排除环境/硬件/时序问题。

## 本地验证

```bash
python -m unittest discover -q
python build.py
```

测试使用CPU PyTorch2.1.2，覆盖恢复后下一次AdamW更新等价、数据顺序、预检哈希拒绝、
正常Gather/Sort梯度等价、索引变异及越界截获、原生版本检查、快照预算、失败不更新参数、
后台超时与无自动重试。没有在用户4090上执行GPU复现，不声称已修复CUDA根因。
