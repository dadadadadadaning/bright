# 前两张图片显存对照

2026-10-22 的 `memcheck1500_bv0vugq4` 在第2张图片准备阶段 OOM，完成1/356张。完整 Sanitizer 标题显示6条 CUDA API 错误：3条 cudaMalloc、3条 cudaGetLastError，全部是 cudaErrorMemoryAllocation。它没有到达原先索引355的 parent.ids 单比特异常，不能用于证明或排除那次故障。

此工具用于定位这次 OOM 的显存增长。每个进程从同一个1500步检查点恢复，依次处理规范Test名单前两张图，保留全部六分支、原始FP32/路由校验以及先前的R操作观察。先跑普通Python进程；只有其两图评估及完整性检查通过后，才在另一个新进程中跑同样的两图加 Compute Sanitizer。没有第三次尝试，也不自动继续356图或完整Test。

这不是降低分辨率、减少类别或跳过失败样本的性能评估。图片前缀被明确限定为诊断，不用其分数替代Test878。测试标签仅作原有评分，不参与训练或配置选择。

## 运行

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memcheck_memory_v1.pyz start
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memcheck_memory_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memcheck_memory_v1.pyz status
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memcheck_memory_v1.pyz stop
```

默认每个子进程最多600秒，两段合计执行预算最多1200秒，启动/收尾的哈希检查另外计时。`--timeout` 可指定每段60至1800秒。后台进程不依赖SSH连接；日志查看的Ctrl+C仅退出查看。启动要求GPU0空闲并沿用原实验排他锁。

输出新建在 `SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/memory1500_*`。使用该目录中的 `runner.pyz status` 可固定查询这个任务，避免与后续任务的latest记录混淆。

## 测量与判读

- `plain/memory.jsonl` 和 `memcheck/memory.jsonl` 保存加载、恢复六个头/优化器、前端各主要子模块、支持构建和分支前向边界的显存记录。
- 包括 PyTorch allocated/reserved/peak，设备free/total，分配重试及OOM计数，图像/分支位置和梯度开关。
- `memory_latest.json` 保存最后一条记录，`status` 即使仍在运行也会显示。
- 保留原显存分配器环境，记录有关环境变量；没有清缓存、改allocator、强制GC、移动优化器、裁图、减小模型或改损失。
- 测量不保存模型张量，不产生新的训练检查点。原始路由观察仍可能在异常时保存数十至数百MB的CPU证据；其它文件主要是日志。
- CUDA异常后不再查询显存。失败点的显存用最后一次成功采样和原异常描述判断；不能将峰值所在区间当成已定位到具体泄漏行。
- 原算法和源文件不改。额外观测和Sanitizer会改变时序/布局；这不是精确历史重放。

普通两图失败就停止，保留证据，不启动Sanitizer。普通通过、Sanitizer失败表示差异与仪器化运行有关，但不能仅据此断定是工具bug、模型泄漏或硬件。两段都通过只说明该短流程可运行；原356图故障仍未解决。

两段完整通过时 `diagnostic_complete=true`，但 `complete=false`、`full_test_evidence=false`、`historical_failure_resolved=false`、`hardware_cleared=false` 保持不变。Sanitizer阶段必须有显式 `ERROR SUMMARY: 0 errors`，仅Python退出0不能通过。缺失日志、超时和异常都不重试或提升为完整Test证据。

## 开发验证

`build.py` 核对原memcheck发布包SHA256，并逐字节继承其模块，另生成显式命名的两图适配器。`test_memory.py` 使用历史模型CPU样本验证六分支输出/参数/RNG不变和钩子恢复，用两图合成记录验证报告边界，并运行真实子进程超时测试。

开发环境没有服务器数据、检查点或CUDA；本地测试不声称GPU显存问题已修复。
