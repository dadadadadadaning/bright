# Step1500 父区域整数索引异常：定向 Memcheck 诊断

已知失败证据：`test_sweep_hzld12ds/test1500`，规范 Test 顺序索引355，图片 `la_palma-volcano_00000812`，R 前向前后 `parent.ids` 一个 int64 从25变为25+2^51。对象、GPU存储地址和版本号相同。前后CPU快照不同不等于已证明GPU硬件故障。

本工具从同一个1500步检查点恢复，保持六分支 E/F/B/D/R/L 顺序，在 **Compute Sanitizer memcheck** 下仅执行规范 Test 名单前356张。没有优化器更新，没有训练，没有修改路由、索引或校验容差。它只生成诊断结果，不补成完整Test878，不生成新的模型性能排名。

## 观察内容

- 所有分支保留 forward/概率校验前后的冻结路由检查。
- 每张图片细查 R 分支的槽编码、Reader、parent/slot/memory还原、区域编码、像素投影、分类头和插值操作前后；目标图片细查全部六分支。
- 每张图片为父区域索引建立独立设备副本与独立CPU参考，核对初始一致性；在后续观察点通过设备端比较检查原索引与副本是否一致。
- 首次观察到父索引快照变化或设备比较不一致时，保留设备比较结果、参考副本CPU内容和两次新的原索引CPU拷贝。所有读数发生在不同时刻，只用于区分证据，不自动判定故障来源。
- 已经出现CUDA异常时不启动这组额外读数；确认期间若CUDA出错，立即停止进一步设备读取并保留已取得的CPU证据。异常不会触发修复、重新归一化或继续累计指标。

完整性观察、额外分配和Sanitizer都会改变CUDA时序及内存布局。本次通过不代表精确重放了历史故障，也不代表硬件已排除问题。

## 运行

使用发布的 `bright_route_memcheck_v1.pyz`：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_route_memcheck_v1.pyz start
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_route_memcheck_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_route_memcheck_v1.pyz status
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_route_memcheck_v1.pyz stop
```

默认后台运行 **最多4小时**，`--timeout` 可指定60至43200秒。这是上限，不是预计时长。Sanitizer可能显著减慢推理，超时后保留已取得证据，不宣称前缀通过。关闭SSH后继续；查看日志时Ctrl+C只退出查看。

任务目录新建在：

`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/memcheck1500_*`

每个目录保存程序副本 `runner.pyz`，可直接用该副本查询对应任务。不会覆盖旧失败目录、检查点或三个已经完成的Test报告。启动时核对原始五检查点谱系、此次失败证据哈希、旧评估程序和Sanitizer可执行文件；要求GPU0空闲，并持有已有实验的排他锁。没有自动重试。

Sanitizer命令为 `--tool memcheck --error-exitcode 86 --launch-timeout 0 --print-limit 20 --log-file ...`。工具自身等待连接不限时，外层进程组监督仍受上述4小时上限约束。没有使用额外的 `--force-blocking-launches`；子进程明确沿用 `CUDA_LAUNCH_BLOCKING=1`。

## 结果判读

`status` 同时输出整体状态、首次失败信息、父索引重复读数、最近操作窗口及Sanitizer日志末尾。

只有Python诊断正常退出、356张全部完成、模型/优化器/RNG/输入来源检查通过，且Sanitizer日志明确报告零错误，才能得到：

```text
phase = diagnostic_pass_no_error_observed
diagnostic_complete = true
job_finished = true
complete = false
full_test_evidence = false
historical_failure_resolved = false
```

后面三个字段保持false，因为这是前缀诊断。`job_finished=true`单独仅表示任务结束。

若Sanitizer报告非法读写，优先分析它的第一条错误和调用栈；若路由快照失败，则同时读取 `parent_ids_corroboration.json`、`operator_window.json`。设备比较不一致支持设备侧也观察到了差异，但不能据此区分软件、驱动和硬件。设备比较一致、重复CPU拷贝恢复一致，则需继续调查拷贝/观察时序/主机内存等因素，也不自动归因。参考副本也改变时会单独记录，不能把它误认成原索引变化。

`ERROR SUMMARY: 0 errors` 只表示该次memcheck没有检测到其覆盖范围内的问题，不是显卡健康证明，也不能排除所有竞态或逻辑错误。

文件主要是逐图统计、简短日志和首个异常的CPU张量。不会保存全套预测图或新训练检查点；异常快照可能占用数十至数百MB。

## 开发验证

原模型和支持实现的小型CPU样本用于比较六分支预测、测试R分支整数单比特变化、仅CPU拷贝损坏、参考副本损坏，以及CUDA类异常退出。另以356条合成记录验证诊断报告不被标为完整Test，以真实子进程验证超时退出。开发环境没有用户的CUDA设备、数据和检查点；真实Sanitizer检查需要用户服务器运行。
