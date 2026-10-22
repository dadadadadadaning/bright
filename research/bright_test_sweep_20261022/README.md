# 六分支五个检查点的 Test878 评估与路由定位

固定评估 **500、1000、1500、2000、2500** 步，每个节点包含 Grid 及 E/F/B/D/R/L。默认后台串行运行，每个节点使用一个新进程；没有优化器更新。这个任务是用户明确要求的多检查点探索性 Test 比较，不用 Test 自动选择模型。

## 为什么同时增加路由定位

独立缓冲区检查已在用户服务器完成155轮、约20分钟，七种模式均通过；这并没有解决模型评估时共享 route_q 出现单比特差异的问题。新程序监测所有六分支的 forward 前后及概率校验前后；对 D 分支所有图片，进一步监测槽编码、Memory Reader、三次 sparse_unpool、区域/像素编码、分类头及插值操作前后。已知目标图则对六分支均做细化监测。这样保留共同的输入不变性检查，同时限制逐算子CPU拷贝开销。出现变化时保留最近两个边界及原始路由校验信息，然后终止该检查点进程。

它不改变路由值、归一化、容差、分槽、检索、标签、权重或分类公式。快照会同步CUDA，改变执行时序；即使通过也不声称历史故障已解决。观察到某操作前后变化只定位时间区间，不等于证明该操作导致错误。

为限制磁盘占用，正常图片只持久保存提取和完成阶段的简短路由摘要；已知目标图 `turkey-earthquake_00000627` 保存全部阶段记录。异常时保存路由张量及相邻边界的 CPU 证据。正常推理不保存全分辨率概率图或预测图片，不复制训练检查点。

## 检查点来源

所有路径均在 `/root/shared-nvme/BRIGHT-checkpoints` 下：

|步数|来源|
|---|---|
|500、1000|`SRAC_SLOT_MEMORY_ABLATION_V1/run_20261020_025710_3840142/seed_3407/train/checkpoints`|
|1500|`SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261020_161047_4184092/checkpoints`|
|2000|`SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261021_112222_495713/checkpoints`|
|2500|`SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/recover2100_dlhrjb52/checkpoints`|

加载前核对每个文件及 sidecar 哈希、六分支身份、训练日志前缀和恢复谱系；加载后核对模型权重及优化器计数。500/1000使用原始检查点格式，后续使用续训格式。缺少检查点会明确停止，不用别的实验替代，也不会重新训练补造节点。

Test878 使用规范名单顺序，核对与 Train/Val 不重叠。查询真值仅用于评分；训练集支持库保持原样。旧 Control/Anchor、Fresh V3 及诊断更新检查点不会混入。

## 运行

使用已有 bright4090 Python：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_sweep_v1.pyz start
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_sweep_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_sweep_v1.pyz status
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_sweep_v1.pyz report
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_sweep_v1.pyz stop
```

默认每个检查点最多2小时，整个任务最多10小时。可用 `--case-timeout` 和 `--total-timeout` 指定秒数。这是上限，不是预计时长。断开 SSH 后继续；`logs` 的 Ctrl+C 只退出查看。`stop` 检查 PID 启动时间与命令后请求停止，保留已完成结果。

输出新建在 `SRAC_SLOT_MEMORY_ABLATION_TEST_SWEEP_V1/test_sweep_*`。每个输出目录内有固定程序副本，可用该 `runner.pyz status` 查询特定任务。启动时检查 GPU 0 空闲并持有与续训/独立评估相同的排他锁，拒绝重复启动。

五个检查点最多各评估一次。评估异常或节点超时会保留失败状态，并在新进程继续下一个预先指定节点；配置、代码来源或完整性故障会终止余下工作。不会在 CUDA 出错后复用同一个进程，也不会跳过失败图片后冒充完整878张结果。没有自动修复或无限重试。

## 输出和解释

- `summary.json`：整个任务及各节点完成状态，`job_finished`与`complete`分别表示任务结束和五节点全通过。
- `comparison.md`：完整节点的中文指标表。
- `metrics.csv`：步数、分支、四类mIoU、前景mIoU、四类IoU、OA、建筑IoU、受损/毁坏precision与recall。
- `contrasts.csv`：B−E、D−F、F−E、D−B、D−R、D−L及各分支−Grid，单位均为百分点。
- `per_event.csv`：逐事件、逐检查点、逐分支指标。
- `testXXXX/test_step_XXXXXX/result.json`：完整混淆矩阵和同像素配对转移（纠正与误伤）统计。
- 异常节点：`evaluation_failure.json`、`traceback.txt`、可能存在的 `route_failure.json`、`route_change.json`、`operator_window.json`及对应CPU张量文件。

报告从完整878条图片记录重建混淆矩阵及事件统计，核对原内存统计与保存文件；模型、优化器、RNG、输入文件、代码、guard恢复检查都通过才进入汇总。失败节点在汇总中保持缺失，不填0、不沿用别的步数。没有完整节点时不会生成虚假的性能结果。

四类mIoU是背景/完好/受损/毁坏IoU的算术平均；前景mIoU只取后三类。使用像素混淆矩阵，行GT、列预测，忽略255。建筑IoU将后三类合并，OA为有效像素准确率。没有擅自定义 F_ch/F_cls。缺失类别按该评分适配器原有的0规则计入固定四类平均；若某个事件完全没有有效像素，其导出指标留空，不能记为0分。

对各节点的冻结Grid混淆矩阵另外进行一致性比较。若不一致，`comparability_review_required=true`，应先检查推理可重复性，不把微小跨步变化直接解释为模型收益。

分析顺序：先检查完整性和Grid一致性，再看D相对Grid的收益是否跨节点保留；比较E/F/B/D区分槽数和历史支持的作用；D−R、D−L检验检索对应关系和支持标签的作用；最后用受损→完好、非毁坏→毁坏的配对统计解释类别变化。单种子小差值不能直接称稳定或显著收益。

## 验证范围

本地CPU测试使用历史真实模型和支持构造的小样本，验证监测前后六分支输出逐位一致、注入D Reader路由变化被定位、CUDA类异常保留原错误并恢复绑定；用完整878条合成记录验证五个步数标签、混淆矩阵、报告及失败节点处理。后台调度测试包含真实子进程超时。

本地没有用户服务器的检查点、真实数据或CUDA。发布前的CPU合同测试不等于服务器评估已通过，也不保证异常不会再次出现。
