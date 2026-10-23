# Original 基线正式恢复：150 → 2500

用途：在已完成 observed/plain 两次151–158重放之后，推进原 `original` 基线。
这不是新的 Memory 方法，不是故障修复证明，也不会自动启动另外三个变体。

## 固定输入

- 失败父运行：`SRAC_SEVERITY_MEMORY_V1/night_3407_zi_sjcjz/train_original`
- 保存状态：`checkpoints/checkpoint_000150.pt`，核验 sidecar、父摘要及实际 SHA256。
- 有观察器重放：`backward158_btuleshk`，8次更新完成。
- 普通重放：`backward158_plain_82l__kmb`，8次更新完成。
- 原实现：`/root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023`

所有相对运行目录均位于 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_SEVERITY_MEMORY_V1`。
旧目录、源代码、Memory 库和检查点保持只读；新结果写入唯一 `original_resume_*` 目录。

## 运行

发布 zipapp 下载并校验后，服务器执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_severity_original_resume_v1.pyz start
```

它会脱离 SSH 会话后台执行。每30秒输出监督心跳；心跳说明监督程序在等待，不代表步数增长。
默认最长12小时，可用 `--timeout-seconds` 设置1至86400秒。失败或超时停止，不自动重新尝试。
启动前检查这组实验的租约、当前 GPU 占用及磁盘余量，不终止其他任务。

查看最近一次运行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_severity_original_resume_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_severity_original_resume_v1.pyz status
```

启动输出也会提供本次目录内的 `runner.pyz logs/status` 命令；这种方式不会误读后来启动的任务。
`check` 只核验现有证据，不加载 Torch 检查点、不使用 GPU，也不启动训练。

## 实际计算

1. 恢复150步的模型、AdamW和全部 RNG；重建并核对前150个原始 batch 后再恢复 RNG。
2. 新 journal 继承原1–150步并明确标记来源，重新计算151–2500步，共2350次新更新。
   父运行151–157步、两次诊断151–158步均不拼接进此次训练轨迹。
3. 沿用原 `exp.update`：NLL＋Lovasz、`read_weight=0`、原学习率调度及数据顺序。
   保持 CUDA_LAUNCH_BLOCKING=1 与原检查；关闭新增反向索引观察器和 anomaly detection。
4. 每50步保存，新增200、250……2500共47个检查点，采用原 payload 格式和真实 roundtrip 核验。
5. 500、1000、1500、2000、2500步分别运行完整Val450；最后运行父manifest指定的固定Train诊断子集。
6. 成功评估恢复 RNG 与模型模式。最终来源、模型冻结状态、PoolGuard 等检查通过后，才确认本次新评估报告完成。

父运行 `initial_val` 只作为带SHA的既有证据引用。其原状态可能仍是
`evaluation_complete_integrity_pending`，本程序不会把旧报告改成完成，也不会用150步模型伪造step0评估。

## 如何读取状态

本次分支完整完成应同时满足：

```text
phase = complete
complete = true
training_complete = true
validation_complete = true
full_validation_evidence = true
last_complete_step = 2500
new_complete_updates = 2350
test_accessed = false
```

`complete` 仅指本次 original 分支训练及其计划评估完成；不代表四变体研究、Test或故障排查完成。
`historical_failure_resolved` 始终为false。2500步更新完成但后续评估失败时，保留
`training_complete=true`，同时 `complete=false`；可从持久化检查点判断后续恢复起点。

异常时保存首个错误和阶段；若原 update 在返回前失败，程序不会猜测该步优化器是否执行过。
没有 clamp 索引、归一化修复路由、跳过图片、静默继续或修改原损失。

## 产物及存储

写入训练JSONL、47个检查点、5份Val450逐图评估、固定Train评估及小型状态/来源文件。
不保存逐步完整激活或预测图片。磁盘检查按当前checkpoint体积估算留存余量；它不是绝对占用上限。

## 比较边界

此分支以父检查点为起点，保留恢复来源，不伪称新进程从零训练。
旧比较器若只接受fresh运行，需要显式支持恢复来源后再比较；不要手改完成/fresh标志绕过限制。
此包只提供可靠的 original 对照输出，不能单独证明 severity Memory 提分。
