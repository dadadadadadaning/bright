# read_aux checkpoint800 的独立恢复入口

本包仅对应 `read_aux_001_3407_night_01`：最后完整日志803、update804失败、已保存checkpoint800。它不修改已安装的实验包或任何原始运行。校验特定800步checkpoint SHA256及原源码版本，其他失败运行不能复用此入口。

## 执行内容

1. 验证父目录plan/manifest/journal、源码、checkpoint800和正式bank/source。
2. 恢复模型与AdamW；严格核对恢复状态和800次optimizer更新。沿原数据loader读取并核对前800个样本（不进行这800步的前向和训练），再恢复Python/NumPy/Torch CPU/CUDA随机状态。原loader使用无增强全图数据和独立随机generator。
3. 从801重新更新；仅801–804启用anomaly detection、canonical路由及autograd整数索引快照检查。父目录801–803日志不继承为新更新。CPU快照按零步长维度压缩，每步总额128MiB；异常时最多写32MiB张量快照。
4. 四步通过后，以同一进程、同一配方继续805–2500。整个恢复进程设置CUDA_LAUNCH_BLOCKING=1。800后新增1700步；保存804、850、900…2500检查点。每500步进行Val450；终点完成固定Train诊断和最终完整性检查。
5. 完整2500步Val通过后可用`compare`比较原original基线。只允许运行协议中明确记录的CUDA同步设置差异，其余配方、数据、初始化等严格匹配。

恢复配方保持 read_aux weight0.01、seed3407、原NLL+Lovasz、AdamW、原5000步调度跨度及500步后0.2倍学习率。此次不访问Test，不新增模型结构。

## 命令

把已校验的pyz下载到 `/tmp/bright_readout_recovery_v1.pyz` 后：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_readout_recovery_v1.pyz start
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_readout_recovery_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_readout_recovery_v1.pyz status
```

start自身后台启动，无需另加nohup；关闭Xshell不结束进程。每次新建 `read_aux_recover800_*` 目录并保存独立runner.pyz。logs/status默认指向最近一次；也可指定`--output 完整新目录`。GPU忙或已持有readout训练锁时拒绝启动，不终止其他程序。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_readout_recovery_v1.pyz compare
```

compare仅在恢复实验及基线均完整完成后读取Val450结果。前者必须是新恢复read_aux，后者默认`original_3407_01`。它会记录执行协议差异，不把该续训冒充未间断历史轨迹。原始compare_runs.py严格要求协议完全相同，所以不要用它隐去同步诊断差异。

## 完成与失败

- 成功必须同时有phase=complete、last_complete_step=2500、training_complete=true、full_validation_evidence=true，并可核对Val450报告。
- 局部四步通过并不证明历史GPU错误已修复；historical_failure_resolved和exact_historical_replay始终为false。
- 如果再次失败，停止本次进程，保留summary.json、traceback.txt、recovery_markers.jsonl、diagnostic_step_*/failure.json。不会自动重试、跳过图片、修补索引或宣称训练完整。
- 不要删除父运行或其checkpoint800。本包没有自动清理磁盘或修改旧complete标志。
- 检查点保存正常模型/优化器状态，会使用磁盘；不保存每步全图预测、attention或大规模重复GPU快照。
- 已有日志与checkpoint分别记录hash；旧checkpoint没有journal-prefix签名，不把后验顺序核对称为历史日志与权重的密码学绑定。

## 验证范围

本地CPU测试验证恢复、错误输入拒绝、数据顺序、源文件只读、明确的协议差异，以及诊断模式与原update在CPU样例上的loss/梯度/AdamW结果一致。没有连接用户RTX4090运行；不保证该环境的间歇性索引/显存异常已修复。
