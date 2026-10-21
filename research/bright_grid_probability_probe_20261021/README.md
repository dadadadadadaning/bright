# Grid 概率前向诊断

针对已记录的六分支续训 `evaluate_val_1500` 异常，检查原始 Val 顺序的前 98 张图。
默认候选为 `morocco-earthquake_00000402`。先核对已有逐图日志、1500 步检查点 SHA256
和原程序/来源身份，再调用原冻结前端、支持构建器和原概率校验器。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u bright_grid_probability_probe_v1.pyz
```

可用 `--scope candidate` 仅运行候选图。默认 `--scope prefix` 检查前缀。
诊断只写一个新建的 `SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/grid_probability_*` 目录。
不修改原文件，不训练、不创建优化器、不计算训练损失、不改变概率或放宽校验、不访问 Test。
检查点只核验字节身份：不会加载六个训练头或优化器，也不声称重放此前1500步的执行状态。

输出 `summary.json` 和 `trace.jsonl`，记录支持构建前后概率的形状、设备、精度、
有限值范围、越界数量及幅度、NaN/Inf计数、归一化偏差和原校验器的实际错误。
第一处无效概率即停止继续读取图片；不会跳过它以生成完整评估指标。

`invalid_grid_probability_observed` 表示诊断捕获到问题；
`frozen_forward_passed` 只表示本次冻结前向通过，不能将旧任务改成完成，也不能据此
声称历史故障修复。如果通过，后续应使用保存的1500步状态执行带详细诊断的完整评估。
`diagnostic_failed` 表示探针自身运行失败，查看新目录中的 traceback.txt。

代码复用服务器当前运行目录中已固定的 trainer.pyz 的只读预检助手，不调用 start/worker。
原始代码或运行身份发生变化时会停止；不会自动修改哈希清单以绕过来源检查。
