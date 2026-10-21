# 六分支 checkpoint1500 独立 Val450 评估

用途：恢复已有第 1500 步的六分支 checkpoint，在新目录中完成 Grid + E/F/B/D/R/L 的 Val450 评估。保存的模型、AdamW 状态和 RNG 均通过原恢复程序校验，但执行零次优化更新。本入口没有训练或续训命令。

## 适用的已有运行

默认源目录：

`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261020_161047_4184092`

要求：原运行仍记录 `evaluate_val_1500` 失败；旧验证日志的有序前缀之后为 `morocco-earthquake_00000402`；checkpoint1500 的字节摘要与 sidecar 和原摘要一致；源运行未运行或被其他恢复进程锁定。

原 trainer 必须为 SHA256：

`0f7fd0a85ca2db5727509875c84c1beba1011d2bd4300a8b6b5ea86b59e75f07`

这些要求有意限定此工具的适用范围。其他实验不应通过改写原记录来满足它们。

## 使用

校验下载包的 SHA256 后执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /tmp/bright_checkpoint_val450_v1.pyz
```

启动器在 `BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/eval1500_*` 创建独立目录，将自身复制为 `evaluator.pyz` 后启动后台进程。标准输出会给出 PID、输出目录、完整的 `tail` 和 `cat` 命令。请只启动一次；排他锁会拒绝并发的本工具评估。

无需预设 `BRIGHT_RUN`、`BRIGHT_PY` 或 `BRIGHT_RECOVERY_LOG`。后台进程继承当前已激活环境，采用原记录的 CUDA 设置。完整的执行协议检查仍由原 runtime 负责。

## 保持的评价协议

- 原来的冻结前端、Memory、候选构造、六个分类头和 Val 顺序。
- 原概率校验器及其容差；原 NLL + Lovasz；原逐图、逐事件混淆矩阵和配对计数。
- 原数据划分与输入身份检查。评估实际使用的文件重新核验；未在本次评估使用的历史 Train 原图不宣称已重新核验。
- Test 不读取，不选择最佳 checkpoint，不进行反向传播或 optimizer step。

新增观察器只记录图像和路径阶段，先执行原校验器。若原校验器抛出 `ValueError`，保存 CPU 概率样本及分类后的异常统计并重新抛出；CUDA/运行时异常后不主动继续读取 GPU 张量。原评估器本身的错误清理保持不变。

## 输出与结论边界

`summary.json` 记录恢复、检查和最终状态；`evaluate.log` 记录进度；`markers.jsonl` 记录每张图的校验阶段。若概率校验失败，查看 `probability_failure.json` 和可能生成的 `probability_failure.pt`；其他失败查看 `traceback.txt`。

只有全部 Val450 完成，原报告/逐图日志验证、模型和优化器未改动、RNG 恢复、冻结 runtime 检查、PoolGuard 退出和源文件检查均通过后，新 `val_step_001500/result.json` 才标记 `standalone_evaluation_complete`，并保存完成前版本 `result.pending.json`。全局四类 mIoU、前景 mIoU 和各类 IoU 以百分数列在摘要中；完整混淆矩阵和事件统计在结果中。

此时仅说明“该保存 checkpoint 的一次独立 Val450 评估完成”，不说明旧训练运行已完成至 2500 步，不说明历史异常根因已消除，也不是历史进程状态的完全重放。旧目录、失败摘要和旧结果保持原样。

## 本地验证

见 `EVALUATION_VALIDATION.md`。CPU 和合成测试只能检验实现契约，不能代替真实 BRIGHT/GPU 实验。
