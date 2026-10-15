# Soft Memory Test V2 FIX1：恢复训练时的完整数值协议

适用于服务器 `soft_memory_q_rebased_v1_fix2` 训练包与已保存的 A/B/C/D 检查点。
这是独立的 Test878 评估工具，不训练、不重建库、不修改原训练包或来源协议。

## 本次修复

旧 Test V2 启动器设置 OMP/MKL 等线程环境为 2，评估器又固定 Torch intra-op=2、interop=1。
Q 来源的严格运行协议包含上述字段，因而在加载前端时发生：

```
ValueError: Current runtime differs from Q training's recorded numerical profile
```

FIX1 从指定检查点 JSON → 训练 manifest → 绑定的 source protocol → E0 元数据读取完整配置，逐层核验哈希。
在导入 Torch/NumPy 前恢复已记录的环境变量（null 表示移除），在加载权重或初始化 CUDA 前恢复线程、精度和确定性设置。
原 `fresh_source.py` 的完整运行配置相等检查保留，不修改原实现，不扩大任何容差。

不能只增加 `--cpu-threads 48`：它不能恢复 interop、OMP/MKL 等其余字段。本版不提供线程覆盖参数。
Python/PyTorch/CUDA、系统与 GPU 身份只能检查，不能伪造；如仍不一致，程序会停止并打印字段差异。
完整 CUDA 配置查询会初始化 CUDA，因此本工具不在原加载器之前进行该查询。

## 手动下载与启动

下载 `RUN_SOFT_MEMORY_TEST_V2_FIX1.py` 到：

```
/root/shared-nvme/BRIGHT-master/New/RUN_SOFT_MEMORY_TEST_V2_FIX1.py
```

它包含经过 SHA256 校验的完整工具，不需要服务器联网下载其他代码。
只启动一次 3000 步：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u /root/shared-nvme/BRIGHT-master/New/RUN_SOFT_MEMORY_TEST_V2_FIX1.py --step 3000
```

启动后读取打印出的 **log 文件路径**；不要对 TOOL_DIR 目录使用 tail。
新目录名前缀为 `test_fix1_step_003000_...`，旧失败日志保留。
默认输出父目录仍为 `SRAC_SOFT_MEMORY_Q_REBASED_TEST_V2`，以共享同一检查点的防重复锁。
不同步数仍可分别启动，因此不要重复启动 2500 与 3000 来确认同一个修复。

可先作 CPU 来源检查（不读取 Test，不执行真实 CUDA 前向）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u /root/shared-nvme/BRIGHT-master/New/RUN_SOFT_MEMORY_TEST_V2_FIX1.py --step 3000 --plan
```

选项：`--step`、`--train-root`、`--tool`、`--output-parent`、`--workers`（默认1）。
步骤 5000 使用 `checkpoint_last.pt`，其他明确指定步骤使用 `checkpoint_XXXXXX.pt`。
指定权重不存在时停止，不自动改用其他步骤。

## 如何读结果

- `NUMERIC_PROFILE_RESTORE`：准备恢复的完整线程与环境；尚未证明真实 CUDA 通过。
- `NUMERIC_RUNTIME_COMPARISON []`：原加载器完整数值协议检查通过；仍需继续完成推理与终检。
- `numeric_runtime_restoration.json`：绑定的配置与来源哈希。
- `numeric_runtime_comparison.json`：预期、实际运行配置和逐字段差异；原门槛失败时也记录。
- `progress.json` / `images.jsonl`：已完成的图像数及逐图整数混淆矩阵。
- `result.json` / `metrics.csv`：878 张完成、来源及头状态终检、统计回读通过之后的五路结果。
- `summary.json` 的 `complete=true` / `full_test_evidence=true` 只表示本次 Test 通过，不表示5000步训练完成。

报告同时提供前景三类 mIoU 和四类 mIoU，必须区分口径；所有结果均为指定中间检查点的 exploratory Test。
查看 Test 后选择最佳步骤属于探索性模型选择，不能再把同一 Test 称作未参与选择的最终盲测。

A：固定空间双槽＋图内参考；B：建筑概率 soft 双槽＋图内参考；
C：固定空间双槽＋跨事件 Train Memory；D：建筑概率 soft 双槽＋跨事件 Train Memory。
同时输出相同前向的 `grid_probq`，不将 Test 与既有 Val 数字混作增益。

测试仍使用原冻结前端、真实四组头和原检索实现。Test 查询真值只用于计分，历史支持标签来自 Train。
训练元数据和已保存权重须保留；不能删除、重建协议或关闭校验来通过检查。
本工具只保存计数与报告，不保存大体积逐像素预测数组。

## 验证边界

CPU 回归覆盖检查点绑定、原分类头推理、指标统计、完整线程环境恢复、来源篡改拒绝和 CUDA 初始化顺序。
具体测试结果见 `CPU_VALIDATION.json`。当前云工作区没有服务器4090、真实权重和BRIGHT数据；实际 Test878 尚待服务器运行。
