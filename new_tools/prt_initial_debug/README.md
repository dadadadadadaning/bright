# P/R 初始概率差异：固定输入诊断

本工具用于排查既有实验的 P/R 初始概率一致性失败。它不会训练、修改原工程、替换权重、生成 Test 成绩或降低任何检查门槛。

已收到的正式失败证据：历史五路径和 fresh E0 复现通过，原生 A1、权重、代码、运行时、Q 检查通过；P 有 5 张初始概率不一致，R 有 4 张，T 完全一致。卷积 TF32 允许，矩阵乘 TF32 禁止。TF32 目前是待检验的原因，不能提前判定。

默认诊断两张图片：原报告中 P 差异最大的一张，以及 R 差异最大的一张，去重。当前对应 `mexico-hurricane_00000208` 与 `ukraine-conflict_00000356`。

## 运行

`prt_initial_debug/` 位于新离线包内，入口按脚本自身位置定位，不需要网络获取工具。原工程仍为 `/root/shared-nvme/BRIGHT-master`，诊断结果仍写到其外的 `BRIGHT-checkpoints/`。

先做 CPU 测试和 CLI 检查，不加载模型或读取 Train/Val：

```bash
PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python bash /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/prt_initial_debug/run.sh --preflight
```

只查看路径和计划、不运行 GPU：

```bash
bash /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/prt_initial_debug/run.sh --plan-only
```

计划模式检查工具校验和、原工程和失败报告文件是否存在；实际诊断仍需通过原协议和 source hash 门禁。执行诊断：

```bash
bash /root/shared-nvme/BRIGHT-master/New/BRIGHT-new-tools_v1/prt_initial_debug/run.sh
```

脚本先检查 GPU 0 是否存在计算进程，忙时停止而不终止其他任务。输出目录位于 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_INIT_DIAGNOSTIC_V1`，每次使用新目录；不要预先创建具体的 run 目录。

详细报告是新输出目录的 `result.json`。脚本末尾自动打印需要判读的摘要；把末尾 JSON 文字或这份报告提供给下一轮分析。也可以独立执行 `python summarize.py /实际输出目录/result.json`，它只读取 JSON，不加载模型或使用 GPU。

需要指定其他经过验证的失败来源时，可以显式设置：

```bash
FAILED_RUN=/实际失败目录 OUTPUT_PARENT=/新的输出父目录 bash /实际工具目录/run.sh
```

默认不会接着启动夜间训练。错误时保存诊断记录并停止，不在损坏的 CUDA 上下文内重试。

默认 `FAILED_RUN` 是 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34`。它是历史结果来源，不是工具安装目录，不能替换为 `New/`。如果历史结果也搬走，需通过上述 `FAILED_RUN` 显式指定并重新通过已有身份绑定检查；不改写旧协议、manifest 或 hash。

## 检查内容

1. 读取原失败报告及通过的 fresh E0，验证绑定的输入与源码。
2. 在原 E0 的运行时设置下执行冻结前端和原五路径顺序。
3. 捕获原 `feature_branches` 本次真正使用的区域均值、分类分数，以及候选 P/R 输入；不把额外重算冒充原始输出。
4. 固定这些张量，比较原分类头、私有副本及函数调用的分数、softmax 和回填概率；记录布局、数值差和输入变化。
5. 可选精度对照只在固定分类输入上执行，例如临时关闭卷积 TF32；不会使用改变后的设置重算整套 A1。每个对照后恢复设置。

诊断为 `diagnostic_only=true`、`training=false`、`test_accessed=false`、`full_validation_evidence=false`。即使本次未复现原失败，也不能宣布修复，更不能替代完整 Val450。

## 判读

- 原与私有头的低分辨率分数首先不同：检查分类算子与布局。
- R 的两份区域均值首先不同：先排查区域汇总输入及计算过程。
- 固定分数相同而后续概率不同：检查 softmax、回填、插值与归一化。
- 单因素精度对照使差异缩小：它支持相关数值路径值得排查，不自动证明 TF32 是唯一根因。

修正实现或正式计算策略后仍须重新验证匹配的初始化与完整 Val450；不得直接冒用本次诊断生成正式成绩。
