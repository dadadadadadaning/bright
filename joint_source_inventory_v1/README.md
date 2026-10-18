# 连续训练缺来源：只读盘点

`inspect_surviving_sources.py` 是独立的标准库脚本。无需旧运行的 summary、500步权重、V3库或`.source.json`即可运行。扫描 BRIGHT-checkpoints 和 BRIGHT-master/New，列出正式Q候选、其他权重（含A1候选）、来源/数据划分元数据及案例库候选。旧chain、训练器、模型和校验规则均不修改。

只在New新建小型REPORT.txt与inventory.json；不加载权重、不使用CUDA、不读图像、不训练、不恢复、不删除文件。文件存在或summary声明完成均不是完整来源验证。脚本不跟随目录符号链接，排除PFLlib/PFLlip和环境、数据目录。没有扫描到不代表所有本地备份都不存在。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/inspect_surviving_sources.py
```

终端输出可直接提供分析，或提供末尾REPORT路径的文本。JSON保留所有候选，终端每类最多40条。不要仅凭同名文件将不同检查点改名冒充原来源；本脚本不自动选择Q或A1。

下一步需要区分：旧成员身份可用时正规恢复；成员身份消失但A1/Q/数据可用时，需要新的Train成员选择与自包含来源模式（现有chain_v1尚不支持）；必要基础权重也消失时，需要真实备份或另外建立训练起点。本工具不会宣称已经实现新成员训练模式。

元数据仅识别名称含source或protocol的JSON，不解包备份。常见A1文件名单列，其余权重按大小列出。名称只用于发现候选，不能代替哈希配对。

验证：7项临时文件测试涵盖缺协议的独立发现、smoke排除、坏JSON、备份引用、符号链接、缺根目录和A1展示。未进行远程BRIGHT数据验证。

## 找到Q和A1、但旧E0及成员库消失之后

`verify_surviving_q.py` 用于下一层CPU核查，直接读取本机受信任的Q检查点，不加载大A1模型，仅计算A1文件哈希。它复用固定fix2 `fresh_source.py` 中的正式Q校验函数，明确使用仍保存的训练期末步Val450证据。独立Val的summary存在而result缺失，不会被当作完整独立验证，也不修改旧报告。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/verify_surviving_q.py
```

无需旧E0协议、旧联合头或旧V3库。自动在BRIGHT的bda_benchmark目录内查找train_set.txt、val_set.txt，按Q中记录的SHA256选择原名单，检查2918/450的唯一性和划分不重叠。寻找含pre-event/post-event/target的数据目录，仅检查所列Train/Val文件是否存在，不打开影像，不读取Test名单或Test像素。

若数据位于自动搜索范围之外，可显式传 `--dataset`、`--train-list`、`--val-list`；名单仍须通过原哈希校验。脚本不跟随符号目录自动搜索。结果写入新的 `New/surviving_q_verify_*/REPORT.txt` 与 `result.json`。运行只会影响本次CPU检查进程及新报告目录，不修改原模型、库、代码、来源或运行状态。

`cpu_foundation_verified=true` 仅表示本工具覆盖的基础核查通过。它不是严格模型装载、原E0复现、真实CUDA前向或训练完成证明。架构参数完整性仍需后续严格装载；像素内容及数据质量仍需真实数据检查。`source_ready_for_training` 始终为false，避免把文件检查说成新训练入口已实现。

后续需要单独实现的新来源适配层应记录现存Q/A1、新数据来源、新选Train成员与代码指纹，复用原双槽、Reader、训练目标、配对引擎和统计函数。不能把新来源写成旧E0格式，不能把新成员写成原V3选择，也不能取消原来源检查后直接调用chain_v1。

CPU验证器的14项模拟文件/检查点测试通过，包括实际调用原正式Q验证器的缺E0、缺独立result情形。只将仓库依赖/代码哈希代理为测试对象，正式校验函数未替换；受信本地Q使用CPU装载。加上初始盘点脚本的7项测试共21项。以上不等于远端实际文件或真实CUDA验证已经通过。
