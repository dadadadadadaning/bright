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
