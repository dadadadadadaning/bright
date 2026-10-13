# BRIGHT 联合训练池化越界诊断包

[下载工具包 ZIP](https://github.com/dadadadadadaning/bright/raw/refs/heads/codex/joint-pool-diagnose-v1/bright_joint_pool_diagnose_v1.zip)

[使用说明](joint_pool_diagnose_v1/README.md)

代码请解压到 `/root/shared-nvme/BRIGHT-master/New` 的新目录。读取旧失败现场，分别重放原池化与连续索引候选，并执行原完整初始验证定位。不会自动启动正式训练。

42 项 CPU 合成测试及独立 CPU CLI 重放通过；真实服务器快照、BRIGHT 与 CUDA 尚需服务器验证。候选通过不等于已找到并解决根因。
