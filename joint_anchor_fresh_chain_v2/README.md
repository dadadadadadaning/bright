# 新来源联合分类配对实验（V2）

## Fix 1

修复启动器 `_run_stage` 内重复导入 `SimpleNamespace` 导致 `values_smoke` 在调用真实双槽建库前出现 `UnboundLocalError`。本修复不改变 Q、A1、双槽、Memory、损失函数或实验预算。旧失败目录保留为失败记录；重新启动会生成新的运行目录。已加入该阶段的回归测试。

这是独立的新实验。现存 Q 正式检查点、A1、Train2918/Val450 名单已由此前 CPU 核查确认；原 V3 成员库和 E0 协议已删除。本工具重新从 Train 选择成员，并使用固定 Q、原 soft 双槽与原 Reader 做配对训练，之后统计双槽与案例质量。它不恢复旧轨迹，不读 Test。

流程：来源核验 → 四事件小库 → 双槽小库 → 真实 CUDA 配对 smoke → 全 Train 扫描、按事件/标签有界选最多 3411 个新成员 → 双槽案例重建 → D 共同前 500 步 → 从相同头、优化器和 RNG 派生原目标与原目标+0.1 KL，两组第 501–1500 步均为原调度的 0.2 倍学习率 → 完整 Val450 → 500/1500 步双槽及历史案例诊断。

与旧结果不能直接比较：新成员身份和数值配置已改变。真实数值只能读取完成且终检通过的 Val450 报告。500 步及 1500 步提前设定为观察点。KL 使用同次前向的冻结 grid_probq，包含四类有效像素；没有“采纳/拒绝 Memory”的门控。

训练工具源码在 `deps/original` 与 `deps/study` 中原样复用；新的文件负责来源、成员身份与流程接线。`deps/audit` 复用之前的区域、检索和像素统计。输出同时记录原输入文件和本次工具哈希，未提供原 E0/成员库复现声明。

服务器代码路径：`/root/shared-nvme/BRIGHT-master/New/joint_anchor_fresh_chain_v2`。运行 `bash start_server.sh` 启动后台流水线；它不会等待空闲 GPU，也不会停止其他进程。结果目录的 `workflow_state.json`、`train/summary.json` 和两个 `audit_*/result.json` 为结束证据。磁盘吃紧时保留来源、成员、案例值和训练检查点，才能复验；不涉及 PFLlib。
