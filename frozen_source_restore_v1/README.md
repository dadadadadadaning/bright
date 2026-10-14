# BRIGHT 冻结来源文件恢复

适用于四组独立训练在 `build_smoke` 加载 E0 时，因清理空间删除旧快照而报 `FileNotFoundError` 的情况。

这次缺失的 `SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint/manifest.json` 是冻结来源的一部分。四组的 independent 模式解除的是等待新 R0/RT/RF 完成的要求，仍需要 E0 原本绑定的旧 J1/J2/J3、前端和旧成员库。新 FP32 配对库没有封装这些全部源文件。

## 恢复办法

1. 在服务器创建 `/root/shared-nvme/BRIGHT-master/New/source_restore_inputs`。
2. 将本地保存的原始 `SRAC_SNAPSHOT_JOINT123_V1` 备份上传到这个目录。可以上传原文件夹、ZIP、TAR、TAR.GZ；目录内的压缩包会被搜索，不要求压缩包顶层目录固定。若还删除了其他 E0 来源文件，也将对应备份放进同一目录。
3. 将本工具 ZIP 上传到 `New`，按发布消息提供的 SHA256 校验后解压到新目录。
4. 执行 `bash "$REPAIR_TOOL/run_server.sh" restore-start`。它先恢复和校验 E0 全部依赖；全通过后调用既有独立四组工具，创建新的 full 运行，自动进行 smoke、全量子库、5000 步与 Val450。

仅查看缺什么：

```bash
bash "$REPAIR_TOOL/run_server.sh" audit
```

仅恢复，不训练：

```bash
bash "$REPAIR_TOOL/run_server.sh" restore
```

备份在其他位置时：

```bash
bash "$REPAIR_TOOL/run_server.sh" restore-start --backup /实际位置/备份.zip
```

多个 `--backup` 可以一起使用。`start` 模式只检查已在原路径的依赖，再启动新的四组运行。

## 工具做什么

`restore_sources.py` 只使用 Python 标准库，读取 E0 `protocol.json` 的 `input_paths` 和 `input_fingerprints`。默认源协议是 `SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json`。

- 对每个原路径计算 SHA256，显示已存在、缺失、内容冲突。
- 从备份查找同名且 SHA256 完全相同的文件，恢复到协议记录的原路径。
- 既有内容不匹配时不覆盖；缺失且没有匹配备份时停止并列出原文件路径。
- 不使用 `extractall`；ZIP/TAR 只读取匹配的普通文件内容，不照搬档案路径。
- 安装前再检查内容；优先临时文件加 hardlink 的无覆盖安装，文件系统不支持时使用独占创建并复验。此回退不宣称原子安装。
- 保留 E0 协议字节、旧报告、原模型文件和训练源码；不生成替代权重或猜造 manifest。

E0 snapshot ledger 通常有14项：前端、A1、旧 cache manifest/bank、metric、Train/Val 列表、J1/J2/J3 权重、joint manifest/summary/validation、snapshot.json。只恢复 ledger 中确实缺失的文件，不必整包解压旧 cache shards。

若本地备份也缺少对应原件，工具会给出缺失清单并停止。仅有报告不能重建模型权重；仅有新版 FP32 key 库也不能代替冻结前端。

## 状态解释

恢复报告保存于 `New/source_restore_reports/run_.../report.json`。依赖检查通过表示 E0 ledger 文件与历史 SHA256 一致，不表示真实 CUDA 前向、数据集完整性、四组训练或 Val450 已通过。原训练工具仍负责 V3、源码、图像、冻结状态、实际 smoke 与最终验证。

新启动的结果仍在 `SRAC_SOFT_MEMORY_JOINT_INDEPENDENT_V1` 的新目录。原失败运行 `run_20261014_182835_full_4lSwuY5D` 不续训、不改成成功。

本工具不修复 pooling 数值故障；它修复的是来源文件删除后不可访问的问题。恢复成功后不需要重训旧快照实验。

## 测试

合成文件测试覆盖目录/ZIP/TAR 恢复、错误 SHA 拒绝、已有文件保护、路径与符号链接、报告保护；不涉及真实 BRIGHT 数据或 CUDA。最终测试结果见 `CPU_VALIDATION.json`。
