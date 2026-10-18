# 从 evidence_collect 恢复原报告

此工具解决原 ABCD `train/summary.json` 等报告被删除、但 `evidence_collect_*/snapshots` 中仍保留原件的情况。只依赖 Python 标准库。

```bash
python restore_reports.py --restore
```

默认备份和原实验路径已填写当前服务器的 `evidence_collect_20261016_032748_d1wmpcny` 与 `run_20261014_202057_full_fEYBZv7D`。不带 `--restore` 仅预览。

校验 inventory 中的原路径、快照相对路径、复制状态、采集时文件是否变化、字节数和 SHA256。只恢复固定白名单中的原报告字节，要求 summary/workflow 已正式完成；全部候选先检查再复制。同名原文件存在且不同则停止，绝不覆盖，也不制造新实验的 complete 状态。

可恢复：原 summary、manifest、workflow、外置 source 协议、所选500/1500/5000步已有Val报告（默认500步）。报告缺失于备份会列出。不会恢复训练日志尾部，也不会把日志尾部当作完整日志。

检查并列出缺失的500步权重、权重旁文件、input_fingerprints、协议等。**收集包不包含模型权重、input_fingerprints 或案例库**。exit 3 表示报告处理结束但关键文件仍缺失；exit 2 是核验/复制错误；exit 0 仅表示列出的关键文件存在，不代表Q依赖、模型及案例库校验通过。后续必须运行原 source_restore 和 audit 校验。

不训练、不加载权重、不使用CUDA、不访问Test、不修改原JSON字段。若原权重全部丢失，应另建目录训练，不能将重训结果标成原500步模型。

测试：`python -m unittest discover -s . -p 'test_*.py' -v`。临时文件测试不代表真实服务器恢复成功。
