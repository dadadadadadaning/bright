# 恢复双槽／Memory诊断依赖的原Q来源文件

用于处理：CPU72项检查通过，但真实smoke加载冻结Q来源时提示原 `formal/train_baseline/last.pth` 不存在。

这是文件恢复工具，不是训练器，不包含模型权重，不更改Q、案例库或任何损失。仅使用Python标准库；不导入Torch，不使用CUDA。

## 为什么已有500步分类头还不够

原四组检查点保存联合分类头，前向仍需要当时被冻结的DamageFormer/A1及Q-baseline。来源协议固定了它们的路径和文件SHA256。即使另一份权重也叫 `last.pth`，也不能直接替代。

程序读取已完成训练的 `train/input_fingerprints.json`，先验证其SHA与 `train/summary.json`一致，再检查来源协议与原检查点侧记、训练manifest的一致性。该输入清单给出原Q权重及报告的字节指纹。

## 检查范围

Q-baseline至少需要：

- `formal/train_baseline/last.pth`
- `formal/train_baseline/summary.json`
- `formal/train_baseline/manifest.json`
- 来源为训练末步验证时：同目录 `validation.jsonl`
- 来源为独立验证时：`formal/val_baseline/summary.json` 和 `result.json`

还会检查A1及E0元数据来源是否存在且匹配。工具仅自动恢复Q-baseline这一组文件；A1或E0文件缺失时会报告阻塞。恢复完成并不等于全部数据集、子槽库及CUDA前向已通过；后续诊断Runtime保留其完整检查。

## 如何运行

只查找，不复制：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/restore_slot_audit_source_v1.py
```

查找并恢复相同SHA的缺失文件：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/restore_slot_audit_source_v1.py --restore
```

默认扫描自己的 `BRIGHT-checkpoints`，以及 `BRIGHT-master/New/source_restore_inputs`。优先查看历史上出现过的 `restored_20261014_134825_xno2hi6o` 目录。不会扫描PFLlib或Conda环境，不删除任何文件，不改写来源协议。

程序只有在所有需要恢复的文件均找到相同SHA副本、所有现有依赖均通过检查后才开始复制。目标已存在但内容不同会停止；目标软链接也不作为恢复结果接受。不能用软链接代替复制，因为原加载器会解析真实路径，可能改变来源身份。

若返回缺少备份，需从本地备份上传原始文件到 `New/source_restore_inputs` 后重试。当前工具扫描普通文件，不解压ZIP；请保留备份目录结构上传或先解压至该目录。工具不会重建或伪造manifest、summary或权重。

恢复成功后启动已有修复版诊断：

```bash
bash /root/shared-nvme/BRIGHT-master/New/slot_memory_audit_v1_fix1/start_server.sh full
```

该操作创建新的诊断结果目录，旧失败记录保留。它不是重新训练。

## 验证

恢复行为在临时文件夹进行合成文件测试，包括相同SHA恢复、不同SHA拒绝、已有目标保护、清单校验、缺失依赖、重复运行和写入失败处理。并未在用户远程服务器执行恢复。
