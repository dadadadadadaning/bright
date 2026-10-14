# Q-baseline 来源的双槽区域与 Memory 四组实验

修订版 fix2：修复真实 Q 前端包含 0 维 Long 缓冲区时的权重哈希错误。上一版的 135 项预检通过，但在 `parent_smoke` 首次校验真实模型状态时停止；本版新增标量缓冲区回归测试。先前失败运行目录保留，请用此包创建新运行。

修订版 fix1：修正预检 CPU 测试对云工作区路径的假设。原包在服务器上误找 `BRIGHT-master/bright`，因此在建库及 CUDA smoke 前停止；本版使用启动时指定的 BRIGHT 仓库路径。既有失败运行目录保留，本版须创建新的运行目录。

此包在服务器上启动一轮**新的** A／B／C／D 匹配实验。它从已完成的 `formal/train_baseline/last.pth` 和 A1 权重重建冻结前端。旧 E0 协议只用于识别数据集与 Train2918／Val450 列表；已删除的 E0 前端、J1／J2／J3 和旧 Memory 权重不会被伪造、替换或宣称重新校验。上一轮 R0／RT／RF 与本轮前端不同，不能作为同源单槽对照。

本轮四组共用同一个 Q、同一批经当前 Q 有效性检查的 Train 案例、像素特征、父区域特征、`grid_probq`、分类头容量、初始化、训练顺序和 5000 步预算。它们只改变两件事：区域内如何分槽，以及 Reader 参考图内区域还是 Train 历史案例。

| 组别 | 两槽组织 | Reader 参考 |
| --- | --- | --- |
| A | 固定空间规则 | 当前图像的空间近邻区域 |
| B | 建筑概率 soft：`Q×B` 与 `Q×(1−B)` | 当前图像的空间近邻区域 |
| C | 固定空间规则 | 跨事件 Train 案例 |
| D | 建筑概率 soft | 跨事件 Train 案例 |

默认由 T128 查找 Top64 个父案例。64 个父案例各有两槽，最多 128 个槽，仍是 64 个案例。Memory 案例的特征与 Train 软标签都按相同槽权重汇总；图内参考的标签使用模型预测，不使用查询真值。背景槽不被强制判成背景。像素与区域信息一直保留，由联合分类器共同判断四类损毁，没有 Memory 采纳／拒绝阈值。

## 服务器输入与新库

默认读取：

```text
Python       /root/shared-nvme/conda-envs/bright4090/bin/python
代码         /root/shared-nvme/BRIGHT-master
Q 检查点     /root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1/run_20261014_003226_vdy7ey5t/sar_q/formal/train_baseline/last.pth
A1 权重      /root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
数据元信息   /root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json
成员来源     /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665/keys
结果根目录   /root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1
```

Q 检查点旁还需正式训练 `summary.json`、`manifest.json`、`validation.jsonl`；若已有独立 `formal/val_baseline/{summary.json,result.json}`，也会作为更强的 Val 证据检查。V3 `keys` 目录需保留 `paired_bank.pt`、`manifest.json`、`summary.json`、`protocol.json`、`images.jsonl`。V3 旧向量只用于验证成员来源，**不会作为新 Q 的案例特征**。若这些文件实际已经删除，脚本会停止并指出缺失路径，不会生成假成员。

新程序对旧候选成员用冻结 Q 重新提取 FP32 T128、F256 和同权重 Train 软标签。当前 Q 下无效的候选会从新库排除，报告保留原成员 ID、原因和新旧 ID 对应。正式建库检查保留比例、跨事件来源及 Damaged／Destroyed 标签覆盖；四组使用相同的新库。随后为两种分槽方式重新提取案例内容。查询端与支持端始终使用同一 Q 和分槽规则。

## 启动

将发布 ZIP 放进服务器的 `/root/shared-nvme/BRIGHT-master/New/`，按照发布时给出的 SHA256 检查，再解压到**新目录**。进入工具目录后运行：

```bash
bash /root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2/start_server.sh check
```

`check` 在后台完成来源预检、四张跨事件 Train 来源图像的新父库、对应双槽案例、四组各 20 步真实 CUDA smoke 和两张 Val 检查。它不会启动 5000 步正式训练。检查通过后，直接运行：

```bash
bash /root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2/start_server.sh full
```

`full` **自己先重新做 smoke**，通过后自动建立完整父库和双槽内容，四组各训练 5000 次完整图像更新，并从序列化末步权重执行 Val450。两条命令都立即返回 PID、日志与结果目录；后台进程创建不等于实验完成。查看示例：

```bash
tail -n 60 -f /root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_具体名称.log
```

脚本默认不会等待 GPU 空闲，不会停止现有任务，不覆盖旧实验。已知 native pooling 越界没有在此包中放宽或绕过；如果发生，流程会停下并保存日志。请勿把失败的 smoke 当作方法效果。

默认路径可用 `BRIGHT_Q_CKPT`、`BRIGHT_A1_CKPT`、`BRIGHT_E0_METADATA`、`BRIGHT_MEMBERSHIP`、`BRIGHT_RESULT_BASE`、`BRIGHT_DEVICE` 等环境变量覆盖。代码仅需解压到 `New`，无需修改原仓库模块。程序仍依赖服务器现有 `bda_benchmark` 实验模块和数据集。

## 报告解读

结果目录中的 `workflow_state.json` 显示阶段与错误；`parent*/summary.json` 显示成员保留和重建状态；`values*/summary.json` 显示双槽建库；`smoke/summary.json` 是小样本运行；`train/summary.json` 与 `train/final_val/result.json` 才是正式训练和末步 Val450 证据。也可运行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2/run.py report --root /实际结果目录
```

比较 B−A 看 soft 相对固定空间双槽；D−C 看使用 Memory 时 soft 的额外作用；C−A 与 D−B 看相同分槽下 Train 案例的增量。四组都输出 Val 前景三类 mIoU、每类 IoU、逐事件及救回／误伤统计。只有终检、保存权重和预测回读都通过，才能作为正式成绩。训练中的中间 Val、smoke 分数和 Test 四类 mIoU 不与正式 Val 前景 mIoU 混用。此轮不声称历史 E0 重放、Memory 已涨分或 soft＋Memory 一定提高。

包内 CPU 合约测试仅验证接口和计算约束；本云工作区无法替代服务器 4090 的真实 CUDA／BRIGHT smoke。
