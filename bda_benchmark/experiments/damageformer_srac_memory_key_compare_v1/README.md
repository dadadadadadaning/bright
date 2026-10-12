# 同成员 T128/F256 检索诊断与 grid_probq 可拒绝纠错

这是独立新增实验。只新增本目录，原 A1、Q、Embedding、旧库、旧投影、J1/J2/J3 和历史结果均保持只读。默认服务器来源仍由已通过的 **fresh E0 legacy/native/all-path Val450** 绑定；不会恢复或伪造丢失的旧 protocol/SHA。

## 阶段一：先验证检索

`build_keys.py` 保留原 bank 每一行、原顺序、T128、四类软标签、事件/图像/区域身份，额外提取同一冻结 Q 下 `mean_Q(F256)`。只前向原支持成员所在的 Train 图像，正式输出包含所有原支持成员；不宣称重新扫描了全部 Train2918。两种支持特征采用相同原始存储精度，查询均用 FP32。

提取时核验同一次 A1 原生分类器输出；检查重算 T128 在原存储精度下与旧支持匹配、Q 区域标签与原支持标签匹配。均值使用既有分组 gather/sum 算子；原前端的 native 运算保持原样。已有检查容差不放宽，没有失败后切 CPU、跳图或换权重的行为。原支持图片的 pre/post/target 内容哈希在提取前后核验，并在后续加载、诊断和纠错前后核验。

`diagnose.py` 在一次前端计算中重放旧五路径，再比较：

| 新诊断路径 | 检索键 | 支持策略 |
| --- | --- | --- |
| t128_global | 原 T128 | 全局 Top8 |
| f256_global | Q 均值 F256 | 全局 Top8 |
| t128_per_class | 原 T128 | 四类各2条 |
| f256_per_class | Q 均值 F256 | 四类各2条 |

这四条均为参数为零的余弦检索，不把旧 J3 的128维投影混用于F256。两种键共享支持有效性掩码、查询有效性掩码及同事件/同图排除规则。并列排名按原库行号稳定排序。类别分组只读支持库的 Train 标签，不读取查询真值，也不按查询预测只检索某一类。缺少某类支持保留空槽，不用其他类补满配额。

保存 Top8/Top64、相似度、四类软标签投票、来源重复/覆盖、软标签相容性和逐图预测。分类配额的 Top64 为每类16条。标签投票的回投是**未训练诊断**：仅对有支持的 Q 质量归一化，没有支持的像素回到 grid_probq；它不是训练好的 Reader，也不是校准类别后验。GT只在预测与检索完成之后用于评分和区域混合分层。

## 阶段二：诊断之后才训练纠错

`train_correction.py` 正式运行必须显式绑定已通过的本阶段诊断，核验同一 protocol、bank、源码及 K/温度/策略。比较三个私人小头：

| 组 | 输入 |
| --- | --- |
| query_only | F256 + grid_probq；10维历史证据置零 |
| t128_memory | 相同输入 + T128检索的4类证据/6维质量 |
| f256_memory | 相同输入 + F256检索的4类证据/6维质量 |

每组都是270通道输入、1×1隐藏层、4通道delta和1通道gate。同结构、同参数量、同初始化、同Train图序、同5000更新预算。Query-only保留相同F256支持覆盖掩码以控制作用范围。训练不更新前端、Q、区域T128、库、投影或旧Reader。三头共享每步的前端计算；不是分别前向主干15000次。

输出为基础概率的有界logit修正：`p_new = normalize(p_base * exp(coverage * sigmoid(gate) * 2*tanh(delta)))`。末层零初始化，初始预测精确等于 grid_probq，同时保留有效delta梯度。空支持和force-off直接返回原概率张量。门控可下调错误证据，但不承诺训练后提分或每张图无伤害。输入不含查询GT。

正式fit先完整初始Val450，随后Train2918全图、无增强、FP32、batch1、各头5000步，最后完整Val450。只保存同一步 `heads_last.pth`，没有按Test或不同最佳步数选头。独立验证增加正常/随机支持身份/支持标签打乱/关闭支持对照。随机身份对照保留正常的相似度权重和有效槽位；它不是重新计算随机近邻余弦的另一种随机实验。所有干预复用同一个已训练头。

正式指标固定为 Intact/Damaged/Destroyed 三类IoU平均。完整Val450还必须通过原五路径的历史和fresh E0复现（绝对1e-5）、同次原生A1检查、来源/运行配置/文件/权重完整性检查。任何smoke、失败、零测试或中间训练loss都不能当正式结果。报告完整通过只是验证有效，不等于已经证明方法增益。

## 上传和安全安装

代码包名为 `bright_memory_key_compare_v1.zip`。先把实际代码包通过SFTP上传到：

```text
/root/shared-nvme/bright_memory_key_compare_v1.zip
```

包内有校验和清单和非交互安装器。安装器只创建本实验目录，不覆盖原工程其他文件。已有同目录且内容完全一致时只检查；内容不同则退出并保留现有文件。使用随代码包给出的外部SHA256先验证ZIP，然后执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -c 'import zipfile; exec(compile(zipfile.ZipFile("/root/shared-nvme/bright_memory_key_compare_v1.zip").read("INSTALL_SERVER.py"), "INSTALL_SERVER.py", "exec"))'
```

## 第一轮服务器命令

请在现有GPU队列完成后运行真实数据阶段，不要重复启动或修改正在运行的代码。CPU preflight可以先运行。下面不使用read交互，也不依赖旧DIAG_OUTPUT。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export KEY_EXP=bda_benchmark/experiments/damageformer_srac_memory_key_compare_v1

bash "$KEY_EXP/run.sh" preflight
```

真实运行时，首次设置一个新根目录，此后的所有命令在同一shell保留该变量：

```bash
export KEY_OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_KEY_COMPARE_V1/run_$(date +%Y%m%d_%H%M%S)_$$"
unset E0_PROTOCOL

bash "$KEY_EXP/run.sh" build_smoke &&
bash "$KEY_EXP/run.sh" smoke &&
bash "$KEY_EXP/run.sh" report_smoke
```

未显式设置E0_PROTOCOL时，入口只从已知真实启动日志读取唯一 `OUTPUT_ROOT=`：

```text
/root/shared-nvme/bright_region_night_20261013_001703_3671794.log
```

并使用其 `fresh_e0/protocol.json`。找不到或协议未通过就退出，不挑“最新目录”、不使用plan-only目录、不另起夜间队列。若本次fresh E0实际来自另一日志，请非交互地 `export E0_PROTOCOL=/实际根目录/fresh_e0/protocol.json`。路径应来自实际日志，不复制旧丢失协议的示例。

四张支持源图的建库smoke和两张Val诊断只证明短流程可执行。它们不是完整库/完整Val。通过后用同一根目录运行不同子目录：

```bash
bash "$KEY_EXP/run.sh" build &&
bash "$KEY_EXP/run.sh" diagnose &&
bash "$KEY_EXP/run.sh" report
```

先提供 `keys/summary.json`、`diagnose/summary.json` 与 `diagnose/result.json` 分析。观察F256是否提高软标签相容性、Damaged支持覆盖，是否降低Intact被Damaged证据误导。不要只看未训练labelvote的mIoU。逐图NPZ会占用磁盘并增加诊断耗时，先检查 `df -h /root/shared-nvme /dev/shm`。

## 第二阶段命令（诊断分析后使用）

不要与第一阶段或夜间队列并行争用GPU；不自动执行这部分。

```bash
bash "$KEY_EXP/run.sh" train_smoke
```

这是20步和两图Val，正式训练重新初始化：

```bash
bash "$KEY_EXP/run.sh" train &&
bash "$KEY_EXP/run.sh" evaluate
```

独立验证结果在 `correction_independent_val/result.json`，比较 `f256_memory-query_only`、`t128_memory-query_only` 和 `f256_memory-t128_memory`，同时看Damaged救回/伤害、Intact→Damaged误报和逐事件变化。已有Test878不用于选方法。

## 输出与失败处理

```text
KEY_OUTPUT_ROOT/
  keys_smoke/                 仅短流程支持子集
  diagnose_smoke/             非正式两图诊断
  keys/                      manifest.json + paired_bank.pt + 源图哈希
  diagnose/                  summary.json + result.json + images.jsonl + predictions/*.npz
  correction_smoke/          非正式20步头
  correction_fit/            summary/manifest + heads_last.pth + initial_val/ + final_val/
  correction_independent_val/ 独立Val450及支持干预
```

阶段目录存在就拒绝覆盖。失败保留summary、图名、阶段及已有逐图产物。先查失败记录，再选择新的输出目录；没有精确断点续训、跳过失败图或合并旧失败结果的入口。原前端仍含native CUDA运算，本新增包不是历史illegal-memory-access根因修复。

源码与依赖SHA贯穿各阶段，运行中不得同步修改这些Python或shell文件。完整源码更新后旧新实验产物不能混接。

## 验证边界

本地使用PyTorch2.2.1+cpu进行合成单元/集成测试及真实shell入口检查。真实服务器权重、GPU、BRIGHT图片、Train/Val450没有在当前云端运行；最终效果需上述服务器验证。CPU测试验证数学和执行合同，不代表F256优于T128或Memory已提分。
