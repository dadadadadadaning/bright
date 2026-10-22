# BRIGHT：读取监督改进、匹配消融与验证说明

这是独立的新实验包。它实现一个明确假设：历史候选包含损毁信息，但最终分类监督未必能使 Reader 读出正确的类别信息；显式标签读取辅助监督可能改善这种情况。**没有新精度结果，也没有在用户 RTX4090／BRIGHT 数据上验证过。** CPU 测试只能验证实现契约。

本包不修改旧源码、数据、Memory bank、checkpoint 或报告；不延长旧六分支训练；不访问 Test。旧训练已经达到2500步，新方法从相同种子的新初始化开始，与新控制组比较。

## 1. 文件与实现

| 文件 | 用途 |
|---|---|
| `model_adapter.py` | 原模型兼容子类；保留 state_dict 名称，增加小型读取分布，或关闭指定分支 |
| `supervision.py` | 原路由上的查询 GT 汇总接口、辅助软标签交叉熵、候选覆盖统计 |
| `experiment.py` | 真正的 plan／smoke／train／status 命令；复用已安装的原 runtime |
| `analyze_results.py` | 读取一个完整旧报告，比较其中两分支；纯 CPU 标准库 |
| `compare_runs.py` | 严格对齐两个本包新实验的初始化、划分、配方、逐图日志后比较 |
| `HISTORY_AND_FIGURES.md` | 旧实验用途、区域设计、当前收益解释、论文展示方案 |
| `reported_results.csv`、`make_figures.py`、`figures/` | 根据用户已报告结果制作的图，不冒充重新核验的原始报告 |
| `test_*.py`、`VALIDATION.md` | 实现测试与验证边界 |

模型模式：

| mode | 精确定义 | 能回答什么 |
|---|---|---|
| `original` | 原始 D 结构、D 支持、原 NLL＋Lovasz | 同版本的新控制组 C0 |
| `no_reader` | 从训练起将 memory_context 置零；保留像素／父区域／查询槽／Grid 和融合宽度 | 支持读取整体有没有价值；它不是纯分类头 |
| `no_label` | D 候选与视觉支持不变，仅令 label_embedding 输出为零 | 显式历史标签是否有价值；它不是关闭 Reader |
| `read_aux` | 原 D 前向预测不变，额外返回注意力加权标签分布，训练增加读取监督 | 明确读取目标能否改善最终识别 |
| `observe` | 仅模型 API；返回读取统计但不增加损失 | 固定模型机制诊断；不是 CLI 自动加载历史模型的入口 |

`read_aux --read-weight 0` 使用完全原始路径，不计算辅助 GT 汇总。固定旧权重做临时关闭干预只能测依赖性，不能替代从头训练的 no_reader／no_label。

本包 CLI 没有提供旧 checkpoint 的自动诊断／恢复入口；不要把旧 checkpoint 传给训练命令。已有权重的观察可通过 `make_classifier(base_class, 'observe')`、`load_state_dict(..., strict=True)` 在现有评估代码中接入。复杂的旧实验恢复不属于这次方法改进。

## 2. 第一项改动的数学定义

现有多头注意力 a 对合法支持槽的标签分布 y 加权：

```text
p_read(q,c) = sum_j mean_heads(a(q,j)) * y(j,c)
L_total = L_original + read_weight * L_read
L_read = mean_valid_query_slots[-sum_c target(q,c) * log(p_read(q,c))]
```

- 所有候选、粗检索、路由、原 logits 融合、原损失与 LR 调度保留。
- 查询 GT 只在 forward 结束后用于训练损失。调用原 `slots.slot_labels`，先对有效 one-hot 质量做 area 汇总，再按现有路由池化；ignore255不进入分母。不用最近邻替换原汇总。
- 支持标签在新增读取分支 detach；辅助损失主要经 attention 更新 Q/K 和相关编码。原主损失仍更新原有效路径。
- 有效槽等权平均。背景角色／建筑角色的统计分别记录；本轮不同时加类别重权、门控、对比损失或新候选采样。
- 无合法候选、无有效 GT 质量的槽排除；不因某类候选缺失而悄悄删除该类目标。
- `candidate_class_max` 是各类分别能达到的最大候选标签质量；多个最大值不一定能同时达到，不能称 oracle mIoU。
- `clamp_min(1e-8)` 只用于辅助损失的 log。任何异常路由、索引或概率仍报错，不修补后继续。
- 读取分布更准确而最终 mIoU 不提高，不能称方法有效。

## 3. 验证顺序与停止条件

| 步骤 | 操作 | 通过标准／不通过时的处理 |
|---|---|---|
| A | CPU 契约测试 | 原始／观察输出和梯度一致；标签与路由不被改写；禁用分支确实无对应依赖；异常输入拒绝 |
| B | plan／validate-source | 核对原发布包、source 和输入清单；尚未执行 GPU，也不代表能正常训练 |
| C | 每个拟使用版本的3步 Train smoke | 前向、反向、优化、保存和重新读取正常；summary complete_smoke；不能作为精度证据 |
| D | 新 original 与新 no_reader，同 seed／初始化／2500步 | 确定读取价值；先分析 C0−H，再决定标签路径和新设计是否值得继续 |
| E | 按候选／读取证据选择一个改动 | 可选 read_aux，示例权重0.01；仅在Train／Val选配置；0.01不是已证实最佳值 |
| F | 完整 Val450＋配对分析 | 四类mIoU和类别误差改善；注意受损召回与误报、毁坏召回与误报的取舍 |
| G | 对有希望的核心对照增加配对种子 | 每个seed保持匹配，报告差值本身和波动；3个种子只是初步稳定性检查 |
| H | 固定设计后统一 Test | 本包不运行Test；沿用已验证的评估工具和统一检查点规则，说明此前Test已被探索使用 |

如果 no_reader 与 original 接近，应先承认目前读取价值有限；不要强行增加所有模块。若候选缺少相关等级信息，先研究语义检索投影，而非增大辅助权重。若标签读取改善但最终指标不改善，转向检查信息冗余／融合，结束当前假设。

## 4. 服务器放置与 CPU 验证

将压缩包上传到 `/root/shared-nvme/bright_memory_readout_design_20261022.zip`，然后执行（目录名已经包含在 zip 中）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -m zipfile -e /root/shared-nvme/bright_memory_readout_design_20261022.zip /root/shared-nvme/BRIGHT-master/New
```

第一次解压到新目录；不要覆盖已经用于运行的版本。以下每条命令都可单独执行，不依赖上次SSH会话中的变量。

```bash
cd /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022 && CUDA_VISIBLE_DEVICES='' BRIGHT_REFERENCE_MODEL=/root/shared-nvme/BRIGHT-master/New/joint_anchor_extended_v2/deps/original/model.py /root/shared-nvme/conda-envs/bright4090/bin/python -B -m unittest discover -s . -p 'test_*.py' -v
```

若 Reader 源码结构不匹配，测试会明确拒绝。此时应比较版本差异，不应删掉指纹检查。这个指纹约束的是被复制的读取算法，并不声称全部 CUDA 算子已经通过验证。

## 5. 计划检查与3步 smoke

默认source是之前的ExtendedV2运行，默认addon是服务器现有 `New/slot_memory_ablation_v1`，均可显式覆盖。plan只读取文件，不导入Torch、不占GPU、不创建结果目录。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py plan --variant read_aux --read-weight 0.01
```

先在GPU没有其他实验同时运行时依次执行，输出目录必须不存在：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py smoke --variant original --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/original_smoke_01
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py smoke --variant no_reader --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/no_reader_smoke_01
```

准备检验读取监督时，再执行它的 smoke：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py smoke --variant read_aux --read-weight 0.01 --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/read_aux_smoke_01
```

smoke仅使用Train小子集；会执行3次更新及小范围评估。完整状态为 `phase=complete_smoke`，`complete=true`，`last_complete_step=3`；`full_validation_evidence=false`是正确行为。

## 6. 正式新控制组与改进组

下面是前台命令，各自从0到2500步，不接着旧2500步训练。先运行 original／no_reader；结果表明读取值得继续研究后，再做 read_aux 或 no_label。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py train --variant original --seed 3407 --steps 2500 --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/original_3407_01
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py train --variant no_reader --seed 3407 --steps 2500 --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/no_reader_3407_01
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py train --variant read_aux --read-weight 0.01 --seed 3407 --steps 2500 --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/read_aux_3407_01
```

若核心问题是标签路径，使用 `--variant no_label`、独立新目录，其他配方相同。不要同时运行四个GPU进程。本包同一父目录的进程互斥锁不管理其他旧工具，所以仍应查看 `nvidia-smi`。

正式运行包含初始Val450、每500步Val450、终点固定Train诊断；每50步保存checkpoint。每个2500步运行约50份后端模型／优化器checkpoint，不保存全图概率和完整attention；实际磁盘占用以 `du -sh` 为准。

检查点记录模式、辅助权重、模型、AdamW和RNG。state_dict本身不编码“关闭Reader”的行为，加载时必须按记录重新构造适配类；本包尚无自动续训入口，不能用旧六分支恢复脚本加载它。

### 后台运行示例

先创建日志目录，再运行下面一条命令；PID只代表启动，不代表完成。它只运行original，后续版本按上面的顺序单独启动。

```bash
mkdir -p /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/logs
```

```bash
nohup /root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py train --variant original --seed 3407 --steps 2500 --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/original_3407_night_01 > /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/logs/original_3407_night_01.log 2>&1 < /dev/null &
```

```bash
tail -n 40 -f /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/logs/original_3407_night_01.log
```

Ctrl+C退出tail；训练由nohup进程继续。不要再次执行train命令来“刷新进度”。

## 7. 如何判读日志与比较结果

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/experiment.py status --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/original_3407_01
```

正式完成需要 `complete=true`、`training_complete=true`、`full_validation_evidence=true`、`last_complete_step=2500`。若最后更新已完成但Val失败，training_complete仍为true，完整实验complete为false；不要因此误认为需要重训旧六分支。

`train.jsonl`分别记录 original_loss、training_loss、read_loss、label_grad_abs_sum、reader_grad_abs_sum、有效读取槽、缺失类别目标质量、角色读取损失。不同配方的 training_loss 包含不同目标，不宜直接画成一条优劣曲线；使用 original_loss 和最终指标比较。Val报告不计算辅助目标，只报告原主损失及预测混淆指标。

新版本间的严格比较示例：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/compare_runs.py --before /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/original_3407_01/val_step_002500/result.json --after /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/read_aux_3407_01/val_step_002500/result.json --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/comparison_read_aux_3407_01
```

它检查完整状态、同seed／步数／样本名单、初始化哈希、配方、运行协议、逐图GT数量；生成总体与逐事件差值和逐图CSV。全数据mIoU从累计混淆矩阵重算，不是逐图mIoU均值。若冻结Grid在两次运行间不同，会明确记录，微小差异需据此审慎解释。不生成虚假的显著性或多种子区间。

分析已完成旧Val1500里B与D（不推理、不训练）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022/analyze_results.py --result /root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/eval1500_y5qlliyp/val_step_001500/result.json --before b --after d
```

脚本拒绝partial、smoke、完整性待定及类别顺序不明的报告；拒绝后应核对证据，不应手改complete。逐图混淆矩阵可做配对图像／事件分析；不同模型之间的逐像素救回／误伤还需要对应预测。本包评估记录的是同一次前向的 Grid→当前模型配对转移，不声称已有 original→read_aux 的逐像素配对。

## 8. 避免再次混淆方法改进与工程排障

本包以新格式只计算原主损失、混淆矩阵、逐事件指标和配对转移，不复用旧“类别NLL分解相加”的额外校验器；这不改变原训练NLL，不放宽路由／概率检查，也不是对旧数值故障的根因修复。

首次异常保存阶段、图片、调用栈及完成数；不自动重试、不跳过坏图、不在CUDA断言后继续运行、不把部分评估标成完整结果。此前观察到的索引／浮点位变化仍是另一条未解决的执行证据链，不能由CPU测试或新方法假设宣布解决。

本轮首要产出：**明确无支持读取后还剩多少收益；在一个小改动下验证读取目标能否改善最终损毁分类。** 区域设计如何写入论文、旧实验如何归类、哪些结论还缺控制，详见 `HISTORY_AND_FIGURES.md`。
