# soft 双槽＋Memory：共享训练器与输出一致性实验

这是一个**新候选实验**，不是已证实有效的修复。它复用原 `soft_memory_q_rebased_v1_fix2` 的模型、前端、Q、双槽、检索和监督损失；没有删除背景槽，也没有 Memory 接受／拒绝开关。

## 为什么做这一步

原 A/B/C/D 从500到5000步都下降，包含没有历史 Memory 的 A/B。D 的 Val 原损失降低而前景 mIoU 降低；后期建筑二类 IoU改善，Intact→Damaged误报却增加。低学习率只部分缓解，删除背景Lovász未解决退化。证据支持先检查可训练后端如何修改原预测，不能证明双槽结构本身有错。

Q、建筑概率、前端与库冻结，同图原始槽内容不会随训练步数改变。变化发生在槽编码、Reader、联合分类头。四组都双槽，因此历史结果也不能替代同配置单槽对照。

## 模型设计保持一个实现

```text
冻结 DamageFormer → 像素 F256、建筑概率 B、Q、父区域 T128
                               ↓
                    Q(1−B) 与 QB 两槽
                               ↓
                  原 Reader（B图内 / D历史案例）
                               ↓
     像素 + 父区域 + 槽回投 + Memory回投 + grid_probq
                               ↓
              原联合头：log(grid_probq) + delta
                               ↓
                         四类概率 p
```

保留原先的加权合并回投。本包没有加入独立双槽回投、调制前特征、新Q、新定位或新检索。

共享工程：

| 文件 | 唯一职责 | 复用来源 |
|---|---|---|
| `legacy.py` | 来源检查、明确恢复/新实验入口 | 原损失实验的文件检查、RNG、Adam状态与调度 |
| `engine.py` | 一个共同前缀与一个配对更新循环 | 同一个原模型类、同一个 `Runtime.loss` |
| `anchor.py` | 四类KL及分解、梯度诊断 | 不实现第二套NLL/Lovász |
| `evaluate.py` | 一套评估循环 | 原混淆矩阵指标、配对变化及损失分解 |
| `run.py` | 流程、检查点、报告、终检 | 原运行时与冻结检查 |
| `install_shared.py` | 固定版本共享工具仅安装一次 | 已有正确版本直接复用；冲突不覆盖 |

旧发布包不修改、不复制出 `train_kl.py`、`model_kl.py`，不通过修改旧脚本全局函数切换目标。共享依赖 `joint_loss_after500_v1` 中被调用的是状态与统计工具；**不会运行旧的去背景Lovász分支**。

## 唯一训练变量

- control: `runtime.loss(p, GT)`，即原 NLL＋原 Lovász。
- anchor: 同一原目标＋`lambda * KL(stopgrad(grid_probq) || p)`。

默认lambda=0.1、T=1，预先指定的探索值，不是已验证最优。KL覆盖所有有效像素和四类，255忽略。先每图有效像素平均，再按图平均。主路径用logits的log_softmax，避免概率下溢后梯度消失。GT只用于原监督、ignore和事后分类别统计，不筛选“教师正确”像素、不作置信度门控。

KL不知道哪些基线判断正确，也可能锁住教师错误。它是在同任务中的输出保持正则化，不是一个新的Memory或知识集成模块。在T=1下，NLL+lambda KL对logits的梯度可写为真值与教师软分布混合的交叉熵梯度，原Lovász仍保留；若要严格归因于教师知识，候选有效后还应增加同预算的均匀软目标／监督权重对照，而不能把任何收益都归给“记忆保持”。

## 两种来源：不能混淆

### checkpoint500（默认）

从原完成实验的500步模型、AdamW状态、随机状态、图像顺序分叉。严格校验原报告、sidecar、模型SHA、来源和全量库。原500步Val450先重放。两支从501继续到1500；1000仅监测，1500为本轮预定主终点。

### fresh（明确指定）

旧分类头丢失、但冻结A1/Q、成员清单、父库、双槽库和数据仍可用时：

1. 新初始化一个同结构模型。
2. 原目标训练共同前缀1..500，只计算一次，不冗余训练两份相同模型。
3. 连同Adam状态和随机状态复制为独立control/anchor。
4. 两支501..1500按原目标／原目标＋KL训练。

这是新的实验轨迹，不重现历史D500，不用旧分数代替新基线。支持`--arm b`用完全相同驱动研究图内参考；默认`--arm d`。fresh seed由配置指定；相同500步模型分叉不是独立种子重复。

两种来源均固定原5000步调度的长度；500之后两支都乘0.2，不把调度终点改成1500。原模型/库不更新。训练查询来自Train2918的原加载规则，这次1500步只覆盖首轮1500张，不宣称完整训练集已遍历。

## 下载、解压与准备

ZIP根目录为 `joint_anchor_study_v1/`。解压到 `/root/shared-nvme/BRIGHT-master/New`。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -m zipfile -e /root/shared-nvme/BRIGHT-master/New/bright_joint_anchor_study_v1.zip /root/shared-nvme/BRIGHT-master/New
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_anchor_study_v1/install_shared.py
```

第二行只检查／安装共享代码，不会下载模型、重建库或训练。要求原 `soft_memory_q_rebased_v1_fix2` 完整安装。不要修改其绑定代码或SHA。

先执行只读计划：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_anchor_study_v1/run.py plan
```

只有输入齐全才启动默认续接实验：

```bash
bash /root/shared-nvme/BRIGHT-master/New/joint_anchor_study_v1/start_server.sh full
```

`full`执行CPU测试→真实20步配对smoke→重新从原500步恢复→1500步和Val450。smoke不会续接到正式训练。`check`只做smoke。

### 原500权重缺失时

复制配置示例到包外，然后将里面的路径改成**确实存在且通过来源验证的文件／库**：

```bash
cp -n /root/shared-nvme/BRIGHT-master/New/joint_anchor_study_v1/fresh_config.example.json /root/shared-nvme/BRIGHT-master/New/joint_anchor_source.json
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_anchor_study_v1/run.py plan --start fresh --config /root/shared-nvme/BRIGHT-master/New/joint_anchor_source.json
bash /root/shared-nvme/BRIGHT-master/New/joint_anchor_study_v1/start_server.sh full --start fresh --config /root/shared-nvme/BRIGHT-master/New/joint_anchor_source.json
```

示例中的`steps:5000`是调度长度，实际共同前缀500＋配对1000，终点1500。fresh smoke为共同2步＋配对20步，明确标记小样本检查，不等于正式训练500步。

如果plan列出缺失的A1/Q权重或案例库，代码不会凭空恢复，也不会伪造summary。先恢复原依赖；若来源彻底丢失，需要另行建立新前端／案例库。报告副本不能替代权重。不得直接复制其他Q的last.pth冒充原来源。

## 日志与报告

```bash
tail -n 60 -f "$(cat /root/shared-nvme/BRIGHT-master/New/joint_anchor_latest.txt).log"
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/joint_anchor_study_v1/run.py report
```

后台启动不代表CUDA前向成功。启动器不停止其他进程、不等待GPU空闲。新研究使用进程锁避免重复启动同一类流水线；失败不覆盖旧结果、不自动继续。

报告包含：

- 同轮grid/control/anchor原loss、新目标、NLL、Lovász、KL。
- 四类及逐事件混淆；前景mIoU按全split混淆汇总。
- 配对grid→control、grid→anchor、control→anchor的救回／误伤与双方都错。
- KL精确分解：建筑二类KL + 教师建筑概率 × 条件等级KL；这只是同一项的分析，不是增加两项新loss。
- GT四类KL贡献、教师正确／错误位置贡献；不会让这些GT统计进入Reader或教师筛选。
- 稀疏记录原目标与加权KL在logits空间的梯度范数/夹角、实际参数更新；这不是全参数梯度夹角。
- 固定Train450前缀子集初始/末步诊断。这个子集不是完整Train2918。
- 重新读取紧凑JSONL验证混淆和配对计数。默认不写大张量/debug packet或全图预测。

终检通过后才写正式完成标记。无Test入口，不按Test挑lambda、步数或结构。

## 怎样解释结果

1. 先看1500步anchor−control，而不是候选训练loss大小。
2. 再看anchor是否超过同轮grid；仅回到教师水平不算新信息收益。
3. 检查Intact→Damaged／Destroyed是否减少，同时基线错误是否仍能被纠正。
4. KL梯度几乎为零时，先确认作用强度；无收益不能直接否定所有输出保持方法。
5. 若该候选有效，再以同设置训练B，比较新D−新B；最后才统一A/B/C/D与至少三个配对种子探索稳定性。不要将小于0.1pp的单次差异直接称作稳定提升。

后续结构改变按证据选择：混合区域确有互补内容→独立双槽回投的容量匹配对照；低B处信息已受抑制→槽读取调制前特征；Top64缺合适案例→改检索投影和困难负例。每次只新增一个因素，不把KL、PCGrad、定位伪标签、独立回投一起上。

完整论文/作者代码对照见 `RESEARCH.md`。验证范围见 `VALIDATION.md`。
