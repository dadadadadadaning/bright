# BRIGHT：历史标签显式修正损毁等级的配对实验

这是一个**待验证的改进假设**，不是已提分的方法。发行日期：2026-10-23。
代码与已发表方法的关系是借鉴并做受控组合，不声称该残差公式本身是新的论文贡献。

## 为什么做这一轮

已有单种子、2500 步 Val450 的匹配差值：

| 比较 | 四类 mIoU 差值，百分点 | 支持的解释 |
|---|---:|---|
| original − no_reader | +0.04011465 | Reader 的增量很小；不能说全部收益来自 Memory |
| original − no_label | +0.01176822 | 正确标签路径有很小数值差异，未证明稳定性 |
| read_aux − original | −0.01137960 | 这次辅助读出监督未带来额外收益 |

最后一个比较涉及同步执行设置差异且 Grid 混淆矩阵不完全相同，不能把小差值当作可靠负效应。
这也是本轮重新运行 `original` 的原因：以同一发行版、同一执行设置建立匹配对照。
旧 `read_aux` 将注意力读出的四类分布用于辅助损失，没有将该分布显式融合到最终分类分数。

## 核对的论文与作者代码

本次实际核对了作者公开 README、论文元信息和实现。论文站点在当前环境返回 403，未声称逐篇阅读全文。

1. **Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification**, ECCV 2022。
   [论文](https://arxiv.org/abs/2207.09519)；[作者代码](https://github.com/gaopengcuhk/Tip-Adapter)。
   `utils.py` 中使用 `base_logits + alpha * exp(-beta * (1-affinity)) @ cache_values`。
   借鉴点：训练标签缓存直接贡献分类分数。原任务是 CLIP 小样本图像分类，不能把其收益外推到 BRIGHT。
   **没有移植其公开训练脚本中使用 Test 挑选 epoch 的流程。**
2. **Generalization through Memorization: Nearest Neighbor Language Models**, ICLR 2020。
   [论文](https://arxiv.org/abs/1911.00172)；[作者代码](https://github.com/urvashik/knnlm)。
   根据相似邻居标签形成分布，再与模型分布插值。借鉴显式类别信息路径，不移植语言模型任务设定。
3. **Object-Contextual Representations for Semantic Segmentation**, ECCV 2020。
   [论文](https://arxiv.org/abs/1909.11065)；[作者代码](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR)。
   预测类别引导图内上下文汇聚，说明“区域”也可以按语义组织；它不是历史案例检索。
4. **Exploring Cross-Image Pixel Contrast for Semantic Segmentation**, ICCV 2021。
   [论文](https://arxiv.org/abs/2101.11939)；[作者代码](https://github.com/tfzhou/ContrastiveSeg)。
   类别像素／区域队列、困难与容易锚点、跨图同类拉近异类推远。是后续无需推理 Memory 的替代方案。
5. **Rethinking Semantic Segmentation: A Prototype View**, CVPR 2022。
   [论文](https://arxiv.org/abs/2203.15102)；[作者代码](https://github.com/tfzhou/ProtoSeg)。
   每类多个原型、训练期分配及更新。适合后续类别均衡、多事件外观的原型库；本轮没有同时重建库。

## 本轮实现

保持冻结前端、Q、双槽、Top64 候选、Train 标签库、原 Reader 和 NLL+Lovasz 不变。
关闭旧的读出辅助损失。所有新增参数从零初始化，不消耗初始化 RNG。

1. 用原 Reader 注意力读取候选 Train 软标签，得到槽级四类分布。
2. 用原 `route_q/route_ids` 回填到特征网格，再双线性插值到全图，得到 `m_c(x)`。
3. 对三个前景类别构造证据 `e_c = m_c - (m_1+m_2+m_3)/3`。
   不除以很小的前景质量；背景占优的 Memory 不会被强行归一化成很自信的等级证据。
4. 预测修正 `d_c = b_c + g * e_c`，其中 `b=2*tanh(b_raw)`，`g=2*tanh(g_raw)`。
   **g 是有符号系数**；若学成负值，只能解释为反向使用证据，不能宣传为正向标签迁移。
5. 令原前景 logits 为 `z_fg`，新 logits 为：

```text
z'_0  = z_0
z'_fg = z_fg + d - [logsumexp(z_fg+d) - logsumexp(z_fg)]
p'    = softmax(z')
```

这在数值精度内保持本次 forward 的背景概率和前景概率总和，调整前景内部类别分配。
**不保证合并四类 argmax 后的建筑掩膜／IoU 不变**，也不保证训练后整个模型的定位概率不变：
共同后端仍然接受训练梯度。分类协议依然是原四类 argmax，没有增加阈值解码。
零初始化时 logits 与原模型逐位相同，系数仍有可训练梯度。

| 版本 | 新增路径 | 比较目的 |
|---|---|---|
| original | 原完整后端 | 本轮同环境基线 |
| calibration | 三类偏置，Memory 系数固定为零 | 简单等级校准能否改善 |
| severity_memory | 同样偏置＋正确历史标签证据 | 超过校准之后，正确标签是否有额外价值 |
| shuffled_memory | 同样结构，仅打乱新增路径的候选标签对应关系 | 新增收益是否依赖正确对应关系 |

置乱只作用于新概率读出；原 Reader 的特征、标签 embedding 仍使用原始支持。
置乱使用独立 CPU generator，不推进训练 RNG，保持本图已选合法候选标签的边际分布。
置乱供体仅来自本图 `support_ids` 实际选中的合法父区域，不能从整库引入原先排除的同图／同事件标签。
它是确定性的标签对应关系对照，不是从 Test 标签生成噪声。

## 安装后运行

Python 使用服务器已有环境。新包不修改旧模型、库或旧结果，也不覆盖旧的恢复程序。
建议一次启动下面的夜间串行队列；不要同时再启动同一个训练命令。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023/night.py start
```

默认四个版本、种子 3407、从头训练到 2500 步。先全部 smoke，再依次训练。
所有队列子进程统一设置 `CUDA_LAUNCH_BLOCKING=1`；既不证明旧内存故障已消失，也不是故障修复。
启动前检查 GPU 上已有计算进程；不杀已有任务。每阶段默认上限 10800 秒，超时或失败停止，保留日志。
运行目录自动唯一，位于 `SRAC_SEVERITY_MEMORY_V1/night_3407_*`。SSH 断开后后台队列继续。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023/night.py logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023/night.py status
```

以上默认读取本包最后一次启动目录；启动时也会打印可直接使用的完整路径。
`logs` 打印队列心跳和当前子任务末尾；`tail -f` 可实时查看，退出查看不会停止后台训练。
预算不足时，可先运行三个版本：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023/night.py start --variants original calibration severity_memory
```

单独操作也有明确入口（输出路径必须不存在）：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B experiment.py plan --variant severity_memory
CUDA_LAUNCH_BLOCKING=1 /root/shared-nvme/conda-envs/bright4090/bin/python -B -u experiment.py smoke --variant severity_memory --output /tmp/bright_severity_smoke_NEW
CUDA_LAUNCH_BLOCKING=1 /root/shared-nvme/conda-envs/bright4090/bin/python -B -u experiment.py train --variant severity_memory --seed 3407 --steps 2500 --output /root/shared-nvme/BRIGHT-checkpoints/SRAC_SEVERITY_MEMORY_V1/severity_memory_3407_NEW
```

单独命令中的相对脚本要求先进入本包目录；夜间命令已使用绝对路径。
没有训练续跑、坏图跳过或失败自动重试。遇到原有硬件／软件内存异常仍会停止并保存首个错误。

## 每一步如何验收

1. **CPU 合约**：真实发布模型上验证零初始化等价、非零有效梯度、分布归一化、背景质量约束、
   标签置乱隔离、非法路由拒绝及检查点往返。它不能替代服务器 GPU smoke。
2. **GPU smoke**：每个版本少量训练＋评估；确认损失／梯度有限、原状态与源文件校验通过。
   smoke 的 `complete_smoke` 不是正式结果。
3. **正式 Val450**：每 500 步评估、固定 2500 步比较；每张图都记录混淆矩阵与原目标损失。
   完整输出、逐图累计和最终完整性检查全部通过才标为 complete。
4. **机制比较**：`severity_memory−calibration` 衡量标签证据增量；
   `severity_memory−shuffled_memory` 衡量正确对应关系；`calibration−original` 衡量校准收益。
   队列完成后自动产生 `comparison_*.json`，包括 Grid 差异、类别错误方向与事件分组。
5. **主要错误方向**：受损→完好、完好→受损、非毁坏→毁坏及各类精确率／召回率。
   不接受只提升受损召回、却明显增加毁坏误报的单指标宣传。混淆矩阵差是净变化，不是配对像素救回数量。
6. **稳定性与 Test**：有希望的版本做 3407/3408/3409 配对种子，再固定配置和 2500 步进行 Test878。
   单种子有小差值不能称显著；已看过的 Test 应保留探索性说明。当前包只训练／验证，
   **没有 Test 命令**；新模型需对应本 adapter 的独立评估入口，不能拿旧六分支恢复器直接加载。

人工比较完整结果的接口：

```bash
python compare_runs.py --before /完整队列路径/train_calibration/val_step_002500/result.json --after /完整队列路径/train_severity_memory/val_step_002500/result.json --output /新的比较输出目录
```

正式数据之前，CPU 测试可这样执行：

```bash
cd /root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023
BRIGHT_REFERENCE_MODEL=/root/shared-nvme/BRIGHT-master/New/joint_anchor_extended_v2/deps/original/model.py /root/shared-nvme/conda-envs/bright4090/bin/python -B -m unittest discover -v
```

测试生成的临时数据会清理；正式任务保留每 50 步检查点与指标，不保存全图概率或注意力。
磁盘仍会增加，四个独立训练都有各自的模型／优化器检查点；Memory 库和数据集通过原路径读取。

## 如果这轮没有提升，下一步怎样收敛

* 若只 calibration 提升，说明当前收益主要可由等级偏置校准解释；不要把同样收益写成检索创新。
* 若正确证据超过 calibration，却不超过 shuffled_memory，尚不能证明正确标签对应关系有用。
* 若两项差值都稳定为正，进一步做配对种子、受损／毁坏和事件分析，再冻结配置做 Test。
* 若系数接近零且没有增益，不继续叠加同类辅助损失；检查训练集候选的类别覆盖与区分能力。
* 候选质量不足：按 Train 类别建立多个模式原型，存特征、软标签、有效质量、图像／事件 ID 和特征版本；
  限制同图／同事件重复，比较等候选数的原库、多原型库与随机合法支持。可参考 ProtoSeg。
* 特征区分不足：保留相同投影头，对 Train 完好／受损／毁坏做困难与容易像素对比学习；
  以同投影头无对比损失为对照。参考 ContrastiveSeg，推理可以完全不查 Memory。

## 区域收益的解释与下一阶段区域设计

当前双槽 `Q*(1-B)` 和 `Q*B` 对应预测背景／建筑分配，不天然等价于三种损毁等级。
现有定位改善可以与区域聚合、去噪、前景分离一致，但缺少纯像素后端及分支消融时不能唯一归因于 Q。
`no_reader` 仍有区域／槽／分类头，故它也不是无区域对照。

先对 Train/Val 的已有区域统计类别熵、受损像素被完好主导区域吸收的比例，以及与面积／边界的关系。
如果混合确实严重，再保留背景路由，将前景分成 K 个可学习软子区域：
`q_bg=Q*(1-B)`，`q_fg,k=Q*B*softmax_k(a)`，因此总路由仍等于 Q。
K=2 或 3 是外观模式，不预先宣布就是受损语义；用等容量空间／随机分槽、相同 token 数及多种子验证。
本包没有同时实现这一结构改变，以便本轮的成败可以明确归因。

展示建议：四版本固定终点指标表；受损/毁坏精确率-召回率变化；逐事件配对差值；
固定 Train/Val 样例的原预测、修正后预测、GT、区域边界。样例提前按事件和错误类型确定，避免只挑成功图。
