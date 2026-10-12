# ROI 后续验证：先拆开汇总／回投，再独立验证 Memory 或 Q 学习

本目录是独立新增实验，不修改旧 `roi_exchange_v1`、四阶段模型及其权重。所有入口限定 Train2918 / Val450，不读取 Test，不自动删除结果，不自动串行启动其他实验。

本地验证：30项CPU测试通过，包含真实小尺寸前端重放、审计入口、模拟Train建库、Memory与Q各20步训练入口、保存和报告。模拟测试不读取BRIGHT数据，也不等于服务器CUDA验证；四个Python入口的`--help`已检查。尚未运行真实数据训练。

## 1. 三项实验分别回答什么

| 入口 | 固定什么 | 改变什么 | 可以回答的问题 |
|---|---|---|---|
| `audit` | 两个已训练读出分别固定，前端/Q/B固定 | 汇总与回投是否跨组 | 同一个模型的收益／伤害对哪个操作敏感？ |
| `bank` 后 `memory` | 前端、Q、B、选定区域Embedding及分组固定 | 新分类器是否读取Train邻居 | 历史支持是否比同结构无历史分类器提供增量信息？ |
| `q_train` | 前端、RGB金字塔、Soft ROI内容、B固定 | 分区解码器是否更新 | 让Q继续适应任务是否优于固定Q？ |

本轮没有新的图像调制、主干、类别损失、残差头或预测建筑实例。后两项不是必须依次做完：Memory和Q学习是独立路线，不在同一次训练里合并。Memory无需先超过历史A1才有资格测试。

当前来源是A1初始化后训练的V2 **3000步前端快照**和已完成的 **ROI_EXCHANGE_V1 5000步读出**，不是原始A1。`old_pixel`约60.8998%，不能称作A1原始61.6866%。脚本使用来源 `summary.json` 复现而不把这些文字里的四舍五入数作为硬编码目标。

## 2. 审计的四种情况

| 名称 | 区域汇总 | 像素回投 |
|---|---|---|
| HH | hard | hard |
| EH | exchange | hard |
| HE | hard | exchange |
| EE | exchange | exchange |

H使用epsilon=0，E使用来源manifest中的epsilon（本轮0.05），不重新扫描超参数。每个来源head独立做四种推理。`exchange_EE -> exchange_EH`只关闭回投交换；`hard_HH -> hard_EH`只增加汇总交换。

先重放 `hard_original` 和 `exchange_original`，再检查HH/EE的概率和完整Val指标。分类头不更新，GT不决定路由。这些是推理扰动，不是重新训练消融；不能挑出最高的一项就宣布新模型有效。

HE可能读取硬汇总中的空槽。代码用实际汇总质量 `mass>0` 屏蔽严格空槽，在特征分辨率上重新归一化回投权重，再上采样；不删除质量很小但非空的槽。记录被阻断质量、实际跨组回投比例、无对侧支持的像素比例，以及两种Pooling分母触发数。`clamp_min(1e-6)`仍沿用旧代码，只是数值保护，不是假定纯度或有效建筑阈值。

## 3. Memory 对照具体怎样实现

选定一个已有source head（默认hard_last），冻结其完整Embedding；直接提取分类器前的128维Token。新bank不能复用旧四阶段缓存：内容来源、Embedding权重、路由和槽位定义都必须一致。

```text
冻结前端 → 原图Soft ROI特征/Q/B → 固定分组 → 冻结区域Embedding → Token
                                                   ├→ 新nomemory头
                                                   └→ 新memory头 + Train邻居
                                                       ↓
                                                    同一回投
```

- bank只从Train原图1024确定性提取，按事件×多数类别随机优先级抽样，每桶最多64个，包括背景；标签用Train四类软分布，ignore255不计质量。
- 只保存限额Token、软标签、来源ID；不保存整图特征、Q、预测图或每张图的全量缓存。14事件×4类×64最多3584个Token，128维FP32向量本体约1.75MiB，实际文件还含标签/索引/元数据。
- query在Train随机512裁剪、Val原图1024提取。bank与query使用同一冻结表示，但裁剪上下文并非完全相同；若检索很差，这是以后可以单独检验的因素，不与本轮分类器对照混改。
- top-k=8余弦检索；默认排除同图和同事件。同一图的其他裁剪仍按原图ID排除。支持不足时用有效邻居掩码和self fallback，不让`-inf`邻居参与Attention。`ALLOW_SAME_EVENT=1`只是可选独立消融，仍禁止同图；不要在首轮同时改。
- 两个新分类器参数与初值相同、独立优化，都是标签嵌入+Cross-Attention+LayerNorm+线性分类。nomemory使用可学习全局槽，不输入历史样本或查询标签；没有新增灾害/地理/几何元信息。
- 两组均使用当前**回投后的像素NLL+Lovasz**，不是旧Stage C的区域CE。每次读取相同图像/裁剪、相同预算；只比较原线性头与更大Memory头不能归因于Memory。
- 两个新分类器各67332个参数，独立同初始化；参数量匹配不等于两种输入提供的信息或计算量相同。Memory增益应在这个明确的无历史对照下解释。
- 默认每组5000次优化器更新，EBS16，共同看到相同80000个裁剪；两组各有自己的优化器，并非各2500步。初始/每500步Val450，最终详细诊断。
- 最后在固定memory头上做random、shuffle_labels、off。它们只用于依赖性诊断，不能替代独立训练的nomemory对照。shuffle仅打乱允许支持的标签，仍遵循来源排除。

静态bank包含front SHA、readout SHA、路由、拆分阈值、数据列表及代码哈希。改变其中任何表示要素都要重建bank。**仅改输出路径不会改变Token空间；重新训练Embedding会改变。**

## 4. 固定Q／可训练Q具体怎样实现

两个分支都复制同一已训练完整Readout；解码器从同一3000步前端复制，彼此不共享可训练参数。不是新随机读出与旧训练头混比。

- fixed_q：使用前端固定Q，只训练完整Readout。
- learn_q：用第一次原RGB编码的冻结金字塔，在no_grad外运行复制的分区解码器和三轮迭代，训练解码器+完整Readout。
- 内部 `iterative_assignment` 仍用 `zeros_like(B)`，与V2保持一致，B只在之后的路由拆分中使用；避免不小心增加第二次建筑硬约束。
- 不对新Q执行detach；内容和B仍detach。分区梯度、实际参数变化、Q变化量都必须通过检查，不能只看requires_grad。
- 初始化检查新Q、两个分支概率、来源读出重放；fixed解码器和整个前端不可变。
- 两分支总结构匹配，但trainable参数数不同（这正是是否学习Q的实验因素），不能宣称训练参数量相同。
- 仍不保证连通或一栋建筑一个区域；不把楼内差异当局部破损GT；没有Memory。

## 5. 先上传整个新目录并检查

本地目录：`D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_roi_followup_v1`。

上传到服务器同项目的 `bda_benchmark/experiments/damageformer_roi_followup_v1`。依赖服务器原有匹配版本的实验代码，不覆盖旧目录。运行期间不要修改这些代码，否则哈希保护会中止而不是混用版本。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export FRONT_CKPT=/root/shared-nvme/BRIGHT-checkpoints/REGION_EVIDENCE_V1/run_20261003_003133/front_snapshot.pth
export SOURCE_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_EXCHANGE_V1/formal5000_seed3407_20261005_133348
export SOURCE_HEAD=hard
export POOL_MODE=hard
export READ_MODE=hard
export BATCH=2
export ACCUM=8
export SEED=3407
export DEVICE=cuda:0

"$PYTHON" -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.cuda.is_available())"
ls -lh "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth" "$SOURCE_RUN/exchange_last.pth"
bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh preflight

export FOLLOWUP_ROOT="/root/shared-nvme/BRIGHT-checkpoints/ROI_FOLLOWUP_V1/run_$(date +%Y%m%d_%H%M%S)"
```

不使用旧`EXP`、`VARIANTS`环境变量，避免上次`/run.sh`和错误变体残留。`PYTHON`使用绝对路径，无需依赖新SSH会话的conda激活状态。遇到源哈希错误应核对来源代码，不要删掉检查或以strict=False绕过。

## 6. 第一项：先smoke，再完整只读审计

```bash
export OUTPUT_ROOT="$FOLLOWUP_ROOT/smoke_audit"
bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh smoke_audit

# smoke通过后再运行。不要重复使用已有OUTPUT_ROOT。
export OUTPUT_ROOT="$FOLLOWUP_ROOT/audit"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/ROI_FOLLOWUP_V1
export AUDIT_LOG="/root/shared-nvme/BRIGHT-train-logs/ROI_FOLLOWUP_V1/$(basename "$FOLLOWUP_ROOT")_audit.log"
nohup bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh audit > "$AUDIT_LOG" 2>&1 &
echo "审计PID=$!"
tail -f "$AUDIT_LOG"
```

如果只想先查最小问题，可在启动前设 `AUDIT_HEADS=exchange`、`COMBINATIONS="EE EH"`；默认对两个头各做四种情况。不会重新训练。

完成后读取：

```bash
export OUTPUT_ROOT="$FOLLOWUP_ROOT/audit"
export PAIR=exchange_EE__exchange_EH
export BOOTSTRAP=2000
bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh report
unset PAIR BOOTSTRAP
```

报告包括类别IoU/P/R/F1、背景误报、Damaged→Intact伤害、ROI正确/漏检/误检、边界两侧、内部、小对象代理、事件差异和占用槽。bootstrap是固定事件组成下配对图像重采样，不以像素为独立样本；不代表训练种子方差或未知事件泛化，仍不能解决邻近裁剪的空间相关性。种子稳定性需另行运行配对种子。

出现明确正向信号后，建议使用同一bank做至少三个配对训练种子，分别报告均值、标准差和每个种子的差值；这是固定bank条件下的优化波动。bank抽样种子的影响若要验证，应另列对照，不与训练种子波动混为一谈。

## 7. 第二项：确定路径后再建库、训练Memory

默认先保留hard/HH，不从一次扰动最大值自动选方案。若要研究其他路由，明确设置SOURCE_HEAD/POOL_MODE/READ_MODE，并用相同设置重建bank；不要挪用hard库。

先检查完整流程：

```bash
export SOURCE_HEAD=hard POOL_MODE=hard READ_MODE=hard
export BATCH=2 ACCUM=8
export OUTPUT_ROOT="$FOLLOWUP_ROOT/smoke_bank"
bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh smoke_bank
export BANK_ROOT="$OUTPUT_ROOT"
export OUTPUT_ROOT="$FOLLOWUP_ROOT/smoke_memory"
bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh smoke_memory
```

smoke只从两个不同事件的Train图提取库、20步训练并验证Val2。它不判断效果，不能把smoke库用于正式训练。没有命令会自动终止其他GPU任务，先确认资源空闲。

正式建库和配对训练：

```bash
export OUTPUT_ROOT="$FOLLOWUP_ROOT/bank_hard"
bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh bank

# 只有建库成功、manifest complete=True后再启动训练。
export BANK_ROOT="$OUTPUT_ROOT"
export OUTPUT_ROOT="$FOLLOWUP_ROOT/memory5000"
export STEPS=5000 EVAL_EVERY=500
export MEMORY_LOG="/root/shared-nvme/BRIGHT-train-logs/ROI_FOLLOWUP_V1/$(basename "$FOLLOWUP_ROOT")_memory.log"
nohup bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh memory > "$MEMORY_LOG" 2>&1 &
echo "Memory对照PID=$!"
```

完成后设`OUTPUT_ROOT="$FOLLOWUP_ROOT/memory5000"`、`PAIR=nomemory__memory`再运行report。正式默认5000次更新；不要根据短程smoke的0分判断Memory失败。

## 8. 第三项：独立Q学习对照（不是Memory后的自动步骤）

这是一项可选的独立路线，需要GPU空闲后手动运行；不读取BANK_ROOT，不继承训练好的Memory分类器。

```bash
export SOURCE_HEAD=hard POOL_MODE=hard READ_MODE=hard
export OUTPUT_ROOT="$FOLLOWUP_ROOT/smoke_q"
bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh smoke_q

export OUTPUT_ROOT="$FOLLOWUP_ROOT/q5000"
export STEPS=5000 EVAL_EVERY=500
export Q_LOG="/root/shared-nvme/BRIGHT-train-logs/ROI_FOLLOWUP_V1/$(basename "$FOLLOWUP_ROOT")_q.log"
nohup bash bda_benchmark/experiments/damageformer_roi_followup_v1/run.sh q_train > "$Q_LOG" 2>&1 &
echo "Q对照PID=$!"
```

完成后使用`PAIR=fixed_q__learn_q`。本轮正式Q比较固定5000步/每500步验证，以免无意改变原定预算。后续要调整预算应另开协议，不直接覆盖旧结果。

## 9. 结果如何决定下一步

| 观察 | 支持继续做什么 | 不应声称什么 |
|---|---|---|
| EH保留漏检纠错、减少背景伤害 | 另做hard与EH的独立训练对照 | 仅凭固定EE权重扰动已证明新方法 |
| memory优于同结构nomemory且没有严重类别/事件代价 | 再做配对种子与支持标签/邻居质量分析 | 仅超过旧线性头就证明Memory |
| memory只增加D召回同时大增误报 | 查混淆来源、支持标签分布与跨事件适配 | Damaged全面改善或牺牲其他类已被证明 |
| learn_q优于fixed_q且梯度/参数改变确认 | 可研究任务监督分区，重复种子 | Q变化、熵下降或图变好看本身代表有效 |
| 所有差值接近零或互相抵消 | 保留简单hard路径，暂停对应复杂化 | 再堆Memory/边界损失一定能修复 |

分类别TP/FP/FN变化可以回答“Damaged是否以Intact/Destroyed为代价”：须同时看三类IoU/P/R/F1及误报来自Background还是其他损毁类别，不能仅凭D召回上升推断。GT建筑等级标签不是楼内局部破损定位；连通域只作评估代理。B错误分组不能用GT纠正后再当成真实方法。

本版本是机制验证工具，不宣称SRAC严格复现或达到75%。所有训练判断限定Train/Val。

## 10. 保存与恢复边界

- audit：JSON和小统计，不保存预测图、模型副本或特征缓存。
- bank：一份限额`bank.pt`和manifest。
- Memory：每组`best.pth`/`last.pth`小分类器，共4份，不保存前端/Embedding/优化器。
- Q：每组best/last，只存decoder+readout，共4份，不保存Swin或优化器。
- 默认128维下，Memory头FP32张量约0.257MiB/份，Q的decoder+readout约2.596MiB/份；文件元数据与诊断JSON另计。这不是GPU显存估计。
- 不支持精确断点续训；不要把best/last小模块权重当完整前端。来源front_snapshot和source run必须保留。
- 新增目录不修改旧代码；没有终止任务、覆盖输出或删除结果的行为。
