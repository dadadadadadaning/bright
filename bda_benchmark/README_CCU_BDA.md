# CCU-BDA 修订实验路线

CCU-BDA 的目标不是拼接多个变化检测模块，而是验证一个具体命题：

> 灾前 RGB 与灾后 SAR 只应在“局部对应可靠且建筑未变化”的内容上统一；
> 对可靠变化保留差异，对不可靠对应拒绝 SAR 证据。

当前仓库中的 C0-C4 是可训练原型，不应直接整批运行。正式路线必须先完成
E0 局部对应审计；若 H6-A2 的现有特征中没有稳定的局部匹配几何信号，就停止
CCU 训练并重做 SAR 表征，而不是继续增加融合模块。

## 1. 研究变量重新拆分

| 阶段 | 唯一问题 | 允许改变的内容 | 禁止混入的内容 |
|---|---|---|---|
| E0 | 现有特征是否含局部对应信号 | 只做验证集诊断 | 训练、测试集、标签参与推理 |
| E1 | 干净异构双流能达到什么水平 | 非共享 RGB/SAR 编码器、晚期融合 | 匹配、对齐、层级头 |
| E2 | 哪个差异算子最适合 RGB-SAR | abs-diff / 3D-diff / local-correlation | 迁移预训练、层级分类 |
| E3 | 哪种对齐范围不会抹掉变化 | 无对齐 / 全局 / Intact-only / reliable-Intact | 多个新增损失、教师蒸馏 |
| E4 | SAR 迁移是否改善泛化 | SARDet/CROMA/DSTCD 风格初始化 | 同时修改融合和分类头 |
| E5 | 层级严重程度是否改善 Damaged | Building→Affected→Destroyed | 再改编码器或匹配器 |

E0-E5 是论文消融顺序；原 C0-C4 仅作为已有代码模块库。当前 C0 使用项目内
Swin + 原生 SAR ConvNeXt/GN，并不是严格复刻的 HeteCD MiT 基线；C1 也只是
无条件全局统一的架构对照，不应写成 HeteCD FCA 复现。

## 2. E0：先做局部对应审计

E0 冻结 H6-A2，直接读取 Stage3/4 中尚未进行传感器原型修正的 RGB/SAR
common projection。然后使用固定恒等投影的小窗口相关体，统计：

- correlation peak、entropy、top1-top2 margin；
- 局部期望偏移与偏移幅度；
- RGB→SAR→RGB 循环误差与一致性；
- 按 Intact/Damaged/Destroyed 及每个灾害事件分别统计；
- Cohen's d 与事件方向一致率。

验证标签只用于事后分组，不输入模型，不生成可部署规则。程序拒绝 test split。

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_CCU_BDA_5090.sh preflight
bash run/run_CCU_BDA_5090.sh stats
bash run/run_CCU_BDA_5090.sh audit_smoke

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/CCU_BDA
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/E0_local_correspondence_val_$(date +%Y%m%d_%H%M%S).log"

nohup bash run/run_CCU_BDA_5090.sh audit_val > "$LOG" 2>&1 &
echo $!
tail -F "$LOG"
```

默认输入是
`/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth`。
如果位置不同：

```bash
H6_A2_CKPT=/path/to/H6_A2_best_val_miou.pth \
bash run/run_CCU_BDA_5090.sh audit_val
```

完整结果默认保存到
`/root/shared-nvme/BRIGHT-eval-results/CCU_BDA/E0_H6A2_local_correspondence_val.json`。

自动放行门槛为：最佳 Intact→Destroyed 信号 `|Cohen's d| >= 0.25`，至少有
10 个可比较事件，且至少 10 个事件方向一致。门槛只是继续做 E2 的最低条件，
不是论文结论，也不能用 smoke 结果作决定。

## 3. E0 之后怎样决策

### E0 不通过

停止 C2/C3/C4。说明当前 A2 common feature 不适合做局部匹配，下一步进入
E4，但只比较 SAR 表征迁移：当前随机/监督 SAR、SARDet 风格迁移、CROMA
风格雷达编码器初始化。不要先调窗口、top-k 或堆对齐损失。

### E0 通过（本次实测分支）

当前完整验证审计发现，只有 Stage4 `one_minus_peak` 达到可用强度：全局
AUROC=0.7448、事件宏平均=0.6971、11/13 个可比较事件方向一致。Stage3、
margin、entropy、offset 均不足以作为首轮训练信号。因此本轮不运行旧 C2，
而改为参数严格匹配的 E1/E3：

```text
E1：冻结 H6-A2 + Stage4 同位置 RGB/SAR 差分
E2：冻结 H6-A2 + Stage4 多尺度 3D 时空差异
E3：冻结 H6-A2 + Stage4 5x5 参数无关软局部匹配后差分
```

E1/E3使用相同的两层残差适配器、相同参数量、相同损失和相同5000步协议；
唯一自变量是局部匹配。残差系数从严格零开始，因此训练前二者都必须逐元素
复现 H6-A2。这样可以直接回答“E0 发现的局部对应信号是否真的能改善下游
损毁分类”，而不会把冷启动骨干、Stage3、复杂可靠性门混入结论。

E2是独立的强差异算子基线：将RGB/SAR Stage4特征堆成长度为2的时序，使用
1x1、3x3和膨胀3x3深度可分离3D卷积提取同位置、局部和上下文差异。三个分支
初始化为对应尺度的`SAR-RGB`，随后允许监督学习；它仍通过零门控接入冻结A2，
因此初始预测也严格等于H6-A2。E2参数量与E1/E3不同，不能用于孤立证明局部
匹配，只用于回答“学习型3D差异是否比简单差分更强”。

重新运行E0并训练、验证、测试E2：

```bash
LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/CCU_BDA
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/controller_E0_E2_$(date +%Y%m%d_%H%M%S).log"

nohup env RUN_TEST=1 \
  bash run/run_CCU_E0_E2_sequence.sh \
  > "$LOG" 2>&1 &

echo $!
tail -F "$LOG"
```

运行完整受控实验：

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/CCU_BDA
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/controller_E1_E3_5000_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  VARIANTS="e1 e3" \
  CONTINUE_ON_FAILURE=0 \
  bash run/run_CCU_E1_E3_sequence.sh \
  > "$LOG" 2>&1 &

echo $!
tail -F "$LOG"
```

训练只使用验证集选 checkpoint，测试集不会被读取。完成后先比较 E1/E3 的
Valid mIoU、Damaged/Destroyed IoU、Event mean/worst 以及逐事件结果；只有
验证锁定的一个候选才允许执行一次 `eval_test`。

## 4. E3 的最小对齐设计

E3 不采用当前 C3 中未经审计的复杂预测门作为第一版主实验。固定 E2 最优结构，
只添加一个 `L_CCU`：

```text
E3-A: no alignment
E3-B: global alignment
E3-C: GT Intact-only alignment（仅训练时构造掩膜）
E3-D: reliable GT Intact-only alignment
```

推荐形式：

```text
L_CCU = sum[1(y=Intact) * stopgrad(reliability) * distance(F_RGB,F_SAR)]
        / sum[1(y=Intact) * stopgrad(reliability)]
```

只增加这一项对齐损失；可靠性在该损失中 detach，避免模型通过把可靠性降为零
获得伪最优。Background 不作为默认不变区，Damaged/Destroyed 不强制对齐，
也不额外强制二者远离。推理时不使用真值标签；标签只规定训练中的对齐区域。

## 5. 现有 C0-C4 原型如何使用

在 E0/E2 通过前，不运行 C0-C4 控制器。单个原型仍可用于接口检查：

```bash
VARIANT=c0 bash run/run_CCU_BDA_5090.sh smoke
VARIANT=c2 bash run/run_CCU_BDA_5090.sh smoke
```

| 变体 | 当前实现 | 修订后的定位 |
|---|---|---|
| C0 | 模态独立编码 + 晚期融合 | E1 候选，但不是严格 HeteCD |
| C1 | 无条件全局统一 | E3 的负对照候选 |
| C2 | Stage3/4 局部相关体 | E2-C 候选实现 |
| C3 | 预测变化门控统一/保留/拒绝 | 仅在 E3-D 成功后再审计 |
| C4 | C3 + 层级严重程度 | E5 候选 |

## 6. 统一评估与停止规则

所有选择只看验证集；测试集仅在候选结构、超参数和最佳验证 checkpoint 冻结后
评估一次。主指标为 Final mIoU、Valid mIoU、Damaged/Destroyed IoU、Loc F1；
跨事件必须同时报告 Event mean、worst、std，以及正收益事件数。

1. E0 不通过：停止局部相关路线；
2. E2-C 不优于简单 abs-diff：停止复杂匹配；
3. E3-D 不优于 E3-C：可靠性没有增益，删除可靠性门；
4. E4 只提升验证、不提升事件均值/最差事件：不进入 Full；
5. E5 牺牲 Destroyed 换取 Damaged：不作为主模型。

## 7. 与相关工作的边界

- HeteCD 提供非共享双流及异构差异建模参考；
- IM4CD 提供局部模板匹配、响应峰和空间偏移视角；
- SARDet-100K 说明 SAR 迁移差距不仅在输入层，也包括结构与迁移过程；
- CROMA 提供雷达/光学独立编码与跨模态预训练参考；
- Keep the Balance 提醒无条件全局交互可能引入跨模态噪声；
- BRIGHT 固定你的灾前 RGB、灾后 SAR、建筑定位和严重程度评估任务。

这些是设计依据，不等于这些论文已证明 CCU 对 BRIGHT 有效。CCU 的核心贡献
只有在 E0→E2→E3 的证据链成立后才能宣称。
