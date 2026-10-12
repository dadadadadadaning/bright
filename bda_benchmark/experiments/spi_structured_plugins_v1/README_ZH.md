# Structured Plug-ins V1：插入位置、作用与实验顺序

## 1. 设计原则

当前父系统固定为：

```text
Pre RGB + Post SAR
        ↓
锁定 P3（定位头 + 损毁头）
        ↓
LocCC 伪建筑
        ↓
A6 区域可靠性规则
        ↓
fixed S2 软建筑上下文
```

本目录不把多个新模块一次塞进 P3。每个模块只处理一个已经被实验定位的问题，并独立开关、独立训练、独立评估。这样可以判断收益来自哪里，也不会破坏目前唯一稳定有效的 `P3 + fixed S2`。

## 2. 五个模块应插在哪里

### 2.1 StripSpatialBlock：插入 RGID 几何头

位置：

```text
P3 fused_roi（冻结、detach）
        ↓
RGID 64通道投影
        ↓
StripSpatialBlock ← 新增
        ↓
center head / boundary head
        ↓
Watershed 候选分裂
```

原因：LocCC 的问题是建筑粘连、碎裂和长边界不完整。普通 `3×3` 卷积只能看到很小邻域；水平/垂直条带卷积能沿建筑长轴、屋顶轮廓和建筑间缝隙聚合上下文。它只影响伪实例候选，不改变 P3 的 RGB/SAR 编码、定位 logits 或损毁 logits。

实现：`StripRGIDHead` 保持原 RGID 的输入、输出字典和 center/boundary 目标不变，只替换共享空间混合器。因此现有 RGID decoder、Selective RGID 风险规则和 fixed S2 均可复用。

### 2.2 CrossScaleBoundaryFusion：插入 Stage3→Stage4 边界支路

位置：

```text
Stage3 detail ── 1×1投影 ── resize ── boundary gate ─┐
Stage4 semantic ──────────────────────────────────── + ─→ 后续特征
```

原因：Stage4 语义强但小建筑边界粗；Stage3 有细节但噪声较多。该模块只在边界先验允许的位置加入低层细节。`gain=0` 初始化，刚创建时输出严格等于原 Stage4 特征。

限制：当前 P3 的 `return_aux` 只公开最终 `rgb_feature/sar_feature/fused_roi`，没有公开 Stage3。因此该块已经实现，但现在不要强接入 P3。应先增加只读 feature-tap 并复现锁定 P3，再开展独立 A/B 实验。

### 2.3 LocalCrossModalMatcher：插入 RGB/SAR 融合之前的证据旁路

位置：

```text
P3 rgb_feature ─── query ───────┐
                                ├─ 5×5局部匹配 → residual/confidence/offset
P3 sar_feature ─── local keys ──┘
                                             ↓
                                  只作为RGID/Verifier证据
```

原因：RGB 与 SAR 有配准残差，同坐标直接相减会把位置误差当成灾损。局部匹配在小窗口里寻找最相似 SAR token，并输出：匹配后 SAR、残差、置信度、熵和位移。第一阶段只审计这些量是否能解释粘连/有害修改，不能直接替换 P3 的 SAR 特征。

该实现无可学习参数，适合先做 E0。先前 IM4CD 简单池化证据失败，说明不能再次只做整栋均值；下一次应统计候选切割带、子区域两侧和高风险 LocCC 内的空间分布。

### 2.4 RegionRelationBlock：插入固定区域池化之后

位置：

```text
改进后的伪实例 / fixed SLIC
        ↓
区域池化得到 N 个 token + 区域中心坐标
        ↓
RegionRelationBlock（1层）
        ↓
关系变化证据 / 低容量 readout
```

原因：像素差回答“这个点是否不同”，关系块回答“建筑内部各区域的组织关系是否改变”。它只在一个建筑或有限邻域内计算，避免全图 Transformer 的容量和显存开销。残差 `gain=0` 初始化，默认严格保持原 token。

限制：已有 Grid/SLIC Relation E0 没有通过，所以此模块不能直接进入最终模型。只有 Selective RGID 改善伪实例后，新的 Region Oracle 或无参数关系审计通过，才允许训练这一层。

### 2.5 AffectedOccurrenceResidual：插入损毁 logits 之后

位置：

```text
P3 damage logits + fixed S2候选
        ↓
通过审计的 affected evidence
        ↓
AffectedOccurrenceResidual
        ↓
最终 logits
```

作用：只推动 `Intact → Affected`，同时给 Damaged 和 Destroyed 加相同偏移：

```text
z_damaged'   = z_damaged   + delta
z_destroyed' = z_destroyed + delta
```

因此 `z_destroyed' - z_damaged'` 与原模型完全相同，不破坏已有严重度边界。偏移只在 detached 建筑支持区内生效，且 `gain=0` 时严格复现原 logits。

限制：它是最终安全注入接口，不是当前立即训练的模块。只有某个 Evidence Verifier 在独立 calibration/confirm 上通过，才打开它。

## 3. 当前立即运行的实验：Strip-RGID

### 3.1 本地/服务器预检

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_structured_plugins_v1/run.sh preflight
bash experiments/spi_structured_plugins_v1/run.sh smoke
```

### 3.2 训练 500 optimizer steps

```bash
nohup env \
SEED=7341 \
STEPS=500 \
STRIP_KERNEL=19 \
BATCH_SIZE=1 \
ACCUMULATION_STEPS=4 \
NUM_WORKERS=4 \
bash experiments/spi_structured_plugins_v1/run.sh train-strip \
> /root/data/BRIGHT-train-logs/SPI_STRIP_RGID_V1_seed7341.log 2>&1 &
```

查看训练：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_STRIP_RGID_V1_seed7341.log
```

### 3.3 用现有 Selective RGID-E0 审计新头

训练结束后，不直接访问 `meta_confirm/Val450/test`，而是把新 checkpoint 送入现有 Selective RGID 流程：

```bash
nohup env \
SEED=7341 \
RGID_STEPS=500 \
RGID_CKPT=/root/data/BRIGHT-checkpoints/SPI_STRUCTURED_PLUGINS_V1/strip_rgid_step500_seed7341/rgid_step_500_calibrated.pth \
RGID_PROVENANCE=/root/data/BRIGHT-checkpoints/SPI_STRUCTURED_PLUGINS_V1/strip_rgid_step500_seed7341/provenance.json \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_SELECTIVE_STRIP_RGID_E0_V1/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_SELECTIVE_STRIP_RGID_E0_V1 \
bash experiments/spi_selective_rgid_e0_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_SELECTIVE_STRIP_RGID_E0_seed7341.log 2>&1 &
```

比较对象必须保持为：

```text
R0 原 LocCC + fixed S2
R1 全部分裂 + fixed S2
R2 Selective Strip-RGID + fixed S2
R3 同数量随机分裂 + fixed S2
R4 Split Oracle（只算上限）
```

只有 `R2 > R0、R1、R3`，事件稳定且 bootstrap 下界大于 0，才能保留 Strip-RGID。500 步只是 pilot；pilot 不通过不能靠扩展到 3000 步挽救，除非训练日志明确显示欠拟合而不是机制失效。

## 4. 后续实验顺序

```text
E0 复现 P3 + fixed S2
 ↓
E1 原 RGID 与 Strip-RGID 的候选/Oracle对照
 ↓ 只有Selective Strip-RGID通过
E2 在高风险LocCC切割带审计 Local Matching
 ↓ 只有AUROC、事件稳定性通过
E3 fixed SLIC / 子区域 Relation E0
 ↓ 只有Region Oracle与无参数证据通过
E4 单层 RegionRelationBlock 或低容量Verifier
 ↓
E5 AffectedOccurrenceResidual 安全注入
 ↓
meta_confirm 一次性确认
 ↓
最后才访问 Val450
```

`CrossScaleBoundaryFusion` 是 Stage3 特征接口准备好后的独立竞争实验，不与 Strip-RGID 同轮开启。FSA 可以重新定位为 Stage3 SAR 纹理探针，但此前的 post-fusion FSA 已证明几乎未打开，不能把旧结果解释为有效模块；RCM 当前版本有显著负增益，应停止。

## 5. 代码文件

- `bda_benchmark/model/structured_plugins/blocks.py`：五个独立模块。
- `bda_benchmark/model/structured_plugins/rgid_head.py`：原 RGID/Strip-RGID 工厂与兼容加载器。
- `bda_benchmark/model/structured_plugins/test_blocks.py`：形状、零初始化、支持区和严重度不变性测试。
- `bda_benchmark/experiments/spi_structured_plugins_v1/train_strip_rgid.py`：冻结 P3、只训练 Strip-RGID。
- `bda_benchmark/experiments/spi_structured_plugins_v1/run.sh`：预检、smoke、正式训练入口。

## 6. 绝对不能同时做的事情

- 不同时开启 Strip-RGID、CrossScale、Matcher、Relation 和 Memory。
- 不让 RGID 的梯度进入 P3。
- 不让局部匹配直接覆盖 SAR feature。
- 不在 E0 阶段把 Oracle/GT 特征送入推理模块。
- 不因训练集 AUC 好就访问 `meta_confirm`。
- 不用同一份 calibration 反复挑模块、核大小和阈值。

## 7. Selective RGID加速与断点恢复

原实现每张图对5个阈值分别构造R2/R3，并反复调用完整fixed S2。完整S2会额外生成`6×H×W`诊断特征，但本实验只需要最终类别。这造成大量无效内存写入和CPU计算。

加速版保持S2接受规则、融合强度、R0–R4、随机种子及阈值选择完全不变，只做以下等价替换：

- prediction-only fixed S2，不再生成未使用的诊断图；
- Split Oracle由逐建筑全图扫描改为`bincount`；
- Selective/Random分区由逐建筑mask循环改为编码后的向量化重编号；
- 每`STATE_EVERY`张保存一次原子断点，默认25张；
- 重启时校验P3、RGID、数据名单、阈值网格和随机种子指纹，然后从下一张继续。

断点文件自动位于：

```text
<policy-or-result-json>.partial.pt
```

安全启动建议：

```bash
nohup env NUM_WORKERS=1 STATE_EVERY=25 ... \
bash experiments/spi_selective_rgid_e0_v1/run.sh audit > run.log 2>&1 &
```

若进程被杀，使用完全相同的命令重启，会出现：

```text
Resuming Selective RGID fit at image N
```

不得改变checkpoint、名单、阈值、seed或minimum area后强行读取旧断点；代码会拒绝不一致的状态。
