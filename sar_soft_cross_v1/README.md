# SAR-Q × soft：匹配组合训练工具

这是**新的组合实验**，复用现有区域路由模型和训练器。代码解压在 `/root/shared-nvme/BRIGHT-master/New`；结果新建于 `/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_SOFT_CROSS_V1`。原仓库、旧结果、原模型和Memory库保持只读。没有真实BRIGHT／CUDA成绩的预先承诺。

## 实验问题与文献依据

主问题：经过相同预算训练的 SAR-Q，与建筑概率 soft 路由能否互补？旧 soft +0.3311pp 和旧 SAR +0.1370pp 不属于一个匹配组合，不能相加。

- SSN 的可微区域分配说明Q可以由任务训练；SpixelFCN 的语义重建提供以后约束损毁边界的依据。
- BRIGHT UMIM／CFOG 支持单独检查跨模态空间对应，不把SAR亮度直接当作损毁标签。
- STViT 通过区域汇总、回投和像素残差保留细节；MCIBI 将像素与Memory上下文融合。这些支持后续联合分类方向，不是本项目已验证的收益。
- 本轮**只训练区域读出组合**，不训练新像素联合头或Memory Reader。它是最终像素＋区域＋Memory方案之前的一项分阶段实验。

完整源码证据、固定论文代码链接与访问限制见 [RESEARCH.md](RESEARCH.md)。源码中出现soft不代表都是同一模块：Q软归属、建筑soft路由、区域软标签、Reader attention应分别说明。

## 两段实验，不混淆预算

前驱是已发布的 `bright_sar_q_matched_v1.zip`：baseline／rgb_capacity／shifted_sar／sar_cue 四组Q，各2000次更新、batch2×accum8、512裁块，冻结A1与原下游。它回答新增SAR对Q的作用。

本包使用前驱中**完整通过的baseline和sar_cue末步last.pth**。两者都经过相同2000步Q训练；不拿未继续训练的E0 Q充当baseline。先重新检查四组完整预测档案；不要求SAR先正涨分，允许进一步探索交互。

新阶段冻结Q，对每个Q分别重新训练四个路由头：

|Q来源|unsplit|capacity_control|hard|soft|
|---|---|---|---|---|
|匹配baseline-Q|A|棋盘对照|硬分组|C|
|匹配SAR-Q|B|棋盘对照|硬分组|D|

四头复制相同的E0 embedding/classifier初始权重，参数量相同。unsplit是K槽、9路；另外三组是2K槽、18路。棋盘对照用于检查“增加槽位与分组”的影响，不是增加更多分类器参数。hard按预测B>=0.5二分；soft使用Q×(1−B)、Q×B，floor=0。每组只更新自己的embedding/classifier。

每个Q源共享一次冻结前向和同一整图Train流供四头学习；每头2000次整图更新、AdamW lr1e-4、wd0.005、NLL+Lovasz、梯度裁剪1、warmup/cosine，FP32读出。两Q源顺序运行，图像序列哈希必须一致。加载Q后统一重置E0 seed+31，避免两个Q构造器消耗不同随机数；因此它是新匹配实验，不冒充旧soft的精确重跑。

Train2918是采样池；每头2000次整图更新不等于遍历2918张至少一轮，也不能与Q阶段2000×16裁块的曝光量混称相同。第一轮沿用2000步作为匹配筛选；若需延长，所有对照一起用新的完整学习率计划重训。

注意：改变Q也会改变边界/熵及其注入后的adapted features。本实验测试整条Q前端干预与路由的组合，不能把差异全部解释为只换了一个矩阵而所有特征数值不变。

## 为什么使用独立适配器

原 `building_route_v1` 仅允许building_reconstruction来源，本包不伪造这一身份。`core_adapter.py` 在独立子进程内显式替换训练/评价模块的来源加载、实验格式与代码哈希引用，允许**正式baseline或sar_cue**。源Q模块、原core与磁盘文件不改；退出或异常时恢复所有引用。

原Q检查点完整性、冻结A1/下游、E0复现1e-5、原生池化、初始化unsplit概率一致性等检查保留。新格式 `SRAC_SAR_SOFT_ROUTE_V1` 单独记录，不覆盖旧实验。代码不替换CUDA算子，不放宽容差，不自动重试或回退CPU。

## 自动运行流程

1. 固定一项新版SAR四组任务。默认选择SRAC_SAR_Q_MATCHED_V1中名字最新的声明任务；打印并锁定来源，失败时不会自动改用更老任务。
2. 若该任务仍在运行，组合父进程每30秒读取状态等待，不建立CUDA上下文。若前驱失败，组合停止。若任务被人为结束却没写failure，会保持等待；先检查前驱日志，等待本身不能证明它仍运行。
3. 全量重读四组训练与Val450存档，生成来源证书。不只看450/450进度。
4. 原路由CPU预检；通过nvidia-smi等目标GPU连续两次空闲。它只观察空闲、不预留GPU、不杀已有任务。
5. 两个Q源各四头，先各2步真实smoke＋2图独立验证；两源smoke均通过才正式训练。
6. 两个Q源顺序各四头2000步，训练过程中初始和末步完整Val；每个训练结束另起进程回读heads_last.pth，再做独立Val450。
7. 重新核对两源的450张存档预测：所有历史路径、Q-A1和四路由的混淆矩阵、图序、真值、每事件统计、训练初始权重及完整图像流。
8. 输出八格指标、主四格差异/交互、Damaged P/R、每类救回/误改和逐事件结果。

所有worker均CUDA_LAUNCH_BLOCKING=1，CUBLAS/TF32等继承原E0；原数值策略不变。等待、CPU测试、smoke、正式训练、完整验证分开标记。任一步失败保留新现场并停止后续，无resume开关。

## 下载与启动

前驱包：[固定GitHub下载](https://github.com/dadadadadadaning/bright/raw/7c80abec37330aa79b4eda2e79c5a512d30cf97d/bright_sar_q_matched_v1.zip)。如果它尚未启动，先按前驱包README启动；本包不会把缺失前驱当作已完成。

把本包 **bright_sar_soft_cross_v1.zip** 上传到：

```
/root/shared-nvme/BRIGHT-master/New/bright_sar_soft_cross_v1.zip
```

按发布消息校验ZIP SHA256，然后：

```bash
(
  set -euo pipefail
  PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
  ZIP=/root/shared-nvme/BRIGHT-master/New/bright_sar_soft_cross_v1.zip
  test -s "$ZIP"
  UNPACK="$(mktemp -d /root/shared-nvme/BRIGHT-master/New/sar_soft_XXXXXXXX)"
  "$PYTHON" -m zipfile -e "$ZIP" "$UNPACK"
  bash "$UNPACK/sar_soft_cross_v1/start_server.sh"
)
```

脚本检查包内SHA、运行CPU测试，再创建后台队列，打印准确的结果和日志路径。代码保存在New，不会安装覆盖bda_benchmark。

如需指定某个新版SAR来源，启动脚本支持 `SAR_SOURCE_ROOT=/完整结果目录` 环境变量；不指定时选最新任务并打印。不要把旧semantic_sar目录或某个last.pth直接传入，它需要整套四组结果目录。

只查看计划（不需要数据、不启动训练）：

```bash
"$PYTHON" "$TOOL/run.py" plan
```

实际TOOL由启动器打印，关闭终端后可按打印的完整路径调用。报告：

```bash
"$PYTHON" "$TOOL/run.py" report --root "$OUTPUT_ROOT"
```

`report`是只读，即使失败也能读；不要把占位变量直接复制到未设置变量的新终端。启动器提供了完整展开后的命令。

## 怎样读结果

查看 `comparison/report.json`：

- `cells`：baseline和sar各四路由，前景mIoU与四类IoU，Damaged precision/recall；百分比单位。
- `effects`：B−A、C−A、D−C、D−B及D−A；差值单位百分点。
- `effects.interaction_pp`：D−B−C+A。只表示mIoU尺度上的交互，不是独立像素显著性检验，也不是跨种子稳定性证明。
- `effects.within_q_controls`：各Q下soft−capacity_control与soft−hard。
- `per_event`、`paired_transitions`：分事件与像素救回/误改。像素数不是逐栋建筑数。
- 原独立Val的 `result.json` 还保存预测B分层下的hard→soft统计。这些真值分层仅作事后诊断。

预定主指标为Val前景三类mIoU，单列背景和三损毁类IoU。不能换用Test四类平均，也不能将两条实验的不同最佳中间步拿来比较。

保留组合需要D相对B、C在匹配重复中有可靠增益；交互正值不是硬门槛，组合最好但次可加也可能有实际意义。若D−C不正，说明当前设置中SAR没有给soft额外收益；若D−B不正，soft没有给SAR额外收益。第一轮无收益不等于否定全部SAR或soft方案。

八格回答组合与路由问题。原前驱的RGB/错位SAR对照仍使用冻结旧头，不足以单独证明**在soft下**SAR空间线索的机制。如果八格出现候选收益，下一阶段需在同一soft训练协议下加入rgb_capacity-Q和shifted_sar-Q，再做匹配种子重复。本包当前没有自动跑这两个追加组合，避免预先扩成16头。

## 后续如何接语义重建和Memory

1. 若错误仍集中于Intact/Damaged混合，另建“有/无Train四类语义重建×有/无SAR”的Q因素实验，其他条件相同。已有四类重建失败/未完成记录不能当正向证据。
2. 选定Q与路由后，同规则生成Train支持与当前查询的key/value/软标签。新Q或2K槽位会改变区域身份，不能直接复用旧Q、旧槽位的Memory成员映射。
3. 保留像素F256，区域回投提供图内上下文，Reader提供历史案例；用训练过的联合分类头共同决定四类，无拒绝门控。
4. 比较同容量图内参照、T128检索、F256检索等匹配组合；收益后做随机支持/标签打乱及训练种子重复。当前正在安排的旧Q R0/RT/RF实验不要中途换Q或库。

本包完成是实验工具可运行，不代表真实训练已运行或结果为正。机器、数据、正式来源和CUDA运行仍以服务器日志及最后的完整报告为准。
