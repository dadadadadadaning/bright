# 直接训练 J1 / J2 / J3（不运行 Q 退火、不训练 J0）

这是上一版 qmetric 联合实验的**三组专用入口**，不是另一套主干或融合结构。
新目录避免改变旧实验的源码哈希；不修改 transformer_v1、joint_context_v1 或 qmetric_v1。

## 每组干什么

| 组 | 同一像素证据与当前区域 | 历史参考 | 回答的问题 |
| --- | --- | --- | --- |
| J1 | 原 A1 Soft ROI 像素特征 + 当前区域 | 可学习槽位，无 Train 历史 | 不查库时的联合分类能力 |
| J2 | 完全相同 | 原余弦 Top-8 | J2−J1：Memory 能否提供增量 |
| J3 | 完全相同 | 学习检索映射后的 Top-8 | J3−J2：只改检索距离有没有帮助 |

三组独立分类器初值相同、训练图像相同。冻结 A1、Q、区域 Embedding、建筑概率B、Memory库。
J3检索映射先用Train拟合2000步，再冻结；它只改变邻居行号，不改变库成员、原始支持特征、支持标签或Reader结构。
联合头使用原来的 `JointContextDecoder`。像素分类块复制A1，区域上下文块零初始化，第0步验证A1身份；训练后像素块也会更新，不保证永远不掉分。
数学上输出仍是 `W_pixel F + W_context C`，不能把此线性分块写成新的非残差融合机制。

正式每组5000次优化器更新、每500步完整Val450；每次读取同一张Train完整1024图像，依次更新三个独立头。
因此日志step5000意味着每个头都更新5000次，不是三组平分5000步；不是三个GPU并行进程。
有效batch=1，无随机裁剪/翻转/旋转，AdamW 1e-4、weight decay5e-3、warmup500+cosine，沿用联合头协议。
不继承旧shell的 `BATCH/ACCUM/VARIANTS/METRIC_CKPT`。它不是前端50,000步、有效batch16的训练。

跳过J0后可以回答Memory两项增量，但不能仅凭这三组声称“区域优于同预算像素微调”。A1只读输出仍会评估，不等于训练J0。

## Q 为什么不在这里顺便硬化

1. one-hot/argmax硬归属：每位置只选一个区域；分类损失不能通过该离散选择直接学习归属，冻结Q时这个梯度限制不影响Reader训练。
2. 建筑/背景硬隔离：同一候选区域按预测B拆分；组内Q仍可为软分配。这与one-hot不是同一件事，也不保证一栋建筑一个区域。
3. 小温度或大beta：仍是软归属，只是分布更尖；过度饱和会减弱梯度、放大错误先验。

硬化可能减少混合，也可能固定错误边界、使弱置信度选择变成确定错误。RGB–SAR配准残差和SAR轮廓外有效证据还可能受过窄支持域伤害。
不能把“每位置只进一组”当作“一个组里面全是建筑/同损毁等级”。
改变用于汇总的Q后，Token、统计量和几何信息会变化；旧Memory及旧检索投影不能直接视为匹配的新库。
若以后做硬Q对照，需要用一致的硬Q生成区域、回投和Train库，重新训练匹配Reader。只在推理时硬化是干预，不是公平训练对照。
SSN原文也将硬归属/连通性处理作为可选、不可微的步骤：[论文](https://varunjampani.github.io/papers/jampani18_SSN.pdf)。

本轮保留原Q，不实现硬化，先回答Memory问题。

## 前提

需要原 `damageformer_srac_transformer_v1` 正式实验的：

- `front/best.pth`：整个前端声明的训练预算已完成；不是只检查文件存在。
- `train_memory/manifest.json` 和 `bank.pt`：来自同一前端的完整Train2918缓存。
- 原始A1文件，哈希一致。

无需先运行 q_anneal 50,000步，也不依赖父流程的Memory/无Memory旧Reader训练结果。
但若父流程的前端或建库尚未完成，这一阶段不能用中途文件替代。
本代码只读校验，不停止父流程，也不会绕过完成状态、源码哈希和切分哈希。

## 一条连续命令：检查 → 检索映射 → 短测 → 三组正式训练

上传**整个本目录**到服务器 `bda_benchmark/experiments/damageformer_srac_joint123_v1`。
依赖上一版 `damageformer_srac_qmetric_v1` 全目录，若尚未上传也须一并上传；不要覆盖原父实验目录。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export PARENT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_TRANSFORMER_V1/formal50000_seed3407_20261008_173616
export J123_EXP=bda_benchmark/experiments/damageformer_srac_joint123_v1
export SEED=3407 DEVICE=cuda:0 WORKERS=4 AMP=bf16

bash "$J123_EXP/run.sh" preflight
bash "$J123_EXP/run.sh" check

unset J123_METRIC_CKPT
export J123_ROOT="/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT123_V1/formal5000_seed3407_$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/shared-nvme/BRIGHT-train-logs/SRAC_JOINT123_V1
export J123_LOG="/root/shared-nvme/BRIGHT-train-logs/SRAC_JOINT123_V1/$(basename "$J123_ROOT").log"

nohup bash "$J123_EXP/run.sh" pipeline > "$J123_LOG" 2>&1 &
echo "训练PID=$!"
echo "结果目录=$J123_ROOT"
echo "日志=$J123_LOG"
tail -F "$J123_LOG"
```

`PARENT_RUN`默认采用之前日志中的目录；若实际正式运行目录不同，改成真实目录。
只在GPU资源允许时启动；脚本不自动终止别的训练。`tail`的Ctrl+C仅停止查看日志，不停止后台训练。

pipeline的顺序：

```text
原前端/Train库只读核验
  -> J3检索映射：2000步（不训练Swin或Q）
  -> 三组短测：各20步，Train四个事件图像、Val两张
  -> 短测成功后从相同初值重新建立三组头
  -> J1/J2/J3各5000步、完整Val450
```

短测使用已经完成的**正式**前端、正式库和正式检索映射；只缩小头的训练/验证样本，不冒充父流程smoke。
短测权重不用于正式初始化，不用于挑选配置，也不能作为正式父权重继续传递。
任何阶段报错，后面停止；不会自动放宽稀有类别跨事件正对要求或删除结果。

已有同前端同库的正式 `metric_last.pth` 时：用其真实路径设置 `J123_METRIC_CKPT`，代替上面unset，pipeline核验后跳过2000步拟合。
不能填其他Memory方法的分类头文件。若已有结果，使用新的J123_ROOT。

## 查看结果

```bash
"$PYTHON" -m bda_benchmark.experiments.damageformer_srac_joint123_v1.report \
  --root "$J123_ROOT/joint"
```

结果依次保存在 `metric/`（若新拟合）、`probe/`、`joint/`。
joint包含 `j1_best/last.pth`、`j2_best/last.pth`、`j3_best/last.pth`，没有J0。
保存训练日志、每类IoU/P/R/F1、每事件结果、逐类检索覆盖/同类Recall@8、J3相对J2的Damaged→Intact伤害与救回。

不复制A1和Train缓存。J3高于J2但低于J1，仍不能声称Memory有效；检索指标改善但分割不改善，也不是方法成功。
缺少J0时，区域相对像素微调的因果对照仍待补充。正式结论还需要多种子，不以像素数当独立重复。
