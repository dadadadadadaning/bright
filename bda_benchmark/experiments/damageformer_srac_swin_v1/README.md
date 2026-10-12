# DamageFormer / Swin 四阶段区域检索改编 V1

## CUDA空间NLL布局兼容修正

服务器PyTorch2.2.1+cu121异常追踪定位到区域损失NllLoss2DBackward，提示grad_input must be contiguous。
`losses.region_nll` 在调用空间NLL前将log概率显式转为FP32、标准连续NCHW；不改变损失定义和梯度路径。
preflight新增非连续channels-last输入的数值/梯度等价测试，CUDA可用时测试FP32及BF16。
本地CPU通过不代表服务器完整CUDA训练通过，更新后仍需重新冒烟。使用新输出目录，保留失败日志；不需要升级环境。

## 新增：已核验 A1 初始化入口

`run_a1.sh` 显式选择任务权重微调，不要求 SWIN_PRETRAIN。仅接受 SHA256 为
`cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add` 的已核验A1。
完整严格加载model.base，不读取旧optimizer，不迁移成双分支相同权重。
新增区域模块随机初始化；原主模型lr=1e-5，新模块lr=1e-4，均warmup+cosine。
预算仍为50,000新优化器更新，不是从旧15,000步续跑。初始化方式和学习率与通用预训练路线不同，应分表记录。

正式阶段在任何更新之前先完整Val450验证原像素分支；与0.6168659209491562的绝对差超过1e-5即停止。
记录写入regions/initial_a1_audit.json。冒烟只用两张Val，reproduction_pass=null，不代表完整基线通过。
若正式审计失败，检查数据和评估协议，不要直接放宽阈值。原像素分支通过也不代表随机区域分支达到基线。

```bash
cd /root/shared-nvme/BRIGHT-master
conda activate bright4090
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
bash bda_benchmark/experiments/damageformer_srac_swin_v1/run_a1.sh preflight
export RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/DAMAGEFORMER_SRAC_SWIN_A1_V1/smoke_seed3407
bash bda_benchmark/experiments/damageformer_srac_swin_v1/run_a1.sh smoke_all
```

冒烟成功后，更换RUN_ROOT至全新formal_seed3407，执行run_a1.sh regions仅训练前端，或pipeline依次执行全部阶段。
磁盘保存方式未改变：best/last均带优化器；正式流程约3GiB，建议至少5GB可用。不删除任何旧文件。
本次增加源文件会改变代码哈希，不能混用修改前本实验的checkpoint/cache。旧实验目录未修改。

独立实验，不修改旧实验，不是 SRAC 官方复现，不承诺论文分数。只允许 Train / Val；未提供 Test 入口。

## 实际实现

1. DamageFormer 双 Swin 编码器从同一 torchvision Swin-T 通用预训练初始化。RGB 四尺度特征进入小型 UNet++ 式嵌套解码器，三次像素/中心软分配更新生成局部区域归属。它是 SSN 思路的可微局部聚类，不是官方 VMambaSSN。没有专门的紧致度监督，也不保证硬区域连通；训练依靠损毁分类经 Pool/Unpool 反传。
2. 预测建筑概率衰减背景；硬超像素边界与建筑概率参与特征调制。不是在原始影像上压暗后重新编码。建筑概率与硬边界在此 detach；定位分支另有监督。没有旧 S3 的建筑/背景双组，避免重复约束。
3. 空间注意力加权汇总 + 原融合特征 mean/std 旁路，产生128维区域Token。区域四类概率按软归属回投，再上采样为全图预测。此版最终输出不是 A1 像素残差；保留像素头仅辅助监督和诊断。
4. 冻结最佳前端，在完整 Train 图像上提取区域。事件×多数类别均衡的随机蓄水池每桶最多64个真实区域，不是平均原型。保存Token、四类软标签、事件、图像、区域ID。标签只进入支持 Value，不参与 query 的最近邻选择。余弦 Top-8 在线精确检索，排除同事件及同图。无合格邻居时退回自身Token。

本版不加入地理/灾害类型嵌入、几何嵌入、对比学习、多原型，也没有原论文的 MMAP 预计算邻居表。使用小规模真实样本库与分块检索，接口更易审计。构库背景区域也保留，不能使用 GT 前景筛掉推理区域。

## 训练与对照

阶段A：前端联合训练50,000次优化器更新，batch4×累积4=16，800,000 crops；AdamW lr1e-4、wd5e-3、warmup1000+cosine、梯度裁剪1，默认BF16。训练crop512，完整Val450图1024、FP32，每1000步验证。

总损失 = 定位CE+Lovasz + 区域NLL+Lovasz + 0.25×(像素CE+Lovasz)。辅助权重是本实现预设，不是论文复现参数。该协议不等同旧T0，不能将差异全部归因于某一个模块。

阶段B：冻结前端，用 Train 原图确定性提取并分片保存。区域软标签来自 GT 四类分数面积下采样后按归属汇总；权重为特征网格有效质量，**不是精确原图像素数量**。忽略标签不进入分布。

阶段C：分别训练相同参数量的无历史支持头、历史支持头，各5000步，每500步全图Val。均匀抽取Train图像后抽128个区域，区域质量加权软标签CE。两头从相同随机种子初始化。无Memory头用可学习槽位，不读取历史；计算量不完全相同。前端始终冻结，避免Memory特征过期。

阶段D：固定最佳模型验证区域原头、无Memory头、真实Memory、随机邻居、打乱支持标签、关闭支持。后三项只是同一模型的推理干预，不是重新训练对照；默认单次随机种子，只供诊断，不代表统计显著性。

最终指标为完整图像四类混淆矩阵及前景三类mIoU，含背景误报；同时输出每事件结果。Memory是否有增益首先比较真实Memory和同容量无Memory头，再比较阶段A原头。共用前端不能替代完整方法对T0的公平预算对照。

## 服务器操作

上传本目录完整内容到 `/root/data/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_swin_v1`。依赖现有仓库中的 joint_region_v1、iterative_partition_v1、association 和 DamageFormer代码，不能单独移出仓库。

```bash
cd /root/data/BRIGHT-master
conda activate bright4090
bash bda_benchmark/experiments/damageformer_srac_swin_v1/run.sh preflight
```

从历史T0读取真实通用预训练路径并核验哈希；失败先修路径，不能用任务best.pth代替：

```bash
export SWIN_PRETRAIN="$(python - <<'PY'
import json, hashlib
from pathlib import Path
p=Path('/root/data/BRIGHT-checkpoints/DAMAGEFORMER_JOINT_REGION_IMAGE_V2/formal50000_seed3407/t0/manifest.json')
d=json.loads(p.read_text())
w=Path(d['arguments']['pretrain'])
if not w.is_file(): raise SystemExit(f'预训练不存在: {w}')
if hashlib.sha256(w.read_bytes()).hexdigest()!=d['pretrain_sha256']:
    raise SystemExit('预训练哈希不匹配')
print(w)
PY
)"
test -s "$SWIN_PRETRAIN" && echo 'PRETRAIN OK'
```

全流程冒烟测试：前端20步，四张跨事件Train构库，两个头各20步，Val只前两张。不能判断性能。

```bash
export RUN_ROOT=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_SRAC_SWIN_V1/smoke_seed3407
bash bda_benchmark/experiments/damageformer_srac_swin_v1/run.sh smoke_all
```

成功后，用全新目录串行完整训练（不会停止已有GPU任务）：

```bash
export RUN_ROOT=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_SRAC_SWIN_V1/formal_seed3407
mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_SRAC_SWIN_V1
nohup bash bda_benchmark/experiments/damageformer_srac_swin_v1/run.sh pipeline \
 > /root/data/BRIGHT-train-logs/DAMAGEFORMER_SRAC_SWIN_V1/formal_seed3407.log 2>&1 &
echo "PID=$!"
```

也可以按顺序分别运行 `regions`、`build_memory`、`train_heads`、`evaluate`，共享同一个RUN_ROOT。建议先查看前端完整Val结果，再决定是否构库。`pipeline` 则自动执行全部阶段。正式前端50000步，两个头各5000步，不是多个前端各训练50000步。

## 隔离及失败处理

- 输出必须不存在；不覆盖旧结果，不删除旧缓存，不支持训练中途续跑。
- 每次训练仅覆盖自身best.pth，结束保存last.pth；不保存每步checkpoint。意外中断时last不存在是设计行为。
- 阶段绑定前端权重、代码版本、Train/Val列表、Memory哈希。更换前端或编辑代码后必须重建匹配缓存，不能跳过校验。
- 新代码上传前后的换行变化也可能改变版本哈希；正式训练期间不要再编辑/上传源码。
- 所有torch缓存及权重须来自可信来源；载入包含Python对象的文件不能来自未知下载。
- 如果训练batch4显存不足，允许BATCH=2 ACCUM=8或BATCH=1 ACCUM=16，保持EBS16；这些配置仍须先冒烟验证。
- 本地CPU契约测试不等于真实数据、CUDA、BF16测试。实际GPU资源、数据文件及运行时长由服务器冒烟确认。

## 本轮要回答的问题

是否能够在Swin体系中训练完整区域分类路径？区域加权统计能否保留信息？在同一冻结Token空间中，Train历史证据是否比相同容量的无历史分类器更好？正确邻居/标签是否必要？

这是完整改编的可执行起点，不代表每一设计已被证实。若区域路径仍显著弱于同协议像素基线，应先分析全图误差，不用区域分类准确率代替像素mIoU。
