# BRIGHT：现有权重统一推理与定性对比

这是推理代码，不是重新训练代码。已有 `.pth` 时，先统一推理得到真实预测图。
不能把不同来源权重的比较称为“同协议重新训练的公平对比”。不包含 ChangeOS。

## 1. 在服务器仓库根目录执行

将 `tools/compare_bright_checkpoints.py` 上传到服务器对应仓库的 tools 目录。
使用你原先能运行模型的 Python 环境。脚本依赖 torch、torchvision、numpy、Pillow、matplotlib，
模型本身另需原训练环境的 timm/VMamba 等依赖。不建议为此覆盖现有环境。

```bash
python tools/compare_bright_checkpoints.py inspect \
  --checkpoint /root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth \
  > best_checkpoint_inspection.json

python tools/compare_bright_checkpoints.py init \
  --checkpoints /root/shared-nvme/BRIGHT-checkpoints \
  --output comparison_models.json
```

`inspect` 只读取你自己的可信权重。其输出包含元数据和参数形状，不进行预测。
**文件名 best_val_miou.pth 不能识别模型结构。先检查训练配置，再填写 manifest 中 Ours。**
若元数据不足，需要提供保存该权重的训练脚本、配置或启动命令。

## 2. 填写模型配置（禁止猜测 Ours 的结构）

生成的 manifest 已包含五个基线的构造器、输入方式和损毁输出索引；核对服务器文件名。
ChangeMamba 默认 disabled，因为必须指定与训练一致的 YAML 和可工作的 VMamba 环境。
填写其 `cfg`、必要的 `opts`，再将 `enabled` 改为 true；不是忽略这个对照。

Ours 的 `checkpoint` 已指向用户提供的最佳文件。其余字段必须根据真实训练配置填写。
若它确实是**未经修改的原版 DamageFormer**，才可填：

```json
{
  "name": "Ours",
  "factory": "bda_benchmark.model.DamageFormer:DamageFormer",
  "kwargs": {"use_pretrained_swin": false},
  "input_mode": "pair",
  "output_index": 1,
  "checkpoint": "/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth",
  "enabled": true
}
```

这只是格式示例，不是对该权重架构的判断。修改过的模型使用其真实 `模块:类名` 和构造参数。
如果模型有额外 Memory 缓存或只保存插件参数，应复用原推理封装；本脚本会拒绝把插件权重当作完整网络。
若输出是字典或需要多步后处理，应写明确的适配器返回四分类 logits，不能随便指定 output_index。
所有权重均严格加载，不允许缺失参数后继续比较。非权重参数如 roi_alpha 同样要核对。
模型实例化只取消 ResNet ImageNet 初始化下载，随后用完整 checkpoint 严格覆盖，不改变其结构。

## 3. 先选输入图，不看模型预测选图

`--data` 指向**已解压的数据根目录**，下有 pre-event、post-event、target 三个文件夹。
将下面的数据路径替换为服务器真实路径；它不是 checkpoint 路径。

```bash
python tools/compare_bright_checkpoints.py select \
  --data /YOUR/EXTRACTED/BRIGHT_DATA \
  --output comparison_candidates_v1 \
  --crop 512 --per-event 1 --max-dark 0.15 --max-invalid 0.01
```

使用仓库 standard_ML/val_set.txt，不访问 Test。每个事件最多一个候选；没有合格样本的事件不强行补齐。
每幅图检查九个固定位置的裁剪，过滤 RGB 近黑像素、整体暗区及建筑暗区比例。
SAR 暗响应不作为排除依据。这只是暗区启发式，**不是真实阴影检测器**。
GT 仅用于可视化/建筑像素数量与质量统计，不参与模型预测；这会形成筛选样本偏差。

打开输出目录的 `*_RGB_SAR_GT.png`，确认建筑清楚、没有半幅阴影或黑边后，
在 `selection.json` 中把该条 `approved` 改成 true。不合适的保留 false。
筛选规则固定在看预测之前；不要看哪个模型好再选择。

## 4. 推理与出图

```bash
python tools/compare_bright_checkpoints.py run \
  --data /YOUR/EXTRACTED/BRIGHT_DATA \
  --manifest comparison_models.json \
  --selection comparison_candidates_v1/selection.json \
  --output comparison_predictions_v1 \
  --device cuda:0
```

先调通一个基线可加 `--models DamageFormer`，使用另一个新的输出目录。
CPU 可指定 `--device cpu`，但完整图推理非常慢；ChangeMamba 依赖其实际支持的运行环境。
不会因某个模型失败而偷偷省略该列。缺文件、架构不匹配、输出异常都会报错。
输出目录必须不存在，避免覆盖原结果；失败的目录保留供诊断，下次换新目录。

统一使用原始整图前向，再裁剪到同一展示窗口；没有缩小输入、TTA、建筑投票或定位掩码后处理。
RGB 和三通道重复 SAR 使用仓库的 ImageNet 均值/标准差。SAR 百分位拉伸仅用于显示。
若某个 checkpoint 训练时采用不同输入预处理，必须先改成其正确配置，不能强行套用本协议。

产物：

- `comparison_*.png/pdf`：每页最多五个事件，RGB｜SAR｜GT｜各启用模型，Ours 在最后。
- 各模型 `*_labels.png`：原始类别 ID（0/1/2/3），不是 RGB 彩图。
- `*_color.png`：整图预测，背景白、完整绿、损毁棕、毁坏红。
- `*_probabilities.npz`：真实四分类 softmax 概率，形状 4×H×W。
- `*_damaged_probability.png`：真实 P(Damaged) 热力图，统一 0–1 色条。
- `run_manifest.json`：checkpoint SHA256、配置、完成状态、展示裁剪混淆矩阵。

这里的概率热力图不是 Attention，也不是区域 Q 或 Memory 读取权重。
**筛选案例的混淆矩阵不能作为全量 Val mIoU；不能以此声称模型最好或达到 61.8%。**
论文定量表需要未筛选的完整 Val/Test 协议；重新训练对比还需统一训练集、预算、种子与选模准则。

## 5. 当前需要确认的信息

1. 最佳 checkpoint 的 inspect 输出以及对应训练配置。
2. 服务器已解压数据的真实根目录。
3. ChangeMamba 的训练 YAML 与模型代码版本。

本工具不启动训练、不改权重、不伪造预测，不将官方 DamageFormer 权重冒充改进模型。
