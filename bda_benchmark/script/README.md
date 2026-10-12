# BRIGHT 脚本目录

## 入口分组

| 目录/文件模式 | 用途 | 对应模型 |
|---|---|---|
| `standard_ML/train_UNet.py` 等 | 原始基线训练 | UNet、DeepLabV3Plus、Siam 系列、DamageFormer |
| `standard_ML/train_DamageFormer_BG*.py` | 建筑先验与原型实验 | BG、BGMP、AutoK、Aux、V2、K3 |
| `standard_ML/train_Hierarchical_RGBSAR.py`、`train_S5_S8*.py` | 分层 RGB-SAR 差异实验 | Hierarchical RGBSAR、S5、S8 |
| `standard_ML/train_DARM.py`、`train_DamageFormer_IARM*.py` | 实例/记忆实验 | DARM、IARM |
| `standard_ML/train_Priority_Modular.py` | 模块化优先区域实验 | Priority Modular |
| `cross_event/` | one-shot / zero-shot 域适应 | DamageFormer、MeanTeacher、AdaptSeg、SiamCRNN |
| `distillation/` | 新的教师/学生统一入口 | 注册表内完整任务模型 |
| `_archive/` | 历史备份，只用于追溯 | 不参与新实验 |

## 新脚本规则

1. 不再直接写个人机器的 `/home/...` 地址。
2. 模型通过 `bda_benchmark.model.create_model` 创建。
3. 权重、日志、评估目录通过 `script.common.paths` 解析。
4. 蒸馏代码只接收统一的 `DistillationOutput`，不根据模型文件名重复判断返回格式。
5. 现有训练脚本名称暂不改变，以保持 shell 命令和实验记录可复现。

## 推荐调用

从项目根目录运行：

```bash
python -m bda_benchmark.script.distillation.inspect_pair --list
```

旧脚本仍可从 `bda_benchmark` 目录按原方式运行，例如：

```bash
python script/standard_ML/train_UNet.py ...
```

