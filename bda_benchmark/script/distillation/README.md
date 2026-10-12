# 蒸馏脚本入口

此目录只放教师/学生共有的蒸馏入口，不再复制某个历史训练脚本。

## 1. 列出模型

```bash
python -m bda_benchmark.script.distillation.inspect_pair --list
```

## 2. 检查教师/学生和权重兼容性

```bash
python -m bda_benchmark.script.distillation.inspect_pair \
  --teacher damageformer_iarm \
  --teacher-checkpoint /path/to/teacher.pth \
  --student damageformer_s1c \
  --student-checkpoint /path/to/student_init.pth \
  --offline \
  --experiment KD_IARM_to_S1C
```

命令会检查模型是否可创建、教师是否冻结、权重键是否完全匹配，并打印统一的权重、日志和评估目录。默认不会创建目录；确认无误后增加
`--create-artifact-dirs`。

## 3. 在训练代码中使用

```python
from bda_benchmark.model.distillation import DistillationPair

pair = DistillationPair.from_registry(
    "damageformer_iarm",
    "damageformer_s1c",
    offline=True,
)
outputs = pair(pre_image, post_image)
teacher_out = outputs["teacher"]
student_out = outputs["student"]
```

教师始终处于 `eval` 状态且不计算梯度；学生保持正常训练。可以先做
logit KD，再通过 `feature_maps` 增加特征蒸馏。

