# 网格、SLICO与学习区域的公平可视化

截图中的网格和SLIC使用GT建筑轮廓作掩码，只适合说明区域形态。`visualize_partition_compare.py`改用模型预测建筑范围，并让网格、SLICO和学习区域共享同一范围、特征分辨率和近似区域预算。

GT不进入区域生成，只用于从验证集选择诊断案例、裁剪显示区域和计算事后指标。脚本拒绝名称含`test`的列表。

## 运行

```bash
cd /root/data/BRIGHT-master
conda activate bright4090

python -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.visualize_partition_compare \
  --dataset_path /root/data/BRIGHT-master/bda_benchmark/dataset \
  --val_list /root/data/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/val_set.txt \
  --a1_ckpt /root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth \
  --checkpoint /root/data/BRIGHT-checkpoints/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3/p5_anchored_change_seed3407/stage1_change/best.pth \
  --output_dir /root/data/BRIGHT-eval-results/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3/visual_compare \
  --focus_class 2 \
  --device cuda:0
```

也可以直接使用实验入口：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
bash experiments/damageformer_rgb_anchored_change_superpixel_v3/run.sh visualize p5_anchored_change
```

可通过`SAMPLE_ID=...`、`FOCUS_CLASS=2`或`CHECKPOINT=...`覆盖默认设置。

不指定`--sample_id`时，脚本会从给定验证列表中选择Damaged像素较多的案例。也可以增加：

```bash
  --sample_id haiti-earthquake_00000xxx
```

该样本必须属于传入的验证列表。

## 输出

- `*_partition_comparison.png`和`.pdf`：两行八面板对比图；
- `*_partition_arrays.npz`：预测建筑、网格、SLICO、学习区域和关联熵；
- `*_partition_comparison.json`：同一样本上的区域数量、纯度、边界召回和affected retention。

图中的“Learned hard view of Q”只是把软关联取最大值后用于观察边界；模型训练和特征池化仍使用软概率Q。
