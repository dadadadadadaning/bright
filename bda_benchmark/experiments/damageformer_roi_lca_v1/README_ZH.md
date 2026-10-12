# DamageFormer ROI-LCA V1

这是一套与 P3、Bag、Memory、层级分类完全隔离的原始 DamageFormer 消融。

## 模型组

- `D0`：已经训练好的原始 DamageFormer，只作为共同父权重和评价基线。
- `a2_lca3`：建筑扩展 ROI 内进行 3×3 局部相关匹配，不使用熵置信度。
- `a3_conf3`：A2 + 匹配熵置信度。
- `a4_orient3`：A3 + Scharr 绝对方向响应组成的匹配 Key。
- `a5_conf5`：A3 的 5×5 搜索窗口版本。

四个新模型都通过零初始化残差接入，因此 step 0 必须与 D0 完全一致。训练时 D0
参数全部冻结，只训练新增对齐分支。人工 ±4/±8 像素单轴 SAR 位移用于训练，
验证集只在审计阶段施加位移。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

export D0_CKPT=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_PARENT_V1/D0_original_damageformer_final50000_seed3407/best_val_miou.pth

bash experiments/damageformer_roi_lca_v1/run.sh preflight

for v in a2_lca3 a3_conf3 a4_orient3 a5_conf5; do
  bash experiments/damageformer_roi_lca_v1/run.sh smoke "$v"
done
```

5000步顺序后台运行：

```bash
mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_LCA_V1
nohup bash experiments/damageformer_roi_lca_v1/run_sequential.sh \
  probe a2_lca3 a3_conf3 a4_orient3 a5_conf5 \
  > /root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_LCA_V1/launcher_probe5000.log 2>&1 &
echo "launcher PID=$!"
```

查看状态：

```bash
bash experiments/damageformer_roi_lca_v1/run.sh status
tail -f /root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_LCA_V1/launcher_probe5000.log
tail -f /root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_LCA_V1/a2_lca3_probe.log
```

训练完成后执行完整的干净/错位审计：

```bash
TRAIN_TAG=probe5000 bash experiments/damageformer_roi_lca_v1/run.sh audit_all
TRAIN_TAG=probe5000 bash experiments/damageformer_roi_lca_v1/run.sh summarize
python -m json.tool /root/data/BRIGHT-eval-results/DAMAGEFORMER_ROI_LCA_V1/summary_probe5000_seed3407.json
```

## 准入

快速阶段只筛选，不访问 Test。候选至少应满足：干净 Val 相对 D0 不下降，
Damaged F1 不下降，并且 x4/y4/x8/y8 的平均性能下降小于 D0。之后只把最优候选和
必要对照续训到 20000 步：

```bash
bash experiments/damageformer_roi_lca_v1/run.sh confirm a3_conf3
```
