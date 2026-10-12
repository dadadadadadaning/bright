# D0/A0/A1 一次性锁定 Test878

本流程只用于三个已经冻结模型的最终测试：

- D0：原始 DamageFormer 父模型；
- A0：相同 P3 实现中的像素路径训练对照；
- A1：A0 加 soft 建筑 ROI。

其中真正归因 soft ROI 的比较是 `A1 - A0`，而不是 `A1 - D0`。

先检查权重和代码，不读取测试图像：

```bash
bash experiments/d0_a0_a1_test_locked_v1/run.sh preflight
```

确认三项权重都已冻结且不再根据测试结果调参后，只运行一次：

```bash
ALLOW_TEST_ACCESS=YES \
bash experiments/d0_a0_a1_test_locked_v1/run.sh test_locked
```

结果保存在：

```text
/root/data/BRIGHT-eval-results/D0_A0_A1_TEST878_LOCKED_V1/
```

流程会优先复用此前已经生成的 D0/A0 锁定结果，因此通常只需要新增 A1 推理。
