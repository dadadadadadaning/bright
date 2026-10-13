# BRIGHT 像素＋超像素＋Memory 联合训练工具包

独立工具包分支，包含服务器训练入口与原始ZIP，不包含数据或权重。

- 下载根目录 `bright_region_memory_joint_train_v1.zip`（网页右侧 Download raw file）。
- 上传到服务器 `/root/shared-nvme/bright_region_memory_joint_train_v1.zip`。
- 按 `region_memory_joint_train_v1/README.md` 解压并运行 `start_server.sh`。

流程：预检 → 真实20步smoke → R0/RT/RF各5000更新 → 完整Val450。
使用现有V3 FP32库，固定前端/Q/库；像素、区域与历史案例共同分类，无拒绝门控。

本地50项CPU合约与驱动测试通过。尚未进行真实BRIGHT/CUDA训练；服务器smoke负责实际验证。

ZIP SHA256：

```
dbd01c6c8b6dcb598c0ed0c3817d15a33041c875dd124c058bdfeb206a307576
```
