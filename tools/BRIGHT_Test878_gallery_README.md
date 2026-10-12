# BRIGHT 878张Test全图对比与手工挑选

本地浏览页使用全部标准Test ID，按灾害类型、事件和ID排序。RGB/SAR/GT使用整张输入，不裁局部、不按预测好坏筛选。
每张图同时显示UNet、DeepLabV3Plus、SiamAttnUNet、SiamCRNN、DamageFormer、ChangeMamba、A1。
没有真实预测的模型格明确标注PREDICTION MISSING。标灰不是预测背景。

## 当前交付状态

本地有878张原始数据，只有18个案例的六模型预测归档；ChangeMamba尚未核实训练YAML，未推理。
本地没有CUDA。完整六/七模型预测需要在服务器已有环境运行。

## 在服务器补齐

将新脚本tools/build_test878_gallery.py上传到服务器仓库tools目录。
复用仓库中tools/compare_bright_checkpoints.py；代码包附有当前读取版本，若服务器文件有自己的修改，先比较而不要直接覆盖。
将server_models.json上传到仓库根目录。其配置来自已完成的18案例推理记录，包含权重SHA256。
不要删除哈希校验来绕过错误；权重改变时需要先确认架构和非权重参数，再建立独立记录。

以下数据路径必须包含pre-event、post-event、target三个目录，或三个同名zip。

```bash
cd /root/shared-nvme/BRIGHT-master
conda activate /root/shared-nvme/conda-envs/bright4090
python -u tools/build_test878_gallery.py \
  --data /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --manifest /root/shared-nvme/BRIGHT-master/server_models.json \
  --out /root/shared-nvme/BRIGHT-Test878-gallery-v1 \
  --action all --device cuda:0
```

该过程冻结权重，逐模型执行整图前向，无训练、TTA和额外后处理。与原18案例脚本采用同样的输入标准化。
输出严格要求为1x4xHxW。错误架构、异常输出、缺依赖、显存不足都会记录，禁止随机初始化缺失参数后继续。
每个完成的原始类别PNG会落盘，可以按同一命令断点继续。程序核对缓存模型配置和SHA256，避免混用权重。
脚本完成不等于所有模型成功；务必查看summary.json和inference_errors.json。

如已有原18案例压缩包，可在首次运行加以下参数复用这些完整预测：

```text
--import-zip /实际路径/BRIGHT_test_sixmodels_images.zip
```

ChangeMamba默认为disabled，输出仍保留缺失格。若需要它参与，先将server_models.json中对应项cfg填为实际训练YAML，并将enabled设为true。
不能将A1放在ChangeMamba标题下面。

## 查看与选择

将整个输出目录下载后打开index.html。不要只下载HTML。
页面可筛选灾害类型、事件、样本编号，或仅显示六模型齐全的案例。
点击图片打开整张拼图，勾选后点击“导出已选编号”。浏览器允许时会将选择保存在本地；导出TXT是可靠备份。

主要文件：
- comparisons/灾害类型/事件/样本.png：每个样本的全图模型对比。
- predictions/模型/样本_labels.png：原始类别图，可用于后续高分辨率出图。
- inventory.csv / inventory.json：全量文件索引与缺失清单。
- summary.json：已生成拼图数、各模型预测数和完整案例数。
- inference_errors.json：模型推理错误；仅推理时生成。

## 对已选案例生成更清晰的整图

仍使用原输出目录，利用其中原始类别缓存；加--ids只导出选定ID，浏览页也将更新为选定子集。
建议先备份原index.html和inventory文件，或重新不加--ids执行assemble恢复全量浏览页。

```bash
python tools/build_test878_gallery.py \
  --data /实际数据目录 \
  --out /root/shared-nvme/BRIGHT-Test878-gallery-v1 \
  --action assemble --ids selected_test_ids.txt --tile 1024
```

精选Test例图只用于展示案例。应同时保留失败案例，论文定量结论以完整Test评估为准。
