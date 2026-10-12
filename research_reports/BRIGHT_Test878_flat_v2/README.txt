全部文件放在同一文件夹，例如 /root/shared-nvme/BRIGHT-gallery-tools/。
原有模型代码和数据不需要移动。

conda activate /root/shared-nvme/conda-envs/bright4090
cd /root/shared-nvme/BRIGHT-gallery-tools
python -u build_test878_gallery.py --repo-root /root/shared-nvme/BRIGHT-master --data /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset --manifest server_models.json --out /root/shared-nvme/BRIGHT-Test878-gallery-v2 --action all --device cuda:0

--repo-root 指向包含 bda_benchmark 和 ChangeMamba-master 的源代码仓库。
--data 指向包含 pre-event/post-event/target 三个目录或三个zip的数据目录。
ChangeMamba使用指定权重中的B2 DiffFusion结构及config_dump，不冒充原版。
权重哈希不同则停止该模型，其他模型错误记录在 inference_errors.json。
本机只验证配置和打包，GPU前向需要服务器环境；缺少模块时不会降低strict加载要求。
结果中的 summary.json 用于确认各模型是否完成878张，不以脚本结束代表全部成功。
