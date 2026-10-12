"""Build a flat launcher bundle from the verified local checkpoint metadata."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import torch

root=Path(__file__).resolve().parents[1]
out=root/'research_reports/BRIGHT_Test878_flat_v2'
out.mkdir(exist_ok=False)
ckpt=Path('D:/software/dowmload/BRIGHT-checkpoints/chagemamba/best_model.pth')
p=torch.load(ckpt,map_location='cpu',weights_only=False)
assert p['args_snapshot']['model_type']=='ChangeMamba-MMBDA_B2_DiffFusion'
assert any('diff_fusion.' in k for k in p['model'])
(out/'changemamba_b2_snapshot.yaml').write_text(p['config_dump'],encoding='utf-8')
h=hashlib.sha256()
with ckpt.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
entries=json.loads((root/'research_reports/BRIGHT_Test878_gallery_20261009/server_models.json').read_text(encoding='utf-8'))
keep=['name','factory','kwargs','input_mode','output_index','checkpoint','enabled','expected_sha256','expected_variant','cfg','opts']
entries=[{k:v for k,v in e.items() if k in keep} for e in entries]
cm=next(e for e in entries if e['name']=='ChangeMamba')
cm.update(factory='changemamba_b2_diff',display_name='ChangeMamba B2 DiffFusion',
          checkpoint='/root/shared-nvme/BRIGHT-checkpoints/chagemamba/best_model.pth',
          cfg='changemamba_b2_snapshot.yaml',enabled=True,expected_sha256=h.hexdigest())
(out/'server_models.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
for name in ['build_test878_gallery.py','compare_bright_checkpoints.py']:
    shutil.copy2(root/'tools'/name,out/name)
(out/'README.txt').write_text('''全部文件放在同一文件夹，例如 /root/shared-nvme/BRIGHT-gallery-tools/。
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
''',encoding='utf-8')
with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
    for f in out.iterdir():z.write(f,f.name)
print(out.with_suffix('.zip'))
