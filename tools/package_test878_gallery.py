"""Validate and bundle the offline gallery and server continuation tools."""
import json
from pathlib import Path
import subprocess
import zipfile
from PIL import Image
import build_test878_gallery as g

root=Path(__file__).resolve().parents[1]
out=root/'research_reports/BRIGHT_Test878_gallery_20261009'
g.import_archive(root/'BRIGHT_test_sixmodels_images.zip',out/'predictions',[])
rows=json.loads((out/'inventory.json').read_text(encoding='utf-8'))
ids=set((root/'bda_benchmark/dataset/splitname/standard_ML/test_set.txt').read_text().split())
assert {r['sample_id'] for r in rows}==ids and len(rows)==878
for row in rows:
    assert row['image'] and 'error' not in row
    with Image.open(out/row['image']) as im:
        im.verify()
subprocess.run(['node','-e',
    "const fs=require('fs');const t=fs.readFileSync(process.argv[1],'utf8');new Function(t.split('<script>')[1].split('</script>')[0]);console.log('Gallery JavaScript syntax OK')",
    str(out/'index.html')],check=True)
with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_STORED) as z:
    for f in out.rglob('*'):
        if f.is_file():
            z.write(f,f.relative_to(out))
with zipfile.ZipFile(out.parent/'BRIGHT_Test878_server_tools.zip','x',zipfile.ZIP_DEFLATED) as z:
    for name in ['build_test878_gallery.py','compare_bright_checkpoints.py','BRIGHT_Test878_gallery_README.md']:
        z.write(root/'tools'/name,'tools/'+name)
    z.write(out/'server_models.json','server_models.json')
print('878 images verified; offline gallery and server tools packaged.')
