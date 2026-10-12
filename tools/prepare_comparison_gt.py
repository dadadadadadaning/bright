"""Export exact GT crops for the two existing preview samples."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('research_reports/figures')
out=root/'comparison_gt_v1'
out.mkdir(exist_ok=False)
palette=np.array([[255,255,255],[70,181,121],[228,189,139],[182,70,69]],dtype=np.uint8)
with zipfile.ZipFile('F:/建筑损毁分类数据集/target.zip') as z:
    for row in json.loads((root/'real_region_preview_v2/manifest.json').read_text(encoding='utf-8'))[:2]:
        name='target/'+row['sample_id']+'_building_damage.tif'
        label=np.array(Image.open(io.BytesIO(z.read(name))))
        x,y,w,h=row['crop_xywh']
        label=label[y:y+h,x:x+w]
        assert set(np.unique(label)).issubset({0,1,2,3,255})
        color=np.zeros((*label.shape,3),dtype=np.uint8)+128
        for c in range(4): color[label==c]=palette[c]
        Image.fromarray(color).save(out/(row['sample_id']+'.png'))
        print(row['sample_id'],np.unique(label,return_counts=True))
