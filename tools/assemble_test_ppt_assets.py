import io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

work=Path('D:/software/dowmload/BRIGHT-master/research_reports/test_comparison_ppt')
assets=work/'build/assets';assets.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile('D:/software/dowmload/BRIGHT-master/BRIGHT_test_sixmodels_images.zip')
samples=json.loads(z.read('predictions/selection.json'))
records=json.loads(z.read('predictions/run_manifest.json'))
models=['UNet','DeepLabV3Plus','SiamAttnUNet','SiamCRNN','DamageFormer','Ours_A1']
assert all(any(r['name']==m and r['status']=='complete' and r['split']=='test' for r in records) for m in models)
sources={k:zipfile.ZipFile(f'F:/建筑损毁分类数据集/{k}.zip') for k in ['pre-event','post-event','target']}
colors=np.array([[255,255,255],[70,181,121],[228,189,139],[182,70,69]],np.uint8)
rows=[]
for s in samples:
    sid=s['sample_id'];x,y,w,h=s['xywh'];box=(x,y,x+w,y+h)
    arr=[]
    for k,suffix in [('pre-event','pre_disaster'),('post-event','post_disaster'),('target','building_damage')]:
        matches=[n for n in sources[k].namelist() if n.endswith(f'{sid}_{suffix}.tif')]
        assert len(matches)==1,(sid,k,matches)
        arr.append(np.array(Image.open(io.BytesIO(sources[k].read(matches[0])))))
    rgb,sar,gt=arr;rgb=rgb[...,:3]
    cropgt=gt[y:y+h,x:x+w]
    assert [int((cropgt==c).sum()) for c in range(4)]==s['class_pixels'],sid
    assert all(v>0 for v in s['class_pixels'])
    lo,hi=np.percentile(sar[np.isfinite(sar)],[1,99])
    sar=np.uint8(np.clip((sar-lo)/max(float(hi-lo),1e-8),0,1)*255)
    col=colors[np.minimum(gt,3)];col[gt==255]=128
    paths=[]
    for name,a in [('RGB',rgb),('SAR',sar),('GT',col)]:
        f=assets/f'{sid}_{name}.png';Image.fromarray(a).crop(box).save(f);paths.append(str(f))
    for m in models:
        a=Image.open(io.BytesIO(z.read(f'predictions/{m}/{sid}_color.png')))
        assert a.size==(gt.shape[1],gt.shape[0])
        f=assets/f'{sid}_{m}.png';a.crop(box).save(f);paths.append(str(f))
    rows.append(dict(**s,paths=paths))
(work/'build/data.json').write_text(json.dumps(dict(rows=rows,models=models,records=records),ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified GT crop counts and exported',len(rows)*9,'individual image tiles')
