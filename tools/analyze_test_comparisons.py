"""Recompute selected-crop metrics from label IDs; prepare evidence-only zooms."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

work=Path('D:/software/dowmload/BRIGHT-master/research_reports/test_evidence_details')
assets=work/'build/assets';assets.mkdir(parents=True,exist_ok=True)
src=Path('D:/software/dowmload/BRIGHT-master/research_reports/test_comparison_ppt/build/data.json')
data=json.loads(src.read_text(encoding='utf-8'))
z=zipfile.ZipFile('D:/software/dowmload/BRIGHT-master/BRIGHT_test_sixmodels_images.zip')
gtz=zipfile.ZipFile('F:/建筑损毁分类数据集/target.zip')
index={Path(n).name:n for n in gtz.namelist() if n.endswith('.tif')}
models=data['models'];total={m:np.zeros((4,4),np.int64) for m in models}
arrays={};rows=[]
def metrics(cm):
    tp=np.diag(cm);gt=cm.sum(1);pred=cm.sum(0)
    divide=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    iou=divide(tp,gt+pred-tp);p=divide(tp,pred);r=divide(tp,gt)
    return dict(iou=iou.tolist(),miou3=float(iou[1:].mean()),damaged_precision=float(p[2]),
        damaged_recall=float(r[2]),damaged_f1=float(divide(2*tp,gt+pred)[2]),
        damaged_as_intact=int(cm[2,1]),intact_as_damaged=int(cm[1,2]))
for item in data['rows']:
    sid=item['sample_id'];x,y,w,h=item['xywh']
    gt=np.array(Image.open(io.BytesIO(gtz.read(index[f'{sid}_building_damage.tif']))))[y:y+h,x:x+w]
    preds={};stats={}
    for m in models:
        pred=np.array(Image.open(io.BytesIO(z.read(f'predictions/{m}/{sid}_labels.png'))))[y:y+h,x:x+w]
        assert set(np.unique(pred)).issubset({0,1,2,3})
        mask=gt<4;cm=np.bincount(4*gt[mask].astype(int)+pred[mask],minlength=16).reshape(4,4)
        total[m]+=cm;stats[m]=metrics(cm);preds[m]=pred
    a=preds['Ours_A1'];b=preds['DamageFormer']
    recovery=int(((gt==2)&(b==1)&(a==2)).sum())
    reverse=int(((gt==2)&(b==2)&(a==1)).sum())
    row=dict(**item,metrics=stats,damaged_recovered_from_intact=recovery,damaged_new_intact_errors=reverse,
        delta_miou3=stats['Ours_A1']['miou3']-stats['DamageFormer']['miou3'],
        delta_damaged_iou=stats['Ours_A1']['iou'][2]-stats['DamageFormer']['iou'][2])
    rows.append(row);arrays[sid]=(gt,preds)
for m in models:
    reported=next(r for r in data['records'] if r['name']==m)['confusion_visual_subset']
    assert total[m].tolist()==reported,m
summary={m:metrics(cm) for m,cm in total.items()}
print('SUMMARY',json.dumps(summary,indent=2))
print('CASES',[(r['sample_id'],round(r['delta_miou3']*100,2),round(r['delta_damaged_iou']*100,2)) for r in rows])
# Diagnostic examples chosen by observed differences. Not a representative random sample.
chosen=[]
for key,reverse,kind in [('delta_miou3',True,'Largest foreground mIoU gain'),('delta_damaged_iou',True,'Damaged IoU gain'),('delta_damaged_iou',False,'Damaged IoU regression'),('delta_miou3',False,'Foreground mIoU regression')]:
    row=next(r for r in sorted(rows,key=lambda r:r[key],reverse=reverse) if r['sample_id'] not in [x['sample_id'] for x in chosen])
    row=dict(row);row['case_kind']=kind
    gt,preds=arrays[row['sample_id']];a=preds['Ours_A1'];b=preds['DamageFormer'];sz=gt.shape[0];win=max(128,sz//4)
    if key=='delta_damaged_iou':
        score=((gt==2)&(a==2)&(b!=2)).astype(int)-((gt==2)&(b==2)&(a!=2)).astype(int)
        score-=((gt!=2)&(a==2)&(b!=2)).astype(int)
        score+=((gt!=2)&(b==2)&(a!=2)).astype(int)
    else:score=((gt>0)&(a==gt)&(b!=gt)).astype(int)-((gt>0)&(b==gt)&(a!=gt)).astype(int)
    options=[(int(score[yy:yy+win,xx:xx+win].sum()),xx,yy) for yy in range(0,sz-win+1,32) for xx in range(0,sz-win+1,32)]
    value,xx,yy=(max(options) if reverse else min(options));box=(xx,yy,xx+win,yy+win)
    row['zoom_xywh']=[xx,yy,win,win];row['zoom_rule']='Extremal signed error-change sum on a 32-pixel grid; diagnostic selection after prediction.'
    row['zoom_paths']=[]
    for i,p in enumerate(row['paths']):
        f=assets/f'{row["sample_id"]}_zoom_{i}.png';Image.open(p).crop(box).save(f);row['zoom_paths'].append(str(f))
    row['heatmap_paths']=[]
    for m in models:
        f=assets/f'{row["sample_id"]}_{m}_heatmap.png'
        f.write_bytes(z.read(f'predictions/{m}/{row["sample_id"]}_damaged_probability.png'))
        row['heatmap_paths'].append(str(f))
    chosen.append(row)
report=dict(scope='18 selected Test crops ONLY. Pooled pixel counts, not whole Test or event-balanced metrics.',
    summary=summary,rows=rows,cases=chosen,
    improved_crops=sum(r['delta_miou3']>0 for r in rows),worse_crops=sum(r['delta_miou3']<0 for r in rows),
    recovered=sum(r['damaged_recovered_from_intact'] for r in rows),new_misses=sum(r['damaged_new_intact_errors'] for r in rows))
(work/'build/analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('SELECTED',[(r['sample_id'],r['case_kind']) for r in chosen])
