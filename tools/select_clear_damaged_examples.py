"""Find diagnostic recovery examples in the existing 18-crop predictions only."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label,find_objects

root=Path('D:/software/dowmload/BRIGHT-master')
work=root/'research_reports/test_reselected_examples';assets=work/'build/assets';assets.mkdir(parents=True,exist_ok=True)
data=json.loads((root/'research_reports/test_evidence_details/build/analysis.json').read_text(encoding='utf-8'))
z=zipfile.ZipFile(root/'BRIGHT_test_sixmodels_images.zip');gz=zipfile.ZipFile('F:/建筑损毁分类数据集/target.zip')
index={Path(n).name:n for n in gz.namelist() if n.endswith('.tif')}
ranked=[]
for row in data['rows']:
 sid=row['sample_id'];x,y,w,h=row['xywh']
 gt=np.array(Image.open(io.BytesIO(gz.read(index[sid+'_building_damage.tif']))))[y:y+h,x:x+w]
 b=np.array(Image.open(io.BytesIO(z.read(f'predictions/DamageFormer/{sid}_labels.png'))))[y:y+h,x:x+w]
 a=np.array(Image.open(io.BytesIO(z.read(f'predictions/Ours_A1/{sid}_labels.png'))))[y:y+h,x:x+w]
 labs,n=label(gt==2)
 candidates=[]
 for ci,slices in enumerate(find_objects(labs),1):
  if slices is None:continue
  sy,sx=slices;mask=labs[sy,sx]==ci;num=int(mask.sum())
  if num<64:continue
  aa=a[sy,sx][mask];bb=b[sy,sx][mask]
  recovery=int(((aa==2)&(bb==1)).sum());reverse=int(((aa==1)&(bb==2)).sum())
  recA=float((aa==2).mean());recB=float((bb==2).mean());missB=float((bb==1).mean())
  if recA<.4 or missB<.25 or recovery<50:continue
  size=max(96,int(max(sx.stop-sx.start,sy.stop-sy.start)*1.5));size=min(size,w,h)
  xx=max(0,min(w-size,(sx.start+sx.stop-size)//2));yy=max(0,min(h-size,(sy.start+sy.stop-size)//2))
  gg=gt[yy:yy+size,xx:xx+size];ap=a[yy:yy+size,xx:xx+size];bp=b[yy:yy+size,xx:xx+size]
  def iou(p):return float(((gg==2)&(p==2)).sum()/max(1,((gg==2)|(p==2)).sum()))
  ia,ib=iou(ap),iou(bp)
  newerr=int(((gg<4)&(bp==gg)&(ap!=gg)).sum());fix=int(((gg<4)&(ap==gg)&(bp!=gg)).sum())
  if ia<.25 or ia-ib<.1 or fix<=newerr:continue
  candidates.append(dict(component=ci,area=num,recovered=recovery,reverse=reverse,recall_a=recA,recall_b=recB,baseline_intact_fraction=missB,zoom_xywh=[xx,yy,size,size],iou_a=ia,iou_b=ib,new_errors=newerr,corrected=fix,score=(ia-ib)*np.sqrt(recovery)))
 for item in sorted(candidates,key=lambda c:c['score'],reverse=True)[:2]:
  r=dict(row,local=item);ranked.append(r)
ranked.sort(key=lambda r:r['local']['score'],reverse=True)
for k,r in enumerate(ranked):
 box=r['local']['zoom_xywh'];xx,yy,ww,hh=box;r['zoom_paths']=[]
 for col in [0,1,2,7,8]:
  file=assets/f'candidate_{k+1}_{col}.png';Image.open(r['paths'][col]).crop((xx,yy,xx+ww,yy+hh)).save(file);r['zoom_paths'].append(str(file))
 # Only assemble existing evidence images; do not synthesize prediction pixels.
 canvas=Image.new('RGB',(1200,240),'white')
 for j,p in enumerate(r['zoom_paths']):canvas.paste(Image.open(p).resize((240,240),Image.Resampling.NEAREST),(j*240,0))
 canvas.save(assets/f'candidate_{k+1}_review.png')
 print(k+1,r['sample_id'],json.dumps(r['local']))
(work/'build/candidates.json').write_text(json.dumps(ranked,ensure_ascii=False,indent=2),encoding='utf-8')
