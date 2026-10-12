"""Real paired-image previews; GT-masked SLIC baseline, NOT learned regions."""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries


def read_image(z, name):
    return np.array(Image.open(io.BytesIO(z.read(name))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='F:/建筑损毁分类数据集')
    p.add_argument('--out', required=True)
    p.add_argument('--events', nargs='+', default=['turkey-earthquake', 'bata-explosion', 'beirut-explosion'])
    p.add_argument('--min-area',type=int,default=6000)
    args = p.parse_args()
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    archives = {k: zipfile.ZipFile(root / v) for k, v in {
        'rgb': 'pre-event.zip', 'sar': 'post-event.zip',
        'ann': 'cvprw26_trainval_instance_labels.zip'}.items()}
    rgb_names, sar_names = set(archives['rgb'].namelist()), set(archives['sar'].namelist())
    records = []
    for event in args.events:
        candidates = []
        for name in sorted(archives['ann'].namelist()):
            if not name.endswith('.json') or '/' + event + '_' not in name:
                continue
            doc = json.loads(archives['ann'].read(name).decode('utf-8-sig'))
            sid = doc['images'][0]['sample_id']
            rn, sn = f'pre-event/{sid}_pre_disaster.tif', f'post-event/{sid}_post_disaster.tif'
            if rn not in rgb_names or sn not in sar_names:
                continue
            for a in doc['annotations']:
                x, y, w, h = a['bbox']
                if args.min_area <= a['area'] <= 45000 and min(w, h) > 25 and max(w, h) < 400 and x > 25 and y > 25 and x+w < 999 and y+h < 999:
                    candidates.append((abs(a['area']-16000), name, doc, a, rn, sn))
            if len(candidates) >= 30:
                break
        if not candidates:
            from scipy.ndimage import label as cc_label, find_objects
            with zipfile.ZipFile(root/'target.zip') as targets:
                names=[n for n in targets.namelist() if '/'+event+'_' in n and n.endswith('.tif')][:40]
                for name in names:
                    sid=Path(name).name.replace('_building_damage.tif','')
                    rn,sn=f'pre-event/{sid}_pre_disaster.tif',f'post-event/{sid}_post_disaster.tif'
                    if rn not in rgb_names or sn not in sar_names: continue
                    gt=read_image(targets,name)
                    cc,_=cc_label((gt>0)&(gt<4))
                    for bid,box in enumerate(find_objects(cc),1):
                        if box is None: continue
                        ys,xs=box; h,w=ys.stop-ys.start,xs.stop-xs.start
                        m=cc==bid; area=int(m.sum())
                        if area<400 or min(w,h)<15 or max(w,h)>400: continue
                        a={'id':bid,'category_id':int(np.bincount(gt[m].astype(int)).argmax()),'bbox':[xs.start,ys.start,w,h],'area':area,'segmentation':[],'_mask':m}
                        candidates.append((abs(area-16000),name,{'images':[{'sample_id':sid}]},a,rn,sn))
                    if len(candidates)>=30: break
        if not candidates:
            continue
        _, name, doc, a, rn, sn = min(candidates, key=lambda x: x[0])
        rgb, sar = read_image(archives['rgb'], rn), read_image(archives['sar'], sn)
        assert rgb.shape[:2] == sar.shape[:2]
        mask_img = Image.new('L', (rgb.shape[1], rgb.shape[0]))
        draw = ImageDraw.Draw(mask_img)
        for poly in a['segmentation']:
            draw.polygon(list(zip(poly[::2], poly[1::2])), fill=1)
        mask = a.get('_mask',np.array(mask_img, dtype=bool))
        x,y,w,h = a['bbox']
        side = int(max(w,h)+64)
        left = max(0, min(rgb.shape[1]-side, int(x+w/2-side/2)))
        top = max(0, min(rgb.shape[0]-side, int(y+h/2-side/2)))
        sl = np.s_[top:top+side, left:left+side]
        rgb, sar, mask = rgb[sl], sar[sl], mask[sl]
        base = rgb.astype(float)/255
        if sar.ndim == 3:
            sar = sar[..., 0]
        finite = sar[np.isfinite(sar)]
        lo, hi = np.percentile(finite, [1,99])
        sar_display = np.clip((sar.astype(float)-lo)/max(float(hi-lo),1e-6),0,1)
        labels = slic(base, n_segments=8, compactness=10, sigma=1,
                      mask=mask, start_label=1, enforce_connectivity=True)
        ids, counts = np.unique(labels[mask], return_counts=True)
        selected = int(ids[np.argmax(counts)])
        yy,xx = np.indices(mask.shape)
        mx,my = np.where(mask)[1].min(), np.where(mask)[0].min()
        bw,bh = np.ptp(np.where(mask)[1])+1,np.ptp(np.where(mask)[0])+1
        choices=[]
        for nx in range(1,7):
            for ny in range(1,7):
                g=1+np.minimum((xx-mx)*nx//bw,nx-1)+nx*np.minimum((yy-my)*ny//bh,ny-1)
                g=np.where(mask,g,0)
                n=len(np.unique(g[mask]))
                choices.append((abs(n-len(ids)),abs(bw/nx-bh/ny),g,n))
        _,_,grid,ng=min(choices,key=lambda v:v[:2])
        outline=mark_boundaries(base,mask.astype(int),color=(1,.45,.55),mode='outer')
        grid_view=mark_boundaries(base,grid,color=(1,1,1),mode='inner')
        content=mark_boundaries(base,labels,color=(1,1,1),mode='inner')
        chosen=base.copy()*.65
        hit=labels==selected
        chosen[hit]=.6*base[hit]+.4*np.array([1,.35,.45])
        chosen=mark_boundaries(chosen,labels,color=(1,1,1),mode='inner')
        sid=doc['images'][0]['sample_id']
        dest=out/sid
        dest.mkdir()
        arrays=[base,sar_display,outline,grid_view,content,chosen]
        titles=['Pre-event RGB','Post-event SAR','GT building outline',f'Masked grid ({ng} regions)',f'RGB SLIC ({len(ids)} regions)',f'Selected SLIC region {selected}']
        for i,(arr,title) in enumerate(zip(arrays,titles)):
            Image.fromarray((arr*255).clip(0,255).astype('uint8')).save(dest/f'{i+1:02d}.png')
        np.savez_compressed(dest/'regions.npz',slic_labels=labels,grid_labels=grid,gt_mask=mask,sar_raw=sar)
        fig,axes=plt.subplots(2,3,figsize=(10,8.5))
        for ax,arr,title in zip(axes.flat,arrays,titles):
            ax.imshow(arr,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            ax.set_title(title,fontsize=12)
            ax.axis('off')
        fig.suptitle(sid,fontsize=14)
        fig.text(.5,.025,'Real paired imagery | GT-mask-assisted SLIC baseline, NOT model output\nSame pixel crop; SAR 1–99% display stretch. No alignment correction or learned weights.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.025,right=.975,bottom=.12,top=.88,wspace=.08,hspace=.20)
        fig.savefig(dest/'preview.png',dpi=180)
        fig.savefig(dest/'preview.pdf')
        plt.close(fig)
        records.append(dict(sample_id=sid,annotation=name,instance_id=a['id'],category_id=a['category_id'],
                            crop_xywh=[left,top,side,side],rgb_member=rn,sar_member=sn,
                            sar_display_percentiles=[float(lo),float(hi)],slic_regions=len(ids),grid_regions=ng,
                            selected_region=selected,selection='first 30 eligible buildings per event; area closest to 16000 pixels',
                            caveat='GT-assisted baseline only; not predicted mask, learned Q, memory attention or damage prediction'))
        print(dest/'preview.png',flush=True)
    (out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    for z in archives.values():
        z.close()
    if not records:
        raise RuntimeError('No eligible pairs found')


if __name__ == '__main__':
    main()
