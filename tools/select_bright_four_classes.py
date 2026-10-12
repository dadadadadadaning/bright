"""Select validation crops by GT coverage and RGB quality, never model scores.

Cloud screening is a conservative RGB heuristic, not a cloud segmentation model.
Every accepted crop still needs human review before comparison inference.
"""
import argparse
import io
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import compare_bright_checkpoints as c

ARCHIVES={}
def read_image(root,folder,sid,suffix):
    relative=f'{folder}/{sid}_{suffix}.tif'
    path=Path(root)/relative
    if path.is_file():return np.array(Image.open(path))
    archive=Path(root)/f'{folder}.zip'
    if archive not in ARCHIVES:
        z=zipfile.ZipFile(archive)
        ARCHIVES[archive]=(z,{Path(n).name:n for n in z.namelist() if n.endswith('.tif')})
    z,index=ARCHIVES[archive]
    return np.array(Image.open(io.BytesIO(z.read(index[f'{sid}_{suffix}.tif']))))

def read_pair(root,sid):
    return (read_image(root,'pre-event',sid,'pre_disaster')[...,:3],
            read_image(root,'post-event',sid,'post_disaster'),
            read_image(root,'target',sid,'building_damage'))


def largest(mask):
    labels,n=ndi.label(mask)
    return int(np.bincount(labels.ravel())[1:].max()) if n else 0


def assess(rgb,gt,args):
    counts=[int((gt==k).sum()) for k in range(4)]
    minimum=[args.min_background]+[args.min_class_pixels]*3
    if any(n<m for n,m in zip(counts,minimum)):
        return None,'insufficient_class_pixels'
    components=[largest(gt==k) for k in (1,2,3)]
    if min(components)<args.min_component:
        return None,'small_class_components'
    rgb=rgb.astype(np.float32)
    lum=rgb@np.array([.2126,.7152,.0722],np.float32)
    fg=(gt>0)&(gt<4)
    dark=float((lum<40).mean());dark_fg=float((lum[fg]<40).mean())
    invalid=float((rgb.max(-1)<=2).mean())
    if invalid>.005 or max(dark,dark_fg)>args.max_dark:
        return None,'dark_or_invalid'
    # Large, bright, low-saturation, locally smooth RGB areas may be cloud/haze.
    # Bright roofs and concrete can trigger this; thin/dark cloud can be missed.
    hi=rgb.max(-1);lo=rgb.min(-1)
    sat=(hi-lo)/np.maximum(hi,1)
    mean=ndi.uniform_filter(lum,size=15)
    std=np.sqrt(np.maximum(ndi.uniform_filter(lum*lum,size=15)-mean*mean,0))
    suspect=(lum>170)&(sat<.20)&(std<20)
    suspect=ndi.binary_opening(suspect,iterations=2)
    cloud_largest=largest(suspect)/gt.size
    if cloud_largest>args.max_cloud_component:
        return None,'suspected_cloud_or_bright_surface'
    return dict(class_pixels=counts,largest_class_components=components,
        dark_fraction=dark,building_dark_fraction=dark_fg,near_black_fraction=invalid,
        suspected_cloud_fraction=float(suspect.mean()),largest_suspected_cloud_fraction=cloud_largest),None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset')
    p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--allow-test',action='store_true')
    p.add_argument('--crop',type=int,default=512)
    p.add_argument('--crops',type=int,nargs='+',help='Search multiple display crop sizes, e.g. 512 768 1024')
    p.add_argument('--audit-only',action='store_true',help='GT class coverage only; no visual quality claim')
    p.add_argument('--stride',type=int,default=128)
    p.add_argument('--per-event',type=int,default=2)
    p.add_argument('--min-class-pixels',type=int,default=512)
    p.add_argument('--min-background',type=int,default=1024)
    p.add_argument('--min-component',type=int,default=128)
    p.add_argument('--max-dark',type=float,default=.10)
    p.add_argument('--max-cloud-component',type=float,default=.01)
    args=p.parse_args()
    if min(args.crop,args.stride,args.per_event,args.min_class_pixels,args.min_background,args.min_component)<=0:
        p.error('Sizes, class minima and counts must be positive')
    if args.split=='test' and not args.allow_test:
        p.error('Test requires --allow-test; do not use Test for training or tuning.')
    ids=(c.ROOT/f'bda_benchmark/dataset/splitname/standard_ML/{args.split}_set.txt').read_text().split()
    missing=[]
    for sid in ids:
        for folder,suffix in [('pre-event','pre_disaster'),('post-event','post_disaster'),('target','building_damage')]:
            f=Path(args.data)/folder/f'{sid}_{suffix}.tif'
            if not f.is_file() and not (Path(args.data)/f'{folder}.zip').is_file():missing.append(str(f))
    if missing:
        raise FileNotFoundError(f'{len(missing)} input/GT files missing. A split list is not the dataset. First missing paths: '+str(missing[:8]))
    print(f'{args.split.upper()}: {len(ids)} samples; all input/GT paths exist.',flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    best=[];reasons=Counter();events=Counter();coverage={}
    for i,sid in enumerate(ids):
        event=sid.rsplit('_',1)[0];events[event]+=1
        # Check labels first; most images without all classes cannot qualify.
        gt=read_image(args.data,'target',sid,'building_damage')
        ec=coverage.setdefault(event,dict(images=0,images_with_class=[0]*4,images_meeting_all_class_minima=0,quality_qualified_images=0))
        ec['images']+=1
        for k in range(4):ec['images_with_class'][k]+=int(np.any(gt==k))
        if any(int((gt==k).sum())<m for k,m in enumerate([args.min_background]+[args.min_class_pixels]*3)):
            reasons['image_missing_class_or_area']+=1;continue
        ec['images_meeting_all_class_minima']+=1
        if args.audit_only:continue
        rgb,sar,gt=read_pair(args.data,sid)
        h,w=gt.shape
        options=[]
        positions=[]
        for sz in (args.crops or [args.crop]):
            if sz<=0:raise ValueError('Crop sizes must be positive')
            if min(h,w)<sz:continue
            xs=sorted(set(list(range(0,w-sz+1,args.stride))+[w-sz]))
            ys=sorted(set(list(range(0,h-sz+1,args.stride))+[h-sz]))
            positions.extend((sz,x,y) for y in ys for x in xs)
        for sz,x,y in positions:
                info,reason=assess(rgb[y:y+sz,x:x+sz],gt[y:y+sz,x:x+sz],args)
                if reason:reasons[reason]+=1;continue
                # Prefer substantial representation of all foreground classes.
                score=min(info['class_pixels'][1:])/float(sz*sz)-info['largest_suspected_cloud_fraction']-0.1*info['building_dark_fraction']
                options.append(dict(sample_id=sid,event=event,xywh=[x,y,sz,sz],
                    approved=False,review_status='PENDING_CLOUD_AND_SHADOW_REVIEW',selection_score=score,**info))
        if options:
            best.append(max(options,key=lambda s:s['selection_score']))
            ec['quality_qualified_images']+=1
        print(f'{i+1}/{len(ids)}: {sid}, qualifying crops={len(options)}',flush=True)
    selected=[]
    for event in events:
        group=sorted((s for s in best if s['event']==event),key=lambda s:s['selection_score'],reverse=True)
        selected.extend(group[:args.per_event])
    for s in selected:
        sid=s['sample_id'];rgb,sar,gt=read_pair(args.data,sid);x,y,w,h=s['xywh']
        crop=lambda a:a[y:y+h,x:x+w]
        strip=np.concatenate([crop(rgb),np.repeat(crop(c.sar_display(sar))[...,None],3,axis=2),crop(c.colorize(gt))],axis=1)
        Image.fromarray(strip).save(out/f'{sid}_RGB_SAR_GT.png')
    c.write_json(out/'selection.json',dict(protocol=f'four_class_{args.split}_visual_subset',split=args.split,filters=vars(args),
        note='All four GT IDs present with area thresholds. RGB cloud heuristic ONLY. Human cloud/shadow review required. This subset is NOT full split performance.',samples=selected))
    c.write_json(out/'screening_report.json',dict(images_scanned=len(ids),qualifying_images=len(best),
        selected=len(selected),rejection_counts=dict(reasons),event_coverage=coverage,audit_only=args.audit_only,
        events_without_qualifying_images=[e for e in events if not any(s['event']==e for s in best)]))
    print(f'Done: {len(selected)} candidates. Review PNGs; set approved=true ONLY for cloud-free suitable crops in {out / "selection.json"}.',flush=True)
    if not selected:print('No qualifying samples. Requirements were NOT relaxed. Consider a new run with --crop 768 or --crop 1024.')
    if args.audit_only:print('AUDIT ONLY: visual quality was not assessed. See event_coverage in screening_report.json.')


if __name__=='__main__':main()
