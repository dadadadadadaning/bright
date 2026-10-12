"""Existing-checkpoint BRIGHT comparison. No training; no test-set tuning.

Run from the repository root. Only load trusted checkpoints (torch pickle).
See tools/BRIGHT_comparison_README.md for the server workflow.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bda_benchmark'))
COLORS = np.array([[255,255,255],[70,181,121],[228,189,139],[182,70,69]], dtype=np.uint8)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def state_of(payload, explicit=None):
    import torch
    obj = payload
    if explicit:
        for key in explicit.split('.'):
            obj = obj[key]
    elif isinstance(obj, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not obj or not all(torch.is_tensor(x) for x in obj.values()):
        raise ValueError('Cannot identify a full state dict. Specify state_key; plugin-only weights need the original loader.')
    if all(k.startswith('module.') for k in obj):
        obj = {k[7:]: v for k,v in obj.items()}
    return obj


def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect(args):
    import torch
    p = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = {}
    if isinstance(p, dict):
        for k,v in p.items():
            if k in ('args','config','variant','epoch','step','best_val_miou','metrics','model_name','stage'):
                meta[k] = str(v)[:12000]
    s = state_of(p)
    print(json.dumps({'checkpoint': args.checkpoint, 'sha256': digest(args.checkpoint),
                      'metadata': meta, 'tensor_count': len(s),
                      'state_shapes': {k:list(v.shape) for k,v in s.items()}}, indent=2))


def init(args):
    base = Path(args.checkpoints)
    specs = [
        ('UNet','UNet',dict(in_channels=6,num_classes=4),'concat',None),
        ('DeepLabV3Plus','DeepLabV3Plus',dict(in_channels=6,num_classes=4),'concat',None),
        ('SiamAttnUNet','SiamAttnUNet',dict(in_channels=3,num_classes=4),'pair',None),
        ('SiamCRNN','SiamCRNN',{},'pair',1),
        ('DamageFormer','DamageFormer',dict(use_pretrained_swin=False),'pair',1),
    ]
    entries = [dict(name=n, factory=f'bda_benchmark.model.{m}:{m}', kwargs=kw,
                    input_mode=mode, output_index=idx,
                    checkpoint=str(base/f'ckpt_{n}_standard_ML_split.pth'), enabled=True)
               for n,m,kw,mode,idx in specs]
    entries.append(dict(name='ChangeMamba', factory='changemamba', cfg='REPLACE_WITH_TRAINING_YAML',
                        kwargs={},input_mode='pair',output_index=1,
                        checkpoint=str(base/'ckpt_ChangeMamba_bright_standard_ML_split.pth'),enabled=False))
    entries.append(dict(name='Ours', factory='REPLACE_WITH_EXACT_TRAINING_MODULE:CLASS', kwargs={},
                        input_mode='pair',output_index=1,
                        checkpoint=str(base/'best/best_val_miou.pth'),enabled=True))
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, entries)
    print('Created manifest. Configure Ours and ChangeMamba before inference:',args.output)


def read_sample(root, sid):
    root = Path(root)
    rgb = np.array(Image.open(root/'pre-event'/f'{sid}_pre_disaster.tif'))[...,:3]
    sar = np.array(Image.open(root/'post-event'/f'{sid}_post_disaster.tif'))
    gt = np.array(Image.open(root/'target'/f'{sid}_building_damage.tif'))
    if sar.ndim != 2 or gt.ndim != 2 or rgb.shape[:2] != sar.shape or sar.shape != gt.shape:
        raise ValueError(f'Unexpected dimensions: {sid}')
    if not set(np.unique(gt)).issubset({0,1,2,3,255}):
        raise ValueError(f'Unexpected label IDs: {sid}')
    return rgb,sar,gt


def sar_display(x):
    lo,hi = np.percentile(x[np.isfinite(x)], [1,99])
    return np.uint8(np.clip((x-lo)/max(float(hi-lo),1e-8),0,1)*255)


def colorize(gt):
    rgb = COLORS[np.minimum(gt,3)]
    rgb[gt==255] = [128,128,128]
    return rgb


def select(args):
    # Deliberately restricted to the checked-in validation split.
    ids = (ROOT/'bda_benchmark/dataset/splitname/standard_ML/val_set.txt').read_text().split()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    selected, rejected, counts = [], [], {}
    for sid in ids:
        event = sid.rsplit('_',1)[0]
        if counts.get(event,0) >= args.per_event:
            continue
        rgb,sar,gt = read_sample(args.data,sid)
        h,w = gt.shape
        size = min(args.crop,h,w)
        choices = []
        for y in sorted(set(np.linspace(0,h-size,3).astype(int))):
            for x in sorted(set(np.linspace(0,w-size,3).astype(int))):
                r = rgb[y:y+size,x:x+size].astype(float)
                g = gt[y:y+size,x:x+size]
                lum = r @ np.array([.2126,.7152,.0722])
                fg = (g>0)&(g<4)
                dark = float((lum<args.dark_threshold).mean())
                invalid = float((r.max(-1)<=2).mean())
                dark_fg = float((lum[fg]<args.dark_threshold).mean()) if fg.any() else 1.
                if invalid>args.max_invalid or max(dark,dark_fg)>args.max_dark or fg.sum()<args.min_building_pixels:
                    continue
                choices.append(dict(sample_id=sid,event=event,xywh=[int(x),int(y),size,size],
                                    dark_fraction=dark,building_dark_fraction=dark_fg,
                                    near_black_fraction=invalid,
                                    class_pixels=[int((g==c).sum()) for c in range(4)],approved=False))
        if not choices:
            rejected.append(sid)
            continue
        item = min(choices,key=lambda v: max(v['dark_fraction'],v['building_dark_fraction']))
        selected.append(item)
        counts[event]=counts.get(event,0)+1
        x,y,s,_ = item['xywh']
        strip = np.concatenate([rgb[y:y+s,x:x+s],
                  np.repeat(sar_display(sar)[y:y+s,x:x+s,None],3,axis=2),
                  colorize(gt[y:y+s,x:x+s])],axis=1)
        Image.fromarray(strip).save(out/f'{sid}_RGB_SAR_GT.png')
    write_json(out/'selection.json',{'protocol':'validation_visual_subset',
        'note':'RGB darkness heuristic, NOT a verified shadow detector. Manually review and set approved=true.',
        'filters':vars(args),'samples':selected,'rejected_images':rejected})
    print(f'{len(selected)} candidates. Review PNGs, set approved=true in {out / "selection.json"}.')


@contextlib.contextmanager
def no_download():
    # Full checkpoints replace these initial weights. Do not download ImageNet weights.
    import torchvision.models as tv
    with contextlib.ExitStack() as stack:
        for name in ('resnet18','resnet50'):
            original = getattr(tv,name)
            def ctor(*a,_fn=original,**kw):
                kw.pop('pretrained',None)
                kw['weights']=None
                return _fn(*a,**kw)
            stack.enter_context(patch.object(tv,name,ctor))
        yield


def build(entry):
    if 'REPLACE' in entry['factory']:
        raise ValueError('Ours architecture not configured. Inspect checkpoint and its training configuration first.')
    with no_download():
        if entry['factory'] in ('changemamba', 'changemamba_b2_diff'):
            sys.path.insert(0,str(ROOT/'ChangeMamba-master'))
            from changedetection.configs.config import get_config
            from changedetection.script.script_utils import get_vssm_kwargs
            if entry['factory'] == 'changemamba_b2_diff':
                from changedetection.models.ChangeMambaMMBDA_B2_DiffFusion import ChangeMambaMMBDA_B2_DiffFusion as ChangeMambaMMBDA
            else:
                from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
            cfg = entry['cfg']
            if not Path(cfg).is_file():
                raise FileNotFoundError(f'ChangeMamba requires the exact training YAML: {cfg}')
            config = get_config(SimpleNamespace(cfg=cfg,opts=entry.get('opts',[])))
            return ChangeMambaMMBDA(output_building=2,output_damage=4,pretrained=None,**get_vssm_kwargs(config))
        module, name = entry['factory'].split(':')
        return getattr(importlib.import_module(module),name)(**entry.get('kwargs',{}))


def normalize(x):
    # Matches bda_benchmark/dataset/imutils.py. Display stretch is NEVER used here.
    import torch
    x = (x.astype(np.float32)-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)
    return torch.from_numpy(x.transpose(2,0,1).copy()).unsqueeze(0)


def run(args):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    entries = [e for e in load_json(args.manifest) if e.get('enabled',True)]
    if args.models:
        names = args.models.split(',')
        if set(names)-{e['name'] for e in entries}:
            raise ValueError('Unknown or disabled model requested')
        entries = [e for e in entries if e['name'] in names]
    if not entries:
        raise ValueError('No enabled models')
    for e in entries:
        if 'REPLACE' in e['factory'] or not Path(e['checkpoint']).is_file():
            raise ValueError(f'Configure model/checkpoint before running: {e["name"]}')
    split = getattr(args,'split','val')
    if split == 'test' and not getattr(args,'allow_test',False):
        raise ValueError('Test requires --allow-test with frozen checkpoints.')
    selection = load_json(args.selection)
    if split == 'test' and selection.get('split') != 'test':
        raise ValueError('Test requires a newly generated Test selection; do not reuse Val candidates.')
    samples = [s for s in selection['samples'] if s.get('approved') is True]
    valid_ids = set((ROOT/f'bda_benchmark/dataset/splitname/standard_ML/{split}_set.txt').read_text().split())
    if not samples:
        raise ValueError('Zero approved samples. Review candidates and use bright_selection_review.py approve.')
    if any(s['sample_id'] not in valid_ids for s in samples):
        raise ValueError(f'Sample IDs do not belong to the requested {split} split.')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'selection.json',samples)
    records = []
    for e in entries:
        print('Loading',e['name'],flush=True)
        checkpoint_hash = digest(e['checkpoint'])
        if e.get('expected_sha256') and checkpoint_hash != e['expected_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {e["name"]}')
        model = build(e)
        payload = torch.load(e['checkpoint'],map_location='cpu',weights_only=False)
        if e.get('expected_variant') and payload.get('variant') != e['expected_variant']:
            raise ValueError(f'Checkpoint variant mismatch: {e["name"]}')
        model.load_state_dict(state_of(payload,e.get('state_key')),strict=True)
        del payload
        model.to(args.device).eval()
        d = out/e['name']
        d.mkdir()
        record = dict(e,sha256=checkpoint_hash,status='running',confusion_visual_subset=np.zeros((4,4),np.int64).tolist())
        cm = np.zeros((4,4),np.int64)
        for s in samples:
            sid=s['sample_id']
            print(f'  {e["name"]}: {sid}', flush=True)
            rgb,sar,gt=read_sample(args.data,sid)
            a=normalize(rgb).to(args.device)
            b=normalize(np.repeat(sar[...,None],3,axis=2)).to(args.device)
            with torch.inference_mode():
                result = model(torch.cat([a,b],1)) if e['input_mode']=='concat' else model(a,b)
                idx=e.get('output_index')
                logits=result[idx] if idx is not None else result
                if not torch.is_tensor(logits) or logits.shape!=(1,4,*gt.shape):
                    raise ValueError(f'Wrong output for {e["name"]}; configure output_index. No silent resize.')
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite logits')
                prob=logits.float().softmax(1)[0].cpu().numpy()
            pred=prob.argmax(0).astype(np.uint8)
            Image.fromarray(pred).save(d/f'{sid}_labels.png')
            Image.fromarray(colorize(pred)).save(d/f'{sid}_color.png')
            np.savez_compressed(d/f'{sid}_probabilities.npz',probabilities=prob)
            x,y,w,h=s['xywh']
            g=gt[y:y+h,x:x+w]; p=pred[y:y+h,x:x+w]; valid=g<4
            cm+=np.bincount(4*g[valid].astype(int)+p[valid],minlength=16).reshape(4,4)
            fig,ax=plt.subplots(figsize=(4,3.6))
            im=ax.imshow(prob[2,y:y+h,x:x+w],cmap='viridis',vmin=0,vmax=1)
            ax.set_title(f'{e["name"]}: P(Damaged)'); ax.axis('off')
            fig.colorbar(im,ax=ax,label='Class probability'); fig.tight_layout()
            fig.savefig(d/f'{sid}_damaged_probability.png',dpi=180);plt.close(fig)
        record['status']='complete'; record['confusion_visual_subset']=cm.tolist()
        record['split']=split
        record['note']='Selected crops only: NOT full split performance; no TTA or postprocessing.'
        records.append(record);write_json(out/'run_manifest.json',records)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # Five rows per page keeps event names and column labels readable.
    for start in range(0,len(samples),5):
        batch=samples[start:start+5]
        fig,axs=plt.subplots(len(batch),3+len(entries),figsize=(2*(3+len(entries)),2.25*len(batch)),squeeze=False)
        for r,s in enumerate(batch):
            sid=s['sample_id'];rgb,sar,gt=read_sample(args.data,sid)
            arrays=[rgb,sar_display(sar),colorize(gt)]
            arrays += [np.array(Image.open(out/e['name']/f'{sid}_color.png')) for e in entries]
            x,y,w,h=s['xywh']
            for c,(arr,title) in enumerate(zip(arrays,['Pre RGB','Post SAR','GT']+[e['name'] for e in entries])):
                ax=axs[r,c];ax.imshow(arr[y:y+h,x:x+w],cmap='gray',vmin=0,vmax=255)
                ax.set_xticks([]);ax.set_yticks([])
                if r==0:ax.set_title(title)
                if c==0:ax.set_ylabel(sid.replace('_','\n'),fontsize=8)
        fig.suptitle(f'{split.upper()} selected examples | white: background; green: intact; tan: damaged; red: destroyed',fontsize=10)
        fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/f'comparison_{start//5+1}.png',dpi=220)
        fig.savefig(out/f'comparison_{start//5+1}.pdf');plt.close(fig)
    print('Done:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('inspect');a.add_argument('--checkpoint',required=True)
    a=sub.add_parser('init');a.add_argument('--checkpoints',default='/root/shared-nvme/BRIGHT-checkpoints');a.add_argument('--output',default='comparison_models.json')
    a=sub.add_parser('select');a.add_argument('--data',required=True);a.add_argument('--output',required=True)
    a.add_argument('--crop',type=int,default=512);a.add_argument('--per-event',type=int,default=1)
    a.add_argument('--dark-threshold',type=float,default=40);a.add_argument('--max-dark',type=float,default=.15)
    a.add_argument('--max-invalid',type=float,default=.01);a.add_argument('--min-building-pixels',type=int,default=1000)
    a=sub.add_parser('run');a.add_argument('--data',required=True);a.add_argument('--manifest',required=True)
    a.add_argument('--split',choices=['val','test'],default='val');a.add_argument('--allow-test',action='store_true')
    a.add_argument('--selection',required=True);a.add_argument('--output',required=True)
    a.add_argument('--device',default='cuda:0');a.add_argument('--models',help='Optional comma-separated enabled names')
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
