"""Full Test inventory, strict checkpoint inference, and an offline selection gallery.

Images are whole-frame views. Missing predictions are explicitly marked.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
import traceback
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
TYPES = {'earthquake': '地震', 'volcano': '火山', 'explosion': '爆炸',
         'hurricane': '飓风', 'wildfire': '野火', 'flood': '洪水', 'conflict': '冲突'}
NAMES = ['UNet', 'DeepLabV3Plus', 'SiamAttnUNet', 'SiamCRNN', 'DamageFormer', 'ChangeMamba', 'Ours_A1']
DISPLAY_NAMES = {}
COLORS = np.array([[255,255,255], [70,181,121], [228,189,139], [182,70,69]], dtype=np.uint8)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


class Dataset:
    def __init__(self, root):
        self.root = root
        self.zips = {}
        self.lookup = {}
        for folder in ['pre-event', 'post-event', 'target']:
            if not (root/folder).is_dir():
                z = zipfile.ZipFile(root/(folder+'.zip'))
                self.zips[folder] = z
                self.lookup[folder] = {Path(n).name: n for n in z.namelist() if n.endswith('.tif')}

    def read(self, sid):
        arrays = []
        for folder, suffix in [('pre-event','pre_disaster'), ('post-event','post_disaster'), ('target','building_damage')]:
            name = f'{sid}_{suffix}.tif'
            if folder in self.zips:
                with Image.open(io.BytesIO(self.zips[folder].read(self.lookup[folder][name]))) as im:
                    arrays.append(np.array(im))
            else:
                with Image.open(self.root/folder/name) as im:
                    arrays.append(np.array(im))
        rgb, sar, gt = arrays
        rgb = rgb[..., :3]
        if rgb.ndim != 3 or sar.ndim != 2 or gt.ndim != 2 or rgb.shape[:2] != sar.shape or sar.shape != gt.shape:
            raise ValueError(f'{sid}: incompatible RGB/SAR/GT shapes')
        if not set(np.unique(gt)).issubset({0,1,2,3,255}):
            raise ValueError(f'{sid}: invalid GT labels')
        return rgb, sar, gt


def import_archive(archive, cache, valid):
    if archive is None:
        return
    with zipfile.ZipFile(archive) as z:
        records = json.loads(z.read('predictions/run_manifest.json'))
        server_models = [dict(r) for r in records if r['name'] in NAMES]
        if not any(r['name'] == 'ChangeMamba' for r in server_models):
            server_models.insert(-1, dict(name='ChangeMamba', factory='changemamba',
                checkpoint='/root/shared-nvme/BRIGHT-checkpoints/ckpt_ChangeMamba_bright_standard_ML_split.pth',
                cfg='REPLACE_WITH_EXACT_TRAINING_YAML', kwargs={}, input_mode='pair', output_index=1,
                enabled=False, note='Configure the exact training YAML and enable explicitly. Missing slots stay visible.'))
        for entry in server_models:
            if entry.get('sha256'):
                entry['expected_sha256'] = entry['sha256']
        dump(cache.parent/'server_models.json',server_models)
        members = set(z.namelist())
        for r in records:
            if r['name'] not in NAMES or r.get('status') != 'complete' or r.get('split') != 'test':
                continue
            directory = cache/r['name']
            directory.mkdir(parents=True, exist_ok=True)
            meta = directory/'provenance.json'
            if meta.exists() and json.loads(meta.read_text())['sha256'] != r['sha256']:
                raise ValueError(f'Cache hash mismatch: {r["name"]}')
            dump(meta, r)
            for sid in valid:
                member = f'predictions/{r["name"]}/{sid}_labels.png'
                dest = directory/f'{sid}_labels.png'
                if member in members and not dest.exists():
                    dest.write_bytes(z.read(member))


def infer(args, ids, dataset, cache):
    import torch
    import compare_bright_checkpoints as c
    # The launcher and helper may live together outside the source repository.
    c.ROOT = args.repo_root
    sys.path.insert(0, str(args.repo_root/'bda_benchmark'))
    sys.path.insert(0, str(args.repo_root))
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run inference in the existing server training environment.')
    if not args.manifest:
        raise ValueError('--manifest is required for inference')
    entries = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    for e in entries:
        if e.get('cfg') and not Path(e['cfg']).is_absolute():
            e['cfg'] = str((args.manifest.parent/e['cfg']).resolve())
    failures = []
    for e in entries:
        if not e.get('enabled', True):
            continue
        name = e['name']
        try:
            if name not in NAMES:
                raise ValueError(f'Unrecognized model name: {name}')
            sha = digest(e['checkpoint'])
            if e.get('expected_sha256') and sha != e['expected_sha256']:
                raise ValueError('Checkpoint SHA256 mismatch')
            directory = cache/name
            directory.mkdir(parents=True, exist_ok=True)
            meta = directory/'provenance.json'
            if meta.exists():
                previous = json.loads(meta.read_text())
                for key in ['sha256','factory','kwargs','input_mode','output_index','cfg','opts']:
                    expected = sha if key == 'sha256' else e.get(key)
                    if previous.get(key) != expected:
                        raise ValueError(f'Cached model configuration differs: {key}. Use a new output directory.')
            elif list(directory.glob('*_labels.png')):
                raise ValueError('Unverified cached labels: missing provenance.json')
            pending = [sid for sid in ids if not (directory/f'{sid}_labels.png').exists()]
            if not pending:
                print(name, 'already complete', flush=True)
                continue
            model = c.build(e)
            payload = torch.load(e['checkpoint'], map_location='cpu', weights_only=False)
            if e.get('expected_variant') and payload.get('variant') != e['expected_variant']:
                raise ValueError('Checkpoint variant mismatch')
            model.load_state_dict(c.state_of(payload, e.get('state_key')), strict=True)
            del payload
            model.to(args.device).eval()
            dump(meta, dict(e, sha256=sha, split='test', status='running', inference='whole image, no TTA, no postprocessing'))
            for num, sid in enumerate(pending, 1):
                rgb, sar, gt = dataset.read(sid)
                a = c.normalize(rgb).to(args.device)
                b = c.normalize(np.repeat(sar[...,None],3,axis=2)).to(args.device)
                with torch.inference_mode():
                    result = model(torch.cat([a,b],1)) if e['input_mode']=='concat' else model(a,b)
                    logits = result[e['output_index']] if e.get('output_index') is not None else result
                    if not torch.is_tensor(logits) or tuple(logits.shape) != (1,4,*gt.shape):
                        raise ValueError(f'Unexpected output shape for {sid}')
                    if not torch.isfinite(logits).all():
                        raise ValueError(f'Nonfinite logits: {sid}')
                    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                temp = directory/f'{sid}_labels.tmp.png'
                Image.fromarray(pred).save(temp)
                os.replace(temp, directory/f'{sid}_labels.png')
                del a,b,result,logits
                if num % 20 == 0 or num == len(pending):
                    print(name, num, '/', len(pending), sid, flush=True)
            dump(meta, dict(e, sha256=sha, split='test', status='complete'))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as ex:
            failures.append({'model': name, 'error': str(ex)})
            traceback.print_exc()
    dump(args.out/'inference_errors.json', failures)
    return failures


def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def colorize(a):
    out = COLORS[np.minimum(a,3)].copy()
    out[a == 255] = 128
    return Image.fromarray(out)


def render(sid, rgb, sar, gt, cache, target, tile):
    gap, margin, band = 12, 20, 32
    width = margin*2 + tile*4 + gap*3
    height = 115 + 3*(tile+band+gap) + 48
    canvas = Image.new('RGB',(width,height),'white')
    d = ImageDraw.Draw(canvas)
    d.text((margin,12),sid, font=font(25), fill='#243444')
    d.text((margin,47),f'Full image {gt.shape[1]} x {gt.shape[0]} px; display tiles {tile}px; no spatial crop',font=font(16),fill='#526171')
    values = sar[np.isfinite(sar)]
    lo,hi = np.percentile(values,[1,99]) if values.size else (0,1)
    display = np.uint8(np.nan_to_num(np.clip((sar-lo)/max(float(hi-lo),1e-8),0,1))*255)
    panels = [('Pre-event RGB',Image.fromarray(rgb.astype(np.uint8)),False),
              ('Post-event SAR',Image.fromarray(display),False),('GT',colorize(gt),True)]
    missing=[]
    pred_panels=[]
    for name in NAMES:
        path=cache/name/f'{sid}_labels.png'
        if path.exists() and (cache/name/'provenance.json').exists():
            with Image.open(path) as im:
                pred=np.array(im)
            if pred.shape != gt.shape or not set(np.unique(pred)).issubset({0,1,2,3}):
                raise ValueError(f'Invalid labels {path}')
            pred_panels.append((name,colorize(pred),True))
        else:
            missing.append(name)
            pred_panels.append((name,None,True))
    # First row: inputs and GT. Next two rows: all seven model slots.
    positions=[(0,i,p) for i,p in enumerate(panels)]
    positions += [(1+i//4,i%4,p) for i,p in enumerate(pred_panels)]
    for row,col,(name,im,categorical) in positions:
        x=margin+col*(tile+gap)
        y=95+row*(tile+band+gap)
        d.text((x,y),DISPLAY_NAMES.get(name,name),font=font(19),fill='#243444')
        y+=band
        if im is None:
            d.rectangle((x,y,x+tile-1,y+tile-1), fill='#f0f1f3', outline='#c8cdd2')
            d.text((x+16,y+tile//2-12),'PREDICTION MISSING',font=font(17),fill='#6a6f75')
        else:
            thumb=ImageOps.contain(im.convert('RGB'), (tile,tile),
                                   method=Image.Resampling.NEAREST if categorical else Image.Resampling.LANCZOS)
            canvas.paste(thumb,(x+(tile-thumb.width)//2,y+(tile-thumb.height)//2))
    x=margin+3*(tile+gap)
    d.text((x,143),f'Available: {7-len(missing)}/7 models',font=font(18),fill='#243444')
    for k,(label,color) in enumerate(zip(['Background','Intact','Damaged','Destroyed'],COLORS)):
        yy=183+k*30
        d.rectangle((x,yy,x+18,yy+18),fill=tuple(color),outline='#999999')
        d.text((x+27,yy),label,font=font(16),fill='#243444')
    d.text((margin,height-32),'Original Test outputs; missing cells are not predictions. A1 is the recorded A1 model.',font=font(15),fill='#526171')
    target.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(target)
    return missing


HTML = '''<!doctype html><meta charset="utf-8"><title>BRIGHT Test 878 对比浏览</title>
<style>body{font:16px system-ui;margin:24px;background:#f6f7f9;color:#243444}header{position:sticky;top:0;background:#fff;padding:14px;z-index:1;border-bottom:1px solid #ddd}h1{font-size:24px}select,input,button{font:inherit;padding:7px;margin:4px}article{background:white;margin:22px 0;padding:12px}img{width:100%;max-width:1450px;display:block}small{color:#657080}.warn{color:#9b5014}</style>
<h1>BRIGHT 完整 Test 案例浏览</h1><p>按灾害类型、事件、样本编号排列。每张使用完整视野；灰色格表示缺少预测。挑选用于案例展示，不能替代全量指标。</p>
<header><select id="hazard"><option value="">所有灾害</option></select><select id="event"><option value="">所有事件</option></select><input id="search" placeholder="搜索样本编号"><label><input id="ready" type="checkbox">六模型齐全（不含ChangeMamba）</label><label><input id="chosen" type="checkbox">仅已选</label><button id="export">导出已选编号</button><span id="count"></span></header><main id="list"></main>
<script>const rows=__DATA__, types=__TYPES__;let selected=new Set();try{selected=new Set(JSON.parse(localStorage.getItem('bright878-selected')||'[]'))}catch(e){};
const el=x=>document.getElementById(x);for(const h of [...new Set(rows.map(r=>r.hazard))])el('hazard').add(new Option(types[h]||h,h));for(const e of [...new Set(rows.map(r=>r.event))])el('event').add(new Option(e,e));
function draw(){let found=rows.filter(r=>(!el('hazard').value||r.hazard===el('hazard').value)&&(!el('event').value||r.event===el('event').value)&&r.sample_id.includes(el('search').value)&&(!el('ready').checked||r.six_complete)&&(!el('chosen').checked||selected.has(r.sample_id)));el('count').textContent=`显示 ${found.length} / ${rows.length}，已选 ${selected.size}`;el('list').replaceChildren();for(const r of found){const a=document.createElement('article'),lab=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.checked=selected.has(r.sample_id);box.onchange=()=>{box.checked?selected.add(r.sample_id):selected.delete(r.sample_id);try{localStorage.setItem('bright878-selected',JSON.stringify([...selected]))}catch(e){};el('count').textContent=`显示 ${found.length} / ${rows.length}，已选 ${selected.size}`};lab.append(box,document.createTextNode(`${types[r.hazard]} / ${r.event} / ${r.sample_id}`));a.append(lab);const t=document.createElement('p');t.className='warn';t.textContent=r.missing.length?'缺少：'+r.missing.join(', '):'所有模型齐全';a.append(t);if(r.image){const link=document.createElement('a');link.href=r.image;link.target='_blank';const im=document.createElement('img');im.loading='lazy';im.src=r.image;im.alt=r.sample_id;link.append(im);a.append(link)}else{a.append(document.createTextNode(r.error||'未生成'))}el('list').append(a)}}
for(const n of ['hazard','event','ready','chosen'])el(n).onchange=draw;el('search').oninput=draw;el('export').onclick=()=>{const a=document.createElement('a'),u=URL.createObjectURL(new Blob([[...selected].sort().join('\\n')],{type:'text/plain;charset=utf-8'}));a.href=u;a.download='selected_test_ids.txt';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};draw();</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--repo-root',type=Path,default=ROOT,
                   help='BRIGHT source repository containing bda_benchmark; scripts may live anywhere')
    p.add_argument('--import-zip',type=Path)
    p.add_argument('--action',choices=['assemble','infer','all'],default='assemble')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--tile',type=int,default=320)
    p.add_argument('--ids',type=Path,help='Optional selected IDs for larger exports')
    a=p.parse_args()
    if a.manifest:
        for e in json.loads(a.manifest.read_text(encoding='utf-8-sig')):
            DISPLAY_NAMES[e['name']] = e.get('display_name', e['name'])
    a.repo_root=a.repo_root.resolve()
    split_file=a.repo_root/'bda_benchmark/dataset/splitname/standard_ML/test_set.txt'
    if not split_file.is_file():
        p.error(f'Missing Test list: {split_file}. Set --repo-root to the existing BRIGHT repository.')
    ids=split_file.read_text().split()
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate Test IDs')
    if a.ids:
        requested=a.ids.read_text(encoding='utf-8-sig').split()
        if not set(requested)<=set(ids):
            raise ValueError('Requested IDs include non-Test entries')
        ids=list(dict.fromkeys(requested))
    ids=sorted(ids,key=lambda s:(s.rsplit('_',1)[0].rsplit('-',1)[-1],s))
    a.out.mkdir(parents=True,exist_ok=True)
    cache=a.out/'predictions'
    import_archive(a.import_zip,cache,ids)
    dataset=Dataset(a.data)
    if a.action in ['infer','all']:
        infer(a,ids,dataset,cache)
    rows=[]
    if a.action in ['assemble','all']:
        for i,sid in enumerate(ids,1):
            event=sid.rsplit('_',1)[0]
            hazard=event.rsplit('-',1)[-1]
            row=dict(sample_id=sid,event=event,hazard=hazard,image='',missing=NAMES.copy(),six_complete=False)
            try:
                rgb,sar,gt=dataset.read(sid)
                rel=Path('comparisons')/hazard/event/f'{sid}.png'
                row['missing']=render(sid,rgb,sar,gt,cache,a.out/rel,a.tile)
                row['image']=rel.as_posix()
                row['six_complete']=not(set(row['missing'])-{'ChangeMamba'})
            except Exception as ex:
                row['error']=str(ex)
                print(sid,str(ex),flush=True)
            rows.append(row)
            if i%40==0 or i==len(ids):
                print('Assembled',i,'/',len(ids),flush=True)
        dump(a.out/'inventory.json',rows)
        dump(a.out/'summary.json',dict(samples=len(rows),images=sum(bool(r['image']) for r in rows),
             six_models_complete=sum(r['six_complete'] for r in rows),
             all_seven_complete=sum(not r['missing'] for r in rows),
             available_by_model={n:sum(n not in r['missing'] for r in rows) for n in NAMES}))
        with (a.out/'inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(['sample_id','hazard','event','comparison','missing_models'])
            for r in rows:writer.writerow([r['sample_id'],TYPES.get(r['hazard'],r['hazard']),r['event'],r['image'],';'.join(r['missing'])])
        (a.out/'index.html').write_text(HTML.replace('__DATA__',json.dumps(rows,ensure_ascii=False)).replace('__TYPES__',json.dumps(TYPES,ensure_ascii=False)),encoding='utf-8')
        print((a.out/'summary.json').read_text(),flush=True)


if __name__=='__main__':main()
