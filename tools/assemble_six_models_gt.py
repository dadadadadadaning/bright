"""Read-only prediction assembly: original GT + six models, without Torch/GPU."""
import argparse
import hashlib
import html
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODELS = ['UNet', 'DeepLabV3Plus', 'SiamAttnUNet', 'SiamCRNN', 'DamageFormer', 'Ours_A1']
TITLES = ['Pre-event RGB', 'Post-event SAR', 'GT (reference)', 'UNet', 'DeepLabV3+',
          'SiamAttnUNet', 'SiamCRNN', 'DamageFormer', 'A1']
PALETTE = np.array([[255,255,255], [70,181,121], [228,189,139], [182,70,69]], dtype=np.uint8)
TYPES = {'earthquake':'地震', 'volcano':'火山', 'explosion':'爆炸', 'hurricane':'飓风',
         'wildfire':'野火', 'flood':'洪水', 'conflict':'冲突'}


def font(size):
    for name in ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def colorize(labels):
    image = PALETTE[np.minimum(labels, 3)].copy()
    image[labels == 255] = 128
    return Image.fromarray(image)


class Data:
    def __init__(self, root):
        self.root = root
        self.archives = {}
        self.members = {}
        for folder in ['pre-event', 'post-event', 'target']:
            if not (root/folder).is_dir():
                archive = zipfile.ZipFile(root/(folder+'.zip'))
                self.archives[folder] = archive
                self.members[folder] = {}
                for name in archive.namelist():
                    if name.lower().endswith('.tif'):
                        key = Path(name).name
                        if key in self.members[folder]:
                            raise ValueError('Duplicate dataset filename: '+key)
                        self.members[folder][key] = name

    def read(self, folder, filename):
        if folder in self.archives:
            source = io.BytesIO(self.archives[folder].read(self.members[folder][filename]))
        else:
            source = self.root/folder/filename
        with Image.open(source) as image:
            return np.array(image)

    def close(self):
        for archive in self.archives.values():
            archive.close()


def draw_row(sid, arrays, target, tile):
    gap, margin = 12, 20
    width = 2*margin + 9*tile + 8*gap
    canvas = Image.new('RGB', (width, tile+154), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), sid+' | Full field of view; identical sample and extent', font=font(23), fill='#263445')
    for i, (title, im) in enumerate(zip(TITLES, arrays)):
        x, y = margin+i*(tile+gap), 82
        draw.text((x, 51), title, font=font(20), fill='#263445')
        thumb = ImageOps.contain(im.convert('RGB'), (tile, tile),
                    method=Image.Resampling.NEAREST if i >= 2 else Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x+(tile-thumb.width)//2, y+(tile-thumb.height)//2))
        draw.rectangle((x-1,y-1,x+tile,y+tile), outline='#89929b', width=2 if i==2 else 1)
    y, x = tile+102, margin
    for label, color in zip(['0 Background','1 Intact','2 Damaged','3 Destroyed','255 Ignore'],
                            list(PALETTE)+[np.array([128,128,128])]):
        draw.rectangle((x,y,x+18,y+18), fill=tuple(int(v) for v in color), outline='#777777')
        draw.text((x+25,y),label,font=font(17),fill='#263445')
        x += 190
    draw.text((x+10,y),'GT: original dataset labels, not a prediction. SAR contrast stretched for display only.',font=font(17),fill='#465563')
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def write_browser(out, rows):
    page = '''<!doctype html><meta charset="utf-8"><title>BRIGHT 六模型与原始GT</title>
<style>body{font:16px system-ui;margin:20px;color:#263445;background:#f6f7f9}header{position:sticky;top:0;background:white;padding:12px;z-index:2}article{background:white;padding:14px;margin:20px 0}.strip{overflow-x:auto}img{display:block;max-width:none}select,input,button{font:inherit;padding:6px;margin:4px}a{margin:8px}</style>
<h1>原始GT与六模型对比</h1><p>固定顺序：RGB → SAR → 原始GT → UNet → DeepLabV3+ → SiamAttnUNet → SiamCRNN → DamageFormer → A1。横向滚动查看完整一行，点击图片查看原图。白：背景；绿：完整；浅棕：损毁；红：毁坏；灰：忽略。GT忠实呈现数据集标注，不代表标注绝无误差。</p>
<header><select id="hazard"><option value="">所有灾害类型</option></select><input id="search" placeholder="事件或样本编号"><button id="export">导出已选编号</button><span id="count"></span></header><main id="list"></main>
<script>const rows=DATA, types=TYPES;const $=id=>document.getElementById(id);let selected=new Set();try{selected=new Set(JSON.parse(localStorage.getItem('bright-six-gt-selection')||'[]'))}catch(e){}
for(const h of [...new Set(rows.map(r=>r.hazard))])$('hazard').add(new Option(types[h]||h,h));
function count(n){$('count').textContent=`显示 ${n}，已选 ${selected.size}`}
function render(){const found=rows.filter(r=>(!$('hazard').value||r.hazard===$('hazard').value)&&r.sample_id.includes($('search').value));$('list').replaceChildren();count(found.length);for(const r of found){const a=document.createElement('article'), lab=document.createElement('label'), cb=document.createElement('input');cb.type='checkbox';cb.checked=selected.has(r.sample_id);cb.onchange=()=>{cb.checked?selected.add(r.sample_id):selected.delete(r.sample_id);try{localStorage.setItem('bright-six-gt-selection',JSON.stringify([...selected]))}catch(e){}count(found.length)};lab.append(cb,document.createTextNode((types[r.hazard]||r.hazard)+' / '+r.sample_id));a.append(lab);const gt=document.createElement('a');gt.href=r.gt_color;gt.target='_blank';gt.textContent='查看全分辨率GT';a.append(gt);const div=document.createElement('div');div.className='strip';const link=document.createElement('a');link.href=r.image;link.target='_blank';const im=document.createElement('img');im.src=r.image;im.loading='lazy';im.alt=r.sample_id;link.append(im);div.append(link);a.append(div);$('list').append(a)}}
$('hazard').onchange=render;$('search').oninput=render;$('export').onclick=()=>{const u=URL.createObjectURL(new Blob([[...selected].sort().join('\\n')],{type:'text/plain'}));const a=document.createElement('a');a.href=u;a.download='selected_test_ids.txt';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};render();</script>'''
    page = page.replace('rows=DATA', 'rows='+json.dumps(rows,ensure_ascii=False)).replace('types=TYPES', 'types='+json.dumps(TYPES,ensure_ascii=False))
    (out/'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ids',type=Path)
    parser.add_argument('--samples',nargs='+',help='Explicit subset for preview only')
    parser.add_argument('--tile',type=int,default=320)
    parser.add_argument('--zip',action='store_true',help='Package complete output folder next to it')
    args=parser.parse_args()
    if args.tile < 256:
        parser.error('--tile must be >= 256 for readable headers')
    valid=(args.repo_root/'bda_benchmark/dataset/splitname/standard_ML/test_set.txt').read_text().split()
    ids=args.ids.read_text(encoding='utf-8-sig').split() if args.ids else (args.samples or valid)
    if len(ids)!=len(set(ids)) or not set(ids)<=set(valid):
        raise ValueError('Duplicate or non-Test IDs')
    ids=sorted(ids,key=lambda s:(s.rsplit('_',1)[0].rsplit('-',1)[-1],s))
    # Fail before rendering if any required prediction is absent: no fabricated cells.
    provenance={}
    absent=[]
    for model in MODELS:
        path=args.predictions/model/'provenance.json'
        metadata=json.loads(path.read_text(encoding='utf-8-sig'))
        if metadata.get('split')!='test' or not metadata.get('sha256'):
            raise ValueError('Missing Test provenance or checkpoint hash: '+model)
        provenance[model]=metadata
        absent += [str(args.predictions/model/(sid+'_labels.png')) for sid in ids if not (args.predictions/model/(sid+'_labels.png')).is_file()]
    if absent:
        raise FileNotFoundError(f'{len(absent)} predictions missing; first ten: '+str(absent[:10]))
    # Fresh output avoids packing stale figures from a previous run/subset.
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output directory is not empty. Choose a new --out; existing outputs are preserved.')
    args.out.mkdir(parents=True,exist_ok=True)
    data=Data(args.data)
    rows=[]
    try:
        for i,sid in enumerate(ids,1):
            rgb=data.read('pre-event',sid+'_pre_disaster.tif')
            sar=data.read('post-event',sid+'_post_disaster.tif')
            gt=data.read('target',sid+'_building_damage.tif')
            if rgb.ndim!=3 or rgb.shape[2]<3 or sar.ndim!=2 or gt.ndim!=2 or rgb.shape[:2]!=gt.shape or sar.shape!=gt.shape:
                raise ValueError('RGB/SAR/GT shape mismatch: '+sid)
            if not set(np.unique(gt))<={0,1,2,3,255}:
                raise ValueError('Unknown GT class IDs: '+sid)
            if rgb.dtype!=np.uint8:
                raise ValueError('RGB needs explicit non-uint8 display policy: '+sid)
            values=sar[np.isfinite(sar)]
            low,high=np.percentile(values,[1,99]) if values.size else (0,1)
            sar_view=np.uint8(np.nan_to_num(np.clip((sar.astype(float)-low)/max(float(high-low),1e-8),0,1))*255)
            panels=[Image.fromarray(rgb[...,:3]),Image.fromarray(sar_view),colorize(gt)]
            for model in MODELS:
                with Image.open(args.predictions/model/(sid+'_labels.png')) as im:
                    pred=np.array(im)
                if pred.shape!=gt.shape or not set(np.unique(pred))<={0,1,2,3}:
                    raise ValueError('Invalid prediction: '+model+'/'+sid)
                panels.append(colorize(pred))
            event=sid.rsplit('_',1)[0]
            hazard=event.rsplit('-',1)[-1]
            rel=Path('comparisons')/hazard/event/(sid+'.png')
            gt_rel=Path('ground_truth')/hazard/event/(sid+'_GT_color.png')
            gt_path=args.out/gt_rel
            gt_path.parent.mkdir(parents=True,exist_ok=True)
            colorize(gt).save(gt_path)
            raw_path=gt_path.with_name(sid+'_GT_labels.png')
            Image.fromarray(gt.astype(np.uint8)).save(raw_path)
            with Image.open(raw_path) as check:
                if not np.array_equal(np.array(check),gt):
                    raise ValueError('GT export differs from source: '+sid)
            draw_row(sid,panels,args.out/rel,args.tile)
            rows.append(dict(sample_id=sid,event=event,hazard=hazard,image=rel.as_posix(),gt_color=gt_rel.as_posix(),
                gt_source='target/'+sid+'_building_damage.tif',shape=list(gt.shape),
                gt_uint8_sha256=hashlib.sha256(gt.astype(np.uint8).tobytes()).hexdigest()))
            if i%40==0 or i==len(ids):
                print(f'Assembled {i}/{len(ids)}; GT verified',flush=True)
    finally:
        data.close()
    summary=dict(samples=len(rows),models=MODELS,column_order=TITLES,counts_by_hazard=dict(Counter(r['hazard'] for r in rows)),
        gt='Original dataset target; exported raw PNG checked pixel-for-pixel; annotation accuracy not independently adjudicated.',
        data_root=str(args.data),prediction_root=str(args.predictions),tile=args.tile)
    for filename,obj in [('summary.json',summary),('inventory.json',rows),('model_provenance.json',provenance)]:
        (args.out/filename).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    write_browser(args.out,rows)
    if args.zip:
        archive=args.out.parent/(args.out.name+'.zip')
        if archive.exists():
            raise FileExistsError('Archive already exists: '+str(archive))
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for path in sorted(args.out.rglob('*')):
                if path.is_file():
                    z.write(path,Path(args.out.name)/path.relative_to(args.out))
        print('ZIP:',archive,flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
