"""Build paginated, lossless contact sheets from verified nine-column montages."""
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('D:/software/dowmload/BRIGHT-sixmodels-GT-v1.zip')
OUT=ROOT/'output/pdf/BRIGHT_sixmodels_GT'
ORDER=['earthquake','volcano','conflict','hurricane','explosion','wildfire','flood']
NAMES={'earthquake':'地震','volcano':'火山','conflict':'冲突','hurricane':'飓风','explosion':'爆炸','wildfire':'野火','flood':'洪水'}
HEADERS=['灾前 RGB','灾后 SAR','GT 真实标注','UNet','DeepLabV3+','SiamAttnUNet','SiamCRNN','DamageFormer','A1']
W,H=landscape(A3)
MARGIN=30
STRIP_WIDTH=W-2*MARGIN
SCALE=STRIP_WIDTH/2976
PIC_H=320*SCALE
ROW_H=144
FONT='CN'
COLORS=[('#ffffff','0 背景'),('#46b579','1 完整'),('#e4bd8b','2 损毁'),('#b64645','3 毁坏'),('#808080','255 忽略')]


def text(c,x,y,s,size=10,color='#273648'):
    c.setFillColor(color)
    c.setFont(FONT,size)
    c.drawString(x,y,s)


def footer(c,page,total):
    c.setStrokeColorRGB(.82,.85,.88)
    c.line(MARGIN,37,W-MARGIN,37)
    x=MARGIN
    for color,label in COLORS:
        c.setFillColor(color);c.setStrokeColorRGB(.5,.5,.5)
        c.rect(x,18,9,9,fill=1,stroke=1)
        text(c,x+14,19,label,9)
        x+=88
    text(c,x+12,19,'GT来自数据集原始标注；预测未作修改。',9)
    c.setFont(FONT,9);c.drawRightString(W-MARGIN,19,f'{page} / {total}')


def cover(c,title,pages,rows,event_pages):
    text(c,MARGIN,H-60,title,26)
    text(c,MARGIN,H-95,f'{len(rows)} 组样本  |  六模型  |  每页最多5组、每组9列  |  完整视野',14)
    text(c,MARGIN,H-128,'列序：灾前RGB / 灾后SAR / GT / UNet / DeepLabV3+ / SiamAttnUNet / SiamCRNN / DamageFormer / A1',11)
    text(c,MARGIN,H-155,'查看：使用左侧书签跳转事件；Ctrl+F搜索样本编号；放大查看局部。小方框用于记录待选案例。',11)
    text(c,MARGIN,H-180,'GT是原始数据集标注，不是模型输出，也不表示每个标签经过独立人工复核。',11)
    text(c,MARGIN,H-202,'彩色类别图保持统一配色。图像取自已有320像素展示面板；正式局部放大应回到原分辨率数据。',11)
    text(c,MARGIN,H-224,'本册包含指定范围全部样本，未按模型表现筛选。A1是已有模型，不代表候选区域-Memory架构。',11)
    y=H-272
    text(c,MARGIN,y,'灾害 / 事件',12)
    text(c,W-250,y,'样本数',12)
    text(c,W-150,y,'所在页',12)
    for event,hazard,count,start,end in event_pages:
        y-=27
        text(c,MARGIN,y,f'{NAMES[hazard]}  /  {event}',11)
        text(c,W-250,y,str(count),11)
        text(c,W-150,y,f'{start}-{end}',11)
        c.linkRect('',event,(MARGIN,y-3,W-MARGIN,y+16),relative=0,thickness=0)
    footer(c,1,pages)
    c.showPage()


def plan(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['event']].append(row)
    pages=[];toc=[]
    for event,records in groups.items():
        start=len(pages)+2
        for pos in range(0,len(records),5):
            pages.append((event,records[pos:pos+5]))
        toc.append((event,records[0]['hazard'],len(records),start,len(pages)+1))
    return pages,toc


def make_pdf(path,title,rows,z,prefix):
    pages,toc=plan(rows)
    c=canvas.Canvas(str(path),pagesize=(W,H),pageCompression=1)
    c.setTitle(title);c.setAuthor('BRIGHT comparison gallery')
    c.setSubject('All cases, original dataset GT, six cached model predictions; no performance filtering')
    cover(c,title,len(pages)+1,rows,toc)
    seen=set();hazards=set()
    for page,(event,items) in enumerate(pages,2):
        hazard=items[0]['hazard']
        if hazard not in hazards:
            c.bookmarkPage('hazard_'+hazard)
            c.addOutlineEntry(NAMES[hazard],'hazard_'+hazard,0,False)
            hazards.add(hazard)
        if event not in seen:
            c.bookmarkPage(event)
            c.addOutlineEntry(event,event,1,False)
            seen.add(event)
        text(c,MARGIN,H-27,f'{NAMES[hazard]} / {event}',15)
        text(c,W-265,H-27,f'本页 {len(items)} 组 | 同一行对应同一样本',10)
        for k,header in enumerate(HEADERS):
            x=MARGIN+k*332*SCALE
            text(c,x,H-49,header,10)
        c.setStrokeColorRGB(.72,.77,.81)
        c.line(MARGIN,H-58,W-MARGIN,H-58)
        for n,row in enumerate(items):
            top=H-65-n*ROW_H
            c.setStrokeColorRGB(.4,.46,.5)
            c.rect(MARGIN,top-10,8,8,stroke=1,fill=0)
            text(c,MARGIN+15,top-10,row['sample_id'],9)
            with Image.open(io.BytesIO(z.read(prefix+row['image']))) as im:
                if im.size!=(3016,474):raise ValueError(f'Unexpected montage geometry: {im.size}')
                # Only remove decorative headers/legend/margins; retain all 9 image panels.
                for k in range(9):
                    panel=im.crop((20+k*332,82,340+k*332,402)).convert('RGB')
                    if k<2:
                        buffer=io.BytesIO()
                        panel.save(buffer,format='JPEG',quality=92,subsampling=0)
                        buffer.seek(0)
                        image=ImageReader(buffer)
                    else:
                        image=ImageReader(panel)
                    c.drawImage(image,MARGIN+k*332*SCALE,top-16-PIC_H,width=320*SCALE,height=PIC_H)
            c.setStrokeColorRGB(.84,.86,.89)
            c.line(MARGIN,top-19-PIC_H,W-MARGIN,top-19-PIC_H)
        footer(c,page,len(pages)+1)
        c.showPage()
        if page%30==0:print(path.name,page,'/',len(pages)+1,flush=True)
    c.save()
    reader=PdfReader(str(path))
    if len(reader.pages)!=len(pages)+1:raise ValueError('Page count mismatch')
    extracted='\n'.join(p.extract_text() or '' for p in reader.pages[1:])
    for row in rows:
        if extracted.count(row['sample_id'])!=1:raise ValueError('Missing or repeated sample ID: '+row['sample_id'])
    print('Verified',path.name,len(rows),'samples',len(reader.pages),'pages',flush=True)
    return dict(file=path.name,samples=len(rows),pages=len(reader.pages),bytes=path.stat().st_size)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont(FONT,'C:/Windows/Fonts/simhei.ttf'))
    with zipfile.ZipFile(SOURCE) as z:
        name=next(n for n in z.namelist() if n.endswith('/inventory.json'))
        prefix=name[:-len('inventory.json')]
        rows=json.loads(z.read(name))
        rows.sort(key=lambda r:(ORDER.index(r['hazard']),r['event'],r['sample_id']))
        if len(rows)!=878 or len({r['sample_id'] for r in rows})!=878:raise ValueError('Expected 878 unique records')
        outputs=[make_pdf(OUT/'BRIGHT_878_六模型与GT_完整合集.pdf','BRIGHT 六模型与GT | 完整挑选册',rows,z,prefix)]
        for h in ORDER:
            subset=[r for r in rows if r['hazard']==h]
            outputs.append(make_pdf(OUT/f'BRIGHT_{NAMES[h]}_六模型与GT.pdf',f'BRIGHT 六模型与GT | {NAMES[h]}',subset,z,prefix))
    (OUT/'verification.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(outputs,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
