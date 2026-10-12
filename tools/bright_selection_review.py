"""Non-interactive candidate listing, manual approval and output packing."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('list');a.add_argument('--selection',required=True)
    a=s.add_parser('approve');a.add_argument('--selection',required=True)
    a.add_argument('--ids',type=int,nargs='+',required=True,help='Numbers of visually reviewed cloud-free candidates')
    a.add_argument('--output',required=True)
    a=s.add_parser('pack');a.add_argument('--folder',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.action=='pack':
        root=Path(args.folder)
        if not list(root.glob('comparison_*.png')):p.error('No completed comparison PNGs found')
        with ZipFile(args.output,'x',ZIP_DEFLATED) as z:
            for f in sorted(root.rglob('*')):
                if f.is_file() and f.suffix.lower() in {'.png','.pdf','.json'}:
                    z.write(f,f.relative_to(root.parent))
        print('Saved:',args.output);return
    data=json.loads(Path(args.selection).read_text(encoding='utf-8-sig'))
    rows=data['samples']
    if args.action=='list':
        print('Split:',data.get('split','val'),'Candidates:',len(rows))
        for i,r in enumerate(rows,1):
            print(i,r['sample_id'],'GT counts:',r['class_pixels'],'approved:',r.get('approved'))
        return
    chosen=set(args.ids)
    if not chosen.issubset(range(1,len(rows)+1)):p.error('Candidate number out of range')
    for i,r in enumerate(rows,1):
        r['approved']=i in chosen
        r['review_status']='user_confirmed_cloud_free' if i in chosen else 'not_selected'
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print('Saved',len(chosen),'approved samples:',args.output)


if __name__=='__main__':main()
