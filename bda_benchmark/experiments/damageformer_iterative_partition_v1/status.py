"""Read-only status. Partial JSONL final line is tolerated during training."""
import argparse
import json
from pathlib import Path


def rows(path):
    result=[]
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try: result.append(json.loads(line))
            except json.JSONDecodeError: pass
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    a=p.parse_args()
    for variant in ('pixel','single','iter1','iter3','grid'):
        root=Path(a.root)/variant
        if not root.is_dir(): continue
        print('\n'+variant+' '+str(root))
        print('Final summary:',(root/'summary.json').is_file())
        train=rows(root/'training.jsonl'); val=rows(root/'validation.jsonl')
        if train:
            r=train[-1]
            print('step=',r['step'],'loss=',r['total_loss'],'lr=',r['lr'])
            print('partition=',r.get('partition'),'association_grad_norm=',r.get('association_grad_norm'))
        if val:
            r=max(val,key=lambda x:x['metrics']['valid_miou'])
            print('best step=',r['step'],'foreground mIoU=',100*r['metrics']['valid_miou'])
            print('class IoU=',[100*x for x in r['metrics']['iou']])
            print('Damaged F1=',100*r['metrics']['f1'][2])
        else: print('No validation yet. Missing final summary alone does not prove a running process.')


if __name__=='__main__': main()
