"""Read-only Val evaluation/interventions. Test intentionally unsupported."""
import argparse
import torch
from .common import *
from .retrieval import SupportClassifier

def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('checkpoint','dataset','train_list','val_list','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--cache'); p.add_argument('--head')
    p.add_argument('--intervention',choices=['normal','random','shuffle_labels','off','no_metadata'],default='normal')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); train,val=lists(a.train_list,a.val_list)
    device=torch.device(a.device); model,front=load_front(a.checkpoint,device)
    if bool(front['smoke'])!=a.smoke: p.error('Smoke/formal mismatch')
    if front['val_sha256']!=digest(a.val_list): p.error('Val split changed')
    head=bank=None; mode='memory'; k=8
    if a.head:
        if not a.cache: p.error('--head requires matching Train --cache')
        _,meta,bank=read_cache(a.cache,a.checkpoint)
        if meta['val_sha256']!=digest(a.val_list): p.error('Cache split mismatch')
        ck=torch.load(a.head,map_location='cpu',weights_only=False)
        if ck.get('format')!='SRAC_FOURSTAGE_SUPPORT_V2' or ck['source_hash']!=source_hash(): p.error('Wrong head source/format')
        if ck['checkpoint_sha256']!=digest(a.checkpoint) or ck['bank_sha256']!=meta['bank_sha256']: p.error('Head/front/bank mismatch')
        head=SupportClassifier(meta['config']['dim'],len(meta['event_names'])).to(device); head.load_state_dict(ck['model'],strict=True)
        bank=device_bank(bank,device); mode=ck['mode']; k=ck['topk']
        if mode=='nomemory' and a.intervention!='normal': p.error('Memory intervention not applicable to nomemory')
    elif a.intervention!='normal': p.error('Intervention requires a memory head')
    out=new_output(a.output)
    result=evaluate_front(model,loader(a.dataset,val[:2] if a.smoke else val,workers=a.workers),
                          device,head,bank,mode,k,a.intervention)
    result.update(complete=True,arguments=vars(a),checkpoint_sha256=digest(a.checkpoint),
                  source_hash=source_hash(),scope='ALL valid GT pixels; no GT foreground filtering')
    save_json(out/'summary.json',result); print(result['metrics'],flush=True)

if __name__=='__main__': main()
