import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import parse_batch, seed_everything, event_name
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()

def source_hash():
    h=hashlib.sha256()
    repo=Path(__file__).resolve().parents[3]
    external=['bda_benchmark/experiments/damageformer_joint_region_v1/model.py',
              'bda_benchmark/experiments/damageformer_iterative_partition_v1/model.py',
              'bda_benchmark/experiments/damageformer_a1_cross_modal_superpixel_v1/association.py',
              'bda_benchmark/model/DamageFormer_S8_MICB_DSMIL.py',
              'bda_benchmark/model/DamageFormer_S3_BGMP_K3.py',
              'bda_benchmark/dataset/make_data_loader.py']
    files=sorted(Path(__file__).parent.glob('*.py'))+[repo/x for x in external]
    for p in files:
        h.update(p.relative_to(repo).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def save_pt(path,obj):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp')
    torch.save(obj,tmp); tmp.replace(p)

def new_output(path):
    p=Path(path); p.mkdir(parents=True,exist_ok=False); return p

def worker_seed(_):
    s=torch.initial_seed()%2**32; random.seed(s); np.random.seed(s)

def event(name): return event_name(Path(str(name)).stem)

def lists(train,val):
    if Path(train).name!='train_set.txt' or Path(val).name!='val_set.txt':
        raise ValueError('Only standard train_set.txt and val_set.txt allowed. Test forbidden.')
    a,b=([x.strip() for x in Path(p).read_text().splitlines() if x.strip()] for p in (train,val))
    if len(a)!=2918 or len(b)!=450 or set(a)&set(b) or len(set(a))!=len(a) or len(set(b))!=len(b):
        raise ValueError('Expected disjoint unique Train2918 and Val450')
    return a,b

def loader(dataset,names,train=False,batch=1,workers=4,crop=512,seed=3407):
    from torch.utils.data import DataLoader
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
    ds=MultimodalDamageAssessmentDatset(dataset,names,crop if train else 1024,type='train' if train else 'test')
    return DataLoader(ds,batch_size=batch,shuffle=train,num_workers=workers,drop_last=train,
                      worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(seed))

def load_front(path,device):
    from .model import SRACSwin
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('format')!='SRAC_SWIN_FRONT_V1': raise ValueError('Wrong front checkpoint format')
    if data['source_hash']!=source_hash(): raise ValueError('Source changed since training; use archived matching code')
    model=SRACSwin(**data['config']).to(device)
    model.load_state_dict(data['model'],strict=True)
    return model,data

def read_cache(path,checkpoint):
    root=Path(path); meta=json.loads((root/'manifest.json').read_text())
    if not meta.get('complete') or meta['split']!='train' or meta['test_accessed']:
        raise ValueError('Complete Train-only cache required')
    if meta['checkpoint_sha256']!=digest(checkpoint) or meta['source_hash']!=source_hash():
        raise ValueError('Stale or incompatible cache')
    bank=torch.load(root/'bank.pt',map_location='cpu',weights_only=False)
    if bank['checkpoint_sha256']!=meta['checkpoint_sha256']: raise ValueError('Bank model mismatch')
    if digest(root/'bank.pt')!=meta['bank_sha256']: raise ValueError('Bank hash mismatch')
    if bank['tokens'].ndim!=2 or bank['distribution'].shape!=(len(bank['tokens']),4):
        raise ValueError('Invalid memory dimensions')
    if not torch.isfinite(bank['tokens']).all() or not torch.isfinite(bank['distribution']).all():
        raise ValueError('Nonfinite memory')
    if (bank['distribution']<0).any() or not torch.allclose(bank['distribution'].sum(-1),torch.ones(len(bank['tokens'])),atol=1e-5):
        raise ValueError('Invalid memory label distribution')
    return root,meta,bank

def cm_update(cm,gt,pred):
    ok=(gt>=0)&(gt<4)
    cm+=torch.bincount((gt[ok]*4+pred[ok]).long(),minlength=16).reshape(4,4).cpu()

@torch.inference_mode()
def evaluate_front(model,data,device,head=None,bank=None,mode='memory',topk=8,intervention='normal',seed=3407):
    from .model import project_probabilities
    from .retrieval import retrieve
    model.eval()
    if head is not None: head.eval()
    cms={k:torch.zeros(4,4,dtype=torch.long) for k in ['region','pixel']}
    events={}; image_count=0; supported=total=0
    rng=torch.Generator(device=device).manual_seed(seed)
    for batch in data:
        pre,post,loc,gt,names=parse_batch(batch,device)
        r=model(pre,post)
        prob=r['prob']
        if head is not None:
            if len(pre)!=1: raise ValueError('Evaluation requires batch1')
            z=r['tokens'][0]; outputs=[]
            ev=bank['event_names'].index(event(names[0])) if event(names[0]) in bank['event_names'] else -1
            for start in range(0,len(z),256):
                q=z[start:start+256]
                if mode=='nomemory': logits=head(q,mode=mode)
                else:
                    s,d,v=retrieve(q,bank,ev,-1,topk,intervention,rng)
                    supported+=int(v.any(1).sum()); total+=len(q)
                    logits=head(q,s,d,v)
                outputs.append(logits)
            prob=project_probabilities(torch.cat(outputs)[None],r['q'],r['ids'],r['shape'],gt.shape[-2:])
        pred=prob.argmax(1)
        cm_update(cms['region'],gt,pred); cm_update(cms['pixel'],gt,r['pixel'].argmax(1))
        ev=event(names[0]); events.setdefault(ev,torch.zeros(4,4,dtype=torch.long))
        cm_update(events[ev],gt,pred)
        image_count+=len(pre)
        if image_count%25==0: print(f'evaluate {image_count}',flush=True)
    return dict(metrics={k:metrics(v) for k,v in cms.items()},per_event={k:metrics(v) for k,v in events.items()},
                image_count=image_count,support_coverage=supported/max(1,total),test_accessed=False)

def device_bank(bank,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in bank.items()}
