"""Val-only frozen checkpoint interventions. No training, threshold tuning or writeback."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import JointModel
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics,size_masks,LABELS
from bda_benchmark.experiments.damageformer_srac_approx_v1.audit import boundary_mask,counts

PAIRS=[('T0','T1_full'),('T0','T3_full'),('T1_pixel','T1_full'),
       ('T3_pixel','T3_full'),('T3_full','T3_pixel_context')]


def amplitude(base,delta,mask):
    # Class-centering removes common logit shifts that do not change softmax.
    base=base.float(); delta=delta.float()
    b=base-base.mean(1,keepdim=True); d=delta-delta.mean(1,keepdim=True)
    expanded=mask[None,None].expand_as(base)
    return dict(n=int(expanded.sum()),base_sq=float(base[expanded].double().square().sum()),
        delta_sq=float(delta[expanded].double().square().sum()),
        centered_base_sq=float(b[expanded].double().square().sum()),
        centered_delta_sq=float(d[expanded].double().square().sum()),
        pixels=int(mask.sum()),changed=int(((base.argmax(1)!=(base+delta).argmax(1))[0]&mask).sum()))


def summarize_amplitude(r):
    result=dict(r)
    for prefix in ('','centered_'):
        for name in ('base','delta'):
            result[prefix+name+'_rms']=(r[prefix+name+'_sq']/r['n'])**.5 if r['n'] else None
        result[prefix+'rms_ratio']=(r[prefix+'delta_sq']/r[prefix+'base_sq'])**.5 if r[prefix+'base_sq'] else None
    result['changed_fraction']=r['changed']/r['pixels'] if r['pixels'] else None
    return result


def add(dest,values):
    for k,v in values.items(): dest[k]=dest.get(k,0)+v


def update(store,pred,gt,mask):
    for name,x in pred.items():
        cm=torch.bincount((gt[mask]*4+x[mask]).long(),minlength=16).reshape(4,4)
        store.setdefault('cm',{}).setdefault(name,torch.zeros(4,4,dtype=torch.long)).add_(cm)
    for base,other in PAIRS:
        r=store.setdefault('pairs',{}).setdefault(base+'__'+other,{})
        add(r,counts(pred[base],pred[other],gt,mask))
        j=torch.bincount((gt[mask]*16+pred[base][mask]*4+pred[other][mask]).long(),minlength=64).reshape(4,4,4)
        store.setdefault('joint',{}).setdefault(base+'__'+other,torch.zeros_like(j)).add_(j)


def finalize(store):
    pairs={}
    for name,r in store['pairs'].items():
        j=store['joint'][name]; harm=torch.zeros(4,4,dtype=torch.long); rescue=harm.clone()
        for t in range(4):
            for w in range(4):
                if t!=w: harm[t,w]=j[t,t,w]; rescue[t,w]=j[t,w,t]
        assert int(harm.sum())==r['harmed'] and int(rescue.sum())==r['rescued']
        pairs[name]=dict(r,recovery_rate=r['rescued']/r['base_wrong'] if r['base_wrong'] else None,
            harm_rate=r['harmed']/r['base_correct'] if r['base_correct'] else None,
            harm_gt_to_wrong=harm.tolist(),rescue_gt_from_wrong=rescue.tolist())
    return dict(metrics={k:metrics(v) for k,v in store['cm'].items()},comparisons=pairs)


@torch.inference_mode()
def outputs(model,pre,post,intervene=False):
    _,base,aux=model.base(pre,post,return_aux=True)
    if model.head is None: return base,base,None
    args=(aux['rgb_feature'],aux['fused_roi'],aux['p_building'])
    delta=F.interpolate(model.head(*args).float(),base.shape[-2:],mode='bilinear',align_corners=False)
    replaced=None
    if intervene:
        original=model.head.variant
        try:
            # Exactly replace C with z; preserve all weights and the T3 backbone.
            model.head.variant='t1'
            replaced=base+F.interpolate(model.head(*args).float(),base.shape[-2:],mode='bilinear',align_corners=False)
        finally: model.head.variant=original
    return base,base+delta,replaced


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('run_root','dataset','val_list','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--tolerance_pp',type=float,default=.02)
    a=p.parse_args()
    if Path(a.val_list).name!='val_set.txt': p.error('Only Val allowed')
    from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import parse_batch,event_name,sha256_file
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
    from torch.utils.data import DataLoader
    names=[x.strip() for x in Path(a.val_list).read_text().splitlines() if x.strip()]
    if len(names)!=450 or len(set(names))!=450: raise ValueError('Expected unique Val450')
    models={}; provenance={}; digest=sha256_file(a.val_list)
    for v in ('t0','t1','t3'):
        path=Path(a.run_root)/v/'best.pth'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        m=ck['manifest']; args=m['arguments']
        if args['variant']!=v or m['val_sha256']!=digest: raise ValueError('Variant/Val mismatch')
        model=JointModel(v,args['dim'],args['stride'])
        model.load_state_dict(ck['model'],strict=True)
        models[v]=model.eval().requires_grad_(False).to(a.device)
        provenance[v]=dict(path=str(path),sha256=sha256_file(path),step=ck['step'],
            valid_miou=ck['valid_miou'],manifest=m)
        del ck
    for key in ('pretrain_sha256','train_sha256'):
        if len({v['manifest'][key] for v in provenance.values()})!=1: raise ValueError('Protocol mismatch: '+key)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,data): (out/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    save('protocol.json',dict(arguments=vars(a),checkpoints=provenance,test_accessed=False,
        definitions='pixel = own co-trained pixel branch, NOT independently trained T0; pixel_context = T3 C replaced by z, NOT trained T1',
        classes=LABELS,boundary='GT four-class transition radius2 pixels, evaluation only',
        sizes='GT foreground 8-connected component proxy, <256/256-1023/>=1024 pixels, not building instances',
        transition_axes='rows GT, columns wrong class; conditional group metrics are not full-image mIoU'))
    loader=DataLoader(MultimodalDamageAssessmentDatset(a.dataset,names,1024,type='test'),batch_size=1,num_workers=a.workers)
    groups={}; events={}; amps={}; images=[]
    for index,batch in enumerate(loader,1):
        pre,post,loc,gt,ns=parse_batch(batch,torch.device(a.device))
        target=gt[0].cpu().long(); valid=(target>=0)&(target<4)
        pred={}
        for v in ('t0','t1','t3'):
            base,full,replaced=outputs(models[v],pre,post,v=='t3')
            if v=='t0': pred['T0']=full.argmax(1)[0].cpu()
            else:
                prefix=v.upper()
                pred[prefix+'_pixel']=base.argmax(1)[0].cpu(); pred[prefix+'_full']=full.argmax(1)[0].cpu()
                add(amps.setdefault(prefix,{}),amplitude(base,full-base,valid.to(base.device)))
                if replaced is not None: pred['T3_pixel_context']=replaced.argmax(1)[0].cpu()
            del base,full,replaced
        edge=boundary_mask(target)
        masks=dict(all=valid,boundary=edge,nonboundary=valid&~edge,foreground_interior=valid&(target>0)&~edge)
        for c,label in enumerate(LABELS):
            masks[label]=valid&(target==c)
            masks[label+'__boundary']=valid&(target==c)&edge
            masks[label+'__interior']=valid&(target==c)&~edge
        sizes,_=size_masks(target); masks.update(sizes)
        for name,mask in masks.items(): update(groups.setdefault(name,{}),pred,target,mask)
        ev=event_name(str(ns[0])); update(events.setdefault(ev,{}),pred,target,valid)
        row={}; update(row,pred,target,valid)
        images.append(dict(image=str(ns[0]),event=ev,**finalize(row)))
        if index%10==0: print(f'audit {index}/450',flush=True)
    result=dict(groups={k:finalize(v) for k,v in groups.items()},per_event={k:finalize(v) for k,v in events.items()},
        amplitude={k:summarize_amplitude(v) for k,v in amps.items()},test_accessed=False,image_count=len(images))
    scores=result['groups']['all']['metrics']
    checks={v:abs(scores['T0' if v=='t0' else v.upper()+'_full']['valid_miou']-provenance[v]['valid_miou'])*100<=a.tolerance_pp for v in provenance}
    result.update(reproduction_pass=checks,complete=all(checks.values()))
    save('summary.json',result); save('per_image.json',images)
    for k,v in scores.items(): print(k,100*v['valid_miou'])
    if not all(checks.values()): raise RuntimeError('Full output reproduction failed; do not interpret interventions')
    save('SUCCESS.json',dict(complete=True,test_accessed=False))
    print('BRANCH AUDIT COMPLETE. No training. No Test.',flush=True)


if __name__=='__main__': main()
