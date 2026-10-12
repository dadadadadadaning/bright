"""Train-only BN calibration and Val-only probability projection audit."""
import argparse
import hashlib
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.common import (
    lists, loader, load_front, new_output, save_json, save_pt, digest,
    parse_batch, cm_update, metrics, event, seed_everything, source_hash)
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool


def parameters_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class BNProbe:
    """Route BN buffers by invocation, without duplicating/changing parameters.

    Each target BN must execute twice per full frontend forward: original then
    modulated. All modules stay eval; statistics are exact pixel-weighted input
    moments, not small-batch EMA. Calibration pass N uses pass N-1 buffers.
    """
    def __init__(self,model):
        self.model=model
        self.modules={n:m for n,m in model.named_modules()
                      if isinstance(m,torch.nn.BatchNorm2d) and
                      (n.startswith('base.fusion_layer_1.') or n.startswith('base.fusion_layer_3.'))}
        if len(self.modules)!=4: raise ValueError('Expected four shared fusion BNs')
        self.original={n:(m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone())
                       for n,m in self.modules.items()}

    @contextmanager
    def route(self,stats=None,collect=None):
        handles=[]; counts={}
        def begin(*_): counts.clear()
        def end(*_):
            if any(counts.get(n)!=2 for n in self.modules):
                raise RuntimeError('Expected exactly two calls per shared BN')
        handles.append(self.model.register_forward_pre_hook(begin))
        def hook(name):
            def apply(module,inputs):
                call=counts.get(name,0); counts[name]=call+1
                if call>1: raise RuntimeError('Unexpected BN call order')
                key=f'{name}:{call}'
                if collect is not None:
                    x=inputs[0].detach().float()
                    var,mean=torch.var_mean(x,dim=(0,2,3),unbiased=False)
                    count=x.numel()//x.shape[1]
                    mean=mean.double().cpu(); m2=var.double().cpu()*count
                    if key not in collect: collect[key]=[count,mean,m2]
                    else:
                        old,mu,acc=collect[key]; total=old+count; delta=mean-mu
                        collect[key]=[total,mu+delta*(count/total),acc+m2+delta.square()*(old*count/total)]
                mu,var=(stats[key] if stats is not None else self.original[name][:2])
                module.running_mean.copy_(mu); module.running_var.copy_(var)
            return apply
        for n,m in self.modules.items(): handles.append(m.register_forward_pre_hook(hook(n)))
        handles.append(self.model.register_forward_hook(end))
        try: yield
        finally:
            for h in handles: h.remove()
            for n,m in self.modules.items():
                mu,var,num=self.original[n]
                m.running_mean.copy_(mu); m.running_var.copy_(var); m.num_batches_tracked.copy_(num)

    @staticmethod
    def finish(collected,mode):
        result={}
        for key,(n,mu,m2) in collected.items():
            if mode=='split': result[key]=(mu.float(),(m2/max(1,n-1)).float())
            else:
                name=key.rsplit(':',1)[0]
                a,ma,va=collected[name+':0']; b,mb,vb=collected[name+':1']
                mean=(a*ma+b*mb)/(a+b)
                var=(va+vb+(ma-mb).square()*(a*b/(a+b)))/max(1,a+b-1)
                result[key]=(mean.float(),var.float())
        return result


def probability_paths(r):
    pixel=r['pixel'].float().softmax(1)
    small=F.interpolate(pixel,r['shape'],mode='bilinear',align_corners=False)
    pooled=soft_pool_unpool(small,r['q'].float(),r['ids'],r['tokens'].shape[1])['reconstruction']
    def up(x):
        x=F.interpolate(x,pixel.shape[-2:],mode='bilinear',align_corners=False)
        return x/x.sum(1,keepdim=True).clamp_min(1e-8)
    return dict(pixel=pixel,resize_only=up(small),projection=up(pooled),region=r['prob'])


@torch.inference_mode()
def evaluate(model,data,device):
    model.eval(); cms={}; events={}; trans=torch.zeros(4,4,4,dtype=torch.long); count=0
    for batch in data:
        pre,post,_,gt,names=parse_batch(batch,device)
        preds={k:p.argmax(1) for k,p in probability_paths(model(pre,post)).items()}
        ev=event(names[0]); events.setdefault(ev,{})
        for k,p in preds.items():
            cm_update(cms.setdefault(k,torch.zeros(4,4,dtype=torch.long)),gt,p)
            cm_update(events[ev].setdefault(k,torch.zeros(4,4,dtype=torch.long)),gt,p)
        ok=(gt>=0)&(gt<4)
        trans+=torch.bincount((gt[ok]*16+preds['pixel'][ok]*4+preds['region'][ok]).long(),minlength=64).reshape(4,4,4).cpu()
        count+=len(pre)
        if count%25==0: print(f'audit {count}',flush=True)
    return dict(image_count=count,metrics={k:metrics(v) for k,v in cms.items()},
                per_event={ev:{k:metrics(v) for k,v in rows.items()} for ev,rows in events.items()},
                transitions_gt_pixel_region=trans.tolist(),test_accessed=False)


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('checkpoint','dataset','train_list','val_list','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--calibration_batches',type=int,default=128)
    p.add_argument('--passes',type=int,default=2)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if min(a.calibration_batches,a.passes)<1: p.error('Positive calibration budget required')
    train,val=lists(a.train_list,a.val_list); seed_everything(a.seed)
    out=new_output(a.output); device=torch.device(a.device)
    sha=digest(a.checkpoint); model,ckpt=load_front(a.checkpoint,device)
    if digest(a.checkpoint)!=sha: raise RuntimeError('Checkpoint changed during read; use a fixed copy')
    model.eval(); model.requires_grad_(False); before=parameters_hash(model); probe=BNProbe(model)
    n_batches=2 if a.smoke else a.calibration_batches
    def validation(): return loader(a.dataset,val[:2] if a.smoke else val,workers=a.workers,seed=a.seed+3)
    results={'original':evaluate(model,validation(),device)}
    save_json(out/'original.json',results['original'])
    for mode in ('shared','split'):
        stats=None; calibration_names=[]
        for iteration in range(a.passes):
            # Identical Train crops/order in both modes and every pass. Never use Val to calibrate.
            seed_everything(a.seed+100)
            data=loader(a.dataset,train,True,2,a.workers,seed=a.seed+100)
            collected={}; names_seen=[]
            with torch.inference_mode(),probe.route(stats,collected):
                for i,batch in enumerate(data):
                    pre,post,_,_,names=parse_batch(batch,device)
                    model(pre,post); names_seen.extend(names)
                    if (i+1)%16==0: print(f'calibrate {mode} pass={iteration+1} batches={i+1}',flush=True)
                    if i+1>=n_batches: break
            stats=probe.finish(collected,mode); calibration_names=names_seen
        with torch.inference_mode(),probe.route(stats): results[mode]=evaluate(model,validation(),device)
        save_pt(out/(mode+'_bn_buffers.pt'),stats)
        save_json(out/(mode+'.json'),dict(**results[mode],calibration_names=calibration_names))
    if parameters_hash(model)!=before: raise RuntimeError('Unexpected parameter mutation')
    save_json(out/'summary.json',dict(complete=True,arguments=vars(a),checkpoint_sha256=sha,
              checkpoint_step=ckpt['step'],frontend_source_hash=source_hash(),audit_sha256=digest(__file__),parameters_unchanged=True,
              calibration='Train-only deterministic random 512 crops, batch2, FP32; fusion1/3 BNs only',
              results=results,smoke=a.smoke,test_accessed=False))
    for mode,r in results.items(): print(mode,{k:round(v['valid_miou']*100,4) for k,v in r['metrics'].items()},flush=True)
    print('AUDIT COMPLETE. Parameters unchanged; original checkpoint untouched.',flush=True)

if __name__=='__main__': main()
