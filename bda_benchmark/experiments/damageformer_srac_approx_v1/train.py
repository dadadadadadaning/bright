"""Train structure/region stages only. RetrievalClassifier is not silently enabled."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .modules import RegionModule, partition_loss

VARIANTS = {'s0_grid':(False,False,False), 's1_learned':(True,False,False),
            's2_modulated':(True,True,False), 's3_weighted':(True,True,True)}

def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('checkpoint','dataset','train_list','val_list','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--variant',choices=list(VARIANTS),default='s3_weighted')
    p.add_argument('--steps',type=int,default=5000); p.add_argument('--eval_every',type=int,default=500)
    p.add_argument('--crop',type=int,default=512); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stride',type=int,default=8); p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if any('test' in Path(x).name.lower() for x in (a.train_list,a.val_list)): p.error('Test forbidden')
    if min(a.steps,a.eval_every,a.stride)<1: p.error('Positive steps/interval/stride required')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    from torch.utils.data import DataLoader
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
    from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
    from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent, parse_batch, seed_everything
    from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import read_list
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import fast_hist
    seed_everything(a.seed)
    device=torch.device(a.device)
    parent=DamageFormer_S8_MICB_DSMIL(mil_residual_scale=0).to(device).eval().requires_grad_(False)
    provenance=load_parent(parent,a.checkpoint)
    if provenance['unexpected_keys']: raise ValueError('Incompatible A1 checkpoint')
    (out/'manifest.json').write_text(json.dumps(dict(arguments=vars(a),parent=provenance,test_accessed=False,
       approximation=True, memory_enabled=False),indent=2))
    model=RegionModule(learned=VARIANTS[a.variant][0],modulation=VARIANTS[a.variant][1],weighted=VARIANTS[a.variant][2],stride=a.stride).to(device)
    train_names,val_names=read_list(a.train_list),read_list(a.val_list)
    if set(train_names)&set(val_names): raise ValueError('Train/Val overlap')
    train=DataLoader(MultimodalDamageAssessmentDatset(a.dataset,train_names,a.crop,type='train'),batch_size=1,shuffle=True,num_workers=a.workers)
    val=DataLoader(MultimodalDamageAssessmentDatset(a.dataset,val_names,1024,type='test'),batch_size=1,num_workers=a.workers)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps)
    iterator=iter(train); best=-1
    for step in range(1,a.steps+1):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(train); batch=next(iterator)
        pre,post,loc,gt,_=parse_batch(batch,device)
        with torch.no_grad(): _,_,aux=parent(pre,post,return_aux=True)
        model.train(); region=model(aux['rgb_feature'],aux['fused_roi'],aux['p_building'])
        logits=F.interpolate(region['logits'],gt.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.cross_entropy(logits,gt,ignore_index=255) if (gt!=255).any() else logits.sum()*0
        if VARIANTS[a.variant][0]: loss=loss+0.1*partition_loss(region,aux['rgb_feature'],loc)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); scheduler.step()
        if step==1 or step%50==0: print(f'{a.variant} {step}/{a.steps} loss={loss.item():.6f}',flush=True)
        if step%a.eval_every==0 or step==a.steps:
            model.eval(); hist=torch.zeros(4,4,dtype=torch.float64); basehist=hist.clone()
            with torch.no_grad():
                for batch in val:
                    pre,post,loc,gt,_=parse_batch(batch,device)
                    _,base,aux=parent(pre,post,return_aux=True)
                    r=model(aux['rgb_feature'],aux['fused_roi'],aux['p_building'])
                    pred=F.interpolate(r['logits'],gt.shape[-2:],mode='bilinear',align_corners=False).argmax(1)
                    hist+=fast_hist(pred[0],gt[0],4,255).cpu().double()
                    basehist+=fast_hist(base.argmax(1)[0],gt[0],4,255).cpu().double()
            def miou(h): return float((h.diag()/(h.sum(0)+h.sum(1)-h.diag()).clamp_min(1))[1:].mean())
            score=miou(hist)
            state=dict(model=model.state_dict(),step=step,args=vars(a),parent=provenance,valid_miou=score,base_valid_miou=miou(basehist),confusion=hist,test_accessed=False)
            if score>best: best=score; torch.save(state,out/'best.pth')
            if step==a.steps: torch.save(state,out/'last.pth')
            print(f'Val foreground mIoU={score:.6f} A1={miou(basehist):.6f}',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(best_valid_miou=best,steps=a.steps,memory_enabled=False,test_accessed=False)))

if __name__=='__main__': main()
