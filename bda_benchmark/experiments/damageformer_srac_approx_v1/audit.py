"""Val-only compression and complementarity audit; no training or pixel fusion."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .modules import RegionModule
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map, soft_pool_unpool


def metrics(cm):
    h=cm.double(); tp=h.diag()
    iou=tp/(h.sum(0)+h.sum(1)-tp).clamp_min(1)
    return dict(count=int(h.sum()), valid_miou=float(iou[1:].mean()), miou4=float(iou.mean()),
                iou=iou.tolist(), f1=(2*tp/(h.sum(0)+h.sum(1)).clamp_min(1)).tolist(), confusion=h.long().tolist())


def boundary_mask(gt, radius=2):
    valid=(gt>=0)&(gt<4); edge=torch.zeros_like(valid)
    d=(gt[1:]!=gt[:-1])&valid[1:]&valid[:-1]; edge[1:]|=d; edge[:-1]|=d
    d=(gt[:,1:]!=gt[:,:-1])&valid[:,1:]&valid[:,:-1]; edge[:,1:]|=d; edge[:,:-1]|=d
    return F.max_pool2d(edge[None,None].float(),2*radius+1,1,radius)[0,0].bool()&valid


def counts(base, other, gt, mask):
    a=base==gt; b=other==gt
    rescued=int((mask&~a&b).sum()); harmed=int((mask&a&~b).sum())
    return dict(count=int(mask.sum()), rescued=rescued,harmed=harmed,net=rescued-harmed,
                base_wrong=int((mask&~a).sum()), base_correct=int((mask&a).sum()),
                both_wrong=int((mask&~a&~b).sum()), disagreement=int((mask&(base!=other)).sum()))


@torch.inference_mode()
def main(spatial=False):
    p=argparse.ArgumentParser(__doc__)
    for key in ('a1','s3','dataset','val_list','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--expected_images',type=int,default=450)
    p.add_argument('--expected_a1',type=float,default=0.6168659209491562)
    p.add_argument('--tolerance_pp',type=float,default=0.02)
    a=p.parse_args()
    if Path(a.val_list).name!='val_set.txt': p.error('Only explicit val_set.txt allowed; Test forbidden')
    from torch.utils.data import DataLoader
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
    from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
    from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent, parse_batch, sha256_file, event_name
    from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import read_list
    checkpoint=torch.load(a.s3,map_location='cpu',weights_only=False)
    if checkpoint['args']['variant']!='s3_weighted': raise ValueError('Expected S3 checkpoint')
    if checkpoint['parent']['sha256']!=sha256_file(a.a1): raise ValueError('A1 parent mismatch')
    names=read_list(a.val_list)
    if len(names)!=a.expected_images: raise ValueError('Image count mismatch')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,value): (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    save('protocol.json',dict(arguments=vars(a),a1_sha256=sha256_file(a.a1),s3_sha256=sha256_file(a.s3),
         spatial_audit=spatial, grid_strides=[8,4,2] if spatial else [], predicted_building_threshold=0.5,
         split_definition='Each learned token split into predicted BG/FG children before pooling; same probabilities and q; no output class masking; no GT construction',
         val_list_sha256=sha256_file(a.val_list),s3_step=checkpoint['step'],test_accessed=False,
         boundary_definition='GT four-class transitions, ignore excluded, Chebyshev radius 2 original pixels',
         pooling='Probability downsample -> assignment-weighted mean -> unpool -> bilinear upsample. No S3 attention weights. Not an oracle.'))
    device=torch.device(a.device)
    parent=DamageFormer_S8_MICB_DSMIL().to(device).eval().requires_grad_(False)
    report=load_parent(parent,a.a1)
    if report['unexpected_keys']: raise ValueError('Unexpected A1 keys')
    model=RegionModule(stride=checkpoint['args']['stride']).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'])
    loader=DataLoader(MultimodalDamageAssessmentDatset(a.dataset,names,1024,type='test'),batch_size=1,num_workers=a.workers)
    keys=['A1','resize_only','grid_projection','s3_projection','S3']
    if spatial:
        keys += ['s3_predsplit','grid_stride8','grid_stride4','grid_stride2']
        from .spatial_ops import split_project, grid_project
    global_cm={k:torch.zeros(4,4,dtype=torch.long) for k in keys}
    events={}; group_counts={}; image_rows=[]
    for index,batch in enumerate(loader,1):
        pre,post,loc,gt,sample_names=parse_batch(batch,device)
        _,logits,aux=parent(pre,post,return_aux=True)
        r=model(aux['rgb_feature'],aux['fused_roi'],aux['p_building'])
        size=aux['fused_roi'].shape[-2:]; full=gt.shape[-2:]
        prob=F.interpolate(logits.softmax(1),size=size,mode='bilinear',align_corners=False)
        ids,_,gh,gw=candidate_map(*size,model.stride,device)
        q=torch.zeros_like(r['q']); q[:,4]=1
        grid=soft_pool_unpool(prob,q,ids,gh*gw)['reconstruction']
        learned=soft_pool_unpool(prob,r['q'],r['ids'],r['count'])['reconstruction']
        def up(x): return F.interpolate(x,size=full,mode='bilinear',align_corners=False).argmax(1)[0]
        pred=dict(A1=logits.argmax(1)[0],resize_only=up(prob),grid_projection=up(grid),s3_projection=up(learned),S3=up(r['logits']))
        split_stats=None
        if spatial:
            building=aux['p_building'][:,0]>=0.5
            pred['s3_predsplit']=up(split_project(prob,r['q'],r['ids'],r['count'],building))
            for stride in (8,4,2): pred[f'grid_stride{stride}']=up(grid_project(prob,stride))
            split_stats=dict(feature_height=size[0],feature_width=size[1],original_height=full[0],original_width=full[1],
                             s3_stride=model.stride,feature_building_fraction=float(building.float().mean()),
                             parent_token_count=r['count'])
        target=gt[0]; valid=(target>=0)&(target<4)
        edge=boundary_mask(target)
        masks=dict(all=valid,boundary=edge,nonboundary=valid&~edge,foreground=valid&(target>0),
                   foreground_interior=valid&(target>0)&~edge)
        masks.update({label:valid&(target==c) for c,label in enumerate(['Background','Intact','Damaged','Destroyed'])})
        event=event_name(str(sample_names[0]))
        if event not in events: events[event]={k:torch.zeros(4,4,dtype=torch.long) for k in keys}
        for k in keys:
            cm=torch.bincount(target[valid]*4+pred[k][valid],minlength=16).reshape(4,4).cpu()
            global_cm[k]+=cm; events[event][k]+=cm
        # Compare projections to resize-only too, separating resampling from region mixing.
        pairs=[('A1',k) for k in keys if k!='A1']+[('resize_only','grid_projection'),('resize_only','s3_projection')]
        if spatial:
            pairs += [('s3_projection','s3_predsplit'),('grid_stride8','grid_stride4'),('grid_stride4','grid_stride2')]
        for base,other in pairs:
            pair=base+'__'+other
            for group,mask in masks.items():
                value=counts(pred[base],pred[other],target,mask)
                dest=group_counts.setdefault(pair,{}).setdefault(group,{k:0 for k in value})
                for k,v in value.items(): dest[k]+=v
        image_rows.append(dict(image=str(sample_names[0]),event=event,**counts(pred['A1'],pred['S3'],target,valid)))
        if spatial:
            image_rows[-1]['spatial_metadata']=split_stats
            image_rows[-1]['confusion_by_method']={k:torch.bincount(target[valid]*4+pred[k][valid],minlength=16).reshape(4,4).cpu().tolist() for k in keys}
        if index%10==0 or index==len(names): print(f'audit {index}/{len(names)}',flush=True)
    scores={k:metrics(v) for k,v in global_cm.items()}
    reproduced=abs(scores['A1']['valid_miou']-a.expected_a1)*100<=a.tolerance_pp
    result=dict(metrics=scores,comparisons=group_counts,per_event={e:{k:metrics(h) for k,h in row.items()} for e,row in events.items()},
                a1_reproduction_pass=reproduced,test_accessed=False)
    if spatial:
        result['changes_pp']={f'{new}_minus_{old}':100*(scores[new]['valid_miou']-scores[old]['valid_miou'])
                              for old,new in [('s3_projection','s3_predsplit'),('grid_stride8','grid_stride4'),('grid_stride4','grid_stride2')]}
    save('summary.json',result); save('per_image.json',image_rows)
    for k,v in scores.items(): print(f'{k}: foreground mIoU={100*v["valid_miou"]:.4f}%')
    print('S3 vs A1:',group_counts['A1__S3']['all'])
    if not reproduced: raise RuntimeError('A1 reproduction mismatch. Diagnostics saved but not valid for conclusions.')
    print('VAL AUDIT COMPLETE. No training, no Test access.')


if __name__=='__main__': main()
