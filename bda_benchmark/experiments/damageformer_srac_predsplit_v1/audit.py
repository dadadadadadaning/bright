"""Read-only Val audit of trained split heads against frozen A1."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage
from .modules import SplitRegionModule
from bda_benchmark.experiments.damageformer_srac_approx_v1.audit import boundary_mask, counts

LABELS = ['Background', 'Intact', 'Damaged', 'Destroyed']


def metrics(cm):
    h = cm.double(); tp = h.diag(); rows = h.sum(1); cols = h.sum(0)
    union = rows + cols - tp
    def ratios(num, den):
        return [float(n/d) if d > 0 else None for n,d in zip(num,den)]
    iou = ratios(tp, union)
    present = [x for x in iou[1:] if x is not None]
    return dict(count=int(h.sum()), confusion=h.long().tolist(),
        valid_miou=float((tp/union.clamp_min(1))[1:].mean()) if h.sum()>0 else None,
        present_foreground_miou=sum(present)/len(present) if present else None,
        iou=iou, precision=ratios(tp,cols), recall=ratios(tp,rows),
        f1=ratios(2*tp,rows+cols), support=rows.long().tolist(),
        background_to_building=int(h[0,1:].sum()),
        building_to_background=int(h[1:,0].sum()))


def size_masks(gt, small=256, large=1024):
    # Evaluation proxy only: touching buildings merge; cropped buildings truncate.
    components, n = ndimage.label(((gt>0)&(gt<4)).numpy(), structure=np.ones((3,3)))
    area = np.bincount(components.ravel())
    area[0] = 0
    sizes = torch.from_numpy(area[components])
    masks = dict(size_small=(sizes>0)&(sizes<small),
        size_medium=(sizes>=small)&(sizes<large), size_large=sizes>=large)
    number = dict(size_small=int(((area[1:]>0)&(area[1:]<small)).sum()),
        size_medium=int(((area[1:]>=small)&(area[1:]<large)).sum()),
        size_large=int((area[1:]>=large).sum()))
    return masks, number


def accumulate(dest, predictions, gt, mask, transitions=False):
    for name, pred in predictions.items():
        cm = torch.bincount((gt[mask]*4+pred[mask]).long(), minlength=16).reshape(4,4)
        dest.setdefault('cm',{}).setdefault(name,torch.zeros(4,4,dtype=torch.long)).add_(cm)
    for base, other in [('A1','S3_split'),('A1','Grid_split'),('Grid_split','S3_split')]:
        row = counts(predictions[base],predictions[other],gt,mask)
        target = dest.setdefault('comparisons',{}).setdefault(base+'__'+other,{k:0 for k in row})
        for key,value in row.items(): target[key] += value
        if transitions:
            # Axes: true class, baseline prediction, candidate prediction.
            joint=torch.bincount((gt[mask]*16+predictions[base][mask]*4+
                predictions[other][mask]).long(),minlength=64).reshape(4,4,4)
            dest.setdefault('joint',{}).setdefault(base+'__'+other,
                torch.zeros(4,4,4,dtype=torch.long)).add_(joint)


def finalize(dest):
    result = dict(metrics={k:metrics(h) for k,h in dest['cm'].items()}, comparisons={})
    for pair,row in dest['comparisons'].items():
        result['comparisons'][pair] = dict(row,
            recovery_rate=row['rescued']/row['base_wrong'] if row['base_wrong'] else None,
            harm_rate=row['harmed']/row['base_correct'] if row['base_correct'] else None)
        if pair in dest.get('joint',{}):
            j=dest['joint'][pair]
            harm=torch.zeros(4,4,dtype=torch.long)
            rescue=harm.clone()
            for truth in range(4):
                for wrong in range(4):
                    if truth!=wrong:
                        harm[truth,wrong]=j[truth,truth,wrong]
                        rescue[truth,wrong]=j[truth,wrong,truth]
            if int(harm.sum())!=row['harmed'] or int(rescue.sum())!=row['rescued']:
                raise AssertionError('Transition counts disagree with rescue/harm')
            result['comparisons'][pair].update(joint_gt_base_other=j.tolist(),
                harm_gt_to_wrong=harm.tolist(),rescue_gt_from_wrong=rescue.tolist())
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('a1','s3','grid','dataset','val_list','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--expected_images',type=int,default=450)
    p.add_argument('--expected_a1',type=float,default=0.6168659209491562)
    p.add_argument('--tolerance_pp',type=float,default=0.02)
    p.add_argument('--small',type=int,default=256); p.add_argument('--large',type=int,default=1024)
    p.add_argument('--cross_stats',action='store_true',help='Add class x position groups and paired class transitions')
    a=p.parse_args()
    if Path(a.val_list).name!='val_set.txt': p.error('Only val_set.txt allowed. Test forbidden.')
    if not 0<a.small<a.large: p.error('Require 0 < small < large')
    from torch.utils.data import DataLoader
    from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
    from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
    from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import load_parent,parse_batch,sha256_file,event_name
    from bda_benchmark.experiments.spi_building_prototype_memory_v1.core import read_list
    names=read_list(a.val_list)
    if len(names)!=a.expected_images or len(set(names))!=len(names): raise ValueError('Val count/duplicates mismatch')
    device=torch.device(a.device); digest=sha256_file(a.a1)
    models={}; sources={}
    for name,path,variant in [('S3_split',a.s3,'s3_predsplit'),('Grid_split',a.grid,'grid_predsplit')]:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['args']['variant']!=variant or ck['parent']['sha256']!=digest:
            raise ValueError('Wrong variant or A1 parent: '+name)
        model=SplitRegionModule(stride=ck['args']['stride'],learned=variant=='s3_predsplit')
        model.load_state_dict(ck['model'],strict=True)
        models[name]=model.to(device).eval().requires_grad_(False)
        sources[name]=dict(path=path,sha256=sha256_file(path),step=ck['step'],
            saved_valid_miou=ck['valid_miou'],args=ck['args'])
    parent=DamageFormer_S8_MICB_DSMIL(mil_residual_scale=0).to(device).eval().requires_grad_(False)
    report=load_parent(parent,a.a1)
    if report['unexpected_keys']: raise ValueError('Unexpected A1 keys')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,obj):
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    save('protocol.json',dict(arguments=vars(a),parent=report,checkpoints=sources,
        val_sha256=sha256_file(a.val_list),test_accessed=False,
        class_order=LABELS,confusion_orientation='rows=GT, columns=prediction',
        cross_stats=a.cross_stats,
        transition_axes='joint[GT][baseline prediction][candidate prediction]; harm[GT][new wrong class]; rescue[GT][old wrong class]',
        boundary='GT four-class transitions, ignore excluded, radius2 original pixels',
        size_definition='GT foreground 8-connected components, not true building instances; area in original pixels; evaluation only',
        size_thresholds=[a.small,a.large],size_caution='Foreground-only conditional pixel scores; do not compare their mIoU with full-image mIoU',
        missing_class_policy='IoU/precision/recall null if denominator zero; valid_miou keeps fixed three classes with absent class zero; also report present_foreground_miou'))
    loader=DataLoader(MultimodalDamageAssessmentDatset(a.dataset,names,1024,type='test'),batch_size=1,num_workers=a.workers)
    groups={}; events={}; image_rows=[]; component_counts={}; image_counts={}
    for index,batch in enumerate(loader,1):
        pre,post,loc,gt,sample_names=parse_batch(batch,device)
        _,base,aux=parent(pre,post,return_aux=True)
        target=gt[0].cpu().long()
        predictions={'A1':base.argmax(1)[0].cpu()}
        for name,model in models.items():
            r=model(aux['rgb_feature'],aux['fused_roi'],aux['p_building'])
            predictions[name]=F.interpolate(r['logits'],gt.shape[-2:],mode='bilinear',align_corners=False).argmax(1)[0].cpu()
        valid=(target>=0)&(target<4); edge=boundary_mask(target)
        masks=dict(all=valid,boundary=edge,nonboundary=valid&~edge,
            foreground=valid&(target>0),foreground_interior=valid&(target>0)&~edge)
        masks.update({label:valid&(target==i) for i,label in enumerate(LABELS)})
        if a.cross_stats:
            for i,label in enumerate(LABELS):
                masks[label+'__boundary']=valid&(target==i)&edge
                masks[label+'__interior']=valid&(target==i)&~edge
        sizes,numbers=size_masks(target,a.small,a.large); masks.update(sizes)
        for key,value in numbers.items(): component_counts[key]=component_counts.get(key,0)+value
        for key,mask in masks.items():
            accumulate(groups.setdefault(key,{}),predictions,target,mask,a.cross_stats)
            image_counts[key]=image_counts.get(key,0)+int(mask.any())
        event=event_name(str(sample_names[0]))
        accumulate(events.setdefault(event,{}),predictions,target,valid,a.cross_stats)
        row={}; accumulate(row,predictions,target,valid)
        image_rows.append(dict(image=str(sample_names[0]),event=event,**finalize(row)))
        if index%10==0 or index==len(names): print(f'audit {index}/{len(names)}',flush=True)
    summary=dict(groups={k:finalize(v) for k,v in groups.items()},
        per_event={k:finalize(v) for k,v in events.items()},image_count=len(image_rows),
        group_nonempty_images=image_counts,component_counts=component_counts,test_accessed=False)
    scores=summary['groups']['all']['metrics']
    checks={'A1':abs(scores['A1']['valid_miou']-a.expected_a1)*100<=a.tolerance_pp}
    checks.update({k:abs(scores[k]['valid_miou']-v['saved_valid_miou'])*100<=a.tolerance_pp for k,v in sources.items()})
    summary['reproduction_pass']=checks; summary['complete']=all(checks.values())
    save('summary.json',summary); save('per_image.json',image_rows)
    for name,row in scores.items(): print(f'{name}: foreground mIoU={100*row["valid_miou"]:.4f}%')
    if not all(checks.values()): raise RuntimeError('Reproduction mismatch; inspect results, do not interpret as matched audit')
    save('SUCCESS.json',dict(complete=True,image_count=len(image_rows),test_accessed=False))
    print('VAL COMPARISON COMPLETE. No training. No Test.',flush=True)


if __name__=='__main__': main()
