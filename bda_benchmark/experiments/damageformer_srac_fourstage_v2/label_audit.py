"""Reject invalid targets on CPU before launching a CUDA loss kernel."""
import torch

def transfer_checked_batch(batch,device):
    """Own CPU int64 storage before synchronous transfer; verify before forward."""
    pre,post,loc,damage,names=batch[:5]
    images=[x.detach().to(dtype=torch.float32).contiguous().clone()
            for x in (pre,post)]
    cpu=[x.detach().to(device='cpu',dtype=torch.long).contiguous().clone()
         for x in (loc,damage)]
    moved=[x.to(device=device,non_blocking=False,copy=True) for x in cpu]
    for name,original,actual in zip(('location','damage'),cpu,moved):
        if not torch.equal(original,actual.cpu()):
            raise RuntimeError('CPU/GPU label transfer mismatch: '+name+'; samples='+str(list(names)))
    return (images[0].to(device,non_blocking=False),
            images[1].to(device,non_blocking=False),*moved,list(names))

def validate_batch_labels(batch,step,microstep):
    report=dict(step=step,microstep=microstep,samples=[],valid=True)
    names=batch[4]
    for i,name in enumerate(names):
        row=dict(name=str(name))
        for key,target,classes in [('location',batch[2],2),('damage',batch[3],4)]:
            value=torch.as_tensor(target[i]).detach().cpu()
            ids,counts=torch.unique(value,return_counts=True)
            legal=((value>=0)&(value<classes)&(value==value.round()))|(value==255)
            ok=bool(legal.all())
            row[key]=dict(shape=list(value.shape),dtype=str(value.dtype),
                          values=ids.tolist(),counts=counts.tolist(),valid=ok)
            report['valid']=report['valid'] and ok
        report['samples'].append(row)
    return report
