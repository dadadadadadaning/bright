"""Frozen Train query -> Train bank neighbor IDs, stored in a NumPy mmap table."""
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from .common import read_cache,new_output,digest,save_json,source_hash

def select_ids(q,bank,event,image,k):
    score=F.normalize(q.float(),dim=-1)@F.normalize(bank['tokens'].float(),dim=-1).T
    allowed=(bank['events']!=event)&(bank['images']!=image)
    score=score.masked_fill(~allowed[None],-torch.inf)
    values,ids=score.topk(min(k,len(bank['tokens'])),dim=1)
    ids=ids.masked_fill(~torch.isfinite(values),-1)
    if ids.shape[1]<k:
        ids=torch.cat([ids,ids.new_full((len(ids),k-ids.shape[1]),-1)],1)
    return ids

def gather_support(ids,bank):
    valid=ids>=0
    safe=ids.clamp_min(0).long()
    return (bank['tokens'][safe],bank['distribution'][safe],valid,
            {n:bank[n][safe] for n in ('geometry','events','disasters')})

def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('checkpoint','cache','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--topk',type=int,default=8)
    a=p.parse_args()
    if a.topk<1: p.error('Positive topk required')
    root,meta,bank=read_cache(a.cache,a.checkpoint)
    out=new_output(a.output)
    offsets=[]; total=0
    for record in meta['shards']:
        if digest(root/record['path'])!=record['sha256']: raise ValueError('Shard hash mismatch')
        # PyTorch 2.2 requires a string filename when mmap=True.
        shard=torch.load(str(root/record['path']),map_location='cpu',weights_only=False,mmap=True)
        offsets.append(dict(start=total,count=len(shard['tokens'])))
        total+=len(shard['tokens'])
    table=np.lib.format.open_memmap(out/'neighbors.npy',mode='w+',dtype=np.int32,shape=(total,a.topk))
    with torch.inference_mode():
        for i,(record,offset) in enumerate(zip(meta['shards'],offsets)):
            shard=torch.load(str(root/record['path']),map_location='cpu',weights_only=False,mmap=True)
            for start in range(0,offset['count'],256):
                ids=select_ids(shard['tokens'][start:start+256],bank,shard['event'],shard['image'],a.topk)
                begin=offset['start']+start
                table[begin:begin+len(ids)]=ids.numpy().astype(np.int32)
            print(f'neighbors {i+1}/{len(offsets)}',flush=True)
    table.flush(); del table
    save_json(out/'manifest.json',dict(complete=True,source_hash=source_hash(),
              cache_sha256=digest(root/'manifest.json'),bank_sha256=meta['bank_sha256'],
              table_sha256=digest(out/'neighbors.npy'),topk=a.topk,offsets=offsets,
              test_accessed=False,policy='Exclude same event OR same image; IDs only, no query labels'))

def open_table(path,cache_root,bank_hash,k):
    import json
    from pathlib import Path
    root=Path(path); meta=json.loads((root/'manifest.json').read_text())
    if not meta['complete'] or meta['source_hash']!=source_hash() or meta['topk']!=k:
        raise ValueError('Neighbor table version/topk mismatch')
    if meta['cache_sha256']!=digest(cache_root/'manifest.json') or meta['bank_sha256']!=bank_hash:
        raise ValueError('Neighbor table points to different cache/bank')
    if meta['table_sha256']!=digest(root/'neighbors.npy'): raise ValueError('Neighbor table corrupted')
    return np.load(root/'neighbors.npy',mmap_mode='r'),meta

if __name__=='__main__': main()
