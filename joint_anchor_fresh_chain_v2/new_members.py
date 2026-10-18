"""Train-only bounded event/class selection of NEW parent cases.

Original V3 identities are unavailable. The library contains actual Q-matched
FP32 features and soft labels from Train images, with explicit new provenance.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import heapq
import json
import math
import time
import traceback
import torch
from io_utils import HERE,read,write,digest,verify_release,old_modules
from new_source import load as load_source

FORMAT='BRIGHT_NEW_Q_PARENT_BANK_V2'
CAPACITY=3411
PER_IMAGE_CLASS=2

def priority(seed,image,region):
    payload=f'{seed}:{image}:{region}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')

def select(heaps,row,quota):
    """Keep lowest stable priorities within each Train event and true-label class."""
    cell=(row['event'],row['class']); heap=heaps[cell]
    item=(-row['priority'],-row['image_id'],-row['region_id'],row)
    if len(heap)<quota:heapq.heappush(heap,item)
    elif item[:3]>heap[0][:3]:heapq.heapreplace(heap,item)

def selected_rows(heaps,capacity):
    rows=[item[3] for heap in heaps.values() for item in heap]
    # Fixed global budget is final tie-breaker. Every event/class has been bounded.
    rows=sorted(rows,key=lambda r:(r['priority'],r['image_id'],r['region_id']))[:capacity]
    return sorted(rows,key=lambda r:(r['image_id'],r['region_id']))

def _parent_from_rows(rows,event_names):
    _,old,*_=old_modules()
    n=len(rows)
    if n<1:raise ValueError('No eligible Train supports')
    bank={k:torch.tensor([r[k] for r in rows],dtype=torch.long) for k in
          ('images','events','regions','original_rows')}
    bank['member_ids']=torch.arange(n,dtype=torch.long)
    bank['event_names']=list(event_names)
    for k in ('t128','f256','distribution'):
        bank[k]=torch.stack([r[k] for r in rows]).to(torch.float32).contiguous()
    for k in ('mass','label_mass'):
        bank[k]=torch.tensor([r[k] for r in rows],dtype=torch.float32)
    bank['eligible']=torch.ones(n,dtype=torch.bool)
    bank['active']=bank['eligible'].clone()
    old._validate_bank(bank)
    return bank

def build(protocol,output,repo,workers=4,seed=3407,smoke=False):
    fs,old,_,_,_,_=old_modules()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('New member-bank output required: '+str(output))
    output.mkdir(parents=True)
    started=time.monotonic()
    summary=dict(format=FORMAT,complete=False,execution_complete=False,smoke=smoke,training=False,
                 train_only=True,test_accessed=False,phase='loading',image_count=0)
    write(output/'summary.json',summary)
    capture=native_capture=handle=None;guard=None
    try:
        bundle=load_source(protocol,output)
        names=bundle['train_names']
        audit,native,pool,region,transformer,paired=old._repo_dependencies(repo)
        event_names=sorted({transformer.event(n) for n in names})
        if len(event_names)<2:raise ValueError('Cross-event source set required')
        quota=max(1,math.ceil(CAPACITY/(4*len(event_names))))
        if smoke:
            image_ids=[];seen=set()
            for iid,name in enumerate(names):
                event=transformer.event(name)
                if event not in seen:
                    image_ids.append(iid);seen.add(event)
                if len(image_ids)==4:break
            if len(image_ids)!=4:raise ValueError('Smoke needs four Train events')
            quota=2
        else:image_ids=list(range(len(names)))
        requested=[names[i] for i in image_ids]
        front=bundle['front'];before=old.module_hash(front)
        capture=audit.FrozenFrontCapture(front)
        native_capture=native.NativeA1LogitsCapture(front)
        counter=[0]
        def one_forward(*_):counter[0]+=1
        handle=front.register_forward_hook(one_forward)
        guard=pool.PoolGuard(output/'pool_debug',mode='native',capture='failure',check_reference=False)
        data=audit.full_image_loader(bundle['args'].dataset,requested,False,workers,seed+1)
        heaps=defaultdict(list);observed=defaultdict(int);file_hashes={};source_stats=defaultdict(int)
        with torch.no_grad(),guard,(output/'images.jsonl').open('x') as journal:
            for index,batch in enumerate(data):
                if index>=len(requested):raise ValueError('Extra Train source batch')
                image=requested[index];iid=image_ids[index]
                for p in region.sample_paths(bundle['args'].dataset,image).values():
                    file_hashes[str(Path(p).resolve())]=digest(p)
                pre,post,_loc,target,image_names=audit.parse_batch(batch,bundle['device'])
                if len(pre)!=1 or list(image_names)!=[image]:raise ValueError('Train batch identity changed')
                guard.set_image(image);native_capture.clear();prior=counter[0]
                result,feature=capture(pre,post)
                if counter[0]!=prior+1:raise ValueError('Expected one frozen front forward')
                logits=native_capture.take()
                _,check=native.native_grid_probq(result,feature,logits,target.shape[-2:])
                if check.get('passed') is not True:raise ValueError('A1 native check failed: '+image)
                pair=paired.extract_pair(result,feature,target)
                if pair['q_check'].get('passed') is not True or pair['label_check'].get('passed') is not True:
                    raise ValueError('Q/label conservation failed: '+image)
                event=transformer.event(image);eid=event_names.index(event);sampled=0
                active=pair['active'].detach().cpu();distribution=pair['distribution'].detach().cpu()
                t128=pair['t128'].detach().cpu();f256=pair['f256'].detach().cpu()
                mass=pair['mass'].detach().cpu();label_mass=pair['label_mass'].detach().cpu()
                per_class=defaultdict(list)
                for rid in active.nonzero().flatten().tolist():
                    if not bool(result['valid_regions'][0,rid]):continue
                    if mass[rid]<=0 or label_mass[rid]<=0:continue
                    cls=int(distribution[rid].argmax().item())
                    per_class[cls].append((priority(seed,iid,rid),rid))
                    observed[(event,cls)]+=1
                for cls,candidates in per_class.items():
                    # Prevent one image filling a cell. Bounded per-image selection.
                    for value,rid in sorted(candidates)[:PER_IMAGE_CLASS]:
                        row=dict(images=iid,events=eid,regions=rid,original_rows=iid*1000000+rid,
                                 t128=t128[rid].clone(),f256=f256[rid].clone(),
                                 distribution=distribution[rid].clone(),mass=float(mass[rid]),
                                 label_mass=float(label_mass[rid]),event=event,class_=cls,
                                 image_id=iid,region_id=rid,priority=value)
                        row['class']=row.pop('class_')
                        select(heaps,row,quota);sampled+=1
                source_stats[event]+=1
                frame=dict(index=index,image=image,event=event,image_id=iid,forward_count=counter[0],
                           active_regions=int(active.sum()),candidate_regions=sum(len(v) for v in per_class.values()),
                           considered_after_image_cap=sampled,native_a1=check,q_check=pair['q_check'],label_check=pair['label_check'])
                journal.write(json.dumps(frame,allow_nan=False)+'\n');journal.flush()
                summary.update(phase='scan_train',image_count=index+1,image=image)
                write(output/'summary.json',summary)
                if (index+1)%50==0 or index+1==len(requested):
                    print('NEW TRAIN MEMBERS',index+1,'/',len(requested),flush=True)
        if counter[0]!=len(requested):raise ValueError('Incomplete front forwards')
        if old.module_hash(front)!=before:raise ValueError('Frozen front changed during bank build')
        if not guard.restored or guard.failures or guard.restore_conflicts:
            raise ValueError('Native PoolGuard failed')
        rows=selected_rows(heaps,CAPACITY if not smoke else 32)
        bank=_parent_from_rows(rows,event_names)
        if old._coverage(bank,smoke=smoke)['eligible_event_count']<2:raise ValueError('Missing cross-event support')
        source_ids=bank['images'].unique(sorted=True).tolist()
        if smoke and len(source_ids)!=4:raise ValueError('Smoke selection lost a source event')
        source_images=[names[i] for i in source_ids]
        protocol_record=dict(format='BRIGHT_NEW_MEMBER_SELECTION_V2',source_protocol=str(Path(protocol).resolve()),
                             source_protocol_sha256=digest(protocol),seed=seed,capacity=CAPACITY,
                             per_event_class_quota=quota,per_image_class_cap=PER_IMAGE_CLASS,
                             selection='smallest SHA256(seed,image_id,region_id) per dominant Train soft-label class/event',
                             smoke=smoke,old_v3_members_reused=False,test_accessed=False)
        write(output/'protocol.json',protocol_record)
        with (output/'paired_bank.pt').open('xb') as f:torch.save(bank,f)
        again=torch.load(output/'paired_bank.pt',map_location='cpu',weights_only=False)
        old._validate_bank(again,expected_count=len(rows))
        if old.membership_sha(again)!=old.membership_sha(bank):raise ValueError('Bank roundtrip differs')
        selected_files={str(Path(p).resolve()):file_hashes[str(Path(p).resolve())]
                        for n in source_images for p in region.sample_paths(bundle['args'].dataset,n).values()}
        if not selected_files:raise ValueError('No selected Train source image hashes')
        manifest=dict(format=FORMAT,complete=True,execution_complete=True,smoke=smoke,
            train_only=True,test_accessed=False,source_protocol_sha256=digest(protocol),
            source_input_fingerprints=bundle['input_fingerprints'],source_identity=old._source_identity(bundle),
            split_identity=old._split_identity(bundle),source_image_fingerprints=selected_files,
            source_scan_image_count=len(requested),image_count=len(source_ids),selected_train_image_ids=source_ids,
            selected_source_images=source_images,support_count=len(rows),membership_sha256=old.membership_sha(bank),
            paired_bank_sha256=digest(output/'paired_bank.pt'),rebuild_protocol_sha256=digest(output/'protocol.json'),
            selection_provenance=protocol_record,old_v3_members_reused=False,
            per_event_class_counts={f'{e}:{c}':sum(r['event']==e and r['class']==c for r in rows)
                                    for e in event_names for c in range(4)},
            available_per_event_class={f'{e}:{c}':observed[(e,c)] for e in event_names for c in range(4)},
            event_image_counts=dict(source_stats),release_sha256=verify_release(),
            file_sha256={n:digest(output/n) for n in ('paired_bank.pt','protocol.json','images.jsonl')},
            elapsed_seconds=time.monotonic()-started)
        write(output/'manifest.json',manifest)
        summary.update(complete=True,execution_complete=True,phase='complete_smoke' if smoke else 'complete',
                       support_count=len(rows),source_count=len(source_ids),manifest_sha256=digest(output/'manifest.json'),
                       elapsed_seconds=time.monotonic()-started)
        write(output/'summary.json',summary)
        print('NEW TRAIN MEMBER BANK COMPLETE; no historical member identity claim.',flush=True)
        return summary
    except BaseException as e:
        summary.update(phase='failed',error=type(e).__name__+': '+str(e));write(output/'summary.json',summary)
        (output/'traceback.txt').write_text(traceback.format_exc());raise
    finally:
        if handle is not None:handle.remove()
        for obj in (native_capture,capture):
            if obj is not None:obj.close()

def load(root,bundle,allow_smoke=False):
    _,old,*_=old_modules();root=Path(root).resolve();m=read(root/'manifest.json')
    if m.get('format')!=FORMAT or m.get('complete') is not True or m.get('test_accessed') is not False:
        raise ValueError('New member bank incomplete')
    if m['smoke'] and not allow_smoke:raise ValueError('Smoke members cannot enter formal training')
    if m['source_protocol_sha256']!=bundle['source_protocol_sha256'] or m['release_sha256']!=verify_release():
        raise ValueError('New source/code binding differs')
    if m['source_identity']!=old._source_identity(bundle) or m['split_identity']!=old._split_identity(bundle):
        raise ValueError('New member source/split identity changed')
    if m['paired_bank_sha256']!=digest(root/'paired_bank.pt') or m['rebuild_protocol_sha256']!=digest(root/'protocol.json'):
        raise ValueError('New bank artifact changed')
    for name,sha in m['file_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('New member ledger changed')
    bank=torch.load(root/'paired_bank.pt',map_location=bundle['device'],weights_only=False)
    old._validate_bank({k:v.cpu() if torch.is_tensor(v) else v for k,v in bank.items()},expected_count=m['support_count'])
    if old.membership_sha(bank)!=m['membership_sha256']:raise ValueError('Membership hash changed')
    return bank,m
