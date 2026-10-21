"""Observe parent and single/soft child routes; never repair their values.

Owned CPU copies synchronize CUDA. Passive mode only records storage metadata;
the separate PassiveRoutes retains the original predicate operands on failure.
"""
import hashlib
import json
import math
from pathlib import Path
import traceback

from evaluate_checkpoint import write_json


def describe(torch, value):
    def number(x):
        x=float(x)
        return x if math.isfinite(x) else None
    raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
    result=dict(sha256=hashlib.sha256(raw).hexdigest(),shape=list(value.shape),dtype=str(value.dtype))
    if value.numel():
        result.update(min=number(value.min()),max=number(value.max()))
        if value.is_floating_point():
            result.update(negative_count=int((value<0).sum()),nonfinite_count=int((~torch.isfinite(value)).sum()))
            if value.ndim==3:
                sums=value.double().sum(1)
                result.update(cpu_float64_max_sum_error=number((sums-1).abs().max()),
                    cpu_float64_bad_sum_pixels=int(((sums-1).abs()>4e-5).sum()))
    return result


def metadata(value):
    try:version=value._version
    except RuntimeError:version=None
    return dict(object_id=id(value),device=str(value.device),dtype=str(value.dtype),shape=list(value.shape),
        storage_ptr=value.untyped_storage().data_ptr(),offset=value.storage_offset(),stride=list(value.stride()),version=version)


class TestRouteWatch:
    def __init__(self,torch,runtime,factory,evaluator,output,position,names,canonical_names,mode,target):
        self.torch,self.runtime,self.factory,self.evaluator=torch,runtime,factory,evaluator
        self.output,self.position=Path(output),position
        self.names,self.canonical_index=list(names),{n:i for i,n in enumerate(canonical_names)}
        if mode not in ('passive','watched'):raise ValueError('Unknown route observation mode')
        self.mode,self.target=mode,target
        self.example=None; self.supports=None; self.baseline={}; self.baseline_stats={}; self.latest_cpu={}; self.latest=None
        self.finished=False; self.saved=[]; self.first_exception=None; self.evidence_error=None
        self.snapshots=0; self.images_completed=0; self.changes=0

    def tensors(self):
        if self.example is None:return {}
        result={'parent.'+k:self.example.common[k] for k in ('q','ids')}
        for arm in ('b','d'):
            for key in ('route_q','route_ids'):
                result['source.'+arm+'.'+key]=self.example.supports[arm][key]
        for arm,support in (self.supports or {}).items():
            for key in ('route_q','route_ids'):
                result['arm.'+arm+'.'+key]=support[key]
        return result

    def snapshot(self,stage):
        self.position['watch_stage']=stage
        tensors=self.tensors()
        if not tensors:return
        groups={}
        for key,value in tensors.items():groups.setdefault(id(value),[]).append(key)
        if self.mode=='passive':
            if stage=='supports_after':
                self.journal.write(json.dumps(dict(position=dict(self.position),
                    alias_groups=[dict(keys=keys,metadata=metadata(tensors[keys[0]])) for keys in groups.values()]))+'\n')
                self.journal.flush()
            return
        current={}; changes=[]; records=[]
        for keys in groups.values():
            tensor=tensors[keys[0]]
            cpu=tensor.detach().to(device='cpu',copy=True).contiguous()
            # Compare each aliased CPU baseline once. Reuse statistics only
            # after byte equality, avoiding repeated large CPU reductions.
            comparisons={}; changed=[]
            for key in keys:
                old=self.baseline.get(key)
                if old is not None:
                    if id(old) not in comparisons:
                        comparisons[id(old)]=(old.dtype==cpu.dtype and old.shape==cpu.shape
                            and self.torch.equal(old.view(self.torch.uint8),cpu.view(self.torch.uint8)))
                    if not comparisons[id(old)]:changed.append(key)
            cached=next((self.baseline_stats[key] for key in keys if key in self.baseline_stats),None)
            stats=cached if cached is not None and not changed else describe(self.torch,cpu)
            records.append(dict(keys=keys,metadata=metadata(tensor),statistics=stats))
            for key in keys:
                current[key]=cpu
                if key not in self.baseline:
                    self.baseline[key]=cpu;self.baseline_stats[key]=stats
            changes.extend(changed)
        self.snapshots+=1
        self.latest_cpu=current
        self.latest=dict(position=dict(self.position),groups=records,changed_keys=changes)
        self.journal.write(json.dumps(self.latest,allow_nan=False)+'\n');self.journal.flush()
        if changes:
            self.changes+=1
            # Only previously owned CPU tensors are serialized.
            self.torch.save(dict(record=self.latest,before=self.baseline,after=current),self.output/'route_change_cpu.pt')
            write_json(self.output/'route_change.json',self.latest)
            raise ValueError('Frozen routing tensors changed at '+stage+': '+','.join(changes))

    def record(self,exc):
        if self.first_exception is not None:return
        self.first_exception=exc
        record=dict(error=type(exc).__name__+': '+str(exc),position=dict(self.position),last_snapshot=self.latest,
            traceback=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)),
            new_tensor_reads_after_failure=False,route_values_modified=False,mode=self.mode)
        try:
            write_json(self.output/'watch_failure.json',record)
            if self.baseline:
                self.torch.save(dict(record=record,before=self.baseline,last=self.latest_cpu),self.output/'watch_failure_cpu.pt')
        except BaseException as error:self.evidence_error=type(error).__name__+': '+str(error)

    def finish_image(self):
        if self.example is None or self.finished:return
        self.snapshot('image_complete')
        if self.mode=='watched' and self.example.image==self.target:
            self.torch.save(dict(image=self.target,baseline=self.baseline,last=self.latest_cpu,
                last_snapshot=self.latest),self.output/'target_routes_cpu.pt')
        self.images_completed+=1;self.finished=True

    def call(self,stage,function,*args,**kwargs):
        try:
            self.snapshot(stage+'_before')
            self.position['watch_stage']=stage+'_call'
            value=function(*args,**kwargs)
            self.snapshot(stage+'_after')
            return value
        except BaseException as exc:self.record(exc);raise

    def bind(self,obj,name,value):
        self.saved.append((obj,name,name in vars(obj),vars(obj).get(name)))
        setattr(obj,name,value)

    def __enter__(self):
        self.journal=(self.output/'route_snapshots.jsonl').open('x')
        prepare,make,forward,validate=self.runtime.prepare,self.factory.prepare,self.factory.forward,self.evaluator._validate_probability
        def extract(batch):
            try:
                self.finish_image()
                self.example=None;self.supports=None;self.baseline={};self.baseline_stats={};self.latest_cpu={};self.latest=None;self.finished=False
                index=self.position.get('index',-1)+1
                self.position['canonical_index']=self.canonical_index[self.names[index]] if index<len(self.names) else None
                self.position['watch_stage']='extract_call'
                example=prepare(batch);self.example=example
                self.snapshot('extract_after')
                return example
            except BaseException as exc:self.record(exc);raise
        def supports(example,arms):
            try:
                self.snapshot('supports_before');self.position['watch_stage']='supports_call'
                value=make(example,arms);self.supports=value
                self.snapshot('supports_after');return value
            except BaseException as exc:self.record(exc);raise
        def model(head,example,arm,support):
            self.position['active_arm']=arm
            return self.call('forward',forward,head,example,arm,support)
        def probability(prob,target,path):
            self.position['active_arm']=path
            return self.call('probability',validate,prob,target,path)
        try:
            for obj,name,value in ((self.runtime,'prepare',extract),(self.factory,'prepare',supports),
                                  (self.factory,'forward',model),(self.evaluator,'_validate_probability',probability)):
                self.bind(obj,name,value)
        except BaseException:
            self.restore();self.journal.close();raise
        return self

    def restore(self):
        for obj,name,existed,value in reversed(self.saved):
            if existed:setattr(obj,name,value)
            else:delattr(obj,name)
        self.saved.clear()

    def __exit__(self,kind,error,tb):
        try:
            if error is not None:self.record(error)
            else:
                try:self.finish_image()
                except BaseException as exc:self.record(exc);raise
        finally:
            self.restore();self.journal.close()
            try:
                write_json(self.output/'route_watch_summary.json',dict(mode=self.mode,snapshots=self.snapshots,
                    images_completed=self.images_completed,content_changes=self.changes,
                    failure=self.first_exception is not None,evidence_error=self.evidence_error,
                    route_values_modified=False,observation_synchronizes=self.mode=='watched',
                    exact_historical_replay=False,historical_failure_resolved=False,bindings_restored=True))
            except OSError as error:
                self.evidence_error=str(error)
                if self.first_exception is None:raise
        return False
