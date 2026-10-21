"""Observe frozen parent-Q across an actual paired update; never repair its values.

Blocking GPU-to-CPU snapshots deliberately change timing. A successful monitored
run is not proof that the historical asynchronous fault has been fixed.
"""
import contextlib
import functools
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

from evaluate_checkpoint import write_json
from passive_routes import PassiveRoutes


class ParentQObserver:
    def __init__(self, torch, example, heads, optimizers, factory, runtime, output, step, journal, counters):
        self.torch,self.example,self.heads,self.optimizers = torch,example,heads,optimizers
        self.factory,self.runtime,self.output = factory,runtime,Path(output)
        self.step,self.journal,self.counters = step,journal,counters
        self.position=dict(step=step,image=str(example.image),active_arm=None,stage='before_update')
        self.baseline=None
        self.latest=None
        self.first_exception=None
        self.entered=False
        self.hooks=[]
        self.saved=[]

    def metadata(self,tensor):
        try: version=tensor._version
        except RuntimeError: version=None
        return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device),
                    storage_ptr=tensor.untyped_storage().data_ptr(),offset=tensor.storage_offset(),
                    stride=list(tensor.stride()),version=version,requires_grad=tensor.requires_grad)

    def fingerprint(self,value):
        return hashlib.sha256(value.reshape(-1).view(self.torch.uint8).numpy().tobytes()).hexdigest()

    def describe(self,values):
        q=values['q']; ids=values['ids']
        sums=q.to(self.torch.float64).sum(1)
        def finite(value):
            value=float(value)
            return value if math.isfinite(value) else None
        return dict(q_sha256=self.fingerprint(q),ids_sha256=self.fingerprint(ids),
            q_shape=list(q.shape),ids_shape=list(ids.shape),q_min=finite(q.min()),q_max=finite(q.max()),
            q_negative_count=int((q<0).sum()),q_nonfinite_count=int((~self.torch.isfinite(q)).sum()),
            cpu_float64_max_sum_error=finite((sums-1).abs().max()),
            ids_min=int(ids.min()),ids_max=int(ids.max()))

    def snapshot(self,stage):
        self.position['stage']=stage
        values={k:self.example.common[k].detach().to(device='cpu',copy=True).contiguous() for k in ('q','ids')}
        self.counters['snapshots']+=1
        details=self.describe(values)
        metadata={k:self.metadata(self.example.common[k]) for k in ('q','ids')}
        changed=self.baseline is not None and any(
            values[k].shape!=self.baseline[k].shape or values[k].dtype!=self.baseline[k].dtype
            or not self.torch.equal(values[k].view(self.torch.uint8),
                                    self.baseline[k].view(self.torch.uint8)) for k in ('q','ids'))
        record=dict(**self.position,**details,metadata=metadata,content_changed=bool(changed),
                    comparison='same image frozen Q and IDs before support preparation')
        self.journal.write(json.dumps(record,allow_nan=False)+'\n'); self.journal.flush()
        self.latest=record
        if self.baseline is None:
            self.baseline=values
            # Only metadata: reject a forbidden alias with a writable parameter
            # or optimizer moment; a read-only view of frontend inputs is legal.
            input_storages={(self.example.common[k].device.type,self.example.common[k].device.index,
                             self.example.common[k].untyped_storage().data_ptr()) for k in ('q','ids')}
            for arm,head in self.heads.items():
                tensors=list(head.parameters())
                for state in self.optimizers[arm].state.values():
                    tensors.extend(v for v in state.values() if isinstance(v,self.torch.Tensor))
                for value in tensors:
                    if value.numel() and (value.device.type,value.device.index,value.untyped_storage().data_ptr()) in input_storages:
                        raise ValueError('Frozen Q/IDs alias writable head/optimizer storage: '+arm)
        if changed:
            self.counters['content_changes']+=1
            self.torch.save(dict(position=dict(self.position),before=self.baseline,after=values),
                            self.output/'parent_q_change_cpu.pt')
            write_json(self.output/'parent_q_change.json',record)
            raise ValueError('Frozen parent Q/IDs changed at '+stage+'; paired update rejected')

    def call(self,stage,function,*args,**kwargs):
        try:
            self.snapshot(stage+'_before')
            self.position['stage']=stage+'_call'
            result=function(*args,**kwargs)
            self.snapshot(stage+'_after')
            return result
        except BaseException as exc:
            self.record(exc)
            raise

    def record(self,exc):
        if self.first_exception is not None:return
        self.first_exception=exc
        self.counters['failures']+=1
        record=dict(position=dict(self.position),error=type(exc).__name__+': '+str(exc),
            last_completed_snapshot=self.latest,route_values_modified=False,
            observation_synchronizes=True,historical_failure_resolved=False,
            traceback=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))
        try:
            write_json(self.output/'parent_q_failure.json',record)
            # Only previously owned CPU tensors are saved after any exception.
            # PassiveRoutes separately preserves the original failing reduction.
            if self.baseline is not None:
                self.torch.save(dict(position=dict(self.position),tensors=self.baseline),self.output/'parent_q_before_cpu.pt')
        except BaseException as error:
            self.counters['evidence_error']=type(error).__name__+': '+str(error)

    def replace(self,obj,key,value):
        self.saved.append((obj,key,key in vars(obj),vars(obj).get(key)))
        setattr(obj,key,value)

    def __enter__(self):
        if self.entered:raise RuntimeError('Q observer is one-shot')
        self.entered=True
        try:
            self.snapshot('before_update')
            prepare,forward,loss=self.factory.prepare,self.factory.forward,self.runtime.loss
            def supports(example,arms):
                if example is not self.example:raise ValueError('Observed example changed')
                return self.call('support_prepare',prepare,example,arms)
            def model_forward(head,example,arm,support):
                self.position['active_arm']=arm
                if example is not self.example or head is not self.heads[arm]:raise ValueError('Observed arm/example changed')
                return self.call('forward',forward,head,example,arm,support)
            def objective(*args,**kwargs):
                result=self.call('loss',loss,*args,**kwargs)
                if result.requires_grad:
                    def backward(gradient):
                        try:self.snapshot('backward_begin')
                        except BaseException as exc:self.record(exc); raise
                        return None
                    self.hooks.append(result.register_hook(backward))
                return result
            self.replace(self.factory,'prepare',supports); self.replace(self.factory,'forward',model_forward)
            self.replace(self.runtime,'loss',objective)
            for arm,opt in self.optimizers.items():
                def pre(optimizer,args,kwargs,arm=arm):
                    self.position['active_arm']=arm
                    try:self.snapshot('optimizer_before_after_backward')
                    except BaseException as exc:self.record(exc); raise
                def post(optimizer,args,kwargs,arm=arm):
                    self.position['active_arm']=arm
                    try:self.snapshot('optimizer_after')
                    except BaseException as exc:self.record(exc); raise
                self.hooks.append(opt.register_step_pre_hook(pre))
                self.hooks.append(opt.register_step_post_hook(post))
            return self
        except BaseException as exc:
            self.record(exc); self.restore(); raise

    def restore(self):
        for hook in reversed(self.hooks):hook.remove()
        self.hooks.clear()
        for obj,key,existed,value in reversed(self.saved):
            if existed:setattr(obj,key,value)
            else:delattr(obj,key)
        self.saved.clear()

    def __exit__(self,kind,error,tb):
        try:
            if error is not None:self.record(error)
            else:
                try:self.snapshot('paired_update_returned')
                except BaseException as exc:self.record(exc); raise
        finally:self.restore()
        return False


@contextlib.contextmanager
def observe_bindings(context,output):
    """Scope both parent snapshots and unchanged route-predicate capture."""
    output=Path(output)
    bridge=context['bridge']; named,load_runtime=bridge.named,bridge.load_runtime
    counters=dict(paired_updates_observed=0,snapshots=0,content_changes=0,failures=0,evidence_error=None)
    position={}
    with contextlib.ExitStack() as stack:
        journal=stack.enter_context((output/'parent_q_observations.jsonl').open('x'))
        print('QWATCH: frozen parent Q/IDs snapshots enabled; original route predicate retained; '
              'observations synchronize and change timing',flush=True)
        def load(source):
            bindings=load_runtime(source)
            stack.enter_context(PassiveRoutes(bindings.torch,
                bindings.original_train.JointRegionMemoryClassifier,output,position))
            return bindings
        def module(path,label):
            result=named(path,label)
            if Path(path).name=='ablation_runner.py':
                original=result.update_arms
                @functools.wraps(original)
                def update(heads,optimizers,example,runtime,factory,state,**kwargs):
                    import torch
                    monitor=ParentQObserver(torch,example,heads,optimizers,factory,runtime,output,
                                            kwargs['step'],journal,counters)
                    # Route capture uses the same live stage mapping.
                    monitor.position=position
                    position.clear(); position.update(step=kwargs['step'],image=str(example.image),active_arm=None,stage='before_update')
                    with monitor:
                        rows=original(heads,optimizers,example,runtime,factory,state,**kwargs)
                    counters['paired_updates_observed']+=1
                    return rows
                result.update_arms=update
                stack.callback(setattr,result,'update_arms',original)
            return result
        bridge.named,bridge.load_runtime=module,load
        try:yield counters
        finally:
            bridge.named,bridge.load_runtime=named,load_runtime
            write_json(output/'parent_q_observer_summary.json',dict(**counters,
                route_values_modified=False,observation_synchronizes=True,exact_historical_replay=False,
                historical_failure_resolved=False))
