"""Original computations plus owned CPU snapshots at model operation boundaries."""
import json
from night_routes import TestRouteWatch
from night_config import TARGET
from evaluate_checkpoint import write_json


class CompactJournal:
    def __init__(self,stream):self.stream=stream
    def write(self,line):
        row=json.loads(line);p=row['position']
        if row['changed_keys'] or p.get('image')==TARGET:
            self.stream.write(line)
        elif p.get('watch_stage') in ('extract_after','image_complete'):
            self.stream.write(json.dumps(dict(position=p,changed_keys=[],
                route_sha256={','.join(g['keys']):g['statistics']['sha256'] for g in row['groups']}))+'\n')
    def flush(self):self.stream.flush()
    def close(self):self.stream.close()


class StageRoutes(TestRouteWatch):
    def __init__(self,*args,heads,**kwargs):
        super().__init__(*args,**kwargs)
        self.heads=heads;self.in_forward=False;self.boundary_calls=0;self.global_restores=[];self.unpool_index=0

    def snapshot(self,stage):
        previous=self.latest;previous_cpu=self.latest_cpu
        try:return super().snapshot(stage)
        except BaseException:
            # Only serialize already-owned CPU tensors; no new GPU reads here.
            if self.changes and not (self.output/'operator_window.json').exists():
                try:
                    write_json(self.output/'operator_window.json',dict(previous=previous,current=self.latest,
                        note='Observed interval; not proof that this operator caused the difference'))
                    self.torch.save(dict(before=previous_cpu,after=self.latest_cpu),self.output/'operator_window_cpu.pt')
                except BaseException as exc:self.evidence_error=type(exc).__name__+': '+str(exc)
            raise

    def call(self,stage,function,*args,**kwargs):
        previous=self.in_forward
        if stage=='forward':self.in_forward=True;self.unpool_index=0
        try:return super().call(stage,function,*args,**kwargs)
        finally:self.in_forward=previous

    def wrap(self,label,original):
        def invoke(*args,**kwargs):
            detailed=self.position.get('active_arm')=='d' or self.position.get('image')==TARGET
            if not self.in_forward or not detailed:return original(*args,**kwargs)
            self.boundary_calls+=1
            previous=self.position.get('operator')
            self.position['operator']=label
            self.position['operator_sequence']=self.boundary_calls
            try:
                self.snapshot(label+'_before')
                value=original(*args,**kwargs)
                self.snapshot(label+'_after')
                return value
            except BaseException as exc:self.record(exc);raise
            finally:
                if previous is None:self.position.pop('operator',None)
                else:self.position['operator']=previous
                self.position.pop('operator_sequence',None)
        return invoke

    def __enter__(self):
        super().__enter__();self.journal=CompactJournal(self.journal)
        try:
            namespaces={}
            for arm,head in self.heads.items():
                namespaces[id(type(head).forward.__globals__)]=type(head).forward.__globals__
                for method in ('_encode_slots','_read'):
                    self.bind(head,method,self.wrap(arm+'.'+method,getattr(head,method)))
                for name in ('region_encoder','pixel_projection','joint_head'):
                    module=getattr(head,name)
                    self.bind(module,'forward',self.wrap(arm+'.'+name,module.forward))
            for scope in namespaces.values():
                original=scope['sparse_unpool']
                self.global_restores.append((scope,'sparse_unpool',original))
                def unpool(values,q,ids,_original=original):
                    if self.in_forward:self.unpool_index+=1
                    kind={1:'parent',2:'slot',3:'memory'}.get(self.unpool_index,'extra')
                    return self.wrap('sparse_unpool_'+kind,_original)(values,q,ids)
                scope['sparse_unpool']=unpool
                functional=scope['F'];self.global_restores.append((scope,'F',functional))
                watch=self
                class Proxy:
                    def __init__(self,source):self.source=source
                    def __getattr__(self,name):return getattr(self.source,name)
                    def interpolate(self,*a,**kw):return watch.wrap('interpolate',self.source.interpolate)(*a,**kw)
                scope['F']=Proxy(functional)
            return self
        except BaseException:
            self.restore();self.journal.close();raise

    def restore(self):
        for scope,name,value in reversed(self.global_restores):scope[name]=value
        self.global_restores=[]
        super().restore()

    def __exit__(self,*args):
        try:return super().__exit__(*args)
        finally:
            write_json(self.output/'operator_observer_summary.json',dict(boundary_calls=self.boundary_calls,
                observations_synchronize=True,route_values_modified=False,bindings_restored=True,
                forward_boundary_arms=list(self.heads),detailed_arm_default='d',
                all_arm_detailed_target=TARGET,exact_historical_replay=False))
