"""Cooperative source/member adapter for the unchanged slot and Reader runtime."""
from pathlib import Path
from io_utils import digest,old_modules
from new_source import load as load_source
from new_members import load as load_bank

_,_,base,server,_,_=old_modules()
import torch
import sys

class NewBaseRuntime(base.Runtime):
    def __enter__(self):
        if self._entered:raise RuntimeError('Runtime cannot be entered twice')
        self._entered=True
        try:
            self.deps=base._dependencies(Path(self.args.repo).resolve())
            if self.args.workers<0 or self.args.topk!=64 or self.args.chunk_size<1:
                raise ValueError('Require Top64 and legal loader/chunk settings')
            protocol=Path(self.args.protocol).resolve()
            self.bundle=load_source(protocol,self.output)
            if self.bundle['source_protocol_sha256']!=digest(protocol):raise ValueError('Source protocol changed')
            self.device=torch.device(self.bundle['device'])
            if self.device!=torch.device(self.args.device):raise ValueError('Source/worker device differs')
            front=self.bundle['front']
            if any(m.training for m in front.modules()) or any(p.requires_grad for p in front.parameters()):
                raise ValueError('New Q source must be frozen')
            self.front_before=base._model_fingerprint(front)
            self.source_code_before=base.source_hash()
            self.execution_before=base._execution_settings()
            self.input_ledger=base._source_ledger(self.bundle)
            self.input_ledger[protocol]=digest(protocol)
            self.train_names=list(self.bundle['train_names'])
            self.val_names=list(self.bundle['val_names'])
            if len(self.train_names)!=2918 or len(self.val_names)!=450 or set(self.train_names)&set(self.val_names):
                raise ValueError('Require disjoint Train2918/Val450')
            self._train_ids={n:i for i,n in enumerate(self.train_names)}
            self._val_set=set(self.val_names)
            self.key_root=Path(self.args.bank_root).resolve()
            allow_smoke=base._allow_smoke_parent(self.args)
            self.bank,manifest=load_bank(self.key_root,self.bundle,allow_smoke=allow_smoke)
            self.extra_ledger=base._bank_ledger(self.key_root)
            self.input_ledger.update(self.extra_ledger)
            selected_files=manifest.get('source_image_fingerprints')
            if not isinstance(selected_files,dict) or not selected_files:
                raise ValueError('Selected Train image ledger missing')
            for raw,sha in selected_files.items():
                p=Path(raw).resolve()
                if digest(p)!=sha:raise ValueError('Selected Train image changed: '+str(p))
                self.input_ledger[p]=sha
            count=base._validate_parent_bank(self.bank,self.train_names,self.device)
            if count!=manifest['support_count']:raise ValueError('Parent member count differs')
            self.bank_before=base._bank_fingerprint(self.bank)
            self.protocol_info=dict(source_protocol=str(protocol),source_protocol_sha256=digest(protocol),
                source_provenance=self.bundle['source_provenance'],
                source_checkpoint_sha256=self.bundle['source_checkpoint_sha256'],
                original_front_sha256=self.bundle['original_front_sha256'],a1_sha256=self.bundle['a1_sha256'],
                execution_settings=self.execution_before,runtime_source_sha256=self.source_code_before,
                paired_bank=str(self.key_root/'paired_bank.pt'),paired_bank_sha256=manifest['paired_bank_sha256'],
                rebuild_protocol_sha256=manifest['rebuild_protocol_sha256'],membership_sha256=manifest['membership_sha256'],
                support_count=count,support_source_image_count=manifest['image_count'],
                parent_bank_smoke=manifest['smoke'],member_source='new_train_only_event_class_bounded',
                old_v3_member_identity_reused=False,train_count=2918,val_count=450,
                historical_e0_reproduction_claim=False,old_three_arm_baseline='not_comparable_new_members_and_numeric_policy',
                key_storage='torch.float32',retrieval='global_cosine_exclude_same_image_and_event',
                r0_rule='nearest active Q-centroids including self; squared unit-grid distance, stable region-ID ties',
                r0_include_self=True,r0_labels='same-forward native A1 grid softmax Q-weighted mean; no query GT',
                train_query_rule='entire Train list, frozen full-image loader; no crop/augmentation',test_accessed=False)
            d=self.deps
            self.capture=d.FrozenFrontCapture(front)
            self.native_capture=d.NativeA1LogitsCapture(front)
            self.guard=d.PoolGuard(self.output/'pool_debug',mode='native',capture='failure',check_reference=False)
            self.guard.__enter__()
            return self
        except BaseException:
            self.close(*sys.exc_info());raise

class NewFrozenSlotRuntime(server.FrozenSlotRuntime,NewBaseRuntime):
    pass

class NewRuntime(server.Runtime,NewBaseRuntime):
    pass

assert NewBaseRuntime in NewFrozenSlotRuntime.__mro__
assert NewBaseRuntime in NewRuntime.__mro__
