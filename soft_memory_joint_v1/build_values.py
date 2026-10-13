"""Fresh Train-only child values for unchanged parent Q/keys/member identities.

This is a new value extraction, not a historical token replay claim. All parent
keys remain the original immutable V3 FP32 keys. No model weights are trained.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F

try:
    from .common import digest, source_hash, write_json
    from .slots import slot_labels
    from .server_runtime import (FrozenSlotRuntime, RULES, VALUES_FORMAT, VALUE_SCHEMA,
        LABEL_SCHEMA, json_hash, tree_hash, validate_values)
except ImportError:
    from common import digest, source_hash, write_json
    from slots import slot_labels
    from server_runtime import (FrozenSlotRuntime, RULES, VALUES_FORMAT, VALUE_SCHEMA,
        LABEL_SCHEMA, json_hash, tree_hash, validate_values)


def plan_sources(bank, train_names, val_names, limit=0):
    if type(limit) is not int or limit < 0:
        raise ValueError('limit_images must be a nonnegative integer')
    if limit not in (0,4):
        raise ValueError('This protocol uses all fixed sources or exactly four smoke sources')
    if not train_names or len(train_names) != len(set(train_names)) or set(train_names)&set(val_names):
        raise ValueError('Invalid Train/Val identities')
    image_ids = bank['images'].detach().cpu()
    events = bank['events'].detach().cpu()
    if image_ids.dtype != torch.long or bool(((image_ids<0)|(image_ids>=len(train_names))).any()):
        raise ValueError('Every fixed parent source must be an actual Train image')
    ids = sorted(image_ids.unique().tolist())
    if not ids: raise ValueError('No fixed parent sources')
    if limit:
        selected = []; seen = set()
        for iid in ids:
            eids = events[image_ids==iid].unique().tolist()
            if len(eids)!=1: raise ValueError('One source image has inconsistent event identities')
            if eids[0] not in seen:
                selected.append(iid); seen.add(eids[0])
            if len(selected)==limit: break
        if len(seen)<2: raise ValueError('Smoke needs at least two support source events')
        selected += [iid for iid in ids if iid not in selected][:limit-len(selected)]
        if len(selected)!=limit: raise ValueError('Smoke requires four fixed support source images')
        ids = selected
    return ids,[train_names[i] for i in ids]


def empty_values(parent):
    n = len(parent['t128'])
    identity = {name:parent[name].detach().cpu().clone()
                for name in ('member_ids','original_rows','images','events','regions')}
    identity['event_names'] = list(parent['event_names'])
    def table():
        return dict(slot_f=torch.zeros(n,2,256,dtype=torch.float32),slot_mass=torch.zeros(n,2,dtype=torch.float32),slot_fraction=torch.zeros(n,2,dtype=torch.float32),
                    slot_labels=torch.zeros(n,2,4,dtype=torch.float32),label_mass=torch.zeros(n,2,dtype=torch.float32),slot_valid=torch.zeros(n,2,dtype=torch.bool))
    return dict(format=VALUES_FORMAT,value_schema_sha256=json_hash(VALUE_SCHEMA),
                label_schema_sha256=json_hash(LABEL_SCHEMA),identity=identity,
                written=torch.zeros(n,dtype=torch.bool),rules={rule:table() for rule in RULES})


@torch.no_grad()
def label_reference(target, route_q, route_ids, shape, count, labels, label_mass):
    """Independent CPU FP64 reference, not slots.py's FP32 grouped helper."""
    if target.ndim != 3 or target.shape[0] != 1 or labels.shape != (1,count//2,2,4) or label_mass.shape != (1,count//2,2):
        raise ValueError('Reference requires one image and matching child labels')
    gt = target.detach().cpu().long()
    valid = gt != 255
    if bool((valid & ((gt<0)|(gt>3))).any()): raise ValueError('Invalid target in label reference')
    onehot = F.one_hot(torch.where(valid,gt,0),4).permute(0,3,1,2).double()*valid[:,None]
    fractions = F.interpolate(onehot,size=tuple(shape),mode='area')[0].flatten(1).T.contiguous()
    q = route_q.detach().cpu().double()
    ids = route_ids.detach().cpu()
    if ids.ndim == 3:
        if ids.shape[0]!=1: raise ValueError('Reference requires one route image')
        ids = ids[0]
    if ids.dtype!=torch.long or ids.shape!=q.shape[1:] or bool(((ids<0)|(ids>=count)).any()):
        raise ValueError('Reference route IDs invalid')
    counts = torch.zeros(count,4,dtype=torch.float64)
    for candidate in range(ids.shape[0]):
        counts.scatter_add_(0,ids[candidate,:,None].expand(-1,4),fractions*q[0,candidate,:,None])
    mass = counts.sum(-1)
    distribution = counts / torch.where(mass>0,mass,1)[:,None]
    actual_labels = labels.detach().cpu().double().reshape(count,4)
    actual_mass = label_mass.detach().cpu().double().reshape(count)
    def compare(a,b):
        finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        difference = (a-b).abs()
        outside = difference > (2e-5 + 2e-5*b.abs())
        return dict(passed=finite and not bool(outside.any()),max_abs=float(difference.max()),
                    outside_tolerance=int(outside.sum()),finite=finite)
    reports = dict(distribution=compare(actual_labels,distribution),label_mass=compare(actual_mass,mass),
                   weighted_counts=compare(actual_labels*actual_mass[:,None],counts))
    return dict(passed=all(r['passed'] for r in reports.values()),reference_dtype='torch.float64',
                reference_device='cpu',implementation=LABEL_SCHEMA['reference'],
                routes_checked=ids.numel(),child_slots_checked=count,atol=2e-5,rtol=2e-5,**reports)


def _store(value, parent, image_id, data):
    destination = (value['identity']['images']==image_id).nonzero().flatten()
    if not len(destination) or bool(value['written'][destination].any()):
        raise ValueError('Duplicate or missing child-value source')
    region_ids = value['identity']['regions'][destination]
    count = data['result']['count']
    if region_ids.dtype!=torch.long or bool(((region_ids<0)|(region_ids>=count)).any()):
        raise ValueError('Fixed parent region identity outside current Q')
    selected = region_ids.to(data['feature'].device)
    if not bool(data['result']['valid_regions'][0,selected].all()):
        raise ValueError('An original parent is not valid in fresh Q')
    reports = {}
    for rule in RULES:
        slot = data['slots'][rule]
        labels,mass = slot_labels(data['target'],slot['route_q'],slot['route_ids'],
                                 tuple(data['result']['shape']),slot['slot_count'])
        report = label_reference(data['target'],slot['route_q'],slot['route_ids'],
                                 data['result']['shape'],slot['slot_count'],labels,mass)
        if report['passed'] is not True:
            raise ValueError('Independent FP64 child-label reference failed: '+rule+' '+str(report))
        record = value['rules'][rule]
        for name in ('slot_f','slot_mass','slot_fraction'):
            record[name][destination] = slot[name][0,selected].detach().cpu()
        record['slot_labels'][destination] = labels[0,selected].detach().cpu()
        record['label_mass'][destination] = mass[0,selected].detach().cpu()
        record['slot_valid'][destination] = (slot['slot_valid'][0,selected] & (mass[0,selected]>0)).detach().cpu()
        reports[rule] = report
    value['written'][destination] = True
    return destination,reports


def run_build(args, runtime_factory=None):
    output = Path(args.output).expanduser().resolve()
    if output.exists(): raise FileExistsError('New child-values directory required: '+str(output))
    output.mkdir(parents=True)
    limit = int(getattr(args,'limit_images',0))
    smoke = limit>0
    started = time.perf_counter(); code_before = source_hash()
    summary = dict(format=VALUES_FORMAT,complete=False,execution_complete=False,smoke=smoke,
        phase='load',image_count=0,training=False,train_only=True,test_accessed=False,
        full_validation_evidence=False,historical_reproduction_claim=False)
    write_json(output/'summary.json',summary)
    factory = FrozenSlotRuntime if runtime_factory is None else runtime_factory
    runtime = None
    try:
        runtime = factory(args,output)
        with runtime:
            parent = runtime.bank
            ids,names = plan_sources(parent,runtime.train_names,runtime.val_names,limit)
            value = empty_values(parent)
            info = runtime.protocol_info
            manifest = dict(**summary,source_protocol_sha256=info['source_protocol_sha256'],
                paired_bank_sha256=info['paired_bank_sha256'],rebuild_protocol_sha256=info['rebuild_protocol_sha256'],
                membership_sha256=info['membership_sha256'],source_protocol=info['source_protocol'],
                value_schema=VALUE_SCHEMA,label_schema=LABEL_SCHEMA,value_schema_sha256=json_hash(VALUE_SCHEMA),
                label_schema_sha256=json_hash(LABEL_SCHEMA),code_sha256=code_before,
                support_count=len(parent['t128']),expected_source_images=len(names),source_images=names,
                source_train_ids=ids,query_key_unchanged=True,parent_membership_unchanged=True,
                selection_rule='fixed V3 parent membership; no child GT selects or filters parents',
                value_rule='fresh same-forward child F256; original fixed parent T128 remains Reader parent context',
                no_bitwise_parent_replay_claim=True,forward_count=0)
            write_json(output/'manifest.json',manifest)
            with (output/'images.jsonl').open('x',encoding='utf-8') as journal:
                for index,batch in enumerate(runtime.source_batches(names)):
                    if index>=len(names): raise ValueError('Unexpected extra Train source batch')
                    before = runtime.prepare_count
                    data = runtime.extract(batch)
                    if data['name']!=names[index] or runtime.prepare_count!=before+1:
                        raise ValueError('Source order or single-forward contract changed')
                    destination,references = _store(value,parent,ids[index],data)
                    frame = dict(index=index,image=data['name'],event=data['event'],train_image_id=ids[index],
                        member_ids=value['identity']['member_ids'][destination].tolist(),region_ids=value['identity']['regions'][destination].tolist(),
                        source_forward_count=1,native_a1=data['native_check'],q_check=data['pair']['q_check'],
                        label_reference=references,q_sha256=tree_hash(data['result']['q']),ids_sha256=tree_hash(data['result']['ids']),
                        building_sha256=tree_hash(data['building']),
                        slot_sha256={rule:tree_hash({k:v[destination] for k,v in value['rules'][rule].items()}) for rule in RULES})
                    journal.write(json.dumps(frame,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
                    summary.update(phase='Train_child_values',image_count=index+1,image=data['name'])
                    write_json(output/'summary.json',summary)
                    if (index+1)%25==0 or index+1==len(names):
                        print(f'Fresh paired child values {index+1}/{len(names)}',flush=True)
            if summary['image_count']!=len(names): raise ValueError('Incomplete Train source iteration')
            expected_written = torch.zeros_like(value['written'])
            for iid in ids: expected_written |= value['identity']['images']==iid
            if not torch.equal(value['written'],expected_written): raise ValueError('Member written mask differs from fixed source plan')
            validation = validate_values(value,parent,smoke=smoke)
            immutable = runtime.verify()
            if not immutable or any(v is not True for v in immutable.values()):
                raise RuntimeError('Frozen build input integrity failed: '+str(immutable))
            with (output/'slot_bank.pt').open('xb') as stream: torch.save(value,stream)
            reopened = torch.load(output/'slot_bank.pt',map_location='cpu',weights_only=False)
            validate_values(reopened,parent,smoke=smoke)
            if tree_hash(reopened)!=tree_hash(value): raise ValueError('Saved child values changed after reopening')
            write_json(output/'input_fingerprints.json',{str(p):sha for p,sha in runtime.input_ledger.items()})
            manifest.update(validation=validation,source_integrity=immutable,source_integrity_passed=True,
                label_reference_passed=True,roundtrip_exact=True,stored_parent_count=int(value['written'].sum()),
                forward_count=summary['image_count'],image_count=summary['image_count'],
                slot_counts={rule:dict(valid_child_count=int(value['rules'][rule]['slot_valid'].sum()),
                    feature_positive_child_count=int((value['rules'][rule]['slot_mass']>0).sum()),
                    label_positive_child_count=int((value['rules'][rule]['label_mass']>0).sum())) for rule in RULES})
        guard = runtime.guard_report or {}
        if guard.get('restored') is not True or guard.get('failures') or guard.get('restore_conflicts'):
            raise RuntimeError('Native pooling guard did not finish cleanly: '+str(guard))
        if source_hash()!=code_before: raise RuntimeError('Child-value tool code changed during build')
        manifest.update(complete=True,execution_complete=True,phase='complete_smoke' if smoke else 'complete',
            pool_guard_passed=True,pool_guard=guard,
            file_sha256={name:digest(output/name) for name in ('slot_bank.pt','images.jsonl','input_fingerprints.json')},
            elapsed_seconds=time.perf_counter()-started)
        write_json(output/'manifest.json',manifest)
        summary.update(complete=True,execution_complete=True,phase=manifest['phase'],
            stored_parent_count=manifest['stored_parent_count'],support_count=manifest['support_count'],
            source_integrity_passed=True,label_reference_passed=True,roundtrip_exact=True,pool_guard_passed=True,
            manifest_sha256=digest(output/'manifest.json'),slot_bank_sha256=digest(output/'slot_bank.pt'),
            elapsed_seconds=manifest['elapsed_seconds'])
        write_json(output/'summary.json',summary)
        print('CHILD VALUES COMPLETE. No training. Train only. Parent keys and membership unchanged.',flush=True)
        return summary
    except BaseException as exc:
        summary.update(complete=False,execution_complete=False,error=type(exc).__name__+': '+str(exc),
                       elapsed_seconds=time.perf_counter()-started)
        if runtime is not None: summary['pool_guard']=getattr(runtime,'guard_report',None)
        write_json(output/'summary.json',summary)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
