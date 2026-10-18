#!/usr/bin/env python3
"""Read-only slot/Memory diagnostics for a NEW completed anchor-study run.

This adapter imports the released statistics and original model/runtime. It does
not create an old ABCD manifest, replace a B head with a D head, or train a model.
The old PixelAccumulator's private b/d keys are translated to explicit common500
or control/anchor names before any output is written.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

FORMAT = 'BRIGHT_ANCHOR_SLOT_MEMORY_AUDIT_V1'
STUDY_FORMAT = 'BRIGHT_JOINT_ANCHOR_STUDY_V1'
STUDY_MANIFEST = 'e149b690b4274ee0ecc7947831be7545eb8a7199786a137fff56547fb84a8dd2'
AUDIT_MANIFEST = '7d310fa3c0c75f50f25ee032af7d17e00e32690af6e9b522f03eb0136544c977'
DEFAULT_NEW = '/root/shared-nvme/BRIGHT-master/New'


from chain_io import digest, read, write, package_files


def load_named(path, name):
    path = Path(path).resolve()
    tag = '_bright_chain_'+name+'_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if tag in sys.modules:
        if Path(sys.modules[tag].__file__).resolve() != path: raise ValueError('Import collision')
        return sys.modules[tag]
    spec = importlib.util.spec_from_file_location(tag, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(tag, None)
        raise
    return module


def validate_run(root, step):
    """Only completed, non-smoke fresh D runs; never invent historical evidence."""
    root = Path(root).expanduser().resolve()
    if step not in (500, 1500): raise ValueError('step must be 500 or 1500')
    train = root/'train'
    workflow, summary, manifest = [read(p) for p in (root/'workflow_state.json',train/'summary.json',train/'manifest.json')]
    for record in (workflow, summary):
        if (record.get('format') != STUDY_FORMAT or record.get('complete') is not True
                or record.get('execution_complete') is not True or record.get('training_complete') is not True
                or record.get('full_validation_evidence') is not True or record.get('test_accessed') is not False):
            raise ValueError('Anchor study must finish training, Val450 and final integrity before diagnostics')
    if (summary.get('smoke') is not False or summary.get('final_integrity_pending') is not False
            or summary.get('source_mode') != 'fresh' or summary.get('arm') != 'd'
            or summary.get('endpoint_step') != 1500 or summary.get('branch_step') != 500
            or manifest.get('format') != STUDY_FORMAT or manifest.get('source_mode') != 'fresh'
            or manifest.get('branch_step') != 500 or manifest.get('primary_endpoint') != 1500
            or manifest.get('test_accessed') is not False
            or manifest.get('arguments',{}).get('arm') != 'd'):
        raise ValueError('Expected a fresh common500 -> D control/anchor1500 study')
    train_names, val_names = manifest.get('train_names'), manifest.get('val_names')
    if (not isinstance(train_names,list) or not isinstance(val_names,list)
            or len(train_names)!=2918 or len(set(train_names))!=2918
            or len(val_names)!=450 or len(set(val_names))!=450 or set(train_names)&set(val_names)):
        raise ValueError('Expected disjoint Train2918 and Val450 identities')
    checkpoint = train/'checkpoints'/('checkpoint_%06d.pt'%step)
    sidecar = checkpoint.with_suffix('.pt.json')
    meta = read(sidecar)
    expected_heads = {'control'} if step==500 else {'control','anchor'}
    if meta.get('step') != step or set(meta.get('head_hashes',{})) != expected_heads or digest(checkpoint) != meta.get('sha256'):
        raise ValueError('New study checkpoint/sidecar mismatch')
    reference_path = train/('initial_val' if step==500 else 'val_step_001500')/'result.json'
    reference = read(reference_path)
    checksum = Path(str(reference_path)+'.sha256')
    if checksum.read_text().split()[0] != digest(reference_path): raise ValueError('Val report checksum mismatch')
    if (reference.get('step')!=step or reference.get('split')!='val' or reference.get('image_count')!=450
            or reference.get('whole_val450_processed') is not True or reference.get('complete') is not True
            or reference.get('execution_complete') is not True or reference.get('final_integrity_pending') is not False
            or reference.get('full_validation_evidence') is not True or reference.get('test_accessed') is not False
            or not {'grid_probq','control','anchor'} <= set(reference.get('metrics',{}))):
        raise ValueError('Selected checkpoint has no final-checked full Val450 reference')
    if manifest.get('release_sha256') != STUDY_MANIFEST or summary.get('release_sha256') != STUDY_MANIFEST:
        raise ValueError('Training source differs from released anchor study')
    ledger_path = train/'input_fingerprints.json'
    ledger = read(ledger_path)
    if not ledger: raise ValueError('Completed source input ledger is empty')
    artifacts = (root/'workflow_state.json',train/'summary.json',train/'manifest.json',checkpoint,sidecar,
                 reference_path,checksum,ledger_path)
    return dict(root=str(root),train=str(train),checkpoint=str(checkpoint),meta=meta,manifest=manifest,
                summary=summary,reference=reference,input_ledger=ledger,step=step,
                files={str(p):digest(p) for p in artifacts})


def verify_files(files):
    for path, sha in files.items():
        if not Path(path).is_file() or digest(path)!=sha: raise ValueError('Source changed: '+path)


def validate_payload(saved, static, state, torch):
    manifest, step = static['manifest'], static['step']
    if (saved.get('format')!=STUDY_FORMAT or saved.get('step')!=step or saved.get('source_mode')!='fresh'
            or saved.get('arm')!='d' or saved.get('test_accessed') is not False
            or saved.get('release_sha256')!=STUDY_MANIFEST
            or saved.get('runtime_config')!=manifest.get('runtime_config')
            or saved.get('protocol_info')!=manifest.get('protocol_info')
            or saved.get('schedule_horizon')!=5000 or saved.get('primary_endpoint')!=1500):
        raise ValueError('Checkpoint payload provenance mismatch')
    states = saved.get('heads',{})
    if set(states)!=set(static['meta']['head_hashes']): raise ValueError('Checkpoint head set mismatch')
    for name, tensors in states.items():
        if not tensors or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in tensors.values()):
            raise ValueError('Nonfinite/non-tensor head: '+name)
        if state.state_fingerprint(tensors)!=static['meta']['head_hashes'][name]:
            raise ValueError('Checkpoint tensor fingerprint mismatch: '+name)
    ledger = saved.get('input_file_sha256')
    if not isinstance(ledger,dict) or not ledger: raise ValueError('Checkpoint input ledger missing')
    for path,sha in ledger.items():
        if static['input_ledger'].get(path)!=sha: raise ValueError('Checkpoint/final input ledger mismatch')
    return True


def runtime_input_check(runtime, static):
    if runtime.protocol_info != static['manifest']['protocol_info']:
        raise ValueError('Runtime protocol differs from recorded source')
    if runtime.train_names!=static['manifest']['train_names'] or runtime.val_names!=static['manifest']['val_names']:
        raise ValueError('Runtime Train/Val identities changed')
    if not runtime.input_ledger: raise ValueError('Runtime input ledger missing')
    for path,sha in runtime.input_ledger.items():
        if static['input_ledger'].get(str(path))!=sha:
            raise ValueError('Runtime input differs from completed training: '+str(path))


class NamedPixels:
    """Use existing pixel/paired counting, translate private aliases at boundary."""
    def __init__(self, PixelAccumulator, step):
        self.raw = PixelAccumulator()
        self.step = step

    def add(self, target, predictions, masks):
        names = ('common500','common500') if self.step==500 else ('control','anchor')
        self.raw.add(target,dict(grid_probq=predictions['grid_probq'],b=predictions[names[0]],d=predictions[names[1]]),masks)

    def report(self):
        result = self.raw.report()
        for row in result.values():
            metric,paired = row['metrics'],row['paired']
            if self.step==500:
                row['metrics']={'grid_probq':metric['grid_probq'],'common500':metric['b']}
                row['paired']={'common500_vs_grid':paired['b_vs_grid']}
            else:
                row['metrics']={'grid_probq':metric['grid_probq'],'control':metric['b'],'anchor':metric['d']}
                row['paired']={'control_vs_grid':paired['b_vs_grid'],'anchor_vs_grid':paired['d_vs_grid'],
                               'anchor_vs_control':paired['d_vs_b']}
        return result


def replay_check(pixel, reference, pixel_module, step):
    """Translate existing helper's expected names in memory, never write fake B/D."""
    actual=pixel.raw.report()['all_labeled']['metrics']
    remapped={'metrics':{'grid_probq':reference['metrics']['grid_probq'],
                         'b':reference['metrics']['control'],'d':reference['metrics']['anchor']}}
    replay=pixel_module.check_replay(actual,remapped,tolerance_pp=.001)
    paths=replay.pop('paths')
    replay['paths']={'grid_probq':paths['grid_probq']}
    if step==500: replay['paths'].update(common500_vs_initial_control=paths['b'],common500_vs_initial_anchor=paths['d'])
    else: replay['paths'].update(control=paths['b'],anchor=paths['d'])
    return replay


def analyze(example, data, heads, memory_images, memory_events, chunk_size, modules):
    """Only orchestration: all slot, neighbor and trained-Reader math is reused."""
    torch=modules.torch
    with torch.no_grad():
        decoded={name:head(**example.common,**example.supports['d']) for name,head in heads.items()}
        for name,result in decoded.items():
            p=result['probabilities']
            if (not bool(torch.isfinite(p).all()) or bool((p<0).any())
                    or not torch.allclose(p.sum(1),torch.ones_like(p[:,0]),atol=2e-5,rtol=0)):
                raise ValueError('Invalid prediction: '+name)
        predictions={'grid_probq':example.common['base_probability_full'].argmax(1)[0].cpu().numpy()}
        predictions.update({name:r['probabilities'].argmax(1)[0].cpu().numpy() for name,r in decoded.items()})
        slots=modules.slot.slot_diagnostics(data['feature'],data['result']['q'],data['result']['ids'],
                                           data['building'],example.target,data['slots']['soft'])
        for cls,name in ((1,'intact'),(2,'damaged')):
            slots['weights']['role1_gt_'+name]=slots['query_label_mass'][:,1]*slots['query_labels'][:,1,cls]
        local,memory=example.supports['b'],example.supports['d']
        neighbors=modules.neighbor.neighbor_diagnostics(local['slot_f'][0],slots['query_labels'],local['slot_valid'][0],
            memory['support_ids'],memory['support_f'],memory['support_labels'],memory['support_valid'],
            memory_images,memory_events,local['support_ids'],local['support_labels'],local['support_valid'],chunk_size=chunk_size)
        rows=dict(slots['region_rows']);rows.update(neighbors['region_rows'])
        for name,head in heads.items():
            observed=modules.reader.reader_diagnostics(head,decoded[name],memory,chunk_size=chunk_size)
            modules.reader.add_posthoc_label_scores(observed,slots['query_labels'],slots['query_valid'])
            rows.update({'reader_'+name+'_'+key:value for key,value in observed.items()})
    return dict(slots=slots,rows=rows,predictions=predictions,neighbor_definitions=neighbors['definitions'])


def write_report(output, result):
    lines=['新配对训练：双槽与 Memory 内容诊断','',
           '本报告没有训练，也没有访问 Test。所有分类头使用同一 D 的历史案例支持。',
           'common500 是新共同起点；control/anchor 是新1500步两分支，不是旧 B/D 对照。',
           'local_pred 仅描述图内参考候选质量，没有运行一个重新训练的 B 分类头。',
           '区域/邻居统计冻结时应保持一致；Reader统计与像素预测可以变化。',
           'GT只进入事后分层统计，不参与特征、候选选择或模型输入。',
           '分层含建筑/背景边缘、建筑内部、低B真值建筑；不代表建筑实例边界。','']
    for key in ('complete','checkpoint_step','image_count','smoke','full_validation_evidence','final_integrity_pending'):
        lines.append(key+': '+str(result.get(key)))
    lines+=['','path | fg mIoU % | Damaged IoU % | precision % | recall %']
    for name,row in result['pixels']['all_labeled']['metrics'].items():
        values=(row['foreground_miou'],row['iou'][2],row['precision'][2],row['recall'][2])
        lines.append(name+' | '+' | '.join('NA' if v is None else f'{100*v:.4f}' for v in values))
    lines+=['','逐像素救回/误伤（背景、完好、受损、毁坏）：']
    for name,row in result['pixels']['all_labeled']['paired'].items():
        lines.append(f"{name}: 救回={row['rescue_by_gt_class']}; 误伤={row['harm_by_gt_class']}; 完好误改受损={row['intact_correct_to_damaged']}")
    lines+=['','双槽/候选/Reader重点统计；每项保留自己的有效分母；NA不等于0：']
    interesting=('feature_cosine','label_total_variation','slot0_ess','slot1_ess','both_slots_active',
                 'memory_unique_source_images','memory_unique_source_events','memory_damaged50_unique_images',
                 'memory_intact50_unique_images','memory_similar_intact50_fraction','memory_similar_damaged50_fraction',
                 'memory_cosine_vote_damaged','local_pred_cosine_vote_damaged')
    for name,fields in result['regions']['strata'].items():
        lines.append('['+name+']')
        for key,row in fields.items():
            if key in interesting or key.startswith('reader_') and key.endswith(('damaged_label_mass','intact_label_mass','background_role_attention','effective_slots','posthoc_label_TV')):
                lines.append(f"  {key}: mean={row.get('mean')}; n={row.get('n')}; weight_sum={row.get('weight_sum')}")
    lines+=['','Val重放：'+json.dumps(result['replay'],ensure_ascii=False),
            '有邻居、标签相似或Reader注意力都不单独证明Memory具有稳定增益。']
    (Path(output)/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (Path(output)/'regions_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(('stratum','metric','n','weight_sum','weighted_sum','mean'))
        for stratum,fields in result['regions']['strata'].items():
            for key,row in fields.items():writer.writerow((stratum,key,*(row.get(k) for k in ('n','weight_sum','weighted_sum','mean'))))


def run(args):
    output=Path(args.output).expanduser().resolve()
    if output.exists(): raise ValueError('Diagnostic output must be new: '+str(output))
    static=validate_run(args.study_root,args.step)
    study_files=package_files(args.study_dir,STUDY_MANIFEST)
    audit_files=package_files(args.audit_dir,AUDIT_MANIFEST)
    config=copy.deepcopy(static['manifest']['runtime_config'])
    for src in (static['root'],args.study_dir,args.audit_dir,args.tool_dir,args.helpers,config['bank_root'],config['values']):
        source=Path(src).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output): raise ValueError('Output overlaps source: '+str(source))
    output.mkdir(parents=True)
    state=dict(format=FORMAT,complete=False,execution_complete=False,full_validation_evidence=False,
               final_integrity_pending=True,training=False,test_accessed=False,primary_endpoint=False,
               checkpoint_step=args.step,smoke=args.smoke,image_count=0,phase='loading',error=None,
               source_mode='fresh_anchor_study',comparison='common500' if args.step==500 else 'control_vs_anchor',
               original_BD_replay_claim=False)
    write(output/'summary.json',state);started=time.monotonic()
    try:
        execution=static['manifest']['protocol_info'].get('execution_settings',{})
        if execution.get('cuda_launch_blocking') is None: os.environ.pop('CUDA_LAUNCH_BLOCKING',None)
        else: os.environ['CUDA_LAUNCH_BLOCKING']=str(execution['cuda_launch_blocking'])
        sys.dont_write_bytecode=True
        bridge=load_named(Path(args.study_dir)/'legacy.py','legacy')
        legacy=bridge.load_legacy(args.helpers)
        runtime_static=dict(start='fresh',tool=str(Path(args.tool_dir).resolve()),config=config)
        torch,common,train,original_run,Runtime=bridge.import_runtime(runtime_static,legacy)
        if not torch.cuda.is_available(): raise RuntimeError('Real diagnostic requires CUDA; CPU tests are synthetic')
        saved=torch.load(static['checkpoint'],map_location='cpu',weights_only=False)
        validate_payload(saved,static,legacy.state,torch)
        modules=SimpleNamespace(torch=torch,**{name:load_named(Path(args.audit_dir)/(filename+'.py'),name)
                    for name,filename in (('slot','slot_stats'),('neighbor','neighbor_stats'),('reader','reader_stats'),('pixel','pixel_stats'))})
        config.update(output=str(output),command='train',mode='train',smoke=False,baseline_policy='independent',
                      baseline_root=None,baseline_evidence=None,allow_smoke_values=False)
        runtime_args=SimpleNamespace(**config)
        class CaptureRuntime(Runtime):
            def extract(self,batch):
                data=super().extract(batch);self.audit_data=data;return data
        region=modules.slot.RegionAccumulator();pixel=NamedPixels(modules.pixel.PixelAccumulator,args.step)
        events={};seen=[]
        with CaptureRuntime(runtime_args,output) as runtime:
            train._formal_runtime(runtime,False);runtime_input_check(runtime,static)
            names=list(runtime.smoke_names()[1] if args.smoke else runtime.val_names)
            if args.smoke and (len(names)!=2 or len({runtime.deps.event(n) for n in names})!=2):
                raise ValueError('Smoke requires two Val images from distinct events')
            if any(n not in runtime.val_names for n in names): raise ValueError('Only Val images may be audited')
            write(output/'sample_names.json',dict(split='Val',names=names))
            heads={}
            for key,tensors in saved['heads'].items():
                name='common500' if args.step==500 else key
                head=train.JointRegionMemoryClassifier(chunk_size=config['chunk_size']).to(runtime.device)
                head.load_state_dict(tensors,strict=True);head.eval().requires_grad_(False);heads[name]=head
            before={n:legacy.state.state_fingerprint(h.state_dict()) for n,h in heads.items()}
            memory_images=[runtime.train_names[i] for i in runtime.bank['images'].cpu().tolist()]
            memory_events=[runtime.bank['event_names'][i] for i in runtime.bank['events'].cpu().tolist()]
            table=runtime.values['rules']['soft']
            inventory=modules.neighbor.inventory(table['slot_labels'],table['slot_valid'],memory_images,memory_events)
            coverage=train._blank_coverage()
            with (output/'images.jsonl').open('x',encoding='utf-8') as log:
                for index,batch in enumerate(runtime.batches(names,train=False,seed=config['seed'])):
                    with torch.no_grad():
                        example=runtime.prepare(batch);train._assert_example(example)
                        if index>=len(names) or example.image!=names[index]: raise ValueError('Val order differs')
                        train._merge_coverage(coverage,train._coverage(example))
                        measured=analyze(example,runtime.audit_data,heads,memory_images,memory_events,config['chunk_size'],modules)
                    seen.append(example.image);region.add(measured['rows'],measured['slots']['weights'])
                    masks={k:v.cpu().numpy() for k,v in measured['slots']['pixel_masks'].items()};target=example.target[0].cpu().numpy()
                    pixel.add(target,measured['predictions'],masks)
                    if example.event not in events: events[example.event]=(modules.slot.RegionAccumulator(),NamedPixels(modules.pixel.PixelAccumulator,args.step))
                    er,ep=events[example.event];er.add(measured['rows'],measured['slots']['weights']);ep.add(target,measured['predictions'],masks)
                    ip=NamedPixels(modules.pixel.PixelAccumulator,args.step);ip.add(target,measured['predictions'],masks)
                    log.write(json.dumps(dict(image=example.image,event=example.event,pixels=ip.report()),ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                    state.update(image_count=len(seen),phase='forward',elapsed_seconds=time.monotonic()-started);write(output/'summary.json',state)
                    if len(seen)%10==0 or args.smoke: print(f'ANCHOR SLOT AUDIT step={args.step} Val {len(seen)}/{len(names)}; checks pending',flush=True)
                    definitions=measured['neighbor_definitions'];del measured,example,target,masks,ip,batch;runtime.audit_data=None
            if seen!=names: raise ValueError('Val iterator stopped early')
            if coverage['d']['supported_parent_queries']<=0: raise ValueError('No legal historical support observed')
            runtime_input_check(runtime,static);integrity=train._checked_verify(runtime)
            if before!={n:legacy.state.state_fingerprint(h.state_dict()) for n,h in heads.items()}:raise ValueError('Inference changed classifier')
            input_hashes={str(p):sha for p,sha in runtime.input_ledger.items()}
        guard=train._restored_guard(runtime)
        verify_files(static['files']);verify_files(study_files);verify_files(audit_files);bridge.verify_dependencies(legacy);original_run.verify_release()
        replay=dict(passed=None,reason='Two-image smoke does not replay Val450') if args.smoke else replay_check(pixel,static['reference'],modules.pixel,args.step)
        if not args.smoke and not replay['passed']:
            write(output/'replay_failure.json',replay);raise ValueError('Selected new checkpoint Val450 replay failed')
        result=dict(state,complete=True,execution_complete=True,full_validation_evidence=not args.smoke,
                    final_integrity_pending=False,phase='complete_smoke' if args.smoke else 'complete',
                    elapsed_seconds=time.monotonic()-started,regions=region.report(),pixels=pixel.report(),
                    per_event={e:dict(regions=r.report(),pixels=p.report()) for e,(r,p) in events.items()},inventory=inventory,
                    neighbor_definitions=definitions,coverage=coverage,integrity=integrity,guard_restored=guard,replay=replay)
        write(output/'source_fingerprints.json',dict(training=static['files'],study=study_files,audit=audit_files))
        write(output/'input_fingerprints.json',input_hashes);write(output/'result.json',result);write_report(output,result)
        state.update({k:result[k] for k in ('complete','execution_complete','full_validation_evidence','final_integrity_pending','phase','elapsed_seconds')})
        state['result_sha256']=digest(output/'result.json');write(output/'summary.json',state)
        print((output/'REPORT.txt').read_text(),flush=True)
        print('ANCHOR SLOT/MEMORY AUDIT COMPLETE. No training. No Test.',flush=True)
        return result
    except BaseException as exc:
        state.update(phase='failed',error=type(exc).__name__+': '+str(exc));write(output/'summary.json',state)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','plan'))
    parser.add_argument('--study-root',required=True);parser.add_argument('--output')
    parser.add_argument('--study-dir',default=DEFAULT_NEW+'/joint_anchor_study_v1')
    parser.add_argument('--audit-dir',default=DEFAULT_NEW+'/slot_memory_audit_v1_fix1')
    parser.add_argument('--helpers',default=DEFAULT_NEW+'/joint_loss_after500_v1')
    parser.add_argument('--tool-dir',default=DEFAULT_NEW+'/soft_memory_q_rebased_v1_fix2')
    parser.add_argument('--step',type=int,choices=(500,1500),default=500)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.command=='plan':
        static=validate_run(args.study_root,args.step);package_files(args.study_dir,STUDY_MANIFEST);package_files(args.audit_dir,AUDIT_MANIFEST)
        print(json.dumps(dict(ready_for_forward=True,training=False,test_accessed=False,checkpoint=static['checkpoint']),ensure_ascii=False));return
    if not args.output:parser.error('--output required')
    return run(args)

if __name__=='__main__':main()
