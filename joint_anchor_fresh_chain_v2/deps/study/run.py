#!/usr/bin/env python3
"""Matched output-anchoring study. Reuses the released model/runtime and loss helpers."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FORMAT = 'BRIGHT_JOINT_ANCHOR_STUDY_V1'
DEFAULT_NEW = '/root/shared-nvme/BRIGHT-master/New'
DEFAULT_ROOT = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D'
LATEST = Path(DEFAULT_NEW)/'joint_anchor_latest.txt'


def release_hash():
    manifest = HERE/'SHA256SUMS.json'
    expected = json.loads(manifest.read_text())
    for name, value in expected.items():
        path = HERE/name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError('Invalid/missing release file: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != value:
            raise ValueError('Release checksum changed: '+name)
    if not {p.name for p in HERE.glob('*.py')} <= set(expected):
        raise ValueError('Unbound Python file in study package')
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def validate_options(args):
    if not math.isfinite(args.weight) or args.weight < 0:
        raise ValueError('weight must be finite and nonnegative')
    if not math.isfinite(args.lr_multiplier) or not 0 < args.lr_multiplier <= 1:
        raise ValueError('lr-multiplier must be in (0,1]')
    if args.arm not in ('b','d'):
        raise ValueError('Only matched soft branches B/D are supported')


def dependencies(args):
    from legacy import load_legacy, prepare_source
    validate_options(args)
    legacy = load_legacy(args.helpers)
    static = prepare_source(args, legacy)
    return legacy, static


def make_heads(torch, original_train, config, device, saved, arm):
    head = original_train.JointRegionMemoryClassifier(chunk_size=int(config['chunk_size'])).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config['lr']),
                                 weight_decay=float(config['weight_decay']))
    if saved is not None:
        head.load_state_dict(copy.deepcopy(saved['heads'][arm]), strict=True)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizers'][arm]))
    return head, optimizer


def initial_replay(result, reference, arm, original_train, tolerance_pp=.02):
    rows = {}
    for name in ('grid_probq','control','anchor'):
        old = reference['metrics']['grid_probq' if name == 'grid_probq' else arm]
        current = result['metrics'][name]
        # Recompute from confusion; do not trust a mismatched scalar summary.
        if 'confusion' in old:
            old = original_train._metrics(old['confusion'])
        left, right = current.get('iou'), old.get('iou')
        if left is None or right is None:
            raise ValueError('Replay needs all four class IoUs')
        delta = [0. if a is None and b is None else float('inf') if a is None or b is None
                 else abs(100*(a-b)) for a,b in zip(left,right)]
        rows[name] = dict(max_iou_difference_pp=max(delta), passed=max(delta)<=tolerance_pp)
    return dict(passed=all(v['passed'] for v in rows.values()), tolerance_pp=tolerance_pp, paths=rows)


def save_checkpoint(folder, heads, optimizers, step, runtime, config, study_args, legacy, source, release):
    import torch
    path = Path(folder)/('checkpoint_%06d.pt'%step)
    payload = dict(format=FORMAT, step=step, heads={k:v.state_dict() for k,v in heads.items()},
                   optimizers={k:v.state_dict() for k,v in optimizers.items()}, rng=legacy.state.capture_rng(),
                   source_mode=study_args.start, source_checkpoint=source.get('checkpoint'),
                   runtime_config=config, protocol_info=runtime.protocol_info,
                   input_file_sha256={str(p):h for p,h in runtime.input_ledger.items()},
                   arm=study_args.arm, anchor_weight=study_args.weight,
                   lr_multiplier=study_args.lr_multiplier, schedule_horizon=5000,
                   primary_endpoint=1500, release_sha256=release, test_accessed=False)
    torch.save(payload, path)
    record = dict(step=step, sha256=legacy.study.digest(path),
                  head_hashes={k:legacy.state.state_fingerprint(v.state_dict()) for k,v in heads.items()})
    legacy.study.write_json(path.with_suffix('.pt.json'),record)
    reopened = torch.load(path,map_location='cpu',weights_only=False)
    for name, head in heads.items():
        head.load_state_dict(reopened['heads'][name], strict=True)
        if legacy.state.state_fingerprint(head.state_dict()) != record['head_hashes'][name]:
            raise ValueError('Checkpoint roundtrip changed head')
    if legacy.study.digest(path) != record['sha256']:
        raise ValueError('Checkpoint changed during roundtrip')
    return path,record


def _worker(args, backend=None):
    from legacy import import_runtime, load_checkpoint, verify_dependencies
    from engine import fork_pair, train_step
    from evaluate import evaluate, print_table
    release = release_hash()
    legacy, static = dependencies(args) if backend is None else backend.dependencies(args)
    runtime_import = import_runtime if backend is None else backend.import_runtime
    torch, common, original_train, original_run, Runtime = runtime_import(static,legacy)
    saved = load_checkpoint(static,legacy,torch)
    if not torch.cuda.is_available():
        raise RuntimeError('Real-data execution requires CUDA; CPU contracts are separate')
    smoke = args.mode == 'smoke'
    prefix = (2 if smoke else 500) if args.start == 'fresh' else 0
    branch_step = prefix if prefix else 500
    endpoint = branch_step+20 if smoke else 1500
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'checkpoints').mkdir()
    config = copy.deepcopy(static['config'])
    config.update(command='train',mode='train',smoke=False,steps=5000,eval_every=500,output=str(output),
                  baseline_policy='independent',baseline_root=None,baseline_evidence=None,
                  allow_smoke_values=False)
    if backend is not None:
        config = backend.runtime_config(config, smoke=smoke)
    runtime_args = SimpleNamespace(**config)
    status = dict(format=FORMAT, complete=False,execution_complete=False,training_complete=False,
                  full_validation_evidence=False,final_integrity_pending=True,phase='loading',smoke=smoke,
                  test_accessed=False,source_mode=args.start,arm=args.arm,branch_step=branch_step,
                  endpoint_step=endpoint,schedule_horizon=5000,release_sha256=release,
                  experiment=dict(anchor_weight=args.weight,post500_lr_multiplier=args.lr_multiplier,
                      original_loss_unchanged=True,slots_unchanged=True,model_unchanged=True,
                      teacher='same-forward frozen grid_probq',teacher_GT_filter=False,
                      automatic_old_control_reuse=False,claim_historical_resume=False),
                  new_updates={'control':0,'anchor':0},evaluations={},pid=os.getpid())
    write = legacy.study.write_json
    write(output/'summary.json',status)
    started=time.monotonic()
    def update(phase):
        status['phase']=phase; write(output/'summary.json',status)
    try:
        (original_run.check_inputs if backend is None else backend.check_inputs)(runtime_args)
        with Runtime(runtime_args,output) as runtime:
            original_train._formal_runtime(runtime,smoke if backend is not None else False)
            original_train._names(runtime,False)
            original_train._checked_verify(runtime)
            if saved is not None:
                if runtime.protocol_info != saved['protocol_info']:
                    raise ValueError('Frozen runtime differs from checkpoint500')
                if runtime.train_names != static['manifest']['train_names'] or runtime.val_names != static['manifest']['val_names']:
                    raise ValueError('Source Train/Val identities changed')
                legacy.study.check_inputs_against_checkpoint(runtime,saved)
            write(output/'manifest.json',dict(format=FORMAT,arguments=vars(args),runtime_config=config,
                source_mode=args.start,source_checkpoint=static.get('checkpoint'),
                source_sha256=static['source_sha256'],release_sha256=release,
                helper_files=legacy.files,protocol_info=runtime.protocol_info,
                train_names=runtime.train_names,val_names=runtime.val_names,
                common_prefix_steps=prefix,branch_step=branch_step,primary_endpoint=1500,
                historical_trajectory_claim=False,test_accessed=False))
            original_train._seed(int(config['seed']))
            prototype,opt=make_heads(torch,original_train,config,runtime.device,saved,args.arm)
            if saved is not None:
                if legacy.state.state_fingerprint(prototype.state_dict()) != legacy.state.state_fingerprint(saved['heads'][args.arm]):
                    raise ValueError('Restored head differs')
                if legacy.state.state_fingerprint(opt.state_dict()) != legacy.state.state_fingerprint(saved['optimizers'][args.arm]):
                    raise ValueError('Restored optimizer differs')
            for step in (1,500,501,1500,5000):
                if legacy.state.lr_at_step(step) != original_train._lr(step,5000,1e-4):
                    raise ValueError('Inherited LR schedule mismatch')
            iterator=iter(runtime.batches(runtime.train_names,train=True,seed=int(config['seed'])))
            prefix_names=[]
            if saved is not None:
                update('reconstruct_loader')
                for index in range(500):
                    batch=next(iterator)
                    name=legacy.state.raw_batch_identity(batch)
                    if name != static['rows'][index]['image']:
                        raise ValueError('Historical loader mismatch at '+str(index+1))
                    prefix_names.append(name)
                    if (index+1)%100==0: print('Verified CPU loader position',index+1,'/500',flush=True)
                legacy.state.restore_rng(saved['rng'],require_cuda=True)
            else:
                update('shared_prefix')
                with (output/'prefix_train.jsonl').open('x',encoding='utf-8') as journal:
                    for step in range(1,prefix+1):
                        batch=next(iterator); example=runtime.prepare(batch)
                        legacy.study.validate_example(example,runtime,original_train)
                        prefix_names.append(example.image)
                        rows=train_step({'control':prototype},{'control':opt},example,runtime,
                            step=step,arm=args.arm,state=legacy.state,maximum_lr=config['lr'],
                            lr_multiplier=args.lr_multiplier,anchor_weight=args.weight,prefix=True,
                            diagnose=(step==1 or step%100==0))
                        row=dict(step=step,image=example.image,event=example.event,heads=rows)
                        journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
                        if step%20==0 or step==1: print('SHARED PREFIX',json.dumps(row),flush=True)
                save_checkpoint(output/'checkpoints',{'control':prototype},{'control':opt},
                                branch_step,runtime,config,args,legacy,static,release)
            branch_rng=legacy.state.capture_rng()
            heads,optimizers,initial_hashes=fork_pair(prototype,opt,legacy.state)
            del prototype,opt
            status['initial_pair']=initial_hashes
            legacy.state.restore_rng(branch_rng,require_cuda=True)
            val_names=runtime.val_names[:2] if smoke else runtime.val_names
            fixed_names=prefix_names[:2 if smoke else 450]
            update('initial_val')
            initial=evaluate(runtime,heads,val_names,output/'initial_val',step=branch_step,split='val',
                             seed=config['seed'],arm=args.arm,weight=args.weight,original_train=original_train,legacy=legacy)
            print_table(initial)
            if saved is not None and not smoke:
                replay=initial_replay(initial,static['reference'],args.arm,original_train)
                write(output/'initial_replay.json',replay)
                if not replay['passed']: raise ValueError('Initial Val450 replay failed')
                status['initial_replay']=replay
            status['evaluations'][str(branch_step)]=str(output/'initial_val/result.json')
            update('fixed_train_initial')
            fixed_initial=evaluate(runtime,heads,fixed_names,output/'fixed_train_initial',step=branch_step,
                 split='fixed_train',seed=config['seed'],arm=args.arm,weight=args.weight,original_train=original_train,legacy=legacy)
            legacy.state.restore_rng(branch_rng,require_cuda=True)
            update('training')
            coverage_total=0; reader_updates={k:0 for k in heads}; suffix_names=[]
            with (output/'train.jsonl').open('x',encoding='utf-8') as journal:
                for step in range(branch_step+1,endpoint+1):
                    batch=next(iterator)
                    if saved is not None and legacy.state.raw_batch_identity(batch)!=static['rows'][step-1]['image']:
                        raise ValueError('Historical suffix loader mismatch')
                    example=runtime.prepare(batch)
                    if saved is not None and (example.image!=static['rows'][step-1]['image'] or example.event!=static['rows'][step-1]['event']):
                        raise ValueError('Prepared historical image/event mismatch')
                    coverage=legacy.study.validate_example(example,runtime,original_train)
                    coverage_total+=coverage[args.arm]['supported_slot_queries']
                    suffix_names.append(example.image)
                    rows=train_step(heads,optimizers,example,runtime,step=step,arm=args.arm,state=legacy.state,
                        maximum_lr=config['lr'],lr_multiplier=args.lr_multiplier,anchor_weight=args.weight,
                        diagnose=step==branch_step+1 or step%20==0)
                    for name in heads:
                        status['new_updates'][name]=step-branch_step
                        reader_updates[name]+=int(rows[name]['reader_grad_abs_sum']>0)
                    row=dict(step=step,image=example.image,event=example.event,heads=rows,coverage=coverage,
                             elapsed_seconds=time.monotonic()-started)
                    journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
                    if step==branch_step+1 or step%20==0:
                        print('ANCHOR STUDY',json.dumps(row),flush=True)
                        status['last_step']=step;write(output/'summary.json',status)
                    del example,batch
                    if step==1000 or step==endpoint:
                        path,record=save_checkpoint(output/'checkpoints',heads,optimizers,step,
                                                    runtime,config,args,legacy,static,release)
                        update('val_step_%06d'%step)
                        result=evaluate(runtime,heads,val_names,output/('val_step_%06d'%step),step=step,
                            split='val',seed=config['seed'],arm=args.arm,weight=args.weight,original_train=original_train,legacy=legacy)
                        print_table(result)
                        status['evaluations'][str(step)]=str(output/('val_step_%06d'%step)/'result.json')
                        status['last_checkpoint']=str(path)
                        update('training')
            if coverage_total<=0 or any(v<=0 for v in reader_updates.values()):
                raise ValueError('No legal support or no Reader gradient in a branch')
            if len(set(prefix_names+suffix_names))!=endpoint:
                raise ValueError('Expected unique epoch-zero training images')
            status['training_complete']=True
            update('fixed_train_endpoint')
            fixed=evaluate(runtime,heads,fixed_names,output/'fixed_train_endpoint',step=endpoint,
                split='fixed_train',seed=config['seed'],arm=args.arm,weight=args.weight,original_train=original_train,legacy=legacy)
            if fixed['names_sha256']!=fixed_initial['names_sha256'] or fixed['order_sha256']!=fixed_initial['order_sha256']:
                raise ValueError('Fixed Train identities/order changed')
            update('final_integrity')
            if saved is not None: legacy.study.check_inputs_against_checkpoint(runtime,saved)
            integrity=original_train._checked_verify(runtime)
            legacy.study.check_old_files(static)
            verify_dependencies(legacy)
            original_run.verify_release()
            if static.get('config_path') and legacy.study.digest(static['config_path']) != static['config_sha256']:
                raise ValueError('Fresh source configuration changed during training')
            if release_hash()!=release or common.source_hash()!=static['source_sha256']:
                raise ValueError('Study or original code changed')
            final_hashes={k:legacy.state.state_fingerprint(v.state_dict()) for k,v in heads.items()}
            if final_hashes!=record['head_hashes']: raise ValueError('Evaluation mutated the heads')
            input_hashes={str(p):h for p,h in runtime.input_ledger.items()}
        guard=original_train._restored_guard(runtime)
        report_paths=list(status['evaluations'].values())+[str(output/'fixed_train_initial/result.json'),str(output/'fixed_train_endpoint/result.json')]
        for path in report_paths:
            report=legacy.study.read_json(path)
            if not smoke and report['split']=='val' and (
                    report.get('whole_val450_processed') is not True or report.get('image_count')!=450
                    or report.get('execution_complete') is not True):
                raise ValueError('Formal evidence requires every Val450 image and successful evaluation')
            report.update(complete=True,final_integrity_pending=False,smoke=smoke,final_integrity=integrity,
                          guard_report=guard,primary_endpoint=not smoke and report['split']=='val' and report['step']==1500,
                          full_validation_evidence=not smoke and report['split']=='val')
            write(path,report)
            Path(path+'.sha256').write_text(legacy.study.digest(path)+'  result.json\n',encoding='ascii')
        status.update(complete=True,execution_complete=True,full_validation_evidence=not smoke,
                      final_integrity_pending=False,phase='complete_smoke' if smoke else 'complete',
                      unique_train_queries=endpoint,all_Train2918_visited=False,reader_gradient_updates=reader_updates,
                      final_integrity=integrity,guard_report=guard,elapsed_seconds=time.monotonic()-started,
                      fixed_train_scope='first 450 prefix images' if not smoke else 'two-image smoke')
        write(output/'input_fingerprints.json',input_hashes);write(output/'summary.json',status)
        print('ANCHOR STUDY COMPLETE. No Test.',flush=True)
        return status
    except BaseException as exc:
        status.update(complete=False,execution_complete=False,full_validation_evidence=False,error=type(exc).__name__+': '+str(exc))
        write(output/'summary.json',status)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


def worker(args):
    """Retain early dependency/import failures as well as failures after CUDA entry."""
    try:
        return _worker(args)
    except BaseException as exc:
        output=Path(args.output).expanduser().resolve()
        if not output.exists():
            output.mkdir(parents=True)
            record=dict(format=FORMAT,complete=False,execution_complete=False,
                        full_validation_evidence=False,training_complete=False,test_accessed=False,
                        phase='startup_failed',error=type(exc).__name__+': '+str(exc))
            (output/'summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            (output/'traceback.txt').write_text(traceback.format_exc())
        raise


def child_args(args, mode, output):
    cmd=[sys.executable,'-u',str(HERE/'run.py'),'worker','--mode',mode,'--output',str(output),
         '--start',args.start,'--train-root',args.train_root,'--tool-dir',args.tool_dir,'--helpers',args.helpers,
         '--arm',args.arm,'--weight',str(args.weight),'--lr-multiplier',str(args.lr_multiplier)]
    if args.config: cmd+=['--config',args.config]
    return cmd


def validate_child(report,mode,start):
    endpoint=(22 if start=='fresh' else 520) if mode=='smoke' else 1500
    if (report.get('complete') is not True or report.get('execution_complete') is not True
        or report.get('training_complete') is not True or report.get('final_integrity_pending') is not False
        or report.get('smoke') is not (mode=='smoke') or report.get('endpoint_step')!=endpoint
        or report.get('full_validation_evidence') is not (mode=='train') or report.get('test_accessed') is not False):
        raise ValueError('Child lacks complete matching evidence: '+mode)


def pipeline(args):
    release_hash();legacy,static=dependencies(args)
    root=Path(args.output).expanduser().resolve()
    with legacy.study.exclusive_pipeline_lock(root.parent/'anchor_active.lock'):
        root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
        state=dict(format=FORMAT,complete=False,execution_complete=False,full_validation_evidence=False,
                   training_complete=False,test_accessed=False,arguments=vars(args),stages={},current_stage='cpu_contracts')
        legacy.study.write_json(root/'workflow_state.json',state)
        stages=[('cpu_contracts',[sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-q'])]
        stages.extend((mode,child_args(args,mode,root/mode)) for mode in (('smoke',) if args.command=='check' else ('smoke','train')))
        try:
            for name,cmd in stages:
                state['current_stage']=name
                log=root/'logs'/(name+'.log')
                state['stages'][name]=dict(status='running',command=cmd,log=str(log))
                legacy.study.write_json(root/'workflow_state.json',state)
                env=os.environ.copy()
                env['BRIGHT_ORIGINAL_TOOL']=args.tool_dir;env['BRIGHT_LOSS_HELPERS']=args.helpers
                if name=='cpu_contracts': env['CUDA_VISIBLE_DEVICES']=''
                print('START',name,'LOG='+str(log),flush=True)
                with log.open('x',encoding='utf-8') as stream:
                    child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=HERE,env=env)
                    state['stages'][name]['pid']=child.pid
                    legacy.study.write_json(root/'workflow_state.json',state)
                    for line in child.stdout:
                        stream.write(line);stream.flush();print(line,end='',flush=True)
                    code=child.wait()
                state['stages'][name].update(status='complete' if code==0 else 'failed',returncode=code)
                if code: raise RuntimeError(name+' failed; later stages stopped; see '+str(log))
                if name!='cpu_contracts': validate_child(legacy.study.read_json(root/name/'summary.json'),name,args.start)
                legacy.study.write_json(root/'workflow_state.json',state)
            state.update(complete=True,execution_complete=True,current_stage='complete',
                         training_complete=args.command=='full',full_validation_evidence=args.command=='full')
            legacy.study.write_json(root/'workflow_state.json',state)
            print_report(root)
        except BaseException as exc:
            state.update(complete=False,execution_complete=False,error=type(exc).__name__+': '+str(exc))
            state['stages'][state['current_stage']]['status']='failed'
            legacy.study.write_json(root/'workflow_state.json',state)
            (root/'traceback.txt').write_text(traceback.format_exc())
            raise


def print_report(root):
    root=Path(root).expanduser().resolve()
    print('结果目录：',root)
    if (root/'workflow_state.json').is_file():
        state=json.loads((root/'workflow_state.json').read_text())
        print('阶段:',state.get('current_stage'),'完成:',state.get('complete'),'错误:',state.get('error'))
    folder=root/'train' if (root/'train').is_dir() else root/'smoke' if (root/'smoke').is_dir() else root
    if not (folder/'summary.json').is_file():
        print('尚无训练报告；查看 logs。');return
    summary=json.loads((folder/'summary.json').read_text())
    print('起点:',summary.get('source_mode'),'分支:',summary.get('arm'),'新增更新:',summary.get('new_updates'),
          '正式完整证据:',summary.get('full_validation_evidence'),'错误:',summary.get('error'))
    from evaluate import print_table
    for step,path in sorted(summary.get('evaluations',{}).items(),key=lambda x:int(x[0])):
        if Path(path).is_file(): print_table(json.loads(Path(path).read_text()))
        else: print(step,'报告尚未保存:',path)
    print('主要比较：1500步 anchor−control；两支学习率一致。低KL或低loss本身不是收益。')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','check','full','worker','report'))
    p.add_argument('--start',choices=('checkpoint500','fresh'),default='checkpoint500')
    p.add_argument('--train-root',default=DEFAULT_ROOT)
    p.add_argument('--tool-dir',default=DEFAULT_NEW+'/soft_memory_q_rebased_v1_fix2')
    p.add_argument('--helpers',default=DEFAULT_NEW+'/joint_loss_after500_v1')
    p.add_argument('--config',help='Fresh source JSON; explicit choice, no automatic fallback')
    p.add_argument('--arm',choices=('b','d'),default='d')
    p.add_argument('--weight',type=float,default=.1)
    p.add_argument('--lr-multiplier',type=float,default=.2)
    p.add_argument('--output');p.add_argument('--root')
    p.add_argument('--mode',choices=('smoke','train'),default='train')
    return p


def main():
    args=parser().parse_args()
    if args.command=='report':
        root=args.root or (LATEST.read_text().strip() if LATEST.is_file() else None)
        if not root: raise ValueError('Pass --root')
        return print_report(root)
    if args.command=='plan':
        try:
            release=release_hash();legacy,static=dependencies(args)
            record=dict(ready_for_real_smoke=True,CUDA_forward_verified=False,training=False,test_accessed=False,
                        start=args.start,arm=args.arm,weight=args.weight,lr_multiplier=args.lr_multiplier,
                        branch_step=500,endpoint_step=1500,common_fresh_prefix=args.start=='fresh',
                        release_sha256=release,source_sha256=static['source_sha256'],model_unchanged=True)
            print(json.dumps(record,ensure_ascii=False,indent=2));return
        except (OSError,ValueError,RuntimeError) as exc:
            print(json.dumps(dict(ready_for_real_smoke=False,training=False,CUDA_forward_verified=False,
                      error=type(exc).__name__+': '+str(exc),missing=getattr(exc,'missing',[])),ensure_ascii=False,indent=2))
            raise SystemExit(2)
    if not args.output: raise ValueError('--output required; use start_server.sh for a new directory')
    return worker(args) if args.command=='worker' else pipeline(args)

if __name__=='__main__': main()
