#!/usr/bin/env python3
"""Independent surviving-Q -> new Train Memory -> paired D anchor -> Val diagnostics.

Each CUDA stage runs in its own process. No historical E0/V3 member identity claim.
No Test images. No automatic promotion of the candidate objective.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from io_utils import HERE,read,write,digest,verify_release,old_modules

REPO=Path('/root/shared-nvme/BRIGHT-master')
DATASET=REPO/'bda_benchmark/dataset'
Q=Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1/run_20261014_003226_vdy7ey5t/sar_q/formal/train_baseline/last.pth')
A1=Path('/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth')
TRAIN=DATASET/'splitname/standard_ML/train_set.txt'
VAL=DATASET/'splitname/standard_ML/val_set.txt'
OUTBASE=Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_ANCHOR_FRESH_V2')
NEW=Path('/root/shared-nvme/BRIGHT-master/New')
FORMAT='BRIGHT_JOINT_ANCHOR_FRESH_CHAIN_V2'


def _source_args(root):
    return SimpleNamespace(repo=REPO,dataset=DATASET,q_checkpoint=Q,a1_checkpoint=A1,
         train_list=TRAIN,val_list=VAL,device='cuda:0',workers=4,seed=3407)

def _training_config(root,smoke=False):
    bank=root/('members_smoke' if smoke else 'members')
    return dict(repo=str(REPO),protocol=str(root/'source.json'),membership_root=str(bank),bank_root=str(bank),
         values=str(root/('values_smoke' if smoke else 'values')),device='cuda:0',workers=4,seed=3407,
         key_name='t128',topk=64,chunk_size=32,lr=.0001,weight_decay=.005,steps=5000,eval_every=500)

def _run_stage(root,stage):
    from new_source import create
    from new_members import build
    if stage=='cpu_contracts':
        for command in ([sys.executable,'-m','unittest','-q','test_backend'],
                        [sys.executable,'-m','unittest','discover','-s','deps/study','-p','test_*.py','-q']):
            subprocess.run(command,cwd=HERE,check=True)
        return
    if stage=='source':return create(_source_args(root),root/'source.json')
    if stage in ('members_smoke','members'):
        return build(root/'source.json',root/stage,REPO,workers=4,seed=3407,smoke=stage.endswith('_smoke'))
    if stage in ('values_smoke','values'):
        # Uses the released exact child pooling and labeling implementation.
        _,_,_,_,_,_=old_modules()
        from build_values import run_build
        from runtime_adapter import NewFrozenSlotRuntime
        smoke=stage.endswith('_smoke')
        cfg=_training_config(root,smoke)
        cfg.update(command='build',mode='build',output=str(root/stage),limit_images=4 if smoke else 0,
                   baseline_policy='independent',baseline_root=None,baseline_evidence=None,
                   allow_smoke_values=smoke,smoke=smoke)
        return run_build(SimpleNamespace(**cfg),runtime_factory=NewFrozenSlotRuntime)
    if stage in ('study_smoke','train'):
        from io_utils import named
        from study_backend import dependencies,import_runtime,runtime_config,check_inputs
        study_root=HERE/'deps/study'
        if str(study_root) not in sys.path:sys.path.insert(0,str(study_root))
        study=named(study_root/'run.py','study_run')
        config=root/('config_smoke.json' if stage=='study_smoke' else 'config.json')
        write(config,_training_config(root,stage=='study_smoke'))
        args=SimpleNamespace(mode='smoke' if stage=='study_smoke' else 'train',start='fresh',arm='d',weight=.1,
            lr_multiplier=.2,config=str(config),helpers=str(HERE/'deps/helpers'),
            tool_dir=str(HERE/'deps/original'),output=str(root/stage))
        import study_backend
        return study._worker(args,backend=study_backend)
    if stage in ('audit_500','audit_1500'):
        # The diagnostic opens a saved checkpoint before constructing Runtime;
        # establish the new experiment's thread policy first in this process.
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)
        from io_utils import named
        chain=HERE/'deps/chain'
        if str(chain) not in sys.path:sys.path.insert(0,str(chain))
        audit=named(chain/'post_audit.py','post_audit')
        step=500 if stage.endswith('500') else 1500
        args=SimpleNamespace(study_root=str(root),output=str(root/stage),
            study_dir=str(HERE/'deps/study'),audit_dir=str(HERE/'deps/audit'),
            helpers=str(HERE/'deps/helpers'),tool_dir=str(HERE/'deps/original'),
            step=step,smoke=False)
        return audit.run(args)
    raise ValueError('Unknown stage: '+stage)

STAGES=('cpu_contracts','source','members_smoke','values_smoke','study_smoke','members','values','train','audit_500','audit_1500')

def pipeline(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir()
    state=dict(format=FORMAT,complete=False,execution_complete=False,full_validation_evidence=False,
        test_accessed=False,phase='preflight',current_stage=None,release_sha256=verify_release(),
        stages={},old_v3_members_reused=False,historical_trajectory_claim=False,training_complete=False)
    write(root/'workflow_state.json',state)
    lock=OUTBASE/'active.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in STAGES:
            log=root/'logs'/(name+'.log')
            state['current_stage']=name;state['phase']='running';state['stages'][name]=dict(status='running',log=str(log))
            write(root/'workflow_state.json',state)
            print('START',name,'log',log,flush=True)
            with log.open('x') as out:
                proc=subprocess.Popen([sys.executable,'-u',str(HERE/'fresh_chain.py'),'worker',
                                      '--root',str(root),'--stage',name],stdout=out,stderr=subprocess.STDOUT,
                                      cwd=HERE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                state['stages'][name]['pid']=proc.pid;write(root/'workflow_state.json',state)
                code=proc.wait()
            state['stages'][name].update(status='complete' if code==0 else 'failed',returncode=code,
                                         log_sha256=digest(log))
            if code:
                state.update(phase='failed',error=f'{name} returned {code}; see {log}',current_stage=name)
                write(root/'workflow_state.json',state)
                raise RuntimeError(state['error'])
            if name=='train':
                completed=read(root/'train/summary.json')
                if not (completed.get('complete') is True and completed.get('full_validation_evidence') is True):
                    raise ValueError('Formal paired Val450 final check incomplete')
                state['training_complete']=True
            write(root/'workflow_state.json',state)
    for stage in ('audit_500','audit_1500'):
        result=read(root/stage/'result.json')
        if result.get('complete') is not True or result.get('full_validation_evidence') is not True:
            raise ValueError('Post-training content audit incomplete: '+stage)
    state.update(complete=True,execution_complete=True,full_validation_evidence=True,
                 phase='complete',current_stage=None,error=None)
    write(root/'workflow_state.json',state)
    print('FRESH CHAIN COMPLETE. New Train members, paired Val450 study and slot/Memory audits; no Test.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','run','worker','report'))
    p.add_argument('--root',type=Path);p.add_argument('--stage',choices=STAGES)
    args=p.parse_args()
    if args.command=='plan':
        from new_source import EXPECTED
        paths={'q_checkpoint':Q,'a1_checkpoint':A1,'train_list':TRAIN,'val_list':VAL}
        def present(path):
            try:return path.is_file()
            except OSError:return False
        def repo_present():
            try:return (REPO/'bda_benchmark').is_dir()
            except OSError:return False
        report=dict(release_sha256=verify_release(),remote_paths={k:str(v) for k,v in paths.items()},
            files_present={k:present(v) for k,v in paths.items()},
            expected_sha256=EXPECTED,repo_present=repo_present(),
            training_started=False,cuda_checked=False,old_member_bank_required=False)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if not all(report['files_present'].values()) or not report['repo_present']:raise SystemExit(2)
        return
    if args.root is None:p.error('--root required')
    if args.command=='report':
        state=read(args.root/'workflow_state.json')
        print(json.dumps(state,ensure_ascii=False,indent=2));return
    if args.command=='worker':
        if not args.stage:p.error('--stage required')
        _run_stage(args.root,args.stage);return
    pipeline(args.root)

if __name__=='__main__':main()
