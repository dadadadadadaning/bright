#!/usr/bin/env python3
"""Sequential fresh head training followed by Val-only slot/Memory diagnostics."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = '/root/shared-nvme/BRIGHT-master/New'
FORMAT = 'BRIGHT_JOINT_ANCHOR_CHAIN_V1'


from chain_io import digest, read, write, package_files

def verify_release():
    package_files(HERE)
    return digest(HERE / 'SHA256SUMS.json')


def base_command(config):
    keys = ('repo','membership_root','protocol','device','workers','seed','key_name',
            'topk','chunk_size','lr','weight_decay')
    command = []
    for key in keys:
        command += ['--' + key.replace('_','-'), str(config[key])]
    return command


def bank_commands(plan, args):
    config = plan['config']
    commands = []
    for stage in plan['bank_actions']:
        command = [sys.executable, '-u', str(Path(args.original_tool) / 'run.py'), stage, *base_command(config),
                   '--limit-images', '0', '--output', config['bank_root' if stage == 'parent' else 'values']]
        if stage == 'build':
            command += ['--bank-root', config['bank_root']]
        commands.append((stage, command))
    return commands


def validate_bank(root):
    summary = read(Path(root) / 'summary.json')
    if summary.get('complete') is not True or summary.get('smoke') is not False or summary.get('test_accessed') is not False:
        raise ValueError('Bank lacks complete Train-only non-smoke evidence: ' + str(root))


def validate_training(root):
    workflow = read(Path(root) / 'workflow_state.json')
    summary = read(Path(root) / 'train/summary.json')
    for row in (workflow, summary):
        if (row.get('format') != 'BRIGHT_JOINT_ANCHOR_STUDY_V1'
                or any(row.get(k) is not True for k in ('complete','execution_complete','training_complete','full_validation_evidence'))
                or row.get('test_accessed') is not False):
            raise ValueError('Training returned without complete formal evidence')
    if (summary.get('source_mode') != 'fresh' or summary.get('arm') != 'd'
            or summary.get('smoke') is not False
            or summary.get('branch_step') != 500 or summary.get('endpoint_step') != 1500
            or summary.get('final_integrity_pending') is not False
            or summary.get('new_updates') != {'control':1000,'anchor':1000}):
        raise ValueError('Training settings or paired update counts differ')
    for step, folder in ((500,'initial_val'),(1500,'val_step_001500')):
        report = read(Path(root) / 'train' / folder / 'result.json')
        if (report.get('step') != step or report.get('image_count') != 450
                or report.get('whole_val450_processed') is not True
                or report.get('complete') is not True or report.get('execution_complete') is not True
                or report.get('full_validation_evidence') is not True
                or report.get('final_integrity_pending') is not False):
            raise ValueError('Missing complete Val450 evidence at step ' + str(step))
        path = Path(root) / 'train/checkpoints' / ('checkpoint_%06d.pt' % step)
        meta = read(path.with_suffix('.pt.json'))
        if meta.get('step') != step or digest(path) != meta.get('sha256'):
            raise ValueError('Checkpoint missing/changed at step ' + str(step))
    return summary


def validate_audit(folder, step, smoke):
    summary = read(Path(folder) / 'summary.json')
    result = read(Path(folder) / 'result.json')
    for row in (summary, result):
        if (row.get('format') != 'BRIGHT_ANCHOR_SLOT_MEMORY_AUDIT_V1'
                or row.get('complete') is not True or row.get('execution_complete') is not True
                or row.get('training') is not False or row.get('test_accessed') is not False
                or row.get('final_integrity_pending') is not False
                or row.get('image_count') != (2 if smoke else 450)
                or row.get('checkpoint_step') != step or row.get('smoke') is not smoke
                or row.get('full_validation_evidence') is not (not smoke)):
            raise ValueError('Audit returned without complete matching evidence')
    if summary.get('result_sha256') != digest(Path(folder) / 'result.json'):
        raise ValueError('Audit result checksum mismatch')
    if not smoke and result.get('replay',{}).get('passed') is not True:
        raise ValueError('Audit did not replay its new Val450 checkpoint')


def protected_output(root, paths):
    root = Path(root).resolve()
    for raw in paths:
        path = Path(raw).resolve()
        if root == path or root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError('Output overlaps input/code: ' + str(path))


def run_command(command, log, row, save, lock_fd, env):
    with Path(log).open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=env, pass_fds=(lock_fd,), cwd=HERE)
        row['pid'] = child.pid
        save()
        try:
            for line in child.stdout:
                stream.write(line); stream.flush(); print(line,end='',flush=True)
            code = child.wait()
        except BaseException:
            # Do not kill arbitrary GPU processes. Only terminate this owned child.
            child.terminate()
            child.wait()
            raise
        finally:
            if child.stdout is not None: child.stdout.close()
    row['returncode'] = code
    row['log_sha256'] = digest(log)
    if code:
        raise RuntimeError('Stage failed with returncode %s; later stages stopped; see %s' % (code,log))


def pipeline(args, executor=run_command):
    release = verify_release()
    root = Path(args.output).expanduser().resolve()
    paths = [HERE,args.anchor_tool,args.original_tool,args.audit_tool,args.helpers]
    if args.config: paths.append(args.config)
    protected_output(root,paths)
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent / 'anchor_chain_active.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another anchor chain is running; no duplicate launch')
        root.mkdir(exist_ok=False);(root/'logs').mkdir()
        state = dict(format=FORMAT, complete=False, execution_complete=False, training_complete=False,
                     diagnostic_complete=False, full_validation_evidence=False, test_accessed=False,
                     current_stage='install_code', stages={}, arguments=vars(args), release_sha256=release,
                     started_at=time.time(), error=None)
        def save(): write(root/'workflow_state.json',state)
        save()
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1', 'BRIGHT_ORIGINAL_TOOL':args.original_tool,
             'BRIGHT_ANCHOR_TOOL':args.anchor_tool,'BRIGHT_AUDIT_TOOL':args.audit_tool,'BRIGHT_LOSS_HELPERS':args.helpers}
        def stage(name,command,validator=None,cpu=False):
            state['current_stage']=name
            log=root/'logs'/(name+'.log')
            row=state['stages'][name]=dict(status='running',command=command,log=str(log))
            save(); print('START',name,'LOG='+str(log),flush=True)
            executor(command,log,row,save,lock.fileno(),{**env,**({'CUDA_VISIBLE_DEVICES':''} if cpu else {})})
            if validator: validator()
            row['status']='complete';save()
        try:
            stage('install_code',[sys.executable,str(HERE/'install_deps.py'),'--anchor-tool',args.anchor_tool,
                                  '--audit-tool',args.audit_tool,'--helpers',args.helpers],cpu=True)
            state['current_stage']='source_plan';save()
            from source_plan import prepare_plan
            config_path = args.config or str(Path(args.anchor_tool)/'fresh_config.example.json')
            plan=prepare_plan(config_path,args.anchor_tool,args.original_tool,root)
            write(root/'source_plan.json',plan)
            if not plan['ready_for_build_or_reuse']:
                print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
                raise ValueError('Frozen source/member/data dependencies unavailable; see source_plan.json. No training started.')
            # Exclude newly planned descendants; those are outputs, not old inputs.
            protected_output(root,[p for p in plan['protected_paths'] if not Path(p).resolve().is_relative_to(root)])
            config_path=root/'fresh_source_config.json'
            write(config_path,plan['config'])
            state['stages']['source_plan']=dict(status='complete',plan=str(root/'source_plan.json'),
                                                 bank_actions=plan['bank_actions'])
            save()
            stage('chain_contracts',[sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-q'],cpu=True)
            stage('audit_contracts',[sys.executable,'-m','unittest','discover','-s',args.audit_tool,'-p','test_*.py','-q'],cpu=True)
            for name,command in bank_commands(plan,args):
                destination=plan['config']['bank_root' if name=='parent' else 'values']
                stage(name,command,lambda p=destination:validate_bank(p))
            study=root/'study'
            stage('fresh_training',[sys.executable,'-u',str(Path(args.anchor_tool)/'run.py'),'full',
                 '--start','fresh','--config',str(config_path),'--tool-dir',args.original_tool,
                 '--helpers',args.helpers,'--arm','d','--weight','0.1','--lr-multiplier','0.2',
                 '--output',str(study)],lambda:validate_training(study))
            state['training_complete']=True;save()
            steps=sorted(set(args.audit_steps))
            for smoke, step in [(True,steps[0])]+[(False,s) for s in steps]:
                name='audit_smoke' if smoke else 'audit_%06d'%step
                folder=root/name
                command=[sys.executable,'-u',str(HERE/'post_audit.py'),'run','--study-root',str(study),
                         '--study-dir',args.anchor_tool,'--audit-dir',args.audit_tool,'--helpers',args.helpers,
                         '--tool-dir',args.original_tool,'--step',str(step),'--output',str(folder)]
                if smoke:command.append('--smoke')
                stage(name,command,lambda f=folder,s=step,m=smoke:validate_audit(f,s,m))
            if verify_release()!=release:raise ValueError('Chain code changed during execution')
            state.update(complete=True,execution_complete=True,diagnostic_complete=True,
                         full_validation_evidence=True,current_stage='complete',elapsed_seconds=time.time()-state['started_at'])
            save();report(root)
        except BaseException as exc:
            state.update(complete=False,execution_complete=False,error=type(exc).__name__+': '+str(exc))
            name=state['current_stage']
            state['stages'].setdefault(name,{})['status']='failed'
            save();(root/'traceback.txt').write_text(traceback.format_exc())
            print('CHAIN STOPPED:',state['error'],flush=True)
            raise


def report(root):
    root=Path(root).expanduser().resolve()
    state=read(root/'workflow_state.json')
    print('结果目录:',root)
    print(json.dumps({k:state.get(k) for k in ('complete','training_complete','diagnostic_complete','current_stage','error')},ensure_ascii=False,indent=2))
    for name,row in state.get('stages',{}).items():print(name, row.get('status'),row.get('log',''))
    for path in sorted(root.glob('audit_*/REPORT.txt')):print('诊断报告:',path)
    print('未访问 Test；新共同500步不是旧D500复现。来源、父库、双槽库和检查点应保留。')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','plan','report'))
    p.add_argument('--config',help='Explicit fresh source config; defaults to installed example')
    p.add_argument('--anchor-tool',default=NEW+'/joint_anchor_study_v1')
    p.add_argument('--original-tool',default=NEW+'/soft_memory_q_rebased_v1_fix2')
    p.add_argument('--audit-tool',default=NEW+'/slot_memory_audit_v1_fix1')
    p.add_argument('--helpers',default=NEW+'/joint_loss_after500_v1')
    p.add_argument('--audit-steps',type=int,nargs='+',choices=(500,1500),default=[500,1500])
    p.add_argument('--output');p.add_argument('--root')
    return p


def main():
    args=parser().parse_args()
    if args.command=='report':
        root=args.root or (Path(NEW)/'joint_anchor_chain_latest.txt').read_text().strip()
        return report(root)
    if not args.output:raise ValueError('--output required; start_server.sh chooses a new permanent directory')
    if args.command=='plan':
        verify_release()
        from source_plan import prepare_plan
        plan=prepare_plan(args.config or str(Path(args.anchor_tool)/'fresh_config.example.json'),args.anchor_tool,args.original_tool,args.output)
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        if not plan['ready_for_build_or_reuse']:raise SystemExit(2)
        return
    pipeline(args)


if __name__=='__main__':main()
