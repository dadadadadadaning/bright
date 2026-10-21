"""Detached, bounded Test2500 diagnosis. Never train or retry a failed process."""
import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

from night_config import SOURCE,SOURCE_SHA,BASE,TARGET,TARGET_INDEX,FORMAT,CASES,select_case
from grid_probe import sha256,read_json
from evaluate_checkpoint import write_json,ARMS,REQUIRED_CHECKS
from resume_train import verify_files
from resume_context import inactive_parent
import recover_entry


def now():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def preflight():
    if sha256(SOURCE/'trainer.pyz')!=SOURCE_SHA:raise ValueError('Pinned recovery archive differs')
    recover_entry.read_request(SOURCE)
    c=recover_entry.endpoint_context(SOURCE)
    recover_entry.verify_before_test(SOURCE,c)
    workflow=read_json(SOURCE/'workflow_summary.json')
    failed=SOURCE/'assessments/test2500'
    s=read_json(failed/'summary.json');m=read_json(failed/'manifest.json')
    position=s.get('first_failure',{}).get('position',{})
    names=m['names'];select_case('full_test_watched',names)
    if (workflow.get('training_complete') is not True or workflow.get('validation_complete') is not True
        or workflow.get('test_complete') is not False or workflow.get('failed_stage')!='test2500'
        or workflow.get('last_complete_step')!=2500 or s.get('phase')!='failed'
        or position.get('index')!=TARGET_INDEX or position.get('image')!=TARGET or position.get('active_arm')!='d'
        or s.get('completed_images')!=621
        or m.get('provenance',{}).get('checkpoint_sha256')!=c['parent']['sha256']
        or m.get('provenance',{}).get('trainer_sha256')!=SOURCE_SHA):
        raise ValueError('Require completed paired2500/Val450 and the recorded D Test failure at index621')
    paths=[SOURCE/'trainer.pyz',SOURCE/'request.json',SOURCE/'workflow_summary.json']
    paths += [failed/n for n in ('summary.json','manifest.json','evaluation_failure.json','traceback.txt','test_step_002500/images.jsonl')]
    count=0
    with paths[-1].open() as stream:
        for i,line in enumerate(stream):
            row=json.loads(line)
            if (i>=621 or row.get('index')!=i or row.get('image')!=names[i]
                or set(row.get('confusion',{}))!={'grid_probq',*ARMS}):
                raise ValueError('Failed Test journal is not the expected complete621-image prefix')
            count+=1
    if count!=621:raise ValueError('Failed Test prefix length changed')
    for stage in ('val2500','fixed2500'):
        folder=SOURCE/'assessments'/stage
        paths += [folder/'summary.json',folder/'manifest.json']
        for name in ('result.json','result.pending.json','result.json.sha256','images.jsonl'):
            paths.append(folder/('val_step_002500' if stage=='val2500' else 'fixed_train_step_002500')/name)
    c['evidence_files'].update({str(p):sha256(p) for p in paths})
    c['canonical_test_names']=names
    return c


def proc_start(pid):
    text=Path('/proc/%s/stat'%pid).read_text()
    return text[text.rfind(')')+2:].split()[19]


def launched_start(pid):
    try:return proc_start(pid)
    except (FileNotFoundError,ProcessLookupError):return None


def read_request(root):
    root=Path(root).resolve()
    if root.parent!=BASE or not root.name.startswith('test2500_night_'):raise ValueError('Unexpected night directory')
    request=read_json(root/'request.json')
    if sha256(root/'runner.pyz')!=request['artifact_sha256']:raise ValueError('Night archive changed')
    verify_files(request['evidence_files'])
    return request


def launch(timeout,total):
    if not 60<=timeout<=43200 or not 60<=total<=86400:raise ValueError('Use bounded positive time budgets')
    archive=Path(sys.argv[0]).resolve()
    if not archive.is_file() or archive.suffix!='.pyz':raise ValueError('Use the pinned pyz archive')
    with recover_entry.leases() as locks:
        print('NIGHT PREFLIGHT: completed checkpoint2500 and Val450; no optimizer updates',flush=True)
        c=preflight()
        BASE.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='test2500_night_',dir=BASE))
        artifact=root/'runner.pyz';shutil.copyfile(archive,artifact)
        write_json(root/'request.json',dict(format=FORMAT,created_beijing=now(),artifact_sha256=sha256(artifact),
            evidence_files=c['evidence_files'],checkpoint_sha256=c['parent']['sha256'],source_root=str(SOURCE),
            cases=list(CASES),case_timeout_seconds=timeout,total_timeout_seconds=total,
            no_retries=True,optimizer_updates=0,test_accessed=True,existing_test_exposure=True))
        write_json(root/'summary.json',dict(format=FORMAT,phase='queued',night_job_finished=False,
            complete=False,test_complete=False,source_root=str(SOURCE),output=str(root),cases={}))
        fds=tuple(f.fileno() for f in locks)
        cmd=[sys.executable,'-B','-u',str(artifact),'worker','--root',str(root)]
        with (root/'night.log').open('x') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=fds,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                BRIGHT_FINISH_FDS=','.join(map(str,fds))))
        write_json(root/'launcher.json',dict(pid=proc.pid,proc_start=launched_start(proc.pid),command=cmd))
        (BASE/'latest_test2500_night.txt').write_text(str(root)+'\n')
        print(f'已后台启动夜间诊断，退出SSH后继续运行。\nPID={proc.pid}\nOUTPUT={root}\n日志={root / "night.log"}',flush=True)
        print(f'查看日志：{sys.executable} -B {artifact} logs\n查看结果：{sys.executable} -B {artifact} status',flush=True)
    return 0


def run_case(root,case):
    request=read_request(root)
    if case not in CASES:raise ValueError('Unknown case')
    output=root/case
    if not output.is_dir() or (output/'summary.json').exists():raise ValueError('Case must have a fresh allocated directory')
    c=preflight()
    if c['evidence_files']!=request['evidence_files'] or c['parent']['sha256']!=request['checkpoint_sha256']:
        raise ValueError('Source/evaluation evidence changed since launch')
    c['night_artifact_sha256']=request['artifact_sha256']
    from night_assessment import run_assessment
    with inactive_parent(SOURCE):return run_assessment(c,output,case)


def stop_child(proc):
    if proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=5)


def supervise(root,case,budget,stop):
    output=root/case;output.mkdir()
    log_path=output/'case.log'
    command=[sys.executable,'-B','-u',str(root/'runner.pyz'),'case','--root',str(root),'--case',case]
    started=time.monotonic();last_heartbeat=started-30;reason=None
    with log_path.open('x') as writer:
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=writer,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_LAUNCH_BLOCKING='1'))
    write_json(output/'launcher.json',dict(pid=proc.pid,proc_start=launched_start(proc.pid),command=command))
    try:
        with log_path.open() as reader:
            while True:
                text=reader.read()
                if text:print(text,end='',flush=True)
                elapsed=time.monotonic()-started
                if proc.poll() is not None:break
                if stop['requested'] or elapsed>=budget:
                    reason='stopped_by_user' if stop['requested'] else 'time_budget_exceeded'
                    stop_child(proc);continue
                if time.monotonic()-last_heartbeat>=30:
                    heartbeat=dict(stage=case,child_pid=proc.pid,elapsed_seconds=round(elapsed),updated_beijing=now())
                    write_json(root/'heartbeat.json',heartbeat)
                    print('NIGHT HEARTBEAT '+json.dumps(heartbeat,ensure_ascii=False),flush=True)
                    last_heartbeat=time.monotonic()
                time.sleep(2)
            text=reader.read()
            if text:print(text,end='',flush=True)
    finally:stop_child(proc)
    status=read_json(output/'summary.json') if (output/'summary.json').is_file() else {}
    record=dict(returncode=proc.returncode,supervisor_reason=reason,elapsed_seconds=round(time.monotonic()-started,1),
        summary=str(output/'summary.json'),log=str(log_path),phase=status.get('phase'),error=status.get('error'),
        failure_phase=status.get('failure_phase'),completed_images=status.get('completed_images',0),
        diagnostic_complete=status.get('diagnostic_complete',False),full_test_evidence=status.get('full_test_evidence',False),
        first_failure=status.get('first_failure'),last_position=status.get('last_position'))
    write_json(output/'supervisor.json',record)
    return record


def validate_case_success(root,case,record):
    selected,mode,full=select_case(case,read_json(SOURCE/'assessments/test2500/manifest.json')['names'])
    status=read_json(root/case/'summary.json')
    if (record['returncode']!=0 or record.get('supervisor_reason') or status.get('diagnostic_complete') is not True
        or status.get('image_count')!=len(selected) or status.get('complete') is not full
        or status.get('full_test_evidence') is not full or status.get('optimizer_updates')!=0
        or any(status.get('integrity_checks',{}).get(k) is not True for k in REQUIRED_CHECKS)):
        raise ValueError('Case lacks complete matching inference/integrity evidence: '+case)
    from night_assessment import validate_route_evidence
    validate_route_evidence(root/case,selected,mode)


def coordinate(root):
    stop=dict(requested=False)
    def request_stop(*args):stop['requested']=True
    old={sig:signal.signal(sig,request_stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    started=time.monotonic()
    status=dict(format=FORMAT,phase='starting',complete=False,night_job_finished=False,test_complete=False,
        training_performed=False,optimizer_updates=0,checkpoint_step=2500,source_root=str(SOURCE),output=str(root),
        cases={},historical_failure_resolved=False,original_failed_runs_unchanged=True,started_beijing=now())
    try:
        with recover_entry.leases():
            request=read_request(root)
            for case in CASES:
                remaining=request['total_timeout_seconds']-(time.monotonic()-started)
                if stop['requested'] or remaining<=0:
                    status['phase']='stopped_by_user' if stop['requested'] else 'time_budget_exceeded';break
                if case=='full_test_watched' and not status['cases'].get('prefix_watched',{}).get('accepted'):
                    status['phase']='diagnostics_finished_test_blocked'
                    status['cases'][case]=dict(phase='not_started',reason='Watched622 prefix did not pass all checks')
                    break
                status['phase']=case;write_json(root/'summary.json',status)
                print('\nNIGHT CASE '+case+' '+now(),flush=True)
                record=supervise(root,case,min(request['case_timeout_seconds'],remaining),stop)
                status['cases'][case]=record
                if record['returncode']==0:
                    validate_case_success(root,case,record)
                    record['accepted']=True
                elif record['returncode']==3 or (record['returncode'] not in (2,) and not record['supervisor_reason']):
                    # Configuration/source/integrity faults are not data-case comparisons.
                    status['phase']='aborted';status['error']='Case setup/integrity/process failure; inspect '+case;break
                if case=='full_test_watched':
                    status['test_complete']=record['returncode']==0 and not record['supervisor_reason']
                    status['complete']=status['test_complete']
                    status['phase']='complete' if status['complete'] else 'diagnostics_finished_test_failed'
                write_json(root/'summary.json',status)
            verify_files(request['evidence_files'])
    except BaseException as exc:
        status.update(phase='failed',complete=False,test_complete=False,error=type(exc).__name__+': '+str(exc))
        (root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
    status.update(night_job_finished=True,finished_beijing=now(),elapsed_seconds=round(time.monotonic()-started,1))
    status['diagnostic_failures']=[case for case,record in status['cases'].items() if record.get('returncode',0)!=0]
    write_json(root/'summary.json',status)
    print('NIGHT RESULT '+json.dumps(status,ensure_ascii=False),flush=True)
    return 0 if status['complete'] else 2


def stop_job(root):
    launch=read_json(root/'launcher.json');pid=launch['pid']
    try:
        if proc_start(pid)!=launch['proc_start']:raise ValueError('PID has been reused; no signal sent')
        command=Path('/proc/%s/cmdline'%pid).read_bytes().split(b'\0')
        if str(root/'runner.pyz').encode() not in command or b'worker' not in command:
            raise ValueError('Process does not match this night job')
        os.kill(pid,signal.SIGTERM)
        print('已请求停止本次夜间任务；当前子进程将退出，已有证据保留。')
    except (FileNotFoundError,ProcessLookupError):print('该夜间任务已经退出。')
    return 0


def main():
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','worker','case','logs','status','stop'),nargs='?',default='start')
    parser.add_argument('--root');parser.add_argument('--case',choices=CASES)
    parser.add_argument('--case-timeout',type=int,default=7200,help='per case seconds; default2h')
    parser.add_argument('--total-timeout',type=int,default=28800,help='overall seconds; default8h')
    args=parser.parse_args()
    try:
        if args.command=='start':return launch(args.case_timeout,args.total_timeout)
        own=Path(sys.argv[0]).resolve().parent
        root=(Path(args.root).resolve() if args.root else own if own.parent==BASE and own.name.startswith('test2500_night_')
              else Path((BASE/'latest_test2500_night.txt').read_text().strip()))
        if root.parent!=BASE or not root.name.startswith('test2500_night_'):raise ValueError('Unexpected night directory')
        if args.command=='worker':return coordinate(root)
        if args.command=='case':return run_case(root,args.case)
        if args.command=='stop':return stop_job(root)
        if args.command=='logs':os.execvp('tail',['tail','-n','80','-F',str(root/'night.log')])
        print(json.dumps(read_json(root/'summary.json'),ensure_ascii=False,indent=2))
        if (root/'heartbeat.json').exists():print('最后心跳（时间可能早于任务结束）：',json.dumps(read_json(root/'heartbeat.json'),ensure_ascii=False))
        full=root/'full_test_watched/summary.json'
        if full.exists():
            s=read_json(full)
            if s.get('full_test_evidence'):print('完整Test878指标：',json.dumps(s.get('metrics'),ensure_ascii=False,indent=2))
        return 0
    except Exception as exc:
        print('STOP: '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);return 3


if __name__=='__main__':raise SystemExit(main())
