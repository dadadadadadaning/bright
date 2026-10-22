"""Five fixed checkpoints, fresh process per Test878, bounded unattended execution."""
import argparse
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
from grid_probe import read_json,sha256
from evaluate_checkpoint import write_json
from resume_train import verify_files
from resume_context import inactive_parent
import recover_entry
from night_entry import now,launched_start,supervise,stop_job
from sweep_context import BASE,STEPS,SELECTION,preflight,for_step
from sweep_reports import accept_case,export

FORMAT='BRIGHT_FIVE_STEP_TEST878_SWEEP_V1'
CASES=tuple('test%04d'%s for s in STEPS)


def require_idle():
    p=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                     capture_output=True,text=True,timeout=15)
    if p.returncode or p.stdout.strip():raise RuntimeError('GPU 0 must be idle before launch: '+p.stdout.strip()+p.stderr.strip())
    # Keep the historical CUDA device visibility mapping; protocol verification
    # below rejects changes beyond the declared CUDA_LAUNCH_BLOCKING override.


def read_request(root):
    root=Path(root).resolve()
    if root.parent!=BASE or not root.name.startswith('test_sweep_'):raise ValueError('Unexpected sweep output directory')
    r=read_json(root/'request.json')
    if r.get('format')!=FORMAT or r.get('steps')!=list(STEPS) or r['artifact_sha256']!=sha256(root/'runner.pyz'):
        raise ValueError('Sweep request/archive changed')
    verify_files(r['evidence_files']);return r


def launch(timeout,total):
    if not 60<=timeout<=21600 or not 60<=total<=86400:raise ValueError('Use bounded case/total time budgets')
    archive=Path(sys.argv[0]).resolve()
    if archive.suffix!='.pyz' or not archive.is_file():raise ValueError('Run the published pyz')
    with recover_entry.leases() as locks:
        require_idle();print('PREFLIGHT: verify all five checkpoint hashes and common training trajectory; zero updates',flush=True)
        c=preflight();BASE.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='test_sweep_',dir=BASE));artifact=root/'runner.pyz';shutil.copyfile(archive,artifact)
        request=dict(format=FORMAT,created_beijing=now(),artifact_sha256=sha256(artifact),steps=list(STEPS),
            checkpoints=c['checkpoints'],evidence_files=c['evidence_files'],canonical_test_names=c['canonical_test_names'],
            case_timeout_seconds=timeout,total_timeout_seconds=total,checkpoint_selection=SELECTION,
            optimizer_updates=0,exploratory_test=True,test_used_for_selection=False,existing_test_exposure=True,
            automatic_retries=False,observation_synchronizes=True)
        write_json(root/'request.json',request)
        write_json(root/'summary.json',dict(format=FORMAT,phase='queued',complete=False,job_finished=False,cases={}))
        fds=tuple(f.fileno() for f in locks)
        command=[sys.executable,'-B','-u',str(artifact),'worker','--root',str(root)]
        with (root/'night.log').open('x') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=fds,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                BRIGHT_FINISH_FDS=','.join(map(str,fds))))
        write_json(root/'launcher.json',dict(pid=proc.pid,proc_start=launched_start(proc.pid),command=command))
        (BASE/'latest_test_sweep.txt').write_text(str(root)+'\n')
        print(f'已后台启动500/1000/1500/2000/2500步六分支Test878评估，断开SSH仍继续。\nPID={proc.pid}\nOUTPUT={root}\n日志={root / "night.log"}\n查看结果：{sys.executable} -B {artifact} status',flush=True)
    return 0


def run_case(root,case):
    if case not in CASES:raise ValueError('Unknown fixed step')
    r=read_request(root);output=root/case
    if not output.is_dir() or (output/'summary.json').exists():raise ValueError('Case directory must be fresh')
    c=preflight()
    if c['evidence_files']!=r['evidence_files'] or c['checkpoints']!=r['checkpoints']:
        raise ValueError('Checkpoint inventory changed since launch')
    if c['canonical_test_names']!=r['canonical_test_names']:raise ValueError('Test order changed')
    c=for_step(c,int(case[4:]));c['sweep_artifact_sha256']=r['artifact_sha256']
    from sweep_assessment import run_assessment
    root_parent=Path(c['parent']['path']).parent.parent
    if c['parent']['kind']=='resume':
        with inactive_parent(root_parent):return run_assessment(c,output,case)
    return run_assessment(c,output,case)


def coordinate(root):
    stop={'requested':False};old={s:signal.signal(s,lambda *a:stop.update(requested=True)) for s in (signal.SIGTERM,signal.SIGINT)}
    start=time.monotonic();status=dict(format=FORMAT,phase='starting',complete=False,job_finished=False,
        output=str(root),steps=list(STEPS),cases={},training_performed=False,optimizer_updates=0,
        exploratory_test=True,test_used_for_selection=False,historical_failure_resolved=False,started_beijing=now())
    try:
        with recover_entry.leases():
            r=read_request(root)
            for step,case in zip(STEPS,CASES):
                remaining=r['total_timeout_seconds']-(time.monotonic()-start)
                if stop['requested'] or remaining<=0:
                    status['phase']='stopped_by_user' if stop['requested'] else 'time_budget_exceeded';break
                status['phase']=case;write_json(root/'summary.json',status)
                print('TEST SWEEP '+case+' '+now(),flush=True)
                record=supervise(root,case,min(r['case_timeout_seconds'],remaining),stop)
                status['cases'][case]=record
                if record['returncode']==0 and not record.get('supervisor_reason'):
                    record.update(accept_case(root,step,record,r),accepted=True)
                elif record['returncode']==2 or record.get('supervisor_reason'):
                    record['accepted']=False
                    # Never retry this failed checkpoint or reuse its CUDA process.
                else:
                    status['phase']='aborted';status['error']='Setup, integrity or unexpected process failure at '+case;break
                export(root,status);write_json(root/'summary.json',status)
            else:
                status['complete']=all(status['cases'].get(c,{}).get('accepted') for c in CASES)
                status['phase']='complete' if status['complete'] else 'finished_with_incomplete_steps'
            verify_files(r['evidence_files'])
    except BaseException as exc:
        status.update(complete=False,phase='failed',error=type(exc).__name__+': '+str(exc))
        (root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        for s,handler in old.items():signal.signal(s,handler)
    try:status['comparison']=export(root,status)
    except Exception as exc:status.update(complete=False,phase='report_failed',report_error=str(exc))
    status.update(job_finished=True,finished_beijing=now(),elapsed_seconds=round(time.monotonic()-start,1))
    write_json(root/'summary.json',status)
    print('TEST SWEEP RESULT '+json.dumps(status,ensure_ascii=False),flush=True)
    return 0 if status['complete'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','worker','case','logs','status','report','stop'),nargs='?',default='start')
    parser.add_argument('--root');parser.add_argument('--case',choices=CASES)
    parser.add_argument('--case-timeout',type=int,default=7200)
    parser.add_argument('--total-timeout',type=int,default=36000)
    args=parser.parse_args()
    try:
        if args.command=='start':return launch(args.case_timeout,args.total_timeout)
        own=Path(sys.argv[0]).resolve().parent
        root=Path(args.root).resolve() if args.root else own if own.parent==BASE and own.name.startswith('test_sweep_') else Path((BASE/'latest_test_sweep.txt').read_text().strip())
        if root.parent!=BASE or not root.name.startswith('test_sweep_'):raise ValueError('Unexpected sweep directory')
        if args.command=='worker':return coordinate(root)
        if args.command=='case':return run_case(root,args.case)
        if args.command=='stop':return stop_job(root)
        if args.command=='logs':os.execvp('tail',['tail','-n','80','-F',str(root/'night.log')])
        if args.command=='report':
            print((root/'comparison.md').read_text() if (root/'comparison.md').exists() else '尚无完整评估节点。')
            print('结果目录：'+str(root));return 0
        s=read_json(root/'summary.json')
        print(json.dumps({k:v for k,v in s.items() if k!='comparison'},ensure_ascii=False,indent=2))
        if (root/'heartbeat.json').exists():print('最后心跳：'+json.dumps(read_json(root/'heartbeat.json'),ensure_ascii=False))
        return 0
    except Exception as exc:print('STOP: '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);return 3


if __name__=='__main__':raise SystemExit(main())
