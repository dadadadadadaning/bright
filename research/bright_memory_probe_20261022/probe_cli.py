"""20-minute isolated CUDA data-integrity probe, independent of BRIGHT."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

BASE=Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS')
LOCKS=(BASE.parent/'SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/active.lock',
       BASE.parent/'SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/evaluation.lock')
FORMAT='BRIGHT_INDEPENDENT_MEMORY_PROBE_V1'


def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smi(*args):
    p=subprocess.run(['nvidia-smi','-i','0',*args],capture_output=True,text=True,timeout=15)
    if p.returncode:raise RuntimeError('nvidia-smi failed: '+p.stderr.strip())
    return p.stdout


def require_idle():
    rows=smi('--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()
    unknown=[row for row in rows if row.strip() and not row.strip().isdigit()]
    if unknown:raise RuntimeError('Could not interpret GPU process listing: '+str(unknown))
    others=[int(row.strip()) for row in rows if row.strip().isdigit() and int(row.strip())!=os.getpid()]
    if others:raise RuntimeError('GPU has existing compute processes; no process was stopped: '+str(others))


@contextlib.contextmanager
def leases():
    with contextlib.ExitStack() as stack:
        inherited=os.environ.pop('BRIGHT_MEMORY_PROBE_FDS','')
        if inherited:streams=[stack.enter_context(os.fdopen(int(x),'a+')) for x in inherited.split(',')]
        else:
            streams=[]
            for p in LOCKS:
                p.parent.mkdir(parents=True,exist_ok=True);streams.append(stack.enter_context(p.open('a+')))
        for f in streams:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield streams


def start_tick(pid):
    s=Path('/proc/%s/stat'%pid).read_text();return s[s.rfind(')')+2:].split()[19]


def launch(seconds,mib):
    if not 30<=seconds<=3600 or not 64<=mib<=2048:raise ValueError('seconds:30..3600; test-buffer MiB:64..2048')
    archive=Path(sys.argv[0]).resolve()
    if archive.suffix!='.pyz':raise ValueError('Use the released pyz')
    with leases() as streams:
        require_idle();BASE.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='memory_probe_',dir=BASE))
        artifact=root/'probe.pyz';artifact.write_bytes(archive.read_bytes())
        uuid=smi('--query-gpu=uuid','--format=csv,noheader,nounits').strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:raise RuntimeError('Cannot identify GPU 0 UUID')
        request=dict(format=FORMAT,created_beijing=stamp(),seconds=seconds,test_buffer_mib=mib,
            gpu_uuid=uuid,artifact_sha256=digest(artifact))
        write(root/'request.json',request)
        write(root/'summary.json',dict(format=FORMAT,phase='queued',complete=False,output=str(root)))
        (root/'nvidia_before.txt').write_text(smi('-q'))
        fds=tuple(f.fileno() for f in streams)
        command=[sys.executable,'-B','-u',str(artifact),'controller','--root',str(root)]
        with (root/'probe.log').open('x') as log:
            p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=fds,
                env=dict(os.environ,BRIGHT_MEMORY_PROBE_FDS=','.join(map(str,fds)),PYTHONDONTWRITEBYTECODE='1'))
        try:tick=start_tick(p.pid)
        except FileNotFoundError:tick=None
        write(root/'launcher.json',dict(pid=p.pid,start_tick=tick))
        (BASE/'latest_memory_probe.txt').write_text(str(root)+'\n')
        print(f'已启动独立数据完整性检查：约{seconds}秒，测试缓冲区合计{mib} MiB。\n不读取模型、检查点或数据集。\nPID={p.pid}\nOUTPUT={root}\n日志={root / "probe.log"}\n查看结果：{sys.executable} -B {artifact} status',flush=True)
    return 0


def validated(root):
    if root.parent!=BASE or not root.name.startswith('memory_probe_'):raise ValueError('Unexpected probe directory')
    r=read(root/'request.json')
    if digest(root/'probe.pyz')!=r['artifact_sha256']:raise ValueError('Probe archive changed')
    return r


def work(root):
    r=validated(root);stop=dict(requested=False)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *args:stop.update(requested=True))
    status=dict(format=FORMAT,phase='initializing',complete=False,model_loaded=False,training_performed=False,
        test_dataset_accessed=False,optimizer_updates=0,output=str(root),started_beijing=stamp(),
        hardware_fault_proven=False,hardware_cleared=False)
    write(root/'worker_summary.json',status)
    try:
        require_idle()
        import torch
        from probe_core import exercise,BufferMismatch,PATTERNS
        torch.set_num_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        free,total=torch.cuda.mem_get_info(0)
        if free<(r['test_buffer_mib']+1024)*1024*1024:raise RuntimeError('Insufficient free GPU memory for buffers plus reserve')
        status['environment']=dict(torch_version=str(torch.__version__),torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),gpu_total_bytes=total,gpu_free_bytes=free,
            cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),patterns=list(PATTERNS))
        status['phase']='checking';write(root/'worker_summary.json',status)
        last_print=[0.0]
        def emit(progress):
            write(root/'progress.json',dict(progress,updated_beijing=stamp()))
            if time.monotonic()-last_print[0]>=30:
                print('MEMORY PROBE '+json.dumps(progress),flush=True);last_print[0]=time.monotonic()
        with torch.inference_mode():
            result=exercise(torch,seconds=r['seconds'],mib=r['test_buffer_mib'],folder=root,stop=stop,emit=emit)
        status.update(result,phase='no_mismatch_observed' if result['all_patterns_completed'] else 'insufficient_pattern_coverage',
            complete=result['all_patterns_completed'])
    except InterruptedError as exc:status.update(phase='stopped',error=str(exc))
    except Exception as exc:
        status.update(phase='data_mismatch' if type(exc).__name__=='BufferMismatch' else 'failed',error=type(exc).__name__+': '+str(exc))
        if hasattr(exc,'record'):status['mismatch']=exc.record
        (root/'traceback.txt').write_text(traceback.format_exc())
        # Do not perform CUDA queries or cleanup probes after a CUDA exception.
    status['finished_beijing']=stamp();write(root/'worker_summary.json',status)
    print('MEMORY PROBE RESULT '+json.dumps(status,ensure_ascii=False),flush=True)
    return 0 if status['complete'] else 2


def kill_child(p):
    if p is None or p.poll() is not None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=5)


def completion(worker,returncode,reason):
    result=dict(worker,returncode=returncode,supervisor_reason=reason)
    if reason or returncode!=0:
        result.update(complete=False,phase=reason or worker.get('phase','worker_failed'))
    elif worker.get('phase')!='no_mismatch_observed' or worker.get('complete') is not True:
        result.update(complete=False,phase='invalid_worker_completion')
    return result


def control(root):
    stop=dict(requested=False);p=None;reason=None;started=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *args:stop.update(requested=True))
    with leases():
        r=validated(root)
        status=dict(format=FORMAT,phase='running',complete=False,output=str(root),started_beijing=stamp(),
            hardware_fault_proven=False,hardware_cleared=False)
        write(root/'summary.json',status)
        try:
            p=subprocess.Popen([sys.executable,'-B','-u',str(root/'probe.pyz'),'worker','--root',str(root)],
                stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=r['gpu_uuid'],CUDA_LAUNCH_BLOCKING='1',PYTHONDONTWRITEBYTECODE='1'))
            last=0.0
            while p.poll() is None:
                elapsed=time.monotonic()-started
                if stop['requested'] or elapsed>r['seconds']+180:
                    reason='stopped_by_user' if stop['requested'] else 'wall_time_limit';kill_child(p);break
                if elapsed-last>=30:
                    heartbeat=dict(elapsed_seconds=round(elapsed),child_pid=p.pid,updated_beijing=stamp())
                    write(root/'heartbeat.json',heartbeat);last=elapsed
                time.sleep(1)
            worker=read(root/'worker_summary.json') if (root/'worker_summary.json').exists() else {}
            status.update(completion(worker,p.returncode,reason))
        except BaseException as exc:
            status.update(complete=False,phase='supervisor_failed',error=type(exc).__name__+': '+str(exc))
            (root/'supervisor_traceback.txt').write_text(traceback.format_exc())
        finally:kill_child(p)
        try:(root/'nvidia_after.txt').write_text(smi('-q'))
        except Exception as exc:status['nvidia_after_error']=str(exc)
        status.update(job_finished=True,finished_beijing=stamp(),elapsed_wall_seconds=round(time.monotonic()-started,1))
        write(root/'summary.json',status)
        print('PROBE SUPERVISOR RESULT '+json.dumps(status,ensure_ascii=False),flush=True)
    return 0 if status['complete'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','controller','worker','logs','status','stop'),nargs='?',default='start')
    parser.add_argument('--root');parser.add_argument('--seconds',type=int,default=1200)
    parser.add_argument('--mib',type=int,default=1024)
    args=parser.parse_args()
    try:
        if args.command=='start':return launch(args.seconds,args.mib)
        own=Path(sys.argv[0]).resolve().parent
        root=Path(args.root).resolve() if args.root else own if own.parent==BASE and own.name.startswith('memory_probe_') else Path((BASE/'latest_memory_probe.txt').read_text().strip())
        validated(root)
        if args.command=='controller':return control(root)
        if args.command=='worker':return work(root)
        if args.command=='logs':os.execvp('tail',['tail','-n','60','-F',str(root/'probe.log')])
        if args.command=='status':
            print(json.dumps(read(root/'summary.json'),ensure_ascii=False,indent=2))
            if (root/'progress.json').exists():print('LAST PROGRESS '+json.dumps(read(root/'progress.json'),ensure_ascii=False))
            return 0
        launcher=read(root/'launcher.json');pid=launcher['pid']
        try:
            if start_tick(pid)!=launcher['start_tick']:raise RuntimeError('PID reused; no signal sent')
            cmd=Path('/proc/%s/cmdline'%pid).read_bytes().split(b'\0')
            if str(root/'probe.pyz').encode() not in cmd or b'controller' not in cmd:raise RuntimeError('Process mismatch; no signal sent')
            os.kill(pid,signal.SIGTERM);print('已请求停止本次独立检查，已有证据保留。')
        except (FileNotFoundError,ProcessLookupError):print('该检查已经退出。')
        return 0
    except Exception as exc:print('STOP: '+type(exc).__name__+': '+str(exc),file=sys.stderr);return 3


if __name__=='__main__':raise SystemExit(main())
