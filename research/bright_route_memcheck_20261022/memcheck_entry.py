"""One bounded, detached Compute Sanitizer run of the original Test prefix356."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from grid_probe import read_json, sha256
from evaluate_checkpoint import write_json, REQUIRED_CHECKS
from resume_train import verify_files
from resume_context import inactive_parent
from night_entry import now, launched_start, stop_child, stop_job
from sweep_entry import require_idle
import recover_entry
from memcheck_context import BASE, CASE, COUNT, FAILED, FORMAT, INDEX, STEP, TARGET, preflight


def tool_info():
    name = shutil.which('compute-sanitizer')
    if not name and Path('/usr/local/cuda/bin/compute-sanitizer').is_file():
        name = '/usr/local/cuda/bin/compute-sanitizer'
    if not name: raise ValueError('compute-sanitizer is not installed; no uninstrumented fallback')
    path = Path(name).resolve()
    r = subprocess.run([str(path), '--version'], capture_output=True, text=True, timeout=15)
    if r.returncode: raise ValueError('compute-sanitizer --version failed: '+r.stdout+r.stderr)
    return dict(path=str(path), sha256=sha256(path), version=(r.stdout+r.stderr).strip())


def validate_root(root):
    root = Path(root).resolve()
    if root.parent != BASE or not root.name.startswith('memcheck1500_'):
        raise ValueError('Unexpected diagnostic output directory')
    return root


def read_request(root):
    root = validate_root(root); r = read_json(root/'request.json')
    if (r.get('format') != FORMAT or r.get('case') != CASE or r.get('checkpoint_step') != STEP
            or r.get('requested_images') != COUNT or r.get('target_image') != TARGET
            or r.get('artifact_sha256') != sha256(root/'runner.pyz')):
        raise ValueError('Diagnostic request/archive changed')
    verify_files(r['evidence_files'])
    if sha256(Path(r['sanitizer']['path'])) != r['sanitizer']['sha256']:
        raise ValueError('Compute Sanitizer executable changed')
    return r


def command_for(root, request):
    return [request['sanitizer']['path'], '--tool', 'memcheck', '--error-exitcode', '86', '--launch-timeout', '0',
        '--print-limit', '20', '--log-file', str(root/CASE/'sanitizer.log'),
        sys.executable, '-B', '-u', str(root/'runner.pyz'), 'case', '--root', str(root)]


def launch(timeout):
    if not 60 <= timeout <= 43200: raise ValueError('Timeout must be 60..43200 seconds')
    archive = Path(sys.argv[0]).resolve()
    if archive.suffix != '.pyz' or not archive.is_file(): raise ValueError('Run the published pyz')
    with recover_entry.leases() as locks:
        require_idle(); sanitizer = tool_info()
        print('PREFLIGHT: fixed checkpoint1500, canonical prefix356, R operation watch, zero updates', flush=True)
        c = preflight(); BASE.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='memcheck1500_', dir=BASE))
        artifact = root/'runner.pyz'; shutil.copyfile(archive, artifact)
        write_json(root/'request.json', dict(format=FORMAT, case=CASE, created_beijing=now(),
            artifact_sha256=sha256(artifact), checkpoint_step=STEP, checkpoint=c['parent'],
            requested_images=COUNT, target_image=TARGET, target_index=INDEX,
            canonical_test_names=c['canonical_test_names'], evidence_files=c['evidence_files'],
            sanitizer=sanitizer, timeout_seconds=timeout, incident_root=str(FAILED),
            optimizer_updates=0, full_test_requested=False, no_retries=True,
            observation_synchronizes=True, instrumentation_changes_execution=True))
        write_json(root/'summary.json', dict(format=FORMAT, phase='queued', job_finished=False,
            diagnostic_complete=False, complete=False, full_test_evidence=False, optimizer_updates=0))
        cmd = [sys.executable, '-B', '-u', str(artifact), 'worker', '--root', str(root)]
        fds = tuple(f.fileno() for f in locks)
        with (root/'diagnostic.log').open('x') as log:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=fds, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                BRIGHT_FINISH_FDS=','.join(map(str, fds))))
        write_json(root/'launcher.json', dict(pid=p.pid, proc_start=launched_start(p.pid), command=cmd))
        (BASE/'latest_memcheck1500.txt').write_text(str(root)+'\n')
        print(f'已后台启动单次诊断；不训练、不补完整Test、不自动重试。\nPID={p.pid}\nOUTPUT={root}\n最多运行{timeout}秒。\n日志：{sys.executable} -B {artifact} logs\n结果：{sys.executable} -B {artifact} status', flush=True)
    return 0


def run_case(root):
    r = read_request(root); folder = root/CASE
    if not folder.is_dir() or (folder/'summary.json').exists(): raise ValueError('Fresh diagnostic case directory required')
    c = preflight()
    if (c['parent'] != r['checkpoint'] or c['evidence_files'] != r['evidence_files']
            or c['canonical_test_names'] != r['canonical_test_names']):
        raise ValueError('Checkpoint/source/Test order changed since launch')
    c['sweep_artifact_sha256'] = r['artifact_sha256']
    from memcheck_assessment import run_assessment
    with inactive_parent(Path(c['parent_journal']).parent):
        return run_assessment(c, folder, CASE)


def sanitizer_summary(path):
    if not path.is_file(): return dict(log_present=False, error_counts=[], zero_errors_reported=False)
    text = path.read_text(errors='replace')
    counts = [int(x) for x in re.findall(r'ERROR SUMMARY:\s*(\d+)\s+errors?\b', text)]
    return dict(log_present=True, error_counts=counts,
                zero_errors_reported=bool(counts) and all(n == 0 for n in counts),
                log=str(path), log_sha256=sha256(path))


def supervise(root, request, stop):
    folder = root/CASE; folder.mkdir(); path = folder/'case.log'
    cmd = command_for(root, request)
    start = time.monotonic(); last = start-30; reason = None
    with path.open('x') as writer:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=writer, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_LAUNCH_BLOCKING='1'))
    write_json(folder/'launcher.json', dict(pid=child.pid, proc_start=launched_start(child.pid), command=cmd))
    try:
        with path.open() as reader:
            while True:
                chunk = reader.read()
                if chunk: print(chunk, end='', flush=True)
                elapsed = time.monotonic()-start
                if child.poll() is not None: break
                if stop['requested'] or elapsed >= request['timeout_seconds']:
                    reason = 'stopped_by_user' if stop['requested'] else 'time_budget_exceeded'
                    stop_child(child); continue
                if time.monotonic()-last >= 30:
                    beat = dict(stage=CASE, child_pid=child.pid, elapsed_seconds=round(elapsed), updated_beijing=now())
                    write_json(root/'heartbeat.json', beat)
                    print('MEMCHECK HEARTBEAT '+json.dumps(beat), flush=True); last=time.monotonic()
                time.sleep(2)
            chunk = reader.read()
            if chunk: print(chunk, end='', flush=True)
    finally: stop_child(child)
    result = dict(returncode=child.returncode, supervisor_reason=reason,
        elapsed_seconds=round(time.monotonic()-start, 1), case_log=str(path),
        sanitizer=sanitizer_summary(folder/'sanitizer.log'))
    write_json(folder/'supervisor.json', result)
    return result


def accept(root, record, request):
    folder = root/CASE; s = read_json(folder/'summary.json')
    if (record['returncode'] != 0 or record.get('supervisor_reason')
            or not record['sanitizer']['zero_errors_reported']
            or s.get('diagnostic_complete') is not True or s.get('complete') is not False
            or s.get('full_test_evidence') is not False or s.get('checkpoint_step') != STEP
            or s.get('image_count') != COUNT or s.get('optimizer_updates') != 0
            or any(s.get('integrity_checks', {}).get(k) is not True for k in REQUIRED_CHECKS)):
        raise ValueError('Diagnostic prefix/sanitizer/integrity evidence incomplete')
    from night_assessment import validate_route_evidence
    validate_route_evidence(folder, request['canonical_test_names'][:COUNT], 'watched')
    report = folder/'test_step_001500/result.json'
    if not report.with_suffix('.json.sha256').read_text().startswith(sha256(report)+' '):
        raise ValueError('Diagnostic report hash changed')
    return dict(diagnostic_result=str(report), diagnostic_result_sha256=sha256(report))


def coordinate(root):
    stop = {'requested':False}
    old = {sig:signal.signal(sig, lambda *args:stop.update(requested=True)) for sig in (signal.SIGTERM, signal.SIGINT)}
    start = time.monotonic()
    status = dict(format=FORMAT, phase='starting', complete=False, diagnostic_complete=False,
        full_test_evidence=False, full_validation_evidence=False, training_performed=False,
        optimizer_updates=0, historical_failure_resolved=False, hardware_fault_proven=False,
        hardware_cleared=False, exact_historical_replay=False, observation_synchronizes=True,
        instrumentation_changes_execution=True, checkpoint_step=STEP, requested_images=COUNT,
        target_image=TARGET, output=str(root), job_finished=False, started_beijing=now())
    try:
        with recover_entry.leases():
            r = read_request(root); status['phase']='memcheck_running'; write_json(root/'summary.json', status)
            record = supervise(root, r, stop); status['execution']=record
            s = read_json(root/CASE/'summary.json') if (root/CASE/'summary.json').is_file() else {}
            status.update(completed_images=s.get('completed_images', 0),
                first_failure=s.get('first_failure'), case_error=s.get('error'), last_position=s.get('last_position'))
            if record.get('supervisor_reason'):
                status['phase']=record['supervisor_reason']
            elif record['returncode'] == 0 and record['sanitizer']['zero_errors_reported']:
                status.update(accept(root, record, r), phase='diagnostic_pass_no_error_observed', diagnostic_complete=True)
            else:
                status['phase']='diagnostic_failed'
            verify_files(r['evidence_files'])
    except BaseException as exc:
        status.update(phase='failed', diagnostic_complete=False, error=type(exc).__name__+': '+str(exc))
        (root/'traceback.txt').write_text(traceback.format_exc())
    finally:
        for sig, handler in old.items(): signal.signal(sig, handler)
    status.update(job_finished=True, finished_beijing=now(), elapsed_seconds=round(time.monotonic()-start, 1))
    write_json(root/'summary.json', status)
    print('MEMCHECK DIAGNOSTIC RESULT '+json.dumps(status, ensure_ascii=False), flush=True)
    return 0 if status['diagnostic_complete'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start','worker','case','logs','status','stop'), nargs='?', default='start')
    parser.add_argument('--root'); parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()
    try:
        if args.command == 'start': return launch(args.timeout)
        own = Path(sys.argv[0]).resolve().parent
        root = validate_root(args.root if args.root else own if own.parent == BASE and own.name.startswith('memcheck1500_')
                             else (BASE/'latest_memcheck1500.txt').read_text().strip())
        if args.command == 'worker': return coordinate(root)
        if args.command == 'case': return run_case(root)
        if args.command == 'stop': return stop_job(root)
        if args.command == 'logs': os.execvp('tail', ['tail','-n','80','-F',str(root/'diagnostic.log')])
        print(json.dumps(read_json(root/'summary.json'), ensure_ascii=False, indent=2))
        for name in ('parent_ids_corroboration.json','operator_window.json'):
            p = root/CASE/name
            if p.is_file():
                d = read_json(p)
                if name == 'operator_window.json': d={k:(d.get(k) or {}).get('position') for k in ('previous','current')}
                print(name+': '+json.dumps(d, ensure_ascii=False, indent=2))
        p = root/CASE/'sanitizer.log'
        if p.is_file():
            from collections import deque
            with p.open(errors='replace') as stream: print('SANITIZER LOG TAIL:\n'+''.join(deque(stream, maxlen=60)))
        return 0
    except Exception as exc:
        print('STOP: '+type(exc).__name__+': '+str(exc), file=sys.stderr, flush=True)
        return 3


if __name__ == '__main__': raise SystemExit(main())
