"""Bounded two-image plain/memcheck allocator comparison; no training or repair."""
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
from sweep_entry import require_idle
from night_entry import now, launched_start, stop_child, stop_job
from memcheck_entry import tool_info, sanitizer_summary
from memory_context import BASE, CASES, COUNT, FAILED, FORMAT, STEP, preflight
import recover_entry

PREFIX = 'memory1500_'


def headings(path):
    text = path.read_text(errors='replace')
    return [line for line in text.splitlines() if re.match(
        r'^========= (Program hit|Invalid|Misaligned|Uninitialized|Leaked|Hardware|ERROR SUMMARY|Error:)', line)]


def validate_root(root):
    root = Path(root).resolve()
    if root.parent != BASE or not root.name.startswith(PREFIX):
        raise ValueError('Unexpected memory diagnostic output directory')
    return root


def read_request(root):
    r = read_json(root / 'request.json')
    if (r.get('format') != FORMAT or r.get('cases') != list(CASES)
            or r.get('requested_images') != COUNT or r.get('checkpoint_step') != STEP
            or r.get('artifact_sha256') != sha256(root / 'runner.pyz')):
        raise ValueError('Diagnostic request/archive changed')
    verify_files(r['evidence_files'])
    if sha256(Path(r['sanitizer']['path'])) != r['sanitizer']['sha256']:
        raise ValueError('Sanitizer executable changed')
    return r


def launch(timeout):
    if not 60 <= timeout <= 1800:
        raise ValueError('Use 60..1800 seconds per case; default600')
    archive = Path(sys.argv[0]).resolve()
    if archive.suffix != '.pyz' or not archive.is_file():
        raise ValueError('Run the published pyz')
    with recover_entry.leases() as locks:
        require_idle()
        sanitizer = tool_info()
        c = preflight()
        old = FAILED / 'prefix1500_memcheck/sanitizer.log'
        old_titles = headings(old)
        errors = [s for s in old_titles if 'ERROR SUMMARY' not in s]
        if (len(errors) != 6 or sanitizer_summary(old)['error_counts'] != [6]
                or any('Program hit cudaErrorMemoryAllocation' not in s for s in errors)):
            raise ValueError('Prior sanitizer log is not the six-allocation-error case: ' + '\n'.join(old_titles))
        root = Path(tempfile.mkdtemp(prefix=PREFIX, dir=BASE))
        artifact = root / 'runner.pyz'
        shutil.copyfile(archive, artifact)
        write_json(root / 'request.json', dict(format=FORMAT, cases=list(CASES),
            artifact_sha256=sha256(artifact), created_beijing=now(), checkpoint_step=STEP,
            checkpoint=c['parent'], requested_images=COUNT, names=c['canonical_test_names'][:COUNT],
            canonical_test_names=c['canonical_test_names'], evidence_files=c['evidence_files'],
            sanitizer=sanitizer, timeout_seconds=timeout, prior_sanitizer_headings=old_titles,
            allocator_environment={k: os.environ.get(k) for k in (
                'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING', 'CUDA_MODULE_LOADING')},
            optimizer_updates=0, automatic_retries=False, full_test_requested=False))
        write_json(root / 'summary.json', dict(format=FORMAT, phase='queued', job_finished=False,
            complete=False, diagnostic_complete=False, full_test_evidence=False, optimizer_updates=0))
        cmd = [sys.executable, '-B', '-u', str(artifact), 'worker', '--root', str(root)]
        fds = tuple(f.fileno() for f in locks)
        with (root / 'diagnostic.log').open('x') as log:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=fds, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                BRIGHT_FINISH_FDS=','.join(map(str, fds))))
        write_json(root / 'launcher.json', dict(pid=p.pid, proc_start=launched_start(p.pid), command=cmd))
        (BASE / 'latest_memory1500.txt').write_text(str(root) + '\n')
        print(f'已后台启动两图显存对照：普通进程通过后，再运行Sanitizer进程。每段最多{timeout}秒。\n'
              f'PID={p.pid}\nOUTPUT={root}\n'
              f'日志：{sys.executable} -B {artifact} logs\n结果：{sys.executable} -B {artifact} status', flush=True)
    return 0


def command_for(root, case, request):
    cmd = [sys.executable, '-B', '-u', str(root / 'runner.pyz'), 'case', '--root', str(root), '--case', case]
    if case == 'memcheck':
        cmd = [request['sanitizer']['path'], '--tool', 'memcheck', '--error-exitcode', '86',
            '--launch-timeout', '0', '--print-limit', '20', '--log-file', str(root / case / 'sanitizer.log')] + cmd
    return cmd


def supervise(root, case, request, stop):
    folder = root / case
    folder.mkdir()
    path = folder / 'case.log'
    cmd = command_for(root, case, request)
    start = time.monotonic()
    last = start - 30
    reason = None
    with path.open('x') as writer:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=writer, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_LAUNCH_BLOCKING='1'))
    write_json(folder / 'launcher.json', dict(pid=child.pid, proc_start=launched_start(child.pid), command=cmd))
    try:
        with path.open() as reader:
            while True:
                chunk = reader.read()
                if chunk:
                    print(chunk, end='', flush=True)
                elapsed = time.monotonic() - start
                if child.poll() is not None:
                    break
                if stop['requested'] or elapsed >= request['timeout_seconds']:
                    reason = 'stopped_by_user' if stop['requested'] else 'time_budget_exceeded'
                    stop_child(child)
                    continue
                if time.monotonic() - last >= 30:
                    beat = dict(case=case, child_pid=child.pid, elapsed_seconds=round(elapsed), updated_beijing=now())
                    write_json(root / 'heartbeat.json', beat)
                    print('MEMORY HEARTBEAT ' + json.dumps(beat), flush=True)
                    last = time.monotonic()
                time.sleep(1)
            chunk = reader.read()
            if chunk:
                print(chunk, end='', flush=True)
    finally:
        stop_child(child)
    record = dict(returncode=child.returncode, supervisor_reason=reason,
        elapsed_seconds=round(time.monotonic() - start, 1), case_log=str(path))
    if case == 'memcheck':
        record['sanitizer'] = sanitizer_summary(folder / 'sanitizer.log')
        record['sanitizer_headings'] = headings(folder / 'sanitizer.log') if (folder / 'sanitizer.log').is_file() else []
    write_json(folder / 'supervisor.json', record)
    return record


def run_case(root, case):
    r = read_request(root)
    folder = root / case
    if not folder.is_dir() or (folder / 'summary.json').exists():
        raise ValueError('Fresh case directory required')
    c = preflight()
    if (c['parent'] != r['checkpoint'] or c['evidence_files'] != r['evidence_files']
            or c['canonical_test_names'] != r['canonical_test_names']):
        raise ValueError('Inputs changed since launch')
    c['sweep_artifact_sha256'] = r['artifact_sha256']
    from memory_assessment import run_assessment
    with inactive_parent(Path(c['parent_journal']).parent):
        return run_assessment(c, folder, case)


def memory_digest(folder):
    p = folder / 'memory.jsonl'
    if not p.is_file():
        return dict(rows=0)
    rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    valid = [r for r in rows if 'allocated_bytes' in r]
    summary = dict(rows=len(rows), last=rows[-1] if rows else None, file=str(p), sha256=sha256(p))
    if valid:
        biggest = max(valid, key=lambda r: r['allocated_bytes'])
        summary.update(max_observed_allocated_gib=biggest['allocated_bytes'] / 2**30,
            max_observed_reserved_gib=max(r['reserved_bytes'] for r in valid) / 2**30,
            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in valid) / 2**30,
            at_max=dict(stage=biggest['stage'], position=biggest['position']),
            last_successful_sample=valid[-1])
    return summary


def accept(root, case, record, request):
    folder = root / case
    s = read_json(folder / 'summary.json') if (folder / 'summary.json').is_file() else {}
    good = (record['returncode'] == 0 and not record.get('supervisor_reason')
        and s.get('diagnostic_complete') is True and s.get('complete') is False
        and s.get('image_count') == COUNT and s.get('checkpoint_step') == STEP
        and s.get('optimizer_updates') == 0 and s.get('full_test_evidence') is False
        and all(s.get('integrity_checks', {}).get(k) is True for k in REQUIRED_CHECKS))
    if good:
        from night_assessment import validate_route_evidence
        validate_route_evidence(folder, request['names'], 'watched')
        result = folder / 'test_step_001500/result.json'
        if not result.with_suffix('.json.sha256').read_text().startswith(sha256(result) + ' '):
            raise ValueError('Diagnostic report hash changed')
    if case == 'memcheck':
        good = good and record.get('sanitizer', {}).get('zero_errors_reported') is True
    return dict(record, accepted=bool(good), phase=s.get('phase'), error=s.get('error'),
        completed_images=s.get('completed_images', 0), memory=memory_digest(folder))


def coordinate(root):
    stop = {'requested': False}
    old = {sig: signal.signal(sig, lambda *args: stop.update(requested=True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    status = dict(format=FORMAT, phase='starting', complete=False, diagnostic_complete=False,
        job_finished=False, full_test_evidence=False, full_validation_evidence=False,
        training_performed=False, optimizer_updates=0, historical_failure_resolved=False,
        hardware_fault_proven=False, hardware_cleared=False, checkpoint_step=STEP,
        images_per_case=COUNT, observation_changes_execution=True, cases={}, started_beijing=now())
    try:
        with recover_entry.leases():
            request = read_request(root)
            for case in CASES:
                if stop['requested']:
                    status['phase'] = 'stopped_by_user'
                    break
                require_idle()
                status['phase'] = case + '_running'
                write_json(root / 'summary.json', status)
                record = supervise(root, case, request, stop)
                status['cases'][case] = accept(root, case, record, request)
                if not status['cases'][case]['accepted']:
                    status.update(phase='diagnostic_incomplete', failed_case=case)
                    if case == 'plain':
                        status['memcheck_skipped'] = 'Plain two-image execution did not pass; no instrumented retry'
                    break
            else:
                status.update(phase='two_image_comparison_complete', diagnostic_complete=True,
                    conclusion='Both short cases passed. This does not reproduce the index355 incident or clear hardware.')
            verify_files(request['evidence_files'])
    except BaseException as exc:
        status.update(phase='failed', error=type(exc).__name__ + ': ' + str(exc))
        (root / 'traceback.txt').write_text(traceback.format_exc())
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    status.update(job_finished=True, finished_beijing=now())
    write_json(root / 'summary.json', status)
    print('MEMORY COMPARISON RESULT ' + json.dumps(status, ensure_ascii=False), flush=True)
    return 0 if status['diagnostic_complete'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'worker', 'case', 'logs', 'status', 'stop'), nargs='?', default='start')
    parser.add_argument('--root')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--timeout', type=int, default=600, help='per-case seconds; default600')
    args = parser.parse_args()
    try:
        if args.command == 'start':
            return launch(args.timeout)
        own = Path(sys.argv[0]).resolve().parent
        root = validate_root(args.root if args.root else own if own.parent == BASE and own.name.startswith(PREFIX)
                             else (BASE / 'latest_memory1500.txt').read_text().strip())
        if args.command == 'worker':
            return coordinate(root)
        if args.command == 'case':
            if args.case is None:
                raise ValueError('--case is required')
            return run_case(root, args.case)
        if args.command == 'stop':
            return stop_job(root)
        if args.command == 'logs':
            os.execvp('tail', ['tail', '-n', '60', '-F', str(root / 'diagnostic.log')])
        print(json.dumps(read_json(root / 'summary.json'), ensure_ascii=False, indent=2))
        for case in CASES:
            p = root / case / 'memory_latest.json'
            if p.is_file():
                print(case + ' latest memory: ' + json.dumps(read_json(p), ensure_ascii=False))
            p = root / case / 'sanitizer.log'
            if p.is_file():
                print('\n'.join(headings(p)))
        return 0
    except Exception as exc:
        print('STOP: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
