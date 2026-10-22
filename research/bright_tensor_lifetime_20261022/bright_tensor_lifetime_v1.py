"""Bounded synthetic tensor-lifetime comparison. No BRIGHT model/data/checkpoints."""
import argparse
import contextlib
import datetime
import fcntl
import gc
import hashlib
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
import weakref

BASE = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS')
LOCKS = (BASE.parent / 'SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/active.lock',
         BASE.parent / 'SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/evaluation.lock')
FORMAT = 'BRIGHT_SYNTHETIC_TENSOR_LIFETIME_V1'
CASES = ('plain', 'memcheck')
ROUNDS = 8
ELEMENTS = 2 * 1024 * 1024  # 8 MiB per float32 tensor; three temporary tensors.


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def idle():
    p = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if p.returncode or p.stdout.strip():
        raise RuntimeError('GPU 0 must be idle: ' + p.stdout.strip() + p.stderr.strip())


@contextlib.contextmanager
def leases():
    with contextlib.ExitStack() as stack:
        inherited = os.environ.pop('BRIGHT_LIFETIME_FDS', '')
        if inherited:
            streams = [stack.enter_context(os.fdopen(int(fd), 'a+')) for fd in inherited.split(',')]
        else:
            streams = []
            for path in LOCKS:
                path.parent.mkdir(parents=True, exist_ok=True)
                streams.append(stack.enter_context(path.open('a+')))
        for stream in streams:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield streams


def read_request(root):
    if root.parent != BASE or not root.name.startswith('tensor_lifetime_'):
        raise ValueError('Unexpected output directory')
    r = json.loads((root / 'request.json').read_text())
    if r['format'] != FORMAT or digest(root / 'runner.py') != r['script_sha256']:
        raise ValueError('Diagnostic script/request mismatch')
    if digest(Path(r['sanitizer'])) != r['sanitizer_sha256']:
        raise ValueError('Sanitizer executable changed')
    return r


def start():
    with leases() as locks:
        idle()
        tool = shutil.which('compute-sanitizer') or '/usr/local/cuda/bin/compute-sanitizer'
        tool = str(Path(tool).resolve(strict=True))
        version = subprocess.run([tool, '--version'], capture_output=True, text=True, timeout=15, check=True)
        BASE.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='tensor_lifetime_', dir=BASE))
        shutil.copyfile(Path(__file__).resolve(), root / 'runner.py')
        save(root / 'request.json', dict(format=FORMAT, created=stamp(), cases=list(CASES),
            script_sha256=digest(root / 'runner.py'), sanitizer=tool, sanitizer_sha256=digest(Path(tool)),
            sanitizer_version=version.stdout + version.stderr, timeout_per_case_seconds=180,
            rounds=ROUNDS, elements_per_tensor=ELEMENTS, gc_after_rounds=[4, 8],
            automatic_retries=False, test_accessed=False, training_performed=False,
            environment={k: os.environ.get(k) for k in (
                'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING',
                'TORCH_CUDA_SANITIZER', 'CUDA_MODULE_LOADING', 'PYTHONPATH')}))
        save(root / 'summary.json', dict(format=FORMAT, phase='queued', job_finished=False,
                                        diagnostic_complete=False, cases={}))
        fds = tuple(s.fileno() for s in locks)
        with (root / 'diagnostic.log').open('x') as log:
            p = subprocess.Popen([sys.executable, '-B', '-u', str(root / 'runner.py'), 'worker', '--root', str(root)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=fds, env=dict(os.environ, BRIGHT_LIFETIME_FDS=','.join(map(str, fds)), PYTHONDONTWRITEBYTECODE='1'))
        save(root / 'launcher.json', dict(pid=p.pid, created=stamp()))
        (BASE / 'latest_tensor_lifetime.txt').write_text(str(root) + '\n')
        print(f'已后台启动基础张量释放对照：不加载模型、数据或检查点。每段最多180秒。\nPID={p.pid}\nOUTPUT={root}\n'
              f'日志：{sys.executable} -B {root / "runner.py"} logs\n'
              f'结果：{sys.executable} -B {root / "runner.py"} status', flush=True)


def one_round(torch, device, elements=ELEMENTS):
    # A local scope deliberately drops every strong Python reference on return.
    # Return weak references only, never tensors, views, outputs or closures.
    with torch.no_grad():
        x = torch.full((elements,), 0.25, device=device, dtype=torch.float32)
        y = x + 0.5
        z = y.square()
        if float(z[0].item()) != 0.5625 or float(z[-1].item()) != 0.5625:
            raise ValueError('Synthetic arithmetic result differs')
        return tuple(weakref.ref(t) for t in (x, y, z))


def sample(torch, stage, index, refs):
    torch.cuda.synchronize()
    s = torch.cuda.memory_stats()
    return dict(stage=stage, round=index, allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        allocation_count=s.get('allocation.all.current'), allocated_total=s.get('allocated_bytes.all.allocated'),
        freed_total=s.get('allocated_bytes.all.freed'), active_bytes=s.get('active_bytes.all.current'),
        inactive_split_bytes=s.get('inactive_split_bytes.all.current'),
        python_live_weakrefs=sum(r() is not None for r in refs), grad_enabled=torch.is_grad_enabled(),
        gc_enabled=gc.isenabled(), gc_counts=list(gc.get_count()))


def case(root, name):
    read_request(root)
    idle()
    folder = root / name
    result = dict(format=FORMAT, case=name, complete=False, training_performed=False,
                  test_accessed=False, optimizer_updates=0, rows=[])
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no CPU fallback')
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        result['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0), python=sys.version,
            allocator_backend=torch.cuda.get_allocator_backend(), trace_installed=sys.gettrace() is not None,
            profile_installed=sys.getprofile() is not None, gc_thresholds=list(gc.get_threshold()))
        refs = []
        def record(stage, index):
            row = sample(torch, stage, index, refs)
            result['rows'].append(row)
            save(folder / 'summary.json', result)
            print('LIFETIME ' + json.dumps(row), flush=True)
            if row['allocated_bytes'] > 1024 ** 3:
                raise RuntimeError('Synthetic diagnostic exceeded 1 GiB allocated-byte budget')
        with torch.no_grad():
            record('baseline', 0)
            for index in range(1, ROUNDS + 1):
                refs.extend(one_round(torch, 'cuda:0'))
                record('after_scope_return', index)
                if index in (4, 8):
                    collected = gc.collect()
                    result.setdefault('gc_events', []).append(dict(round=index, collected=collected))
                    record('after_explicit_gc', index)
        result.update(complete=True, phase='synthetic_case_complete')
    except BaseException as exc:
        # No CUDA reads after an exception. Retain the last successful sample.
        result.update(phase='failed', error=type(exc).__name__ + ': ' + str(exc))
        (folder / 'traceback.txt').write_text(traceback.format_exc())
    save(folder / 'summary.json', result)
    return 0 if result['complete'] else 1


def command(root, name, request):
    cmd = [sys.executable, '-B', '-u', str(root / 'runner.py'), 'case', '--root', str(root), '--case', name]
    if name == 'memcheck':
        cmd = [request['sanitizer'], '--tool', 'memcheck', '--error-exitcode', '86', '--launch-timeout', '0',
               '--print-limit', '20', '--log-file', str(root / name / 'sanitizer.log')] + cmd
    return cmd


def kill_child(p):
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=5)
        except ProcessLookupError:
            pass


def supervise(root, name, request, stop):
    folder = root / name
    folder.mkdir()
    start_time = time.monotonic()
    reason = None
    cmd = command(root, name, request)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_LAUNCH_BLOCKING='1', PYTHONDONTWRITEBYTECODE='1')
    with (folder / 'case.log').open('x') as log:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env)
    save(folder / 'launcher.json', dict(pid=p.pid, command=cmd))
    last_heartbeat = -30
    try:
        while p.poll() is None:
            elapsed = time.monotonic() - start_time
            if stop['requested'] or elapsed >= request['timeout_per_case_seconds']:
                reason = 'stopped_by_user' if stop['requested'] else 'time_budget_exceeded'
                kill_child(p)
                break
            if elapsed - last_heartbeat >= 30:
                print('LIFETIME HEARTBEAT ' + json.dumps(dict(case=name, elapsed_seconds=round(elapsed))), flush=True)
                last_heartbeat = elapsed
            time.sleep(0.5)
    finally:
        kill_child(p)
    path = folder / 'summary.json'
    s = json.loads(path.read_text()) if path.is_file() else {}
    record = dict(returncode=p.returncode, supervisor_reason=reason,
                  elapsed_seconds=round(time.monotonic() - start_time, 1), result=s)
    if name == 'memcheck':
        path = folder / 'sanitizer.log'
        content = path.read_text(errors='replace') if path.is_file() else ''
        counts = [int(x) for x in re.findall(r'^========= ERROR SUMMARY: (\d+) errors?', content, flags=re.M)]
        record.update(sanitizer_error_counts=counts, sanitizer_headings=[l for l in content.splitlines()
            if re.match(r'^========= (Program hit|Invalid|Misaligned|Uninitialized|Leaked|Hardware|ERROR SUMMARY|Error:)', l)])
    record['accepted'] = (p.returncode == 0 and reason is None and s.get('complete') is True
                          and (name == 'plain' or record['sanitizer_error_counts'] == [0]))
    return record


def worker(root):
    stop = {'requested': False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(requested=True))
    result = dict(format=FORMAT, phase='starting', job_finished=False, diagnostic_complete=False,
        cases={}, test_accessed=False, training_performed=False, optimizer_updates=0,
        historical_failure_resolved=False, hardware_fault_proven=False, hardware_cleared=False,
        notes='Synthetic lifetime probe only; explicit GC at rounds4/8 changes execution in both cases.')
    try:
        with leases():
            r = read_request(root)
            for name in CASES:
                if stop['requested']:
                    result['phase'] = 'stopped_by_user'
                    break
                result['phase'] = name + '_running'
                save(root / 'summary.json', result)
                record = supervise(root, name, r, stop)
                result['cases'][name] = record
                if not record['accepted']:
                    result.update(phase='diagnostic_incomplete', failed_case=name)
                    break
            else:
                result.update(phase='synthetic_pair_complete', diagnostic_complete=True)
    except BaseException as exc:
        result.update(phase='diagnostic_incomplete', error=type(exc).__name__ + ': ' + str(exc))
        (root / 'traceback.txt').write_text(traceback.format_exc())
    result['job_finished'] = True
    save(root / 'summary.json', result)
    print('LIFETIME COMPARISON RESULT ' + json.dumps(result), flush=True)
    return 0 if result['diagnostic_complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'worker', 'case', 'logs', 'status'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.action == 'start':
        return start()
    own = Path(__file__).resolve().parent
    root = (args.root or (own if own.parent == BASE and own.name.startswith('tensor_lifetime_')
                         else Path((BASE / 'latest_tensor_lifetime.txt').read_text().strip()))).resolve()
    read_request(root)
    if args.action == 'worker':
        return worker(root)
    if args.action == 'case':
        if not args.case:
            parser.error('--case is required')
        return case(root, args.case)
    if args.action == 'logs':
        os.execvp('tail', ['tail', '-n', '60', '-F', str(root / 'diagnostic.log')])
    print((root / 'summary.json').read_text())


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except Exception as exc:
        print('STOP: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
