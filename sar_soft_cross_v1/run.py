"""Queue matched baseline-Q/SAR-Q x four route heads; preserve all old evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
FORMAT = 'SAR_SOFT_CROSS_PIPELINE_V1'
SOURCE_FORMAT = 'BRIGHT_SAR_Q_MATCHED_V1'
DEFAULT_REPO = '/root/shared-nvme/BRIGHT-master'
SOURCE_BASE = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1'
ROUTES = ('unsplit', 'capacity_control', 'hard', 'soft')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def select_source(explicit=None, base=SOURCE_BASE):
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise ValueError('指定SAR结果目录不存在：' + str(root))
        return root
    candidates = []
    parent = Path(base)
    for folder in sorted(parent.glob('run_*')):
        if not folder.is_dir():
            continue
        try:
            if read(folder / 'manifest.json').get('format') == SOURCE_FORMAT:
                candidates.append(folder.resolve())
        except (OSError, ValueError):
            continue
    if not candidates:
        raise ValueError('未找到新版SAR四组任务。请先启动 bright_sar_q_matched_v1.zip 中的 start_server.sh；组合实验不使用旧的不完整SAR结果。')
    # Lock the latest declared run once. Never silently fall back after failure.
    return candidates[-1]


def wait_source(root, log=print, poll_seconds=30):
    while True:
        path = Path(root) / 'workflow_state.json'
        if not path.is_file():
            raise ValueError('SAR来源没有workflow_state.json：' + str(path))
        state = read(path)
        if state.get('format') != SOURCE_FORMAT:
            raise ValueError('Unexpected source experiment format')
        if state.get('failure'):
            raise ValueError('SAR四组来源已失败，组合任务停止：' + str(state['failure']))
        if state.get('complete') is True:
            if state.get('full_validation_evidence') is not True:
                raise ValueError('SAR source finished without full formal evidence')
            return
        log('WAITING_SAR_SOURCE ' + str(state.get('current_stage')), flush=True)
        time.sleep(poll_seconds)


def source_files(repo):
    files = {str(p): digest(p) for p in HERE.iterdir()
             if p.is_file() and p.suffix in ('.py', '.sh') and not p.name.startswith('test_')}
    experiments = Path(repo) / 'bda_benchmark/experiments'
    for package in ('damageformer_srac_building_route_v1', 'damageformer_srac_q_factorial_v1'):
        folder = experiments / package
        if not folder.is_dir():
            raise ValueError('缺少原实验代码：' + str(folder))
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ('.py', '.sh') and not p.name.startswith('test_'):
                files[str(p)] = digest(p)
    return files


def plan(args):
    root = Path(args.output)
    certificate = root / 'source_certificate.json'
    stages = [dict(name='verify_sar_source', kind='certificate', output=str(certificate),
        command=[args.python, '-u', str(HERE / 'prerequisites.py'), 'check', '--sar-root', str(args.sar_root),
                 '--repo-root', str(args.repo_root), '--output', str(certificate)]),
        dict(name='original_route_preflight', kind='preflight', command=[
            'bash', str(Path(args.repo_root) / 'bda_benchmark/experiments/damageformer_srac_building_route_v1/run.sh'), 'preflight'])]
    for smoke in (True, False):
        scope = 'smoke' if smoke else 'formal'
        for arm in ('baseline', 'sar_cue'):
            fit = root / scope / ('fit_' + arm)
            evaluation = root / scope / ('val_' + arm)
            for action, output in (('train', fit), ('evaluate', evaluation)):
                command = [args.python, '-u', str(HERE / 'worker.py'), action,
                    '--repo-root', str(args.repo_root), '--certificate', str(certificate),
                    '--q-arm', arm, '--output', str(output)]
                if action == 'evaluate':
                    command += ['--fit', str(fit)]
                if smoke:
                    command.append('--smoke')
                stages.append(dict(name=scope + '_' + action + '_' + arm, kind=action,
                    command=command, output=str(output), smoke=smoke, arm=arm))
    output = root / 'comparison'
    stages.append(dict(name='compare_eight_cells', kind='compare', output=str(output),
        command=[args.python, '-u', str(HERE / 'compare_routes.py'),
            '--baseline-fit', str(root / 'formal/fit_baseline'),
            '--baseline-evaluation', str(root / 'formal/val_baseline'),
            '--sar-fit', str(root / 'formal/fit_sar_cue'),
            '--sar-evaluation', str(root / 'formal/val_sar_cue'), '--output', str(output)]))
    return stages


def child(stage, repo, env, log):
    print('STAGE=' + stage['name'], flush=True)
    print('COMMAND=' + shlex.join(stage['command']), flush=True)
    with Path(log).open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(stage['command'], cwd=repo, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line); stream.flush()
                print(line, end='', flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise


def stage_gate(stage):
    if stage['kind'] == 'preflight':
        return {'passed': True, 'scope': 'CPU contracts only'}
    out = Path(stage['output'])
    if stage['kind'] == 'certificate':
        data = read(out)
        if (data.get('format') != 'SAR_SOFT_SOURCE_CERTIFICATE_V1'
                or data.get('complete') is not True or data.get('full_validation_evidence') is not True):
            raise ValueError('SAR source admission did not pass')
        return {'passed': True, 'sha256': digest(out)}
    if stage['kind'] == 'compare':
        data = read(out / 'report.json')
        if (data.get('format') != 'SRAC_SAR_SOFT_CROSS_COMPARISON_V1'
                or data.get('complete') is not True or data.get('full_validation_evidence') is not True
                or data.get('test_accessed') is not False):
            raise ValueError('Eight-cell comparison did not pass')
        return {'passed': True, 'report_sha256': digest(out / 'report.json')}
    data = read(out / 'summary.json')
    worker = read(out / 'worker_record.json')
    smoke = stage['smoke']
    expected_format = 'SRAC_SAR_SOFT_ROUTE_V1' + ('_EVAL' if stage['kind'] == 'evaluate' else '')
    if (data.get('format') != expected_format or worker.get('format') != 'SAR_SOFT_WORKER_V1'
            or data.get('complete') is not True or data.get('smoke') is not smoke
            or data.get('full_validation_evidence') is not (not smoke)
            or data.get('test_accessed') is not False or worker.get('complete') is not True
            or worker.get('q_arm') != stage['arm'] or worker.get('action') != stage['kind']
            or worker.get('smoke') is not smoke
            or worker.get('summary_sha256') != digest(out / 'summary.json')):
        raise ValueError('Worker or formal-stage gate failed: ' + stage['name'])
    if stage['kind'] == 'train':
        steps = 2 if smoke else 2000
        if data.get('steps') != steps or data.get('updates_per_arm') != {arm: steps for arm in ROUTES}:
            raise ValueError('Not all four route heads completed the fixed update budget')
    elif data.get('image_count') != (2 if smoke else 450):
        raise ValueError('Independent evaluation image count differs')
    return {'passed': True, 'summary_sha256': digest(out / 'summary.json'),
            'worker_record_sha256': digest(out / 'worker_record.json')}


def report(root):
    root = Path(root)
    state_path = root / 'workflow_state.json'
    state = read(state_path) if state_path.is_file() else {}
    result = dict(format=FORMAT, root=str(root), complete=state.get('complete') is True,
        full_validation_evidence=state.get('full_validation_evidence') is True,
        current_stage=state.get('current_stage'), failure=state.get('failure'),
        source_root=state.get('source_root'), test_accessed=False)
    file = root / 'comparison/report.json'
    if file.is_file():
        try:
            result['comparison'] = read(file)
        except (OSError, ValueError) as error:
            result['comparison_read_error'] = str(error)
    result['note'] = 'Complete describes valid matched evidence, not positive gain. Metric: Val foreground classes 1/2/3, no Test or new Memory training.'
    return result


def run(args):
    args = argparse.Namespace(**vars(args))
    repo, root = Path(args.repo_root).resolve(), Path(args.output).resolve()
    source = select_source(args.sar_root, args.source_base)
    if root.exists() or any(root.is_relative_to(p) for p in (repo, HERE, source)):
        raise ValueError('Use a fresh output outside the original repository, tool and source run')
    interpreter = shutil.which(args.python)
    if not interpreter:
        raise ValueError('Python interpreter unavailable')
    args.python = os.path.abspath(interpreter)
    args.repo_root, args.output, args.sar_root = str(repo), str(root), str(source)
    before = source_files(repo)
    stages = plan(args)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    manifest = dict(format=FORMAT, arguments=vars(args), stages=stages, source_bindings=before,
        train_protocol='Two frozen, matched 2000-update Q sources; four independently trained route heads each; 2000 full-image updates/head; last step only',
        q_sources=['baseline', 'sar_cue'], routes=list(ROUTES), test_accessed=False,
        historical_result_reuse=False, memory_training=False)
    write(root / 'manifest.json', manifest)
    manifest_sha = digest(root / 'manifest.json')
    state = dict(format=FORMAT, complete=False, full_validation_evidence=False,
        current_stage='waiting_sar_source', source_root=str(source), stages=[], test_accessed=False)
    write(root / 'workflow_state.json', state)
    print('OUTPUT_ROOT=' + str(root), flush=True)
    print('SAR_SOURCE=' + str(source), flush=True)
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', PYTHON=args.python)
    certificate = None
    try:
        wait_source(source)
        queued = False
        for stage in stages:
            if stage['kind'] == 'train' and not queued:
                state['current_stage'] = 'waiting_gpu'
                write(root / 'workflow_state.json', state)
                from gpu_queue import wait_for_idle
                protocol = read(certificate['protocol'])
                device = protocol['arguments']['device']
                if not isinstance(device, str) or not device.startswith('cuda:') or not device[5:].isdigit():
                    raise ValueError('Source needs explicit CUDA device')
                state['gpu_queue'] = wait_for_idle(int(device[5:]), log=lambda value: print(value, flush=True))
                queued = True
            state['current_stage'] = stage['name']
            row = dict(name=stage['name'], complete=False, started_at=time.time(),
                       log=str(root / 'logs' / (stage['name'] + '.log')))
            state['stages'].append(row)
            write(root / 'workflow_state.json', state)
            if source_files(repo) != before:
                raise ValueError('Experiment code changed while queued/running')
            code = child(stage, repo, env, row['log'])
            row.update(returncode=code, finished_at=time.time(), log_sha256=digest(row['log']))
            if code != 0:
                raise RuntimeError(stage['name'] + ' failed; later stages stopped')
            row['gate'] = stage_gate(stage)
            row['complete'] = True
            if stage['kind'] == 'certificate':
                certificate = read(stage['output'])
                workspace = certificate['source_runtime_settings'].get('cublas_workspace_config')
                if workspace is None:
                    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
                elif isinstance(workspace, str):
                    env['CUBLAS_WORKSPACE_CONFIG'] = workspace
                else:
                    raise ValueError('Invalid CUBLAS setting in source')
                state['source_certificate_sha256'] = digest(stage['output'])
                state['runtime_environment'] = {key: env.get(key) for key in (
                    'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES')}
            write(root / 'workflow_state.json', state)
        if source_files(repo) != before or digest(root / 'manifest.json') != manifest_sha:
            raise ValueError('Final code/manifest integrity check failed')
        if digest(root / 'source_certificate.json') != state['source_certificate_sha256']:
            raise ValueError('Source certificate changed')
        state.update(complete=True, full_validation_evidence=True, current_stage='complete', finished_at=time.time())
        write(root / 'workflow_state.json', state)
        result = report(root)
        write(root / 'summary.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    except BaseException as error:
        state.update(complete=False, full_validation_evidence=False,
                     failure=type(error).__name__ + ': ' + str(error), finished_at=time.time())
        write(root / 'workflow_state.json', state)
        (root / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write(root / 'summary.json', report(root))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'run', 'report', 'source'))
    p.add_argument('--repo-root', default=DEFAULT_REPO)
    p.add_argument('--sar-root')
    p.add_argument('--source-base', default=SOURCE_BASE)
    p.add_argument('--output', '--root', dest='output')
    p.add_argument('--python', default=sys.executable)
    return p


def main():
    args = parser().parse_args()
    if args.action == 'source':
        print(select_source(args.sar_root, args.source_base))
    elif args.action == 'report':
        if not args.output:
            raise SystemExit('--root is required')
        print(json.dumps(report(args.output), ensure_ascii=False, indent=2))
    elif args.action == 'plan':
        args.output = args.output or '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_SOFT_CROSS_V1/NEW_RUN'
        args.sar_root = args.sar_root or 'LATEST_DECLARED_SAR_RUN'
        print(json.dumps(dict(format=FORMAT, no_training_started=True, stages=plan(args)), ensure_ascii=False, indent=2))
    else:
        if not args.output:
            raise SystemExit('--output is required')
        run(args)


if __name__ == '__main__':
    main()
