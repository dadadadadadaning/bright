#!/usr/bin/env python3
"""Launch the unchanged joint experiment after passed synchronous diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DIAG = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_DIAG_V1/run_20261013_150401_3833455'
TOOL = '/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1'
BASE = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_SYNC_V1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(diag_root, tool, base):
    diag_root, tool, base = (Path(p).expanduser().resolve() for p in (diag_root, tool, base))
    diag_path = diag_root / 'summary.json'
    diag = read(diag_path)
    require(diag.get('diagnostic_complete') is True and diag.get('all_executed_stages_passed') is True,
            'The selected diagnosis has not completed with all stages passed')
    stages = ('replay_cpu', 'replay_native', 'replay_contiguous', 'sequence_failed_image', 'sequence_val450')
    for stage in stages:
        row = diag.get('stages', {}).get(stage, {})
        require(row.get('passed') is True and row.get('returncode') == 0 and not row.get('skipped'),
                'Required diagnostic stage did not pass: ' + stage)
    seq_path = diag_root / 'sequence_val450/summary.json'
    seq = read(seq_path)
    require(seq.get('diagnostic_complete') is True and seq.get('gates_passed') is True
            and seq.get('scope') == 'val450' and seq.get('whole_val450_processed') is True
            and seq.get('image_count') == 450 and seq.get('training') is False,
            'Need the completed no-training initial Val450 diagnostic')
    guard = seq.get('pool_guard_report', {})
    require(guard.get('backend') == 'native' and guard.get('restored') is True
            and not guard.get('failures') and not guard.get('restore_conflicts'),
            'The passed Val450 must use the original native pool and restore its hooks')
    require(seq.get('heads_unchanged') is True, 'Diagnostic initial heads changed')
    for arm in ('r0', 'rt', 'rf'):
        require(seq.get('initial_probability_checks', {}).get(arm, {}).get('passed') is True,
                'Initial head probability gate failed: ' + arm)
    require(seq.get('environment', {}).get('CUDA_LAUNCH_BLOCKING') == '1',
            'Passed sequence was not recorded under the expected synchronous configuration')

    manifest_path = Path(seq['preserved_manifest']).resolve()
    require(digest(manifest_path) == seq['preserved_manifest_sha256'], 'Old run manifest changed after diagnosis')
    manifest = read(manifest_path)
    require(manifest.get('protocol_info') == seq.get('protocol_info'), 'Source/bank/runtime binding mismatch')
    values = manifest['arguments']
    require(values.get('mode') == 'train' and values.get('steps') == 5000, 'Need the original formal 5000-step configuration')
    require(Path(seq['arguments']['tool']).resolve() == tool, 'Tool path differs from passed diagnostic')
    hashes = read(tool / 'SHA256SUMS.json')
    require(bool(hashes), 'Missing original tool checksum manifest')
    for name, expected in hashes.items():
        require(Path(name).name == name and digest(tool / name) == expected, 'Original tool changed: ' + name)
    require({p.name for p in tool.glob('*.py')} <= set(hashes), 'Unbound Python file in original tool')
    h = hashlib.sha256()
    for path in sorted(tool.glob('*.py')):
        if not path.name.startswith('test_'):
            h.update(path.name.encode()); h.update(path.read_bytes())
    require(h.hexdigest() == manifest['source_hash'], 'Original training source hash changed')
    require(digest(tool / 'SHA256SUMS.json') == seq['original_release']['package_sha256'], 'Original release manifest differs')

    runtime = manifest['protocol_info']['source_runtime_settings']
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    cublas = runtime.get('cublas_workspace_config')
    if cublas is None:
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    else:
        require(isinstance(cublas, str), 'Invalid historical CUBLAS configuration')
        env['CUBLAS_WORKSPACE_CONFIG'] = cublas
    # run.py full receives explicit original arguments; old JOINT_* environment
    # variables from other terminal sessions cannot change this command.
    run_id = 'run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
    output = base / run_id
    command = [sys.executable, '-u', str(tool / 'run.py'), 'full']
    for name in ('repo', 'v3_root', 'device', 'workers', 'seed', 'topk', 'chunk_size', 'lr', 'weight_decay', 'protocol'):
        value = values.get(name)
        if value is not None:
            command.extend(['--' + name.replace('_', '-'), str(value)])
    # Pin the previously verified E0 path rather than reselecting through a ledger.
    if not values.get('protocol'):
        command.extend(['--protocol', manifest['protocol_info']['source_protocol']])
    command.extend(['--output', str(output)])
    require(not output.exists(), 'Fresh output already exists')
    require(output != diag_root and diag_root not in output.parents and tool not in output.parents,
            'New results must be outside preserved diagnosis/tool directories')
    record = dict(format='SRAC_JOINT_SYNC_LAUNCH_V1', launched=False,
        output=str(output), command=command, old_tool=str(tool),
        launch_environment={'CUDA_LAUNCH_BLOCKING': '1', 'CUBLAS_WORKSPACE_CONFIG': cublas},
        original_numeric_runtime=runtime, pool_backend='native', arms=['r0', 'rt', 'rf'],
        steps_per_arm=5000, smoke_steps_per_arm=20, training_completed=False,
        full_validation_evidence=False, test_accessed=False, root_cause_claim=False,
        inherited_diagnostic=str(diag_root), diagnostic_summary_sha256=digest(diag_path),
        sequence_summary_sha256=digest(seq_path), old_manifest=str(manifest_path),
        old_manifest_sha256=digest(manifest_path), original_source_hash=h.hexdigest(),
        original_release_sha256=digest(tool / 'SHA256SUMS.json'),
        launcher_sha256=digest(__file__),
        note='Launch provenance only. Current progress and completed metrics are in workflow_state.json and train/summary.json; no exact historical execution claim.')
    return record, env, Path(values['repo'])


def launch(args):
    record, env, cwd = prepare(args.diag_root, args.tool, args.result_base)
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print('Dry run only: no output directory or training process created.')
        return record
    output = Path(record['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.log')
    note = output.with_suffix('.launch.json')
    with note.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    try:
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(record['command'], cwd=cwd, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(launched=True, pid=process.pid, log=str(log))
    except Exception as error:
        record['launch_error'] = type(error).__name__ + ': ' + str(error)
        note.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise
    note.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('后台PID:', process.pid)
    print('结果目录:', output)
    print('启动配置记录:', note)
    print('日志:', log)
    print('查看进度：tail -n 60 -f "' + str(log) + '"')
    print('读取报告："' + sys.executable + '" "' + str(Path(args.tool) / 'run.py') + '" report --root "' + str(output) + '"')
    print('已创建新后台进程。等待预检、20步smoke、三组各5000步及末步Val450；启动不等于训练通过。')
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diag-root', default=DIAG)
    p.add_argument('--tool', default=TOOL)
    p.add_argument('--result-base', default=BASE)
    p.add_argument('--dry-run', action='store_true')
    return p


if __name__ == '__main__':
    launch(parser().parse_args())
