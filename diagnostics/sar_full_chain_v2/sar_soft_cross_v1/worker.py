"""Isolated GPU worker for a declared, matched baseline-Q / SAR-Q route experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


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


def dump_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def certificate_inputs(certificate, repo):
    data = read(certificate)
    if (data.get('format') != 'SAR_SOFT_SOURCE_CERTIFICATE_V1'
            or data.get('complete') is not True
            or data.get('full_validation_evidence') is not True
            or Path(data['repo_root']).resolve() != Path(repo).resolve()):
        raise ValueError('Requires a complete four-arm source certificate for this repository')
    bindings = data.get('source_bindings', data.get('source_file_sha256'))
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Certificate has no source bindings')
    for name, expected in bindings.items():
        if digest(name) != expected:
            raise ValueError('Certified input changed: ' + name)
    if digest(data['protocol']) != data['protocol_sha256']:
        raise ValueError('Source E0 protocol changed')
    return data, dict(bindings, **{str(Path(certificate).resolve()): digest(certificate)})


def fit_bindings(root):
    root = Path(root).resolve()
    return {name: digest(root / filename) for name, filename in (
        ('fit_summary_sha256', 'summary.json'),
        ('fit_manifest_sha256', 'manifest.json'),
        ('fit_checkpoint_sha256', 'heads_last.pth'),
        ('training_journal_sha256', 'training.jsonl'))}


def run(args):
    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Worker output must be fresh: ' + str(output))
    if output.is_relative_to(repo) or output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('Write results outside repository/tool directories')
    if os.environ.get('CUDA_LAUNCH_BLOCKING') != '1':
        raise ValueError('This matched protocol requires CUDA_LAUNCH_BLOCKING=1 before Python starts')
    certificate, inputs_before = certificate_inputs(args.certificate, repo)
    runtime = certificate['source_runtime_settings']
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != runtime.get('cublas_workspace_config'):
        raise ValueError('CUBLAS configuration must match the certified E0')
    protocol = certificate['protocol']
    q_checkpoint = certificate['baseline_checkpoint' if args.q_arm == 'baseline' else 'sar_checkpoint']
    record = dict(format='SAR_SOFT_WORKER_V1', complete=False, action=args.action,
        q_arm=args.q_arm, smoke=args.smoke, test_accessed=False,
        source_certificate=str(Path(args.certificate).resolve()),
        source_certificate_sha256=digest(args.certificate),
        protocol_sha256=certificate['protocol_sha256'],
        environment={key: os.environ.get(key) for key in (
            'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES')})
    original_fit_bindings = None
    if args.action == 'evaluate':
        if not args.fit:
            raise ValueError('Evaluation requires --fit')
        fit = read(Path(args.fit) / 'summary.json')
        if (fit.get('q_source', {}).get('arm') != args.q_arm
                or fit.get('q_source', {}).get('sha256') != digest(q_checkpoint)):
            raise ValueError('Fit does not use the selected certified Q source')
        original_fit_bindings = fit_bindings(args.fit)
        record.update(original_fit_bindings, fit_root=str(Path(args.fit).resolve()))
    sys.path.insert(0, str(repo))
    sys.dont_write_bytecode = True
    from core_adapter import adapted_modules, precheck_fit, source_hash
    try:
        with adapted_modules() as modules:
            record['adapter_source_sha256'] = source_hash()
            if args.action == 'train':
                values = ['--protocol', protocol, '--q-checkpoint', q_checkpoint,
                          '--output', str(output), '--steps', '2' if args.smoke else '2000',
                          '--eval-every', '2' if args.smoke else '2000', '--lr', '0.0001',
                          '--threshold', '0.5', '--soft-floor', '0.0']
                if args.smoke:
                    values.append('--smoke')
                result = modules['train'].run(modules['train'].parser().parse_args(values))
            else:
                precheck_fit(args.fit, protocol, smoke=args.smoke)
                values = ['--protocol', protocol, '--fit', args.fit, '--output', str(output)]
                if args.smoke:
                    values += ['--limit', '2']
                result = modules['evaluate'].run(modules['evaluate'].parser().parse_args(values))
            if source_hash() != record['adapter_source_sha256']:
                raise ValueError('Experiment source changed during worker execution')
        if result.get('complete') is not True:
            raise ValueError('Worker returned incomplete evidence')
        for name, expected in inputs_before.items():
            if digest(name) != expected:
                raise ValueError('Certified input changed while worker ran: ' + name)
        if original_fit_bindings is not None and fit_bindings(args.fit) != original_fit_bindings:
            raise ValueError('Training evidence changed during independent evaluation')
        record['summary_sha256'] = digest(output / 'summary.json')
        if args.action == 'evaluate':
            record['result_sha256'] = digest(output / 'result.json')
        record['complete'] = True
        dump_new(output / 'worker_record.json', record)
        print('Worker complete: ' + str(output), flush=True)
        return record
    except BaseException as error:
        record.update(complete=False, error=type(error).__name__ + ': ' + str(error),
                      traceback=traceback.format_exc())
        if output.is_dir() and not (output / 'worker_record.json').exists():
            dump_new(output / 'worker_record.json', record)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('train', 'evaluate'))
    p.add_argument('--repo-root', required=True)
    p.add_argument('--certificate', required=True)
    p.add_argument('--q-arm', choices=('baseline', 'sar_cue'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fit')
    p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
