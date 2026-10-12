"""Immutable source validation and explicit numerical-protocol binding."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def code_hash():
    root = Path(__file__).parent
    files = [p for p in root.glob('*.py') if not p.name.startswith('test_')]
    files += list(root.glob('*.sh'))
    for package, name in (
        ('damageformer_srac_evidence_followup_v1', 'native_pool.py'),
        ('damageformer_srac_stage_diagnostics_v1', 'grouped_pool.py'),
        ('damageformer_srac_stage_diagnostics_v1', 'memory_core.py'),
        ('damageformer_srac_stage_diagnostics_v1', 'memory_audit.py'),
        ('damageformer_srac_partition_probe_v1', 'numeric_runtime.py'),
    ):
        files.append(root.parent / package / name)
    result = hashlib.sha256()
    for path in sorted(files):
        result.update(path.relative_to(root.parent).as_posix().encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('Artifact path escapes its directory')
    return path


def add_source_arguments(parser):
    for name in ('joint_run', 'a1_checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        parser.add_argument('--' + name, required=True)
    for name in ('checkpoint', 'cache', 'metric_checkpoint'):
        parser.add_argument('--' + name)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--profile', choices=('legacy', 'strict'), default='legacy')
    parser.add_argument('--pool_mode', choices=('native', 'grouped'), default='native')
    for name in ('matmul_tf32', 'cudnn_tf32', 'deterministic'):
        parser.add_argument('--' + name, type=int, choices=(0, 1))
    parser.add_argument('--smoke', action='store_true')
    return parser


def configure(args):
    from .numeric import configure_runtime
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import seed_everything
    if args.workers < 0:
        raise ValueError('workers must be nonnegative')
    configure_runtime(args.profile, **{
        name: None if getattr(args, name, None) is None else bool(getattr(args, name))
        for name in ('matmul_tf32', 'cudnn_tf32', 'deterministic')})
    seed_everything(args.seed)


def artifact_identity(bundle):
    return dict(source_hashes=bundle['hashes'], source_head_hashes=bundle['head_hashes'],
                reference_hashes=bundle['reference_hashes'],
                protocol_sha256=bundle['protocol_sha256'], source_sha256=code_hash())


def validate_protocol(record, bundle, args, settings):
    import torch
    if record.get('format') != 'SRAC_HARDQUERY_PROTOCOL_V1' or record.get('scope') != 'all':
        raise ValueError('Requires the new all-path E0 protocol, not A1-only or an old failed audit')
    if record.get('complete') is not True:
        raise ValueError('E0 did not complete successfully; preserve failure and resolve it first')
    if not args.smoke:
        if (record.get('smoke') is not False or record.get('full_validation_evidence') is not True
                or record.get('image_count') != 450 or record.get('reproduction', {}).get('passed') is not True):
            raise ValueError('Formal work requires successful 450-image E0 with unchanged 1e-5 gate')
    elif record.get('reproduction', {}).get('passed') is False:
        raise ValueError('A failed full E0 cannot be laundered into a smoke protocol')
    expected = dict(source_hashes=bundle['hashes'], source_head_hashes=bundle['head_hashes'],
                    reference_hashes=bundle['reference_hashes'], source_sha256=code_hash(),
                    profile=args.profile, pool_mode=args.pool_mode)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('Protocol identity mismatch: ' + key)
    if record.get('runtime_settings') != settings:
        raise ValueError('Runtime differs from E0; use the same process settings/device/environment')
    if record.get('input_fingerprints') != bundle['inputs_before']:
        raise ValueError('E0 source file ledger differs from current files')
    original_args = record.get('arguments', {})
    def compute_device(value):
        value = torch.device(value)
        return (value.type, (value.index or 0) if value.type == 'cuda' else None)
    if compute_device(original_args.get('device', 'cpu')) != compute_device(args.device):
        raise ValueError('Compute device differs from E0')
    if (not original_args.get('dataset')
            or Path(original_args['dataset']).resolve() != Path(args.dataset).resolve()):
        raise ValueError('Dataset root differs from E0; identical image names are insufficient')


def load_verified_bundle(args):
    import torch
    from .numeric import runtime_settings
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import load_bundle, fingerprint
    configure(args)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    bundle = load_bundle(args, device)
    settings = runtime_settings()
    path = Path(args.protocol)
    if path.is_dir():
        path = path / 'protocol.json'
    record = read_json(path)
    validate_protocol(record, bundle, args, settings)
    bundle.update(protocol=record, protocol_path=path, protocol_sha256=digest(path),
                  runtime_settings=settings, frozen_before=fingerprint(bundle),
                  own_source_sha256=code_hash(), device=device)
    return bundle


def check_artifact(record, bundle, allow_smoke=False):
    if record.get('complete') is not True or record.get('test_accessed') is not False:
        raise ValueError('Incomplete artifact or invalid Test-access declaration')
    if not allow_smoke and record.get('smoke') is not False:
        raise ValueError('Smoke artifacts cannot enter formal training')
    for key, value in artifact_identity(bundle).items():
        if record.get(key) != value:
            raise ValueError('Artifact provenance mismatch: ' + key)


def verify_unchanged(bundle):
    from .numeric import runtime_settings
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
        fingerprint, input_fingerprints, diagnostic_code_fingerprint)
    checks = dict(
        files=input_fingerprints(bundle['input_paths']) == bundle['inputs_before'],
        inherited_code=diagnostic_code_fingerprint(bundle['source_manifest']['format']) == bundle['code_before'],
        new_code=code_hash() == bundle['own_source_sha256'],
        frozen_models=fingerprint(bundle) == bundle['frozen_before'],
        runtime=runtime_settings() == bundle['runtime_settings'],
        protocol=digest(bundle['protocol_path']) == bundle['protocol_sha256'])
    if not all(checks.values()):
        raise RuntimeError('Immutable evidence/runtime changed: ' + str(checks))
    return checks


def finite_positive(value, name):
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(name + ' must be finite and positive')


def fail_summary(output, error):
    write_json(Path(output) / 'summary.json', dict(complete=False, full_validation_evidence=False,
               error=type(error).__name__ + ': ' + str(error), test_accessed=False))
