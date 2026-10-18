"""Read-only provenance boundary for the completed, original fix2 experiment.

Planning uses only the standard library. The historical package is imported
only for an explicitly requested forward audit, after its release is verified.
This module never writes to source artifacts or restores optimizer/RNG state.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from types import SimpleNamespace


EXPECTED_SOURCE = 'b84e2867a4117911b8279e2f97ccf9c5fbd021a7dc539f59e6bbb06396546142'
SOURCE_FORMAT = 'SRAC_SOFT_MEMORY_Q_REBASED_V1'
ARMS = ('a', 'b', 'c', 'd')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError('Non-finite JSON value: ' + value)
    data = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)
    if not isinstance(data, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return data


def _source_hash(tool):
    h = hashlib.sha256()
    for path in sorted(Path(tool).glob('*.py')):
        if not path.name.startswith('test_'):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def _release_files(tool):
    manifest = tool / 'SHA256SUMS.json'
    if manifest.is_symlink():
        raise ValueError('Original release manifest cannot be a symlink')
    hashes = read_json(manifest)
    if not hashes:
        raise ValueError('Original package checksum manifest is empty')
    files = {}
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if (not isinstance(name, str) or '\\' in name or relative.is_absolute()
                or not relative.parts or any(p in ('.', '..') for p in relative.parts)
                or relative.as_posix() != name or name == 'SHA256SUMS.json'
                or not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None):
            raise ValueError('Invalid original package checksum entry: ' + str(name))
        path = tool.joinpath(*relative.parts)
        if (path.is_symlink() or any(p.is_symlink() for p in path.parents if p.is_relative_to(tool))
                or not path.is_file() or not path.resolve().is_relative_to(tool)
                or digest(path) != expected):
            raise ValueError('Historical package file is missing/changed: ' + str(path))
        files[str(path)] = expected
    python_files = {p.relative_to(tool).as_posix() for p in tool.rglob('*.py')}
    if not python_files <= set(hashes):
        raise ValueError('Unexpected Python file in original tool directory')
    if _source_hash(tool) != EXPECTED_SOURCE:
        raise ValueError('Use the exact original fix2 package, unchanged')
    files[str(manifest)] = digest(manifest)
    return files


def validate_source(train_root, tool_dir, step=500):
    """Verify metadata and bytes without importing Torch or opening model payloads.

``train_root`` is the run root, not its ``train`` child. Only the original
500/1500/5000 checkpoints are accepted; later LR/loss branches are distinct
experiments and cannot silently substitute for them.
    """
    if type(step) is not int or step not in (500, 1500, 5000):
        raise ValueError('Audit step must be one of 500, 1500, 5000')
    root = Path(train_root).expanduser().resolve()
    tool = Path(tool_dir).expanduser().resolve()
    train = root / 'train'
    summary = read_json(train / 'summary.json')
    workflow = read_json(root / 'workflow_state.json')
    for record in (summary, workflow):
        if (record.get('format') != SOURCE_FORMAT or record.get('complete') is not True
                or record.get('full_validation_evidence') is not True
                or record.get('smoke') is not False):
            raise ValueError('Original formal run must be complete, non-smoke and verified')
    manifest = read_json(train / 'manifest.json')
    names, val = manifest.get('train_names'), manifest.get('val_names')
    if (not isinstance(names, list) or not isinstance(val, list)
            or any(not isinstance(n, str) or not n for n in names + val)
            or len(names) != 2918 or len(set(names)) != 2918
            or len(val) != 450 or len(set(val)) != 450 or set(names) & set(val)):
        raise ValueError('Source must bind disjoint Train2918 and Val450')
    filename = 'checkpoint_last.pt' if step == 5000 else 'checkpoint_%06d.pt' % step
    checkpoint = train / 'checkpoints' / filename
    meta = read_json(checkpoint.with_suffix('.pt.json'))
    if (meta.get('format') != SOURCE_FORMAT or meta.get('file') != filename
            or meta.get('step') != step or meta.get('mode') != 'train'
            or meta.get('smoke') is not False
            or meta.get('training_complete') is not (step == 5000)
            or digest(checkpoint) != meta.get('sha256')
            or meta.get('source_hash') != EXPECTED_SOURCE
            or manifest.get('format') != SOURCE_FORMAT
            or manifest.get('source_hash') != EXPECTED_SOURCE):
        raise ValueError('Checkpoint / manifest / source binding failed')
    protocol = meta.get('protocol_info')
    if (not isinstance(protocol, dict) or not protocol
            or protocol != manifest.get('protocol_info')
            or not isinstance(protocol.get('execution_settings'), dict)):
        raise ValueError('Checkpoint and training manifest protocol bindings differ')
    reference_path = train / ('final_val' if step == 5000 else 'val_step_%06d' % step) / 'result.json'
    reference = read_json(reference_path)
    if (reference.get('image_count') != 450 or reference.get('execution_complete') is not True
            or not {'grid_probq', *ARMS} <= set(reference.get('metrics', {}))):
        raise ValueError('Original checkpoint full Val450 report is missing')
    if step == 5000 and (reference.get('full_validation_evidence') is not True
                         or reference.get('final_integrity_pending') is not False):
        raise ValueError('Original final Val450 integrity has not completed')
    fingerprints = _release_files(tool)
    for path in (checkpoint, checkpoint.with_suffix('.pt.json'), train / 'manifest.json',
                 train / 'summary.json', root / 'workflow_state.json', reference_path):
        fingerprints[str(path)] = digest(path)
    return dict(root=str(root), tool=str(tool), checkpoint=str(checkpoint), meta=meta,
                manifest=manifest, reference=reference, old_files=fingerprints,
                reference_path=str(reference_path), step=step)


def _check_import_origin(module, tool, known_files):
    location = getattr(module, '__file__', None)
    if location is None:
        raise ValueError('Historical module has no file origin: ' + module.__name__)
    path = Path(location).resolve()
    if (not path.is_relative_to(tool) or str(path) not in known_files
            or digest(path) != known_files[str(path)]):
        raise ValueError('Imported the wrong historical module: ' + str(path))


def import_runtime(static):
    """Import verified source modules after setting recorded execution options."""
    check_source_unchanged(static)
    tool = Path(static['tool'])
    original_names = ('common', 'model', 'slots', 'base_runtime', 'server_runtime', 'train', 'run')
    for name in original_names:
        if name in sys.modules:
            _check_import_origin(sys.modules[name], tool, static['old_files'])
    execution = static['meta']['protocol_info']['execution_settings']
    blocking = execution.get('cuda_launch_blocking')
    if blocking is None:
        os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
    else:
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(blocking)
    # This audit must not even create bytecode beside historical source modules.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(tool))
    torch = importlib.import_module('torch')
    imported = {name: importlib.import_module(name) for name in original_names}
    for module in imported.values():
        _check_import_origin(module, tool, static['old_files'])
    release = imported['run'].verify_release()
    if release.get('code_sha256') != EXPECTED_SOURCE:
        raise ValueError('Imported runtime source hash differs from original fix2')
    return SimpleNamespace(torch=torch, common=imported['common'], train=imported['train'],
                           run=imported['run'], Runtime=imported['server_runtime'].Runtime,
                           Classifier=imported['model'].JointRegionMemoryClassifier)


def load_saved(static, modules):
    """Load the user's hash-bound local checkpoint; no optimizer/RNG restoration."""
    check_source_unchanged(static)
    torch = modules.torch
    saved = torch.load(static['checkpoint'], map_location='cpu', weights_only=False)
    meta, step = static['meta'], static['step']
    if (not isinstance(saved, dict) or saved.get('format') != SOURCE_FORMAT
            or saved.get('source_hash') != EXPECTED_SOURCE
            or saved.get('protocol_info') != meta['protocol_info']
            or saved.get('step') != step or saved.get('mode') != 'train'
            or saved.get('smoke') is not False
            or saved.get('training_complete') is not (step == 5000)
            or saved.get('updates_per_arm') != {a: step for a in ARMS}
            or not isinstance(saved.get('heads'), dict) or set(saved['heads']) != set(ARMS)):
        raise ValueError('Checkpoint payload differs from its bound training provenance')
    if (not isinstance(saved.get('config'), dict) or saved['config'].get('steps') != 5000
            or saved['config'].get('eval_every') != 500
            or saved['config'].get('baseline_policy', 'independent') != 'independent'
            or not isinstance(saved.get('input_file_sha256'), dict) or not saved['input_file_sha256']):
        raise ValueError('Checkpoint config or input ledger is incomplete')
    if saved['config'] != static['manifest'].get('arguments'):
        raise ValueError('Checkpoint configuration differs from original training manifest')
    for arm in ARMS:
        state = saved['heads'][arm]
        if not isinstance(state, dict) or not state:
            raise ValueError('Missing original head state: ' + arm)
        for name, value in state.items():
            if not isinstance(name, str) or not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
                raise ValueError('Non-finite/non-tensor original head state: ' + arm)
        # Use the hash function from the verified original package, including
        # its state_dict insertion order, dtype and zero-dimensional handling.
        holder = SimpleNamespace(state_dict=lambda state=state: state)
        if modules.common.module_hash(holder) != meta.get('head_hashes', {}).get(arm):
            raise ValueError('Original head tensor digest mismatch: ' + arm)
    return saved


def runtime_arguments(saved, output):
    config = copy.deepcopy(saved['config'])
    destination = Path(output).expanduser().resolve()
    original_train = Path(config['output']).expanduser().resolve()
    if destination.is_relative_to(original_train.parent):
        raise ValueError('Audit output must be outside the original experiment directory')
    for key in ('bank_root', 'values'):
        if config.get(key) and destination.is_relative_to(Path(config[key]).expanduser().resolve()):
            raise ValueError('Audit output cannot be inside a source bank')
    config.update(command='train', mode='train', smoke=False, output=str(destination),
                  baseline_policy='independent', baseline_root=None, baseline_evidence=None,
                  allow_smoke_values=False)
    return SimpleNamespace(**config)


def check_runtime_inputs(runtime, saved):
    """Check every accessed input against the saved ledger before/after Val use.

The audit only forwards Val names. These were already accessed by the original
initial Val450; no unseen Train file may be silently enrolled in this ledger.
    """
    if runtime.protocol_info != saved['protocol_info']:
        raise ValueError('Runtime protocol differs from checkpoint provenance')
    expected = saved.get('input_file_sha256')
    actual = getattr(runtime, 'input_ledger', None)
    if not isinstance(expected, dict) or not expected or not isinstance(actual, dict) or not actual:
        raise ValueError('Runtime/checkpoint input ledger is missing')
    for path, sha in actual.items():
        if expected.get(str(path)) != sha:
            raise ValueError('Runtime input differs from checkpoint provenance: ' + str(path))
    return True


def check_source_unchanged(static):
    for name, expected in static['old_files'].items():
        if digest(name) != expected:
            raise ValueError('An original artifact changed during the audit: ' + name)
    if _source_hash(static['tool']) != EXPECTED_SOURCE:
        raise ValueError('Original runtime code changed during the audit')
    return True
