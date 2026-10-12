"""Recover immutable inputs from a completed actual-reader audit, not shell state."""
from argparse import Namespace
from pathlib import Path

from bda_benchmark.experiments.damageformer_memory_readout_audit_v1.provenance import _json
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2 import common


PARENT_FORMAT = 'MEMORY_READOUT_AUDIT_V1'
REQUIRED = ('checkpoint', 'run_root', 'memory_run', 'bank', 'dataset', 'train_list', 'val_list')


def load_parent(parent_root, output, device='cuda:0', workers=4, smoke=False):
    """Read configuration only. The original loaders revalidate all source files.

    A formal parent is required even for a two-image smoke. All model/dataset
    paths are inherited; new shell FRONT_CKPT/TOPK/SEED values are not consulted.
    """
    parent = Path(parent_root)
    target = Path(output)
    if not parent.is_absolute() or not target.is_absolute():
        raise ValueError('Parent and output must be absolute paths')
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    summary = _json(parent / 'summary.json')
    manifest = _json(parent / 'manifest.json')
    for record in (summary, manifest):
        if (record.get('format') != PARENT_FORMAT or record.get('smoke') is not False
                or record.get('test_accessed') is not False):
            raise ValueError('Require the completed formal Train/Val-only actual-reader audit')
    if (summary.get('complete') is not True or summary.get('image_count') != 450
            or any(summary.get(k) is not True for k in
                   ('source_verified', 'bank_verified', 'head_verified'))
            or summary.get('reproduction', {}).get('passed') is not True):
        raise ValueError('Parent must be complete Val450 with provenance and reproduction passed')
    original = manifest.get('arguments', {})
    arguments = {}
    for key in REQUIRED:
        value = original.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('Missing absolute parent argument: ' + key)
        exists = Path(value).is_file() if key in ('checkpoint', 'train_list', 'val_list') else Path(value).is_dir()
        if not exists:
            raise FileNotFoundError('Original input unavailable; do not substitute another checkpoint: ' + value)
        arguments[key] = value
    for key in ('source_head', 'pool_mode', 'read_mode'):
        if original.get(key) != 'hard':
            raise ValueError('Parent must fix hard source/pool/read: ' + key)
        arguments[key] = 'hard'
    # Avoid accidentally placing the new run inside a protected input tree.
    for protected in (parent, Path(arguments['run_root']), Path(arguments['memory_run']),
                      Path(arguments['bank']), Path(arguments['dataset'])):
        if target.resolve() == protected.resolve() or protected.resolve() in target.resolve().parents:
            raise ValueError('Choose an isolated output outside input directories: ' + str(target))
    if target.exists() or target.is_symlink():
        raise FileExistsError('Preserve existing results; choose a new output: ' + str(target))
    arguments.update(output=str(target), parent_audit=str(parent), device=device,
                     workers=workers, smoke=bool(smoke))
    files = {str((parent / name).resolve()): common.digest(parent / name)
             for name in ('manifest.json', 'summary.json')}
    expected_inputs = manifest.get('input_file_sha256')
    if not isinstance(expected_inputs, dict) or not expected_inputs:
        raise ValueError('Parent is missing original input file hashes')
    for path, expected in expected_inputs.items():
        if not Path(path).is_absolute() or common.digest(path) != expected:
            raise ValueError('Parent input hash changed: ' + str(path))
    return Namespace(**arguments), dict(files=files, expected_inputs=expected_inputs,
        source_identity=manifest.get('source_identity'), settings=manifest.get('settings'),
        final_metrics=summary.get('final_metrics', {}))


def verify_parent_state(parent, state):
    if state['files'] != parent['expected_inputs']:
        raise ValueError('Loaded files are not the exact inputs recorded by the parent audit')
    if state['identity'] != parent['source_identity'] or state['settings'] != parent['settings']:
        raise ValueError('Loaded source/settings differ from parent audit')
    assert_parent_unchanged(parent)


def assert_parent_unchanged(parent):
    for path, expected in parent['files'].items():
        if common.digest(path) != expected:
            raise RuntimeError('Parent audit changed during the intervention: ' + path)
