"""Read-only provenance and checkpoint restoration helpers for calibration recovery.

This module contains no diagnostic worker or training entry point.
"""
from __future__ import annotations
import importlib.util
import json
import math
import hashlib
from pathlib import Path

FORMAT = 'BRIGHT_CALIBRATION_RESUME_V1'
PARENT_FORMAT = 'BRIGHT_SEVERITY_MEMORY_V1'
ROOT = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SEVERITY_MEMORY_V1')
PARENT = ROOT / 'night_3407_z_yxvq4s/train_calibration'
PACKAGE = Path('/root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023')
TARGET_IMAGE = 'la_palma-volcano_00000270'
RESTORED_STEP = 800
END_STEP = 2500
IDENTITY = {
    'experiment.py': '79d1deb96ee4ce6ec743b0d63d7aa76662d9ffe7de24e8e0dc724894dbff409b',
    'old_model_adapter.py': '9a81006d0191a3379145286d74a74f7d236d230534a980b0671ff3da4b1e8ba4',
    'severity_adapter.py': '3867e126f4bf5a66cb3dba0144cb2c514d5965c84059db7b65fe102191435ae3',
}


def require(value, message):
    if not value:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')
    temporary.replace(path)

def restore_model_optimizer(torch, state, model, optimizer, saved, step):
    require(saved.get('step') == step, 'Checkpoint step mismatch')
    require(all(k in saved for k in ('model', 'optimizer', 'rng')), 'Checkpoint missing model/optimizer/RNG')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    for key, value in (('model', model.state_dict()), ('optimizer', optimizer.state_dict())):
        require(state.state_fingerprint(value) == state.state_fingerprint(saved[key]), 'Restored '+key+' differs')
    counts = [int(v['step']) for v in optimizer.state.values() if 'step' in v]
    require(counts and all(v == step for v in counts), 'Restored optimizer step differs')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite checkpoint parameters')

def reconstruct_iterator(exp, runtime, manifest, rows, restored_step):
    prefix = validate_prefix(rows, manifest, restored_step)
    iterator = exp.batches(runtime, manifest['actual_training_names'], manifest['seed'])
    for index, row in enumerate(prefix, 1):
        epoch, batch = next(iterator)
        require(epoch == row['epoch'] and str(batch[-1][0]) == row['image'],
                'Reconstructed data order differs at step '+str(index))
        del batch
        if index % 200 == 0:
            print('RECOVERY DATA ORDER verified '+str(index)+'/'+str(restored_step), flush=True)
    return iterator

def assert_unchanged(fingerprints):
    for path, digest in fingerprints.items():
        require(sha(path) == digest, 'Parent evidence changed: '+path)


def validate_prefix(rows, manifest, restored_step):
    require(len(rows) >= restored_step, 'Journal shorter than checkpoint')
    names = set(manifest['actual_training_names'])
    for expected, row in enumerate(rows[:restored_step], 1):
        require(row.get('step') == expected and row.get('optimizer_step') == expected,
                'Nonconsecutive journal/optimizer step')
        require(isinstance(row.get('epoch'), int) and row['epoch'] >= 0 and
                row.get('epoch_seed') == manifest['seed'] + row['epoch'], 'Journal epoch mismatch')
        require(row.get('variant') == manifest['plan']['variant'] and row.get('image') in names,
                'Journal variant or training membership mismatch')
    return rows[:restored_step]
