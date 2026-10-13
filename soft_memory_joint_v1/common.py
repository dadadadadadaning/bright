"""Interfaces and file helpers for the standalone V3 joint training package."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import torch

ARMS = ('a', 'b', 'c', 'd')
ARM_RULES = {'a': 'capacity_control', 'b': 'soft', 'c': 'capacity_control', 'd': 'soft'}
MEMORY_ARMS = ('c', 'd')
FORMAT = 'SRAC_SOFT_MEMORY_JOINT_V1'

@dataclass
class Example:
    image: str
    event: str
    target: torch.Tensor
    common: dict
    supports: dict
    diagnostics: dict

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')
    os.replace(temporary, path)

def source_hash():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        if not p.name.startswith('test_'):
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()

def module_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        t = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
