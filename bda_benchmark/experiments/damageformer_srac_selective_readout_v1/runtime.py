"""Provenance and immutable evidence; never changes the parent completion gate."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    digest, lists, seed_everything, source_hash as parent_source_hash,
)
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import (
    load_snapshot, module_fingerprint, verify_splits,
)
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.core import capture


def experiment_source_hash():
    checksum = hashlib.sha256(parent_source_hash().encode())
    root = Path(__file__).resolve().parents[3]
    files = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'run.sh']
    for rel in (
        'damageformer_srac_joint_context_v1/model.py',
        'damageformer_srac_cause_audit_v1/support.py',
        'damageformer_srac_cause_audit_v1/core.py',
        'damageformer_srac_cause_audit_v1/diagnostics.py',
        'damageformer_srac_approx_v1/audit.py',
    ):
        files.append(Path(__file__).parent.parent / rel)
    for path in files:
        checksum.update(path.relative_to(root).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def load_fixed(args):
    """Accept a stable intermediate snapshot for diagnostics, never for J123."""
    lists(args.train_list, args.val_list)
    path = Path(args.checkpoint)
    if not (path.parent / 'snapshot.json').is_file():
        raise ValueError('Use a stable snapshot with adjacent snapshot.json; not a live best.pth')
    if args.workers < 0:
        raise ValueError('workers must be nonnegative')
    seed_everything(args.seed)
    checksum = digest(path)
    front, payload = load_snapshot(path, args.a1_checkpoint, torch.device(args.device))
    if digest(path) != checksum:
        raise RuntimeError('Checkpoint changed during loading')
    verify_splits(payload, args.train_list, args.val_list)
    if payload['smoke'] and not args.smoke:
        raise ValueError('Smoke-trained front cannot produce full-Val formal evidence')
    a1_score = payload.get('load_report', {}).get('expected_valid_miou')
    if type(a1_score) not in (float, int) or not math.isfinite(a1_score) or not 0 <= a1_score <= 1:
        raise ValueError('Missing audited A1 reproduction score')
    manifest = dict(
        arguments=vars(args), checkpoint=str(path.resolve()), checkpoint_sha256=checksum,
        checkpoint_step=payload['step'], parent_complete=bool(payload.get('run_complete')),
        parent_smoke=payload['smoke'], parent_source_hash=payload['source_hash'],
        a1_sha256=payload['a1_sha256'], train_sha256=digest(args.train_list),
        val_sha256=digest(args.val_list), experiment_source_hash=experiment_source_hash(),
        diagnostic_only=True, reusable_as_parent=False, test_accessed=False, smoke=args.smoke,
        fixed='A1, Soft ROI, Q, B, adapter, region embedding and original region head',
    )
    return front, payload, manifest


def owned(value):
    """Ordinary detached tensors can be saved for a trainable head's backward."""
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


class FrozenEvidence:
    def __init__(self, front):
        self.front = front.requires_grad_(False).eval()

    def __call__(self, pre, post):
        # A caller's inference_mode/autocast must not leak into shared evidence.
        if any(module.training for module in self.front.modules()):
            raise RuntimeError('Frozen source left eval mode')
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(pre.device.type, enabled=False):
            result, aux = capture(self.front, pre.float(), post.float())
            result = {key: owned(value) for key, value in result.items()}
            result['pixel_feature'] = owned(aux['fused_roi'])
            result['building'] = F.interpolate(
                aux['p_building'].detach().float(), result['shape'], mode='bilinear', align_corners=False,
            ).clone()
        return result


def upsample_probability(logits, size):
    return F.interpolate(logits.float(), size, mode='bilinear', align_corners=False).softmax(1)


def reproduction_gate(result, payload, smoke):
    if smoke:
        return dict(passed=None, reason='Two-image smoke is not full-Val reproduction')
    checks = {}
    expected = dict(a1=payload['load_report']['expected_valid_miou'], region=payload['valid_miou'])
    for name, value in expected.items():
        actual = result['metrics'][name]['valid_miou']
        delta = abs(value - actual)
        checks[name] = dict(expected=value, actual=actual, absolute_error=delta, passed=delta <= 1e-5)
    return dict(passed=result['image_count'] == 450 and all(x['passed'] for x in checks.values()),
                tolerance=1e-5, checks=checks)


def assert_independent(front, arms):
    seen = {p.data_ptr() for p in front.parameters()}
    for arm in arms.values():
        current = {p.data_ptr() for p in arm.parameters()}
        if current & seen:
            raise RuntimeError('Parameter storage is shared across source/arms')
        seen |= current
    if any(p.requires_grad for p in front.parameters()) or any(m.training for m in front.modules()):
        raise RuntimeError('Source is not completely frozen/eval')


def cosine_warmup(step, steps):
    warmup = min(1000, max(1, steps // 10))
    return step / warmup if step <= warmup else 0.5 * (
        1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))
    )


def objective(probability, target):
    from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax
    if not (target != 255).any():
        return probability.sum() * 0
    return region_nll(probability, target) + lovasz_softmax(
        probability, target.clone(), classes='present', ignore=255,
    )
