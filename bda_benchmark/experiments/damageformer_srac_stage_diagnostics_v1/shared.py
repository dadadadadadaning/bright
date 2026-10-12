"""Small observation-only utilities. Never overwrite an existing experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def tensor_hash(value):
    cpu = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(cpu.shape), cpu.dtype)).encode())
    h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def source_hash():
    # Existing packages are dependencies, not files this experiment modifies.
    from bda_benchmark.experiments.damageformer_srac_partition_probe_v1.audit import source_digest
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.common import source_hash as joint_hash
    h = hashlib.sha256((source_digest() + joint_hash()).encode())
    for path in sorted(Path(__file__).parent.glob("*.py")) + [Path(__file__).parent / "run.sh"]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def transition_counts(target, reference, candidate, mask=None):
    """joint[GT, reference, candidate]; excluded GT labels never become classes."""
    if target.shape != reference.shape or target.shape != candidate.shape:
        raise ValueError("Transition maps have different shapes")
    y, a, b = (x.detach().cpu().long() for x in (target, reference, candidate))
    keep = (y >= 0) & (y < 4)
    if mask is not None:
        if mask.shape != target.shape or mask.dtype != torch.bool:
            raise ValueError("Transition mask must be bool and match target")
        keep &= mask.detach().cpu()
    if bool((((a < 0) | (a >= 4) | (b < 0) | (b >= 4)) & keep).any()):
        raise ValueError("Invalid predicted class")
    return torch.bincount(y[keep] * 16 + a[keep] * 4 + b[keep], minlength=64).reshape(4, 4, 4)


def transition_summary(joint):
    if joint.shape != (4, 4, 4) or bool((joint < 0).any()):
        raise ValueError("Need nonnegative [GT,reference,candidate] counts")
    classes = {}
    for c, name in enumerate(("Background", "Intact", "Damaged", "Destroyed")):
        gt_count = int(joint[c].sum())
        tp0, tp1 = int(joint[c, c].sum()), int(joint[c, :, c].sum())
        kept = int(joint[c, c, c])
        harm, rescue = tp0 - kept, tp1 - kept
        fp0 = int(joint[:, c].sum()) - tp0
        fp1 = int(joint[:, :, c].sum()) - tp1
        classes[name] = dict(gt_count=gt_count, reference_tp=tp0, candidate_tp=tp1,
                             harm=harm, rescue=rescue, net_tp=rescue-harm,
                             harm_rate=harm/tp0 if tp0 else None,
                             rescue_rate=rescue/(gt_count-tp0) if gt_count > tp0 else None,
                             reference_fp=fp0, candidate_fp=fp1, delta_fp=fp1-fp0)
    return dict(joint_gt_reference_candidate=joint.tolist(), per_class=classes)


def close_report(left, right, *, atol=2e-5, rtol=2e-5):
    """No repair, no retries, and no implicit tolerance increase."""
    a, b = left.detach().double().cpu(), right.detach().double().cpu()
    if a.shape != b.shape:
        return dict(passed=False, shape=list(a.shape), expected_shape=list(b.shape))
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = (a-b).abs()
    return dict(passed=finite and bool((delta <= atol + rtol*b.abs()).all()), finite=finite,
                max_abs=float(delta.max()) if finite else None,
                mean_abs=float(delta.mean()) if finite else None,
                outside_tolerance=int((delta > atol + rtol*b.abs()).sum()) if finite else None,
                atol=atol, rtol=rtol)
