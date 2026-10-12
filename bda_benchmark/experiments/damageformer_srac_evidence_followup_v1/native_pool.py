"""Opt-in, observation-only guard for the original front-end pooling operator.

No existing source file, checkpoint or tensor is changed. Within one context,
project modules which imported the original function are temporarily redirected.
The grouped backend preserves the original mass.clamp_min(1e-6), all duplicate
routes and the weighted reconstruction. In particular, Q * attention is NOT
renormalized. This is an audit operator, not a new training layer.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1 import association
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.grouped_pool import GroupedRouting


_ORIGINAL_POOL = association.soft_pool_unpool
_ACTIVE = None
_PACKAGE = __package__
ATOL = 2e-5
RTOL = 2e-5


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def tensor_layout(tensor):
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'stride': list(tensor.stride()),
            'storage_offset': tensor.storage_offset(), 'contiguous': tensor.is_contiguous()}


def validate_metadata(content, weights, ids, count):
    if not all(isinstance(t, torch.Tensor) for t in (content, weights, ids)):
        raise TypeError('Content, weights and candidate IDs must be tensors')
    if content.ndim != 4 or min(content.shape) < 1:
        raise ValueError('Content must be nonempty [B,C,H,W]')
    b, _, h, w = content.shape
    if tuple(weights.shape) != (b, 9, h*w) or tuple(ids.shape) != (9, h*w):
        raise ValueError('Original pooling requires weights[B,9,H*W], IDs[9,H*W]')
    if type(count) is not int or count < 1:
        raise ValueError('token_count must be a positive Python integer')
    if ids.dtype != torch.long:
        raise TypeError('Candidate IDs must have dtype torch.long')
    if not content.is_floating_point() or weights.dtype != content.dtype:
        raise TypeError('Content and weights must share their floating dtype')
    if content.device != weights.device or content.device != ids.device:
        raise ValueError('Content, weights and IDs must be on the same device')


def validate_cpu_snapshot(snapshot):
    content, weights, ids = (snapshot[key] for key in ('content', 'weights', 'ids'))
    count = snapshot['token_count']
    validate_metadata(content, weights, ids, count)
    if any(t.device.type != 'cpu' for t in (content, weights, ids)):
        raise ValueError('Validation snapshot must contain only CPU tensors')
    if not bool(torch.isfinite(content).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError('Nonfinite content or weights before pooling')
    if bool((weights < 0).any()):
        raise ValueError('Negative pooling weight before pooling')
    if bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError('Out-of-range candidate ID before pooling')
    # Q * attention is intentionally not constrained to sum to one.
    return {'finite': True, 'ids_in_range': True, 'nonnegative_weights': True,
            'id_min': int(ids.min()), 'id_max': int(ids.max()),
            'weight_min': float(weights.min()), 'weight_max': float(weights.max()),
            'weight_sum_min': float(weights.sum(1).min()),
            'weight_sum_max': float(weights.sum(1).max())}


def snapshot_inputs(content, weights, ids, count):
    validate_metadata(content, weights, ids, count)
    # Copies happen BEFORE the pooling kernel. A later CUDA fault must not require
    # any further device access in order to preserve this evidence.
    return {'format': 'native_pool_input_v1', 'token_count': count,
            'content': content.detach().cpu().contiguous().clone(),
            'weights': weights.detach().cpu().contiguous().clone(),
            'ids': ids.detach().cpu().contiguous().clone(),
            'layout': {name: tensor_layout(tensor) for name, tensor in
                       (('content', content), ('weights', weights), ('ids', ids))}}


def cpu_reference(snapshot):
    """Original CPU scatter implementation, including its exact denominator rule."""
    validate_cpu_snapshot(snapshot)
    with torch.no_grad():
        return _ORIGINAL_POOL(snapshot['content'], snapshot['weights'], snapshot['ids'],
                              snapshot['token_count'])


def grouped_pool(content, weights, ids, count, *, plan=None, max_work_elements=2**22):
    """Bounded gather/reduce alternative; no scatter, retry or CPU feature fallback.

    Only integer route planning is CPU-side. All floating arithmetic stays on
    the input device and in the input dtype. The guard is inference-only.
    """
    validate_metadata(content, weights, ids, count)
    if plan is None:
        plan = GroupedRouting(ids, count, max_work_elements=max_work_elements)
    elif plan.count != count or not torch.equal(plan.ids, ids.detach().cpu()):
        raise ValueError('Grouping plan does not match supplied IDs/count')
    batch, channels, height, width = content.shape
    if batch*channels > plan.max_work_elements:
        raise ValueError('Working budget is smaller than one batch/channel contribution')
    n = height*width
    rows = content.flatten(2).transpose(1, 2).contiguous()
    flat_weights = weights.reshape(batch, -1).contiguous()
    numerator = content.new_zeros(batch, count, channels)
    mass = content.new_zeros(batch, count)
    block_regions = min(plan.regions_per_block, plan.max_work_elements // (batch*channels))
    for begin in range(0, len(plan.nonempty), block_regions):
        group = plan.nonempty[begin:begin+block_regions]
        size = len(group)
        longest = max(plan.lengths[k] for k in group)
        chunk = min(longest, plan.max_work_elements // (batch*size*channels))
        block_num = content.new_zeros(batch, size, channels)
        block_mass = content.new_zeros(batch, size)
        for offset in range(0, longest, chunk):
            length = min(chunk, longest-offset)
            routes = torch.zeros(size, length, dtype=torch.long)
            present = torch.zeros(size, length, dtype=torch.bool)
            for j, k in enumerate(group):
                take = min(length, max(0, plan.lengths[k]-offset))
                if take:
                    start = plan.starts[k]+offset
                    routes[j, :take] = plan.order[start:start+take]
                    present[j, :take] = True
            device_routes = routes.reshape(-1).to(content.device)
            ww = flat_weights.index_select(1, device_routes).reshape(batch, size, length)
            ww = torch.where(present.to(content.device)[None], ww, torch.zeros_like(ww))
            values = rows.index_select(1, device_routes.remainder(n)).reshape(batch, size, length, channels)
            block_num = block_num + (values*ww[..., None]).sum(2)
            block_mass = block_mass + ww.sum(2)
        for j, k in enumerate(group):
            numerator[:, k:k+1] = block_num[:, j:j+1]
            mass[:, k:k+1] = block_mass[:, j:j+1]
    tokens = numerator / mass.clamp_min(1e-6)[..., None]
    reconstruction = content.new_zeros(batch, n, channels)
    for candidate in range(9):
        index = ids[candidate].reshape(1, n, 1).expand(batch, n, channels)
        reconstruction = reconstruction + tokens.gather(1, index)*weights[:, candidate, :, None]
    return {'tokens': tokens, 'mass': mass,
            'reconstruction': reconstruction.transpose(1, 2).reshape_as(content)}


def compare_results(actual, expected):
    """Fixed absolute+relative tolerance; no caller option can relax the gate."""
    result = {}
    for key in ('tokens', 'mass', 'reconstruction'):
        aa = actual[key].detach().cpu()
        bb = expected[key].detach().cpu()
        if aa.shape != bb.shape or aa.dtype != bb.dtype:
            result[key] = {'passed': False, 'reason': 'shape_or_dtype_mismatch'}
            continue
        finite = bool(torch.isfinite(aa).all()) and bool(torch.isfinite(bb).all())
        diff = (aa.double()-bb.double()).abs()
        bad = diff > (ATOL+RTOL*bb.double().abs())
        result[key] = {'passed': finite and not bool(bad.any()), 'finite': finite,
                       'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                       'elements_outside_tolerance': int(bad.sum()),
                       'atol': ATOL, 'rtol': RTOL}
    result['passed'] = all(result[key]['passed'] for key in ('tokens', 'mass', 'reconstruction'))
    return result


class PoolGuard:
    """Temporarily intercept native project pooling calls for a single audit.

    ``capture='failure'`` keeps a CPU pre-kernel snapshot per call, writing it
    only if that call fails. ``capture='all'`` writes every input and should be
    reserved for a single-image diagnostic: snapshots may be large.
    """

    def __init__(self, output, mode='native', capture='failure', check_reference=False):
        if mode not in ('native', 'grouped') or capture not in ('failure', 'all'):
            raise ValueError('Unknown pool backend or capture mode')
        self.output = Path(output)
        self.mode, self.capture = mode, capture
        self.check_reference = bool(check_reference)
        self.image = None
        self.calls = 0
        self.failures = []
        self.reference_checks = 0
        self.saved_inputs = []
        self.patched = []
        self.restore_conflicts = []
        self.restored = False
        self.last_check = None
        self.plan = None
        self.wrapper = self._call
        self._entered = False

    def set_image(self, name):
        self.image = str(name)

    def report(self):
        # Deliberately uses no CUDA or tensor operation, even after a poisoned
        # CUDA context. A successful operator is not a completed Val audit.
        return {'backend': self.mode, 'capture': self.capture, 'calls': self.calls,
                'reference_checks': self.reference_checks,
                'reference_enabled': self.check_reference,
                'tolerance': {'atol': ATOL, 'rtol': RTOL},
                'failures': list(self.failures), 'saved_inputs': list(self.saved_inputs),
                'patched': list(self.patched), 'restored': self.restored,
                'restore_conflicts': list(self.restore_conflicts),
                'last_input_check': self.last_check, 'image': self.image,
                'training': False, 'automatic_cpu_fallback': False,
                'full_validation_evidence': False}

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None or self._entered:
            raise RuntimeError('PoolGuard is non-nestable and cannot be reused')
        self.output.mkdir(parents=True, exist_ok=False)
        source = Path(inspect.getsourcefile(_ORIGINAL_POOL))
        _json(self.output/'manifest.json', {
            'backend': self.mode, 'capture': self.capture, 'check_reference': self.check_reference,
            'original_source': str(source), 'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'semantics': 'original clamped mass 1e-6; duplicate routes; unnormalized attention weights',
            'training': False, 'no_source_files_modified': True})
        self._entered = True
        _ACTIVE = self
        for name, module in list(sys.modules.items()):
            if not name.startswith('bda_benchmark.') or name.startswith(_PACKAGE) or module is None:
                continue
            for attr, value in list(vars(module).items()):
                if value is _ORIGINAL_POOL:
                    setattr(module, attr, self.wrapper)
                    self.patched.append(name+'.'+attr)
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE
        # Also restore project modules imported after entry: a from-import of
        # association during the context may have picked up the wrapper.
        for name, module in list(sys.modules.items()):
            if not name.startswith('bda_benchmark.') or name.startswith(_PACKAGE) or module is None:
                continue
            for attr, value in list(vars(module).items()):
                if value is self.wrapper:
                    setattr(module, attr, _ORIGINAL_POOL)
        for name in self.patched:
            module_name, attr = name.rsplit('.', 1)
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, attr, None) is not _ORIGINAL_POOL:
                self.restore_conflicts.append(name)
        self.restored = not self.restore_conflicts
        _ACTIVE = None
        _json(self.output/'guard.json', self.report())
        return False

    def _save(self, snapshot):
        path = self.output/f'input_{self.calls:06d}.pth'
        if path.exists():
            raise FileExistsError('Refusing to overwrite pooling evidence: '+str(path))
        torch.save(snapshot, path)
        self.saved_inputs.append(str(path))
        return str(path)

    def _call(self, content, probabilities, candidate_ids, token_count):
        if not self._entered or _ACTIVE is not self:
            raise RuntimeError('Pooling guard used outside its context')
        self.calls += 1
        snapshot = None
        saved = None
        phase = 'metadata'
        self.last_check = None
        try:
            validate_metadata(content, probabilities, candidate_ids, token_count)
            if torch.is_grad_enabled() and (content.requires_grad or probabilities.requires_grad):
                raise RuntimeError('PoolGuard is audit-only; gradient-enabled training is not supported')
            phase = 'cpu_snapshot_before_kernel'
            snapshot = snapshot_inputs(content, probabilities, candidate_ids, token_count)
            snapshot.update({'image': self.image, 'call': self.calls, 'backend': self.mode})
            caller = inspect.currentframe().f_back
            snapshot['caller'] = {'file': caller.f_code.co_filename,
                                  'function': caller.f_code.co_name, 'line': caller.f_lineno}
            del caller
            phase = 'input_validation'
            self.last_check = validate_cpu_snapshot(snapshot)
            if self.capture == 'all':
                saved = self._save(snapshot)
            phase = 'pool_kernel'
            _json(self.output/'progress.json', {'image': self.image, 'call': self.calls,
                  'phase': phase, 'backend': self.mode, 'input': saved,
                  'layout': snapshot['layout'], 'input_check': self.last_check})
            if self.mode == 'native':
                result = _ORIGINAL_POOL(content, probabilities, candidate_ids, token_count)
            else:
                if (self.plan is None or self.plan.count != token_count
                        or not torch.equal(self.plan.ids, snapshot['ids'])):
                    self.plan = GroupedRouting(snapshot['ids'], token_count)
                result = grouped_pool(content, probabilities, candidate_ids, token_count, plan=self.plan)
            # Synchronizing here attributes a CUDA failure to this call rather
            # than to the next audit/statistics operation.
            if content.device.type == 'cuda':
                torch.cuda.synchronize(content.device)
            phase = 'output_finite'
            if any(not bool(torch.isfinite(t).all()) for t in result.values()):
                raise FloatingPointError('Nonfinite native pooling output')
            if self.check_reference:
                phase = 'cpu_reference_gate'
                comparison = compare_results(result, cpu_reference(snapshot))
                self.reference_checks += 1
                if not comparison['passed']:
                    snapshot['comparison'] = comparison
                    raise ValueError('Original-semantics CPU pooling reference gate failed')
            _json(self.output/'progress.json', {'image': self.image, 'call': self.calls,
                  'phase': 'complete', 'backend': self.mode, 'input': saved})
            return result
        except Exception as error:
            # Do not access original CUDA tensors here; the context can be bad.
            save_error = None
            if snapshot is not None and saved is None:
                try:
                    saved = self._save(snapshot)
                except Exception as write_error:
                    save_error = repr(write_error)
            failure = {'image': self.image, 'call': self.calls, 'phase': phase,
                       'type': type(error).__name__, 'message': str(error),
                       'input': saved, 'snapshot_save_error': save_error,
                       'comparison': snapshot.get('comparison') if snapshot is not None else None,
                       'restart_process_required': 'cuda' in str(error).lower()}
            self.failures.append(failure)
            _json(self.output/'failure.json', failure)
            raise
