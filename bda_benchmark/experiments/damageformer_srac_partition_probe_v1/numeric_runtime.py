"""Strict execution settings for diagnostics; no model or precision changes."""
from __future__ import annotations

import os

import torch


WORKSPACE_OPTIONS = (':4096:8', ':16:8')


def execution_settings():
    return dict(
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        # Record, but do not change the experiment's floating-point policy.
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )


def assert_determinism():
    settings = execution_settings()
    if not (settings['deterministic_algorithms']
            and not settings['deterministic_warn_only']
            and settings['cudnn_deterministic']
            and not settings['cudnn_benchmark']
            and settings['cublas_workspace_config'] in WORKSPACE_OPTIONS):
        raise RuntimeError('Strict deterministic execution is not active: ' + str(settings))
    return settings


def configure_determinism():
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace is None:
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA is already initialized; set CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                               'before starting a NEW Python process')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    elif workspace not in WORKSPACE_OPTIONS:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8; got ' + repr(workspace))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return assert_determinism()


def q_difference(left, right, ids, tolerance=2e-5):
    """Compare Q on CPU and distinguish candidate-slot from actual region-ID flips."""
    if left.shape != right.shape or left.ndim != 3 or min(left.shape) < 1:
        raise ValueError('Expected two Q tensors of the same nonempty [B,S,N] shape')
    if ids.ndim != 2 or tuple(ids.shape) != tuple(left.shape[1:]):
        raise ValueError('Candidate IDs must have shape [S,N] matching Q')
    left, right = left.detach().cpu(), right.detach().cpu()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return dict(passed=False, finite=False, bitwise_equal=False, max_abs=None, mean_abs=None,
                    elements_above_tolerance=None, candidate_argmax_changed_positions=None,
                    region_id_changed_positions=None)
    delta = (left.double() - right.double()).abs()
    a, b = left.argmax(1, keepdim=True), right.argmax(1, keepdim=True)
    candidate_ids = ids.detach().cpu()[None].expand(left.shape[0], -1, -1)
    maximum = float(delta.max())
    bitwise = left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    return dict(passed=maximum <= tolerance, finite=True, bitwise_equal=bitwise,
                max_abs=maximum, mean_abs=float(delta.mean()),
                elements_above_tolerance=int((delta > tolerance).sum()),
                candidate_argmax_changed_positions=int((a != b).sum()),
                region_id_changed_positions=int((candidate_ids.gather(1, a) !=
                                                 candidate_ids.gather(1, b)).sum()))
