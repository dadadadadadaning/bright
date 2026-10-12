"""Explicit runtime profiles; legacy means startup defaults, not proven history."""
from __future__ import annotations

import os
import platform

import torch

from bda_benchmark.experiments.damageformer_srac_partition_probe_v1.numeric_runtime import (
    configure_determinism, execution_settings,
)


def runtime_settings():
    """Stable execution/environment fields; no transient allocator/CUDA state."""
    gpu = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = dict(index=index, name=properties.name,
                   capability=list(torch.cuda.get_device_capability(index)),
                   total_memory=properties.total_memory)
    return dict(
        **execution_settings(), torch_version=str(torch.__version__),
        python_version=platform.python_version(), platform=platform.platform(),
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        cuda_available=torch.cuda.is_available(), device=gpu,
        num_threads=torch.get_num_threads(), num_interop_threads=torch.get_num_interop_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        default_dtype=str(torch.get_default_dtype()), cudnn_enabled=torch.backends.cudnn.enabled,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        environment={key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    )


def configure_runtime(profile="legacy", matmul_tf32=None, cudnn_tf32=None, deterministic=None):
    """Configure before CUDA initialization, and run each contrast in a fresh process.

    ``legacy`` without overrides leaves startup numeric defaults untouched. It
    reproduces the historical entry point's *policy*, not undocumented past
    library/hardware defaults. ``deterministic`` is one explicit policy group:
    algorithms, warn-only, cuDNN deterministic/benchmark, and cuBLAS workspace.
    False sets algorithms=False, warn_only=False, cudnn deterministic=False,
    benchmark=False, and removes the cuBLAS workspace variable.
    """
    if profile not in ("legacy", "strict"):
        raise ValueError("Unknown numeric profile: " + str(profile))
    for name, value in (("matmul_tf32", matmul_tf32), ("cudnn_tf32", cudnn_tf32),
                        ("deterministic", deterministic)):
        if value is not None and type(value) is not bool:
            raise TypeError(name + " must be bool or None")
    if profile == "strict":
        configure_determinism()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if deterministic is True:
        configure_determinism()
    elif deterministic is False:
        if torch.cuda.is_initialized():
            raise RuntimeError("Determinism overrides require a fresh process before CUDA initialization")
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if matmul_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    if cudnn_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
    return runtime_settings()
