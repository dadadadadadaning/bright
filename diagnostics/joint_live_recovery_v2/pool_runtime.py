"""Scoped frozen-front pooling and live evidence; no silent recovery fallback.

cpu_fp32_reference explicitly executes original CPU FP32 arithmetic on the
already validated pre-kernel snapshot and transfers its outputs to the caller's
device. This changes the execution device, not dtype, routing or denominator.
It can be substantially slower. CUDA backbone/Reader execution is unchanged.
Real CUDA operation and performance require server verification.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path

import torch

from pool_ops import contiguous_index_algorithm


CPU_BACKEND = "cpu_fp32_reference"
LIVE_BACKEND = "live_contiguous_probe"
BACKENDS = (CPU_BACKEND, LIVE_BACKEND)
_PATCH_ACTIVE = False


class ProbeComplete(Exception):
    """Expected stop outside a successful guarded stats call, before training."""
    def __init__(self, report):
        self.report = dict(report)
        super().__init__("One stats512 diagnostic captured; no training update permitted")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transfer_outputs(outputs, device):
    """Only transfer dtype-preserving original CPU results to the input device."""
    return {key: value.to(device=device) for key, value in outputs.items()}


def cpu_pool_from_snapshot(snapshot, device, *, original_pool):
    if any(snapshot[key].device.type != "cpu" for key in ("content", "weights", "ids")):
        raise ValueError("CPU backend requires the validated pre-kernel CPU snapshot")
    if snapshot["content"].dtype != torch.float32 or snapshot["weights"].dtype != torch.float32:
        raise TypeError("Frozen CPU pooling requires exact FP32 content and weights")
    result = original_pool(snapshot["content"], snapshot["weights"], snapshot["ids"],
                           snapshot["token_count"])
    if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in result.values()):
        raise TypeError("Original CPU pooling returned a non-CPU or non-FP32 output")
    return transfer_outputs(result, device)


def _execution_state():
    # Capture this before arithmetic. After a CUDA error, report() uses saved
    # Python fields only and never calls this function or queries a CUDA device.
    default_device = str(torch.get_default_device()) if hasattr(torch, "get_default_device") else "unreported"
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "original_num_threads": torch.get_num_threads(),
        "cpu_execution_num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()), "default_device": default_device,
        "grad_enabled": torch.is_grad_enabled(), "inference_mode": torch.is_inference_mode_enabled(),
        "autocast_cuda_enabled": torch.is_autocast_enabled(),
        "autocast_cpu_enabled": torch.is_autocast_cpu_enabled(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def make_guard(native_pool_module, *, backend=CPU_BACKEND, stop_after_stats=False, binding=None):
    """Build a guard subclass; installation is done by patched_guard()."""
    if backend not in BACKENDS:
        raise ValueError("Unknown explicit pooling backend: " + str(backend))
    base_guard = native_pool_module.PoolGuard
    if getattr(base_guard, "execution_backend", None) is not None:
        raise RuntimeError("Scoped live guard factory cannot be nested")
    original_pool = native_pool_module._ORIGINAL_POOL
    original_snapshot = native_pool_module.snapshot_inputs
    original_compare = native_pool_module.compare_results
    source = Path(inspect.getsourcefile(original_pool))
    source_record = {"original_source": str(source), "original_sha256": _sha256(source)}
    stable_binding = json.loads(json.dumps(binding or {}, allow_nan=False))

    class FrozenPoolRuntimeGuard(base_guard):
        execution_backend = backend

        def __init__(self, output, mode="native", capture="failure", check_reference=False):
            if mode != "native":
                raise ValueError("Explicit live/CPU execution requires inherited mode='native', not grouped")
            super().__init__(output, mode=mode, capture=capture, check_reference=check_reference)
            self.reference_all_calls = bool(check_reference)
            self._tested_keys = set()
            self.tested_layouts = []
            self.captured_packets = []
            self.metadata_errors = []
            self._snapshot = None
            self._source_caller = None
            self._execution_meta = None
            self.cpu_arithmetic_calls = 0
            self.cuda_scatter_calls = 0

        def report(self):
            value = super().report()
            value.update(backend=backend, original_backend="native",
                         cpu_reference_backend="captured_original_cpu_fp32",
                         reference_enabled=True,
                         reference_policy="first_unique_layout_or_every_call_if_requested",
                         reference_all_calls=self.reference_all_calls,
                         tested_layouts=list(self.tested_layouts),
                         captured_packets=list(self.captured_packets),
                         metadata_errors=list(self.metadata_errors),
                         cpu_arithmetic_calls=self.cpu_arithmetic_calls,
                         cuda_scatter_calls=self.cuda_scatter_calls,
                         explicit_cpu_execution=backend == CPU_BACKEND,
                         automatic_cpu_fallback=False, no_id_clamping=True,
                         pooling_dtype="torch.float32", precision_intervention=False,
                         execution_device_intervention=backend == CPU_BACKEND,
                         frozen_front_pool_only=True, binding=stable_binding,
                         stop_after_stats=bool(stop_after_stats),
                         slowdown_expected=backend == CPU_BACKEND)
            return value

        def _annotate_records(self):
            for name in ("manifest.json", "progress.json", "failure.json"):
                path = self.output / name
                if not path.is_file():
                    continue
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value.update(backend=backend, original_backend="native",
                                 explicit_cpu_execution=backend == CPU_BACKEND,
                                 automatic_cpu_fallback=False, precision_intervention=False)
                    if name == "manifest.json":
                        value.update(check_reference=True,
                                     reference_policy="first_unique_layout_or_every_call_if_requested",
                                     binding=stable_binding, slowdown_expected=backend == CPU_BACKEND)
                    native_pool_module._json(path, value)
                except (OSError, ValueError, TypeError) as error:
                    self.metadata_errors.append({"file": str(path), "error": repr(error)})

        def __enter__(self):
            if native_pool_module._ORIGINAL_POOL is not original_pool:
                raise RuntimeError("Original callable changed before entry")
            value = super().__enter__()
            self._annotate_records()
            return value

        def _save(self, snapshot):
            snapshot.update(backend=backend, original_backend="native")
            if self._source_caller is not None:
                snapshot["caller"] = dict(self._source_caller)
            return super()._save(snapshot)

        def _capture_compare(self, actual, expected):
            # Parent reached this only after kernel synchronization and finite
            # checks. These copies are the EXACT objects compared by the gate.
            actual_host = {key: value.detach().cpu().clone() for key, value in actual.items()}
            expected_host = {key: value.detach().cpu().clone() for key, value in expected.items()}
            snapshot = dict(self._snapshot)
            snapshot.update(backend=backend, original_backend="native",
                            image=self.image, call=self.calls, caller=dict(self._source_caller))
            meta = {**self._execution_meta, **source_record,
                    "backend": backend, "image": self.image, "call": self.calls,
                    "content_channels": int(snapshot["content"].shape[1]),
                    "arithmetic_device": ("cpu" if backend == CPU_BACKEND else
                                          snapshot["layout"]["content"]["device"]),
                    "returned_device": snapshot["layout"]["content"]["device"],
                    "layout": snapshot["layout"], "binding": stable_binding,
                    "captured_before_original_comparison": True,
                    "original_cpu_reference_callable": original_pool.__module__ + "." + original_pool.__name__}
            packet = {"format": "LIVE_POOL_CAPTURE_V2", "snapshot": snapshot,
                      "actual": actual_host, "expected": expected_host, "meta": meta}
            path = self.output / f"packet_{self.calls:06d}.pth"
            if path.exists():
                raise FileExistsError("Refusing to overwrite live output evidence: " + str(path))
            torch.save(packet, path)
            entry = {"path": str(path.resolve()), "packet": str(path.resolve()), "sha256": _sha256(path),
                     "backend": backend, "image": self.image, "call": self.calls,
                     "content_channels": meta["content_channels"],
                     "captured_before_original_comparison": True}
            self.captured_packets.append(entry)
            native_pool_module._json(path.with_suffix(".json"), {**entry, "meta": meta})
            # Dtype checks and fixed asymmetric tolerances remain the parent's
            # exact function. No backend choice is made based on this verdict.
            comparison = original_compare(actual_host, expected_host)
            entry.update(strict_cpu_gate_passed=comparison["passed"],
                         strict_original_cpu32_gate=comparison)
            native_pool_module._json(path.with_suffix(".json"), {**entry, "meta": meta})
            return comparison

        def _call(self, content, probabilities, candidate_ids, token_count):
            if (native_pool_module._ORIGINAL_POOL is not original_pool
                    or native_pool_module.snapshot_inputs is not original_snapshot
                    or native_pool_module.compare_results is not original_compare):
                raise RuntimeError("Concurrent/nested pooling instrumentation is not supported")
            try:
                signature = {"token_count": token_count,
                    "layout": {name: native_pool_module.tensor_layout(value) for name, value in
                               (("content", content), ("weights", probabilities), ("ids", candidate_ids))}}
                key = json.dumps(signature, sort_keys=True)
            except (AttributeError, TypeError, ValueError):
                signature, key = None, None
            enabled = self.reference_all_calls or key not in self._tested_keys
            previous_reference = self.check_reference
            self.check_reference = enabled
            self._execution_meta = _execution_state()
            caller = inspect.currentframe().f_back
            self._source_caller = {"file": caller.f_code.co_filename,
                                   "function": caller.f_code.co_name, "line": caller.f_lineno}
            del caller

            def snapshot_inputs(c, p, ids, count):
                self._snapshot = original_snapshot(c, p, ids, count)
                return self._snapshot

            def dispatch(c, p, ids, count):
                if c.dtype != torch.float32 or p.dtype != torch.float32:
                    raise TypeError("Frozen pooling execution requires FP32 source values")
                if c.device.type == "cpu":
                    self.cpu_arithmetic_calls += 1
                    return original_pool(c, p, ids, count)
                if c.device.type != "cuda":
                    raise ValueError("Only CPU or CUDA caller tensors are supported")
                if backend == LIVE_BACKEND:
                    self.cuda_scatter_calls += 1
                    return contiguous_index_algorithm(c, p, ids, count)
                if self._snapshot is None or self._snapshot["token_count"] != count:
                    raise RuntimeError("No matching validated pre-kernel snapshot for explicit CPU execution")
                self.cpu_arithmetic_calls += 1
                return cpu_pool_from_snapshot(self._snapshot, c.device, original_pool=original_pool)

            native_pool_module._ORIGINAL_POOL = dispatch
            native_pool_module.snapshot_inputs = snapshot_inputs
            native_pool_module.compare_results = self._capture_compare
            try:
                result = super()._call(content, probabilities, candidate_ids, token_count)
                if enabled and key is not None and key not in self._tested_keys:
                    self._tested_keys.add(key)
                    self.tested_layouts.append({**signature, "first_call": self.calls,
                                               "cpu_reference_passed": True})
            finally:
                # Python/file restoration only; no CUDA query after an error.
                native_pool_module._ORIGINAL_POOL = original_pool
                native_pool_module.snapshot_inputs = original_snapshot
                native_pool_module.compare_results = original_compare
                self.check_reference = previous_reference
                self._snapshot = None
                self._source_caller = None
                self._execution_meta = None
                self._annotate_records()
            # Deliberate diagnostic stop OUTSIDE the parent's _call success
            # path, so it is not misrecorded as a pooling kernel failure.
            if stop_after_stats and content.shape[1] == 512:
                packet = next((entry for entry in reversed(self.captured_packets)
                               if entry["call"] == self.calls), None)
                if packet is None or packet.get("strict_cpu_gate_passed") is not True:
                    raise RuntimeError("Successful stats probe lacks its exact comparison packet")
                raise ProbeComplete({**packet, "diagnostic_stop": True,
                                     "training_started": False})
            return result

    FrozenPoolRuntimeGuard.__name__ = "FrozenPoolRuntimeGuard"
    return FrozenPoolRuntimeGuard


@contextmanager
def patched_guard(native_pool_module, *, backend=CPU_BACKEND, stop_after_stats=False, binding=None):
    """Temporarily install the explicitly selected guard; never auto-switch it."""
    global _PATCH_ACTIVE
    if _PATCH_ACTIVE or native_pool_module._ACTIVE is not None:
        raise RuntimeError("Scoped pooling runtime cannot be nested or installed over an active guard")
    preserved = {key: getattr(native_pool_module, key) for key in
                 ("PoolGuard", "_ORIGINAL_POOL", "snapshot_inputs", "compare_results")}
    guard_class = make_guard(native_pool_module, backend=backend,
                             stop_after_stats=stop_after_stats, binding=binding)
    _PATCH_ACTIVE = True
    native_pool_module.PoolGuard = guard_class
    try:
        yield guard_class
    finally:
        for key, value in preserved.items():
            setattr(native_pool_module, key, value)
        _PATCH_ACTIVE = False
