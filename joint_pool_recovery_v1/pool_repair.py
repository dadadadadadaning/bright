"""Process-local candidate for frozen-front pooling, never an automatic retry.

CUDA calls materialize independent contiguous integer indexing tensors. CPU
calls, including the guard's reference, use the captured original callable.
The original guard still validates and snapshots inputs before every kernel.
This candidate has not been validated on a real GPU by the CPU test suite.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import json

from pool_ops import contiguous_index_algorithm


BACKEND = "native_contiguous_indices_candidate"
_PATCH_ACTIVE = False


def dispatch_pool(content, probabilities, candidate_ids, token_count, *, original_pool,
                  candidate_pool=None):
    """Use the candidate on CUDA only; retain independent original CPU math."""
    if content.device.type == "cuda":
        algorithm = contiguous_index_algorithm if candidate_pool is None else candidate_pool
    elif content.device.type == "cpu":
        algorithm = original_pool
    else:
        raise ValueError("Pool recovery supports only CPU or CUDA inputs")
    return algorithm(content, probabilities, candidate_ids, token_count)


def make_candidate_guard(native_pool_module):
    """Return a subclass of this module's original guard, without installing it."""
    base_guard = native_pool_module.PoolGuard
    if getattr(base_guard, "candidate_backend", None) is not None:
        raise RuntimeError("Candidate guard class factory cannot be nested")
    baseline_pool = native_pool_module._ORIGINAL_POOL

    class ContiguousIndexPoolGuard(base_guard):
        candidate_backend = BACKEND
        original_backend = "native"

        def __init__(self, output, mode="native", capture="failure", check_reference=False):
            if mode != "native":
                raise ValueError("Contiguous index recovery requires mode='native'; grouped is a different intervention")
            super().__init__(output, mode=mode, capture=capture, check_reference=check_reference)
            self.reference_all_calls = bool(check_reference)
            self._tested_keys = set()
            self.tested_layouts = []
            self.metadata_errors = []
            self._source_caller = None

        def report(self):
            value = super().report()
            value.update(backend=BACKEND, original_backend="native",
                         cpu_reference_backend="captured_original_cpu_scatter",
                         reference_enabled=True,
                         reference_policy="first_unique_layout_or_every_call_if_requested",
                         reference_all_calls=self.reference_all_calls,
                         tested_layouts=list(self.tested_layouts),
                         metadata_errors=list(self.metadata_errors),
                         candidate_intervention="clone IDs and materialize expanded integer indices",
                         no_id_clamping=True, automatic_cpu_fallback=False)
            return value

        def _annotate_records(self):
            # File-only operations remain safe after a CUDA assert. Never query
            # tensors, synchronize, clear a cache or retry in this path.
            for name in ("manifest.json", "progress.json", "failure.json"):
                path = self.output / name
                if not path.is_file():
                    continue
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    value.update(backend=BACKEND, original_backend="native")
                    if name == "manifest.json":
                        value.update(check_reference=True,
                                     cpu_reference_backend="captured_original_cpu_scatter",
                                     reference_policy="first_unique_layout_or_every_call_if_requested",
                                     candidate_intervention="clone IDs and materialize expanded integer indices",
                                     no_id_clamping=True, automatic_cpu_fallback=False)
                    native_pool_module._json(path, value)
                except (OSError, ValueError, TypeError) as error:
                    # Preserve the original exception if evidence annotation
                    # encounters a filesystem problem.
                    self.metadata_errors.append({"file": str(path), "error": repr(error)})

        def __enter__(self):
            if native_pool_module._ORIGINAL_POOL is not baseline_pool:
                raise RuntimeError("Original pool callable changed before guard entry")
            # Parent entry must inspect and intercept the unchanged callable.
            result = super().__enter__()
            self._annotate_records()
            return result

        def _save(self, snapshot):
            snapshot.update(backend=BACKEND, original_backend="native")
            if self._source_caller is not None:
                snapshot["caller"] = dict(self._source_caller)
            return super()._save(snapshot)

        def _call(self, content, probabilities, candidate_ids, token_count):
            if native_pool_module._ORIGINAL_POOL is not baseline_pool:
                raise RuntimeError("Concurrent/nested pool dispatch is not supported")
            # Layout queries do not launch CUDA arithmetic. The parent retains
            # responsibility for recording metadata/range/finite failures.
            try:
                signature = {
                    "token_count": token_count,
                    "layout": {name: native_pool_module.tensor_layout(tensor) for name, tensor in
                               (("content", content), ("weights", probabilities), ("ids", candidate_ids))},
                }
                key = json.dumps(signature, sort_keys=True)
            except (AttributeError, TypeError, ValueError):
                signature, key = None, None
            enabled = self.reference_all_calls or key not in self._tested_keys
            previous_check = self.check_reference
            self.check_reference = enabled
            caller = inspect.currentframe().f_back
            self._source_caller = {"file": caller.f_code.co_filename,
                                   "function": caller.f_code.co_name, "line": caller.f_lineno}
            del caller

            def dispatched(c, p, i, count):
                return dispatch_pool(c, p, i, count, original_pool=baseline_pool)

            native_pool_module._ORIGINAL_POOL = dispatched
            try:
                result = super()._call(content, probabilities, candidate_ids, token_count)
                if enabled and key is not None and key not in self._tested_keys:
                    self._tested_keys.add(key)
                    self.tested_layouts.append({**signature, "first_call": self.calls,
                                               "cpu_reference_passed": True})
                return result
            finally:
                # Restore Python references even when CUDA is poisoned. The
                # parent CPU reference sees the dispatcher but dispatches to
                # baseline_pool, never to the candidate algorithm.
                native_pool_module._ORIGINAL_POOL = baseline_pool
                self.check_reference = previous_check
                self._source_caller = None
                self._annotate_records()

    ContiguousIndexPoolGuard.__name__ = "ContiguousIndexPoolGuard"
    return ContiguousIndexPoolGuard


@contextmanager
def patched_pool_guard(native_pool_module):
    """Install the subclass only inside a single non-nested Python scope.

    Use the returned class or code which constructs native_pool_module.PoolGuard
    inside this scope. Source files and project configuration are not edited.
    """
    global _PATCH_ACTIVE
    if _PATCH_ACTIVE or native_pool_module._ACTIVE is not None:
        raise RuntimeError("Pool recovery patch cannot be nested or installed over an active guard")
    original_class = native_pool_module.PoolGuard
    original_pool = native_pool_module._ORIGINAL_POOL
    candidate_class = make_candidate_guard(native_pool_module)
    _PATCH_ACTIVE = True
    native_pool_module.PoolGuard = candidate_class
    try:
        yield candidate_class
    finally:
        native_pool_module.PoolGuard = original_class
        native_pool_module._ORIGINAL_POOL = original_pool
        _PATCH_ACTIVE = False
