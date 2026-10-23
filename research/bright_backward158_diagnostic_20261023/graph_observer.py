"""Bounded, fail-closed observation of autograd's saved gather/sort indices.

This module never replaces PyTorch functions, installs saved_tensors_hooks,
repairs indices, or modifies gradients. Instrumentation synchronizes CUDA and
changes execution timing; a passing replay does not resolve a historical fault.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


class ObservationError(RuntimeError):
    pass


def compact_view(tensor):
    """Remove only redundant zero-stride axes, before any device-to-host copy."""
    value = tensor.detach()
    for axis in reversed(range(value.ndim)):
        if value.shape[axis] > 1 and value.stride(axis) == 0:
            value = value.select(axis, 0)
    return value


def normalize_dim(raw_dim, ndim):
    dim = int(raw_dim)
    if dim >= (1 << 63):
        dim -= 1 << 64  # PyTorch 2.1 saves some negative dims as uint64.
    if ndim < 1 or not -ndim <= dim < ndim:
        raise ObservationError(f"invalid saved dimension {raw_dim} for ndim={ndim}")
    return dim % ndim


def index_statistics(cpu, bound):
    if cpu.device.type != "cpu":
        raise ValueError("index_statistics accepts CPU tensors only")
    integral = cpu.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    low = int(cpu.min()) if cpu.numel() else None
    high = int(cpu.max()) if cpu.numel() else None
    return {
        "numel": cpu.numel(), "min": low, "max": high, "bound": int(bound),
        "integer_dtype": integral,
        "in_bounds": integral and (not cpu.numel() or (low >= 0 and high < bound)),
    }


def _metadata(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        # Inference tensors have no version counter. Saved-tensor getters below
        # still enforce every version check that PyTorch itself supports.
        version = None
    return {
        "device": str(tensor.device), "dtype": str(tensor.dtype),
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "offset": tensor.storage_offset(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "version": version,
    }


def _key(tensor):
    return (str(tensor.device), str(tensor.dtype),
            tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()))


def _same(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)))


def _json_write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


class BackwardIndexObserver:
    evidence_limit_bytes = 32 * 1024 ** 2

    def __init__(self, output, step, max_bytes=128 * 1024 ** 2):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.step = int(step)
        self.max_bytes = int(max_bytes)
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.snapshot_bytes = 0
        self.nodes = []
        self.latest_node = None
        self.coverage_complete = True
        self.graph_nodes_seen = 0
        self.prehook_calls = 0
        self.watched_checks = 0
        self._snapshots = {}
        self._watches = {}
        self._handles = []
        self._node_refs = []
        self._latest_cpu = None
        self._first_bad = None
        self._closed = False
        self._attached = False
        self._failure_written = False
        self._summary = None

    def _path(self, kind, suffix="json"):
        return self.output / f"graph_{kind}_{self.step:06d}.{suffix}"

    def _snapshot(self, tensor, name):
        compact = compact_view(tensor)
        key = _key(compact)
        if key not in self._snapshots:
            required = compact.numel() * compact.element_size()
            if self.snapshot_bytes + required > self.max_bytes:
                self.coverage_complete = False
                raise ObservationError(
                    f"CPU snapshot budget exceeded at {name}: "
                    f"{self.snapshot_bytes}+{required}>{self.max_bytes}; coverage incomplete")
            cpu = compact.to(device="cpu", copy=True).contiguous()
            self._snapshots[key] = cpu
            self.snapshot_bytes += required
        return key, self._snapshots[key]

    def _current_cpu(self, tensor, name):
        compact = compact_view(tensor)
        required = compact.numel() * compact.element_size()
        if required > self.max_bytes:
            self.coverage_complete = False
            raise ObservationError(
                f"CPU current-tensor budget exceeded at {name}: "
                f"{required}>{self.max_bytes}; coverage incomplete")
        # Retain only one current copy; persistent snapshots have a separate
        # max_bytes bound. This prevents copies accumulating across graph nodes.
        self._latest_cpu = None
        return compact.to(device="cpu", copy=True).contiguous()

    def watch(self, name, tensor):
        if self._closed:
            raise ObservationError("observer is closed")
        if name in self._watches:
            raise ValueError(f"duplicate watch: {name}")
        key, _ = self._snapshot(tensor, name)
        self._watches[name] = {"tensor": tensor, "key": key, "metadata": _metadata(tensor)}

    def _mark_bad(self, label, before, after, details):
        if self._first_bad is None:
            self._first_bad = {"label": label, "before": before, "after": after,
                               "details": details}
        error = ObservationError(f"{label}: saved/immutable tensor changed or index out of bounds")
        self.record_failure(error)
        raise error

    def check_watched(self, stage):
        for name, item in self._watches.items():
            self.latest_node = {"kind": "watch", "name": name, "stage": str(stage)}
            metadata = _metadata(item["tensor"])
            after = self._current_cpu(item["tensor"], name)
            before = self._snapshots[item["key"]]
            self._latest_cpu = {"label": name, "before": before, "after": after}
            self.watched_checks += 1
            changed = not _same(before, after)
            metadata_changed = metadata != item["metadata"]
            if changed or metadata_changed:
                self._mark_bad(name, before, after, {
                    "stage": str(stage), "values_changed": changed,
                    "metadata_before": item["metadata"], "metadata_after": metadata,
                })

    @staticmethod
    def _saved(node):
        kind = type(node).__name__
        if kind.startswith("GatherBackward"):
            index = node._saved_index
            sizes = tuple(node._saved_self.shape)
            field = "_saved_index"
        elif kind.startswith("SortBackward"):
            index = node._saved_indices
            sizes = tuple(node._saved_self_sym_sizes)
            field = "_saved_indices"
        else:
            raise ObservationError(f"unsupported index node: {kind}")
        dim = normalize_dim(node._saved_dim, len(sizes))
        return index, sizes, dim, field

    def attach(self, loss):
        if self._closed or self._attached:
            raise ObservationError("attach must be called once on an open observer")
        self._attached = True
        if loss.grad_fn is None:
            self.coverage_complete = False
            raise ObservationError("loss has no autograd graph")
        pending = [loss.grad_fn]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            self._node_refs.append(node)  # Keep wrappers alive while deduplicating IDs.
            self.graph_nodes_seen += 1
            pending.extend(parent for parent, _ in node.next_functions if parent is not None)
            kind = type(node).__name__
            if not kind.startswith(("GatherBackward", "SortBackward")):
                continue
            record = {"id": len(self.nodes), "type": kind, "stage": "attach_begin",
                      "forward_traceback": list(node.metadata.get("traceback_", []))}
            self.nodes.append(record)
            self.latest_node = dict(record)
            index, sizes, dim, field = self._saved(node)
            key, before = self._snapshot(index, f"{kind}:{record['id']}")
            record.update({"field": field, "source_sizes": list(sizes), "dim": dim,
                           "metadata": _metadata(index), "compact_shape": list(before.shape),
                           "statistics": index_statistics(before, sizes[dim]),
                           "stage": "attached", "prehook_calls": 0})
            self.latest_node = dict(record)
            if not record["statistics"]["in_bounds"]:
                self._mark_bad(f"{kind}:{record['id']}", before, before,
                               {"stage": "attach", "statistics": record["statistics"]})

            def prehook(grad_outputs, node=node, record=record, key=key):
                self.latest_node = {**record, "stage": "prehook_saved_tensor_access"}
                # Normal saved-tensor getters intentionally retain native version checks.
                current, current_sizes, current_dim, _ = self._saved(node)
                self.latest_node = {**record, "stage": "prehook_cpu_copy"}
                after = self._current_cpu(current, f"{record['type']}:{record['id']}")
                baseline = self._snapshots[key]
                statistics = index_statistics(after, current_sizes[current_dim])
                metadata = _metadata(current)
                changed = not _same(baseline, after)
                metadata_changed = metadata != record["metadata"]
                self.prehook_calls += 1
                record["prehook_calls"] += 1
                self.latest_node = {**record, "stage": "prehook_checked",
                                    "current_statistics": statistics, "values_changed": changed,
                                    "metadata_changed": metadata_changed}
                label = f"{record['type']}:{record['id']}"
                self._latest_cpu = {"label": label, "before": baseline, "after": after}
                if (changed or metadata_changed or not statistics["in_bounds"] or
                        tuple(current_sizes) != tuple(record["source_sizes"]) or
                        current_dim != record["dim"]):
                    self._mark_bad(label, baseline, after, dict(self.latest_node))
                return None

            self._handles.append(node.register_prehook(prehook))
        return self._base_summary()

    def _base_summary(self):
        return {
            "format": "BRIGHT_BACKWARD_SAVED_INDEX_OBSERVER_V1", "step": self.step,
            "snapshot_budget_bytes": self.max_bytes, "snapshot_bytes": self.snapshot_bytes,
            "current_tensor_budget_bytes": self.max_bytes,
            "unique_snapshots": len(self._snapshots), "graph_nodes_seen": self.graph_nodes_seen,
            "observed_index_nodes": len(self.nodes), "prehook_calls": self.prehook_calls,
            "watched_checks": self.watched_checks, "coverage_complete": self.coverage_complete,
            "coverage_scope": "GatherBackward and SortBackward saved indices plus explicit watches",
            "instrumentation_synchronizes": True, "instrumentation_changes_execution": True,
            "tensor_values_modified": False, "gradients_modified": False,
            "saved_tensors_hooks_used": False, "global_torch_patch_used": False,
            "historical_failure_resolved": False, "hardware_fault_proven": False,
            "full_validation_evidence": False, "full_test_evidence": False,
            "latest_node": self.latest_node, "nodes": self.nodes,
        }

    def record_failure(self, error):
        """CPU-only: safe to call after CUDA assert; never inspect live tensors."""
        if self._failure_written:
            return self._summary
        result = self._base_summary()
        result.update({"phase": "failed", "complete": False,
                       "error": f"{type(error).__name__}: {error}",
                       "failure_recording_cpu_only": True})
        chosen = self._first_bad or self._latest_cpu
        evidence = {"step": self.step, "tensors": {}, "truncated": False}
        tensor_budget = max(0, self.evidence_limit_bytes - 65536)
        used = 0
        if chosen:
            result["first_observed_corruption"] = (
                {"label": chosen["label"], "details": chosen.get("details", {})}
                if self._first_bad else None)
            evidence["label"] = chosen["label"]
            before, after = chosen["before"], chosen["after"]
            if before.dtype == after.dtype and before.shape == after.shape:
                # Integer samples are sufficient to preserve changed index evidence
                # even when the full pair exceeds the disk evidence budget.
                # Scan small CPU chunks so a corrupted large tensor cannot
                # create an unbounded nonzero() temporary during error handling.
                chunks = []
                changed_count = 0
                selected_count = 0
                left, right = before.reshape(-1), after.reshape(-1)
                for start in range(0, left.numel(), 65536):
                    local = (left[start:start + 65536] != right[start:start + 65536]).nonzero().flatten()
                    changed_count += local.numel()
                    take = min(local.numel(), 256 - selected_count)
                    if take:
                        chunks.append(local[:take] + start)
                        selected_count += take
                selected = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)
                evidence["changed_flat_indices"] = selected.clone()
                evidence["changed_before"] = before.reshape(-1)[selected].clone()
                evidence["changed_after"] = after.reshape(-1)[selected].clone()
                evidence["changed_elements"] = changed_count
                used += selected.numel() * (8 + 2 * before.element_size())
            for name, tensor in (("before", before), ("after", after)):
                size = tensor.numel() * tensor.element_size()
                if used + size <= tensor_budget:
                    # clone prevents torch.save from serializing an oversized shared storage.
                    evidence["tensors"][name] = tensor.clone()
                    used += size
                else:
                    evidence["truncated"] = True
            target = self._path("evidence", "pt")
            torch.save(evidence, target)
            actual_size = target.stat().st_size
            if actual_size > self.evidence_limit_bytes:
                target.unlink()
                result["evidence_error"] = "serialization exceeded disk budget; artifact removed"
            else:
                result["evidence"] = {"path": str(target), "bytes": actual_size,
                                      "truncated": evidence["truncated"]}
        self._summary = result
        _json_write(self._path("failure"), result)
        self._failure_written = True
        return result

    def finish(self):
        if self._failure_written:
            return self._summary
        result = self._base_summary()
        unexecuted = [node["id"] for node in self.nodes if not node.get("prehook_calls")]
        complete = self.coverage_complete and self._attached and not unexecuted
        result.update({"phase": "observer_step_complete" if complete else "observer_coverage_incomplete",
                       "complete": complete, "unexecuted_index_nodes": unexecuted})
        _json_write(self._path("summary"), result)
        self._summary = result
        return result

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._node_refs.clear()
        self._watches.clear()
        self._snapshots.clear()
        self._latest_cpu = None
        self._first_bad = None
        self._closed = True
