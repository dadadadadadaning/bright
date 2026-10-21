"""Synchronous diagnostic observation; never repair indices or train a model."""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class IndexContractError(ValueError):
    pass


def metadata(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), device=str(tensor.device),
                storage_offset=tensor.storage_offset())


def compact_cpu(index):
    # Expanded gather channels contain repeated IDs. Preserve singleton axes so
    # the saved CPU view can be expanded back to its original logical shape.
    value = index.detach()
    for dimension in range(value.ndim):
        if value.shape[dimension] > 1 and value.stride(dimension) == 0:
            value = value.narrow(dimension, 0, 1)
    return value.to(device="cpu", copy=True).contiguous()


def index_arguments(func, args, kwargs):
    name = func._schema.name
    bound = dict(zip((a.name for a in func._schema.arguments), args))
    bound.update(kwargs)
    source = bound.get("self")
    if not isinstance(source, torch.Tensor):
        return None
    simple = {"aten::gather", "aten::scatter", "aten::scatter_",
              "aten::scatter_add", "aten::scatter_add_", "aten::scatter_reduce",
              "aten::scatter_reduce_", "aten::index_select"}
    if name in simple and isinstance(bound.get("dim"), int):
        dim = bound["dim"]
        if not -source.ndim <= dim < source.ndim:
            return None  # Original operator retains dimension/type validation.
        dim %= source.ndim
        return source, [(dim, bound.get("index"), 0, source.shape[dim])]
    if name == "aten::index" and isinstance(bound.get("indices"), (list, tuple)):
        dimensions, dim = [], 0
        for index in bound["indices"]:
            if index is None:
                dim += 1
            elif index.dtype in (torch.bool, torch.uint8):
                dim += index.ndim  # Boolean mask semantics remain with PyTorch.
            else:
                if dim >= source.ndim:
                    return None
                dimensions.append((dim, index, -source.shape[dim], source.shape[dim]))
                dim += 1
        return source, dimensions
    return None


class IndexTrace(TorchDispatchMode):
    def __init__(self, output, position, target_index=30):
        super().__init__()
        self.output, self.position = Path(output), position
        self.target_index = target_index
        self.operation_count = self.index_count = self.snapshot_count = 0
        self.first_failure = None
        self.first_exception = None
        self.failure_case = None
        self.last_index = None
        self.rng_restore_skipped = False

    def write(self, name, obj):
        (self.output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

    def record_failure(self, exc, record, case=None):
        if self.first_failure is not None:
            return
        self.first_exception = exc
        self.first_failure = dict(record, error=type(exc).__name__ + ": " + str(exc),
                                  python_traceback=traceback.format_exc(),
                                  call_stack=traceback.format_stack(limit=16))
        self.failure_case = case
        self.write("first_failure.json", self.first_failure)
        print("INDEX DIAGNOSTIC FIRST FAILURE " + json.dumps({
            k: self.first_failure.get(k) for k in ("operation", "position", "phase", "error")}), flush=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.operation_count += 1
        record = dict(operation=str(func), sequence=self.operation_count,
                      position=dict(self.position), phase="operator", operator_called=False)
        case = None
        try:
            indexed = index_arguments(func, args, kwargs)
            if indexed is not None:
                self.index_count += 1
                source, indices = indexed
                record["source"] = metadata(source)
                record["indices"] = []
                case = dict(operation=str(func), source_metadata=metadata(source), indices=[])
                dense = self.position.get("index") == self.target_index
                for dim, index, low, high in indices:
                    if not isinstance(index, torch.Tensor):
                        continue
                    item = dict(dimension=dim, minimum_allowed=low, exclusive_maximum=high,
                                **metadata(index))
                    if dense and index.dtype in (torch.int32, torch.int64):
                        record["phase"] = "pre_operator_index_copy"
                        cpu = compact_cpu(index)
                        self.snapshot_count += 1
                        bad = (cpu < low) | (cpu >= high)
                        item.update(compact_shape=list(cpu.shape),
                                    minimum=int(cpu.min()) if cpu.numel() else None,
                                    maximum=int(cpu.max()) if cpu.numel() else None,
                                    bad_compact_count=int(bad.sum()))
                        case["indices"].append(dict(metadata=dict(item), compact=cpu))
                    record["indices"].append(item)
                if dense and self.position.get("active_arm") == "b" and source.numel() * source.element_size() <= 16 * 2**20:
                    record["phase"] = "pre_operator_source_copy"
                    case["source_cpu"] = source.detach().to(device="cpu", copy=True)
                self.last_index = copy.deepcopy(record)
                if dense:
                    record["callsites"] = [dict(file=f.filename, line=f.lineno, function=f.name)
                                           for f in traceback.extract_stack(limit=10)[:-1]]
                    with (self.output / "target_indices.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                if any(v.get("bad_compact_count", 0) for v in record["indices"]):
                    record["phase"] = "cpu_index_contract"
                    raise IndexContractError("Out-of-range index captured before original operator; no repair performed")
            record.update(phase="original_operator", operator_called=True)
            result = func(*args, **kwargs)
            return result
        except BaseException as exc:
            self.record_failure(exc, record, case)
            raise

    def save(self):
        # All tensors retained here were copied BEFORE the operator failed.
        if self.failure_case is not None:
            torch.save(self.failure_case, self.output / "index_failure_cpu.pt")
        result = dict(operation_count=self.operation_count, index_operation_count=self.index_count,
                      cpu_index_snapshots=self.snapshot_count, first_failure=self.first_failure,
                      last_index_operation=self.last_index,
                      rng_restore_skipped_after_failure=self.rng_restore_skipped)
        self.write("index_summary.json", result)
        return result


@contextlib.contextmanager
def track_evaluation(runtime, factory, evaluator, names, state, trace):
    position = trace.position
    bindings = [(runtime, "prepare"), (factory, "prepare"), (factory, "forward"),
                (evaluator, "_validate_probability"), (state, "restore_rng")]
    saved = [(obj, key, key in vars(obj), vars(obj).get(key), getattr(obj, key))
             for obj, key in bindings]
    prepare_runtime, prepare_support, forward, validate, restore_rng = [x[4] for x in saved]

    def call_stage(stage, function, *args, **kwargs):
        position["stage"] = stage
        try:
            value = function(*args, **kwargs)
            position["stage"] = stage + "_returned"
            return value
        except BaseException as exc:
            trace.record_failure(exc, dict(position=dict(position), phase=stage))
            raise

    def prepare(batch):
        index = position["index"] + 1
        if index >= len(names):
            raise ValueError("Evaluation exceeded diagnostic prefix")
        position.update(index=index, image=names[index], active_arm=None)
        example = call_stage("prepare", prepare_runtime, batch)
        if example.image != names[index]:
            raise ValueError("Evaluation image order changed")
        return example

    def supports(example, arms):
        position["active_arm"] = None
        return call_stage("support_prepare", prepare_support, example, arms)

    def call(model, example, arm, support):
        position["active_arm"] = arm
        return call_stage("forward", forward, model, example, arm, support)

    def probability(prob, target, path):
        position["active_arm"] = path
        result = call_stage("probability_validation", validate, prob, target, path)
        position["stage"] = "after_probability_validation"
        return result

    def restore(*args, **kwargs):
        if trace.first_failure is not None:
            # Failed diagnostic is never promoted. Do not touch CUDA RNG after
            # a device assertion or replace the primary error during unwinding.
            trace.rng_restore_skipped = True
            # Re-raise before the evaluator's subsequent GPU state hashes too.
            raise trace.first_exception
        return restore_rng(*args, **kwargs)

    runtime.prepare, factory.prepare, factory.forward = prepare, supports, call
    evaluator._validate_probability, state.restore_rng = probability, restore
    try:
        with trace:
            yield position
    finally:
        for obj, key, existed, value, original in reversed(saved):
            if existed:
                setattr(obj, key, value)
            else:
                delattr(obj, key)
