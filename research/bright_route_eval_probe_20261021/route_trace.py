"""Observe route validation without replacing inputs, validators or outcomes."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import sys


def _number(value):
    if hasattr(value, "item"):
        value = value.item()
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def _error(exc):
    try:
        message = str(exc)
    except BaseException:
        message = "<exception string unavailable>"
    return {"type": type(exc).__name__, "message": message[:1200]}


def _scalar(value):
    return _number(value) if type(value) in (int, float, bool, str, type(None)) else type(value).__name__


class RouteTrace:
    """Bounded CPU evidence and separately recomputed route predicates.

    Copies and recomputations synchronize device work and alter timing. They do
    not observe the original validator's temporary reduction tensors.
    """

    def __init__(self, torch, classifier_class, emit):
        self.torch, self.classifier_class, self.emit = torch, classifier_class, emit
        self.stage, self.arm = "unassigned", None
        self.validation_call_count = 0
        self.validator_failure_count = 0
        self.diagnostic_error_count = 0
        self.watched_change_count = 0
        self.first_watch_change = None
        self.diagnostic_errors = []
        self._watched, self._validators, self._bindings = {}, {}, []
        self._entered = False

    def _diagnostic_error(self, where, exc):
        self.diagnostic_error_count += 1
        item = dict(where=where, **_error(exc))
        if len(self.diagnostic_errors) < 12:
            self.diagnostic_errors.append(item)
        return item

    def _emit(self, event, **fields):
        try:
            self.emit(event, **fields)
        except BaseException as exc:
            self._diagnostic_error("emit:" + event, exc)

    def set_context(self, stage, arm=None):
        self.stage, self.arm = str(stage), arm

    def _cpu_copy(self, tensor):
        return tensor.detach().to(device="cpu", copy=True).contiguous()

    def _metadata(self, tensor):
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        storage = tensor.untyped_storage()
        return dict(
            dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()),
            storage_offset=tensor.storage_offset(), storage_ptr=storage.data_ptr(),
            storage_nbytes=storage.nbytes(), data_ptr=tensor.data_ptr(),
            device=str(tensor.device), version=version, requires_grad=tensor.requires_grad,
        )

    def _stats(self, cpu):
        torch = self.torch
        flat = cpu.reshape(-1)
        finite = torch.isfinite(flat)
        return dict(
            cpu_sha256=hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest(),
            count=flat.numel(), minimum=_number(flat.min()) if flat.numel() else None,
            maximum=_number(flat.max()) if flat.numel() else None,
            negative_count=int((flat < 0).sum()), nonfinite_count=int((~finite).sum()),
        )

    def _sum_stats(self, sums):
        torch = self.torch
        error = (sums - 1).abs()
        bad = (~torch.isfinite(sums)) | (error > 4e-5)
        return dict(
            minimum=_number(sums.min()) if sums.numel() else None,
            maximum=_number(sums.max()) if sums.numel() else None,
            max_abs_error=_number(error.max()) if error.numel() else None,
            bad_count=int(bad.sum()), threshold=4e-5, atol=2e-5, rtol=2e-5,
        ), bad

    def _weight_details(self, cpu, ids):
        torch = self.torch
        if cpu.ndim != 3 or cpu.shape[1] < 1:
            return {"route_shape_supported": False}
        sums32, bad32 = self._sum_stats(cpu.to(torch.float32).sum(1))
        sums64, bad64 = self._sum_stats(cpu.to(torch.float64).sum(1))
        bad = bad32 | bad64 | ((cpu < 0) | (~torch.isfinite(cpu))).any(1)
        samples = []
        for coordinate in bad.nonzero(as_tuple=False)[:8]:
            batch, pixel = (int(x) for x in coordinate)
            sample = dict(
                batch=batch, flat_pixel=pixel,
                weights=[_number(x) for x in cpu[batch, :, pixel]],
            )
            if ids is not None:
                if tuple(ids.shape) == tuple(cpu.shape):
                    sample["ids"] = [_number(x) for x in ids[batch, :, pixel]]
                elif tuple(ids.shape) == tuple(cpu.shape[1:]):
                    sample["ids"] = [_number(x) for x in ids[:, pixel]]
            samples.append(sample)
        return dict(
            route_shape_supported=True, cpu_float32_sums=sums32, cpu_float64_sums=sums64,
            bad_pixel_count=int(bad.sum()), first_bad_pixels=samples,
        )

    def _gpu_recomputed(self, tensor):
        torch = self.torch
        if tensor.device.type != "cuda":
            return {"performed": False, "reason": "input is not CUDA"}
        tensor = tensor.detach()
        result = dict(
            performed=True, original_intermediates=False,
            negative_any=bool((tensor < 0).any()),
            finite_all=bool(torch.isfinite(tensor).all()),
        )
        if tensor.ndim == 3 and tensor.shape[1] > 0:
            sums = tensor.sum(1)
            result["sum_allclose"] = bool(torch.allclose(
                sums, torch.ones_like(tensor[:, 0]), atol=2e-5, rtol=2e-5
            ))
            result["sums"] = self._sum_stats(sums)[0]
        return result

    def _capture_impl(self, q, ids, stage, count=None):
        record = dict(stage=stage, context_stage=self.stage, arm=self.arm, diagnostic_errors=[])
        case = dict(stage=stage, context_stage=self.stage, arm=self.arm, q=None, ids=None)
        for label, tensor in (("weights", q), ("ids", ids)):
            if tensor is None:
                continue
            try:
                metadata = self._metadata(tensor)
                cpu = self._cpu_copy(tensor)
                record[label] = dict(**metadata, **self._stats(cpu))
                case["q" if label == "weights" else "ids"] = cpu
                if label == "ids" and type(count) is int:
                    record[label]["out_of_range_count"] = int(((cpu < 0) | (cpu >= count)).sum())
            except BaseException as exc:
                record["diagnostic_errors"].append(self._diagnostic_error("capture:" + label, exc))
        if case["q"] is not None:
            try:
                record.update(self._weight_details(case["q"], case["ids"]))
            except BaseException as exc:
                record["diagnostic_errors"].append(self._diagnostic_error("CPU route statistics", exc))
        try:
            record["gpu_recomputed"] = self._gpu_recomputed(q)
        except BaseException as exc:
            record["diagnostic_errors"].append(self._diagnostic_error("GPU recomputation", exc))
        case["metadata"] = copy.deepcopy(record)
        return record, case

    def _capture(self, q, ids, stage, count=None):
        try:
            return self._capture_impl(q, ids, stage, count)
        except BaseException as exc:
            error = self._diagnostic_error("snapshot", exc)
            record = dict(stage=stage, context_stage=self.stage, arm=self.arm,
                          diagnostic_errors=[error])
            return record, dict(stage=stage, q=None, ids=None, metadata=record)

    @staticmethod
    def _signature(record):
        result = []
        for key in ("weights", "ids"):
            item = record.get(key)
            if item is not None:
                result.append((key, item["cpu_sha256"], item["version"], item["storage_ptr"],
                               item["data_ptr"], tuple(item["shape"]), tuple(item["stride"]),
                               item["storage_offset"], item["storage_nbytes"]))
        return tuple(result) if "weights" in record else None

    def watch(self, name, tensor, ids=None):
        name = str(name)
        if name in self._watched:
            entry = self._watched[name]
            if entry["tensor"] is tensor and entry["ids"] is ids:
                return
            raise ValueError("Use a distinct watch name for each tensor binding: " + name)
        record, case = self._capture(tensor, ids, "watch_bind")
        self._watched[name] = dict(
            tensor=tensor, ids=ids, initial=record, latest=record, first_change=None,
            initial_case=case, latest_case=case, first_change_case=None,
        )
        self._emit("route_watch", name=name, changed_from_initial=False,
                   changed_since_previous=False, observation=record)

    def snapshot(self, stage, arm=None):
        self.stage = str(stage)
        if arm is not None:
            self.arm = arm
        for name, entry in self._watched.items():
            record, case = self._capture(entry["tensor"], entry["ids"], self.stage)
            signature = self._signature(record)
            previous = self._signature(entry["latest"])
            initial = self._signature(entry["initial"])
            changed = signature is not None and previous is not None and signature != previous
            if changed:
                self.watched_change_count += 1
                change = dict(name=name, stage=self.stage, arm=self.arm,
                              previous=entry["latest"], current=record)
                if self.first_watch_change is None:
                    self.first_watch_change = copy.deepcopy(change)
                if entry["first_change"] is None:
                    entry["first_change"], entry["first_change_case"] = record, case
            entry["latest"], entry["latest_case"] = record, case
            self._emit(
                "route_watch", name=name,
                changed_from_initial=signature is not None and initial is not None and signature != initial,
                changed_since_previous=changed, observation=record,
            )

    def _wrapper(self, name, original):
        def observed(*args, **kwargs):
            self.validation_call_count += 1
            sequence = self.validation_call_count
            values = {key: args[i] if i < len(args) else kwargs.get(key)
                      for i, key in enumerate(("q", "ids", "count", "batch", "pixels"))}
            frame = sys._getframe(1)
            callsite = dict(path=frame.f_code.co_filename, line=frame.f_lineno,
                            function=frame.f_code.co_name)
            del frame
            record, case = self._capture(values["q"], values["ids"], "validator_pre_call", values["count"])
            record.update(
                sequence=sequence, binding=name, callsite=callsite,
                arguments={key: _scalar(values[key]) for key in ("count", "batch", "pixels")},
            )
            case["metadata"] = copy.deepcopy(record)
            entry = self._validators[name]
            entry["latest"], entry["latest_case"] = record, case
            self._emit("route_validator_pre_call", observation=record)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self.validator_failure_count += 1
                try:
                    record["outcome"] = dict(status="raised", **_error(exc))
                    case["metadata"] = copy.deepcopy(record)
                    if entry["first_failure"] is None:
                        entry["first_failure"], entry["first_failure_case"] = record, case
                    self._emit("route_validator_outcome", observation=record)
                except BaseException as diagnostic:
                    self._diagnostic_error("record original validator exception", diagnostic)
                raise
            try:
                record["outcome"] = {"status": "returned"}
                case["metadata"] = copy.deepcopy(record)
                self._emit("route_validator_outcome", observation=record)
            except BaseException as diagnostic:
                self._diagnostic_error("record original validator result", diagnostic)
            return result
        return observed

    def __enter__(self):
        if self._entered:
            raise RuntimeError("RouteTrace cannot be entered twice")
        self._entered = True
        classifier_globals = self.classifier_class.forward.__globals__
        unpool = classifier_globals["sparse_unpool"]
        candidates = (("classifier", classifier_globals), ("sparse_unpool", unpool.__globals__))
        seen = set()
        try:
            for name, namespace in candidates:
                if id(namespace) in seen:
                    continue
                seen.add(id(namespace))
                original = namespace["validate_routes"]
                self._validators[name] = dict(latest=None, latest_case=None,
                                              first_failure=None, first_failure_case=None)
                wrapper = self._wrapper(name, original)
                self._bindings.append((name, namespace, original))
                namespace["validate_routes"] = wrapper
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exception):
        for name, namespace, original in reversed(self._bindings):
            namespace["validate_routes"] = original
        return False

    def summary(self):
        failures = [entry["first_failure"] for entry in self._validators.values()
                    if entry["first_failure"] is not None]
        return copy.deepcopy(dict(
            validation_call_count=self.validation_call_count,
            validator_failure_count=self.validator_failure_count,
            diagnostic_error_count=self.diagnostic_error_count,
            diagnostic_errors=self.diagnostic_errors,
            watched_change_count=self.watched_change_count,
            watched_change_count_definition="changes between consecutive captured value/version/storage signatures",
            first_watch_change=self.first_watch_change,
            first_validation_failure=min(failures, key=lambda record: record["sequence"])
            if failures else None,
            bindings=[name for name, _, _ in self._bindings],
            watched={name: {key: entry[key] for key in ("initial", "latest", "first_change")}
                     for name, entry in self._watched.items()},
            validators={name: {key: entry[key] for key in ("latest", "first_failure")}
                        for name, entry in self._validators.items()},
            gpu_recomputation_is_original_intermediate=False,
        ))

    def save_cpu_case(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        payload = dict(
            format="BRIGHT_ROUTE_DIAGNOSTIC_INPUTS_V1", summary=self.summary(),
            watched={name: {key: entry[key] for key in ("initial_case", "latest_case", "first_change_case")}
                     for name, entry in self._watched.items()},
            validators={name: {key: entry[key] for key in ("latest_case", "first_failure_case")}
                        for name, entry in self._validators.items()},
            exact_training_replay=False,
        )
        self.torch.save(payload, path)
        return str(path)
