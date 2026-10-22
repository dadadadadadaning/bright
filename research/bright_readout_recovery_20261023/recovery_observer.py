"""Bounded, observational checks for the first four resumed updates.

No tensor repair, retry, optimizer substitution, or global Torch monkeypatch.
CPU copies and saved-tensor hooks change execution timing. Passing is not proof
that the historical CUDA failure has been repaired.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time
import traceback


class ObservationError(RuntimeError):
    pass


def compact_view(tensor):
    """Remove broadcast dimensions before copying expanded gather indices."""
    value = tensor.detach()
    for dimension in reversed(range(value.ndim)):
        if value.shape[dimension] > 1 and value.stride(dimension) == 0:
            value = value.select(dimension, 0)
    return value


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _metadata(tensor):
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                stride=list(tensor.stride()), offset=tensor.storage_offset(),
                device=str(tensor.device), storage_ptr=tensor.untyped_storage().data_ptr(),
                version=_version(tensor))


def _key(tensor):
    compact = compact_view(tensor)
    return (str(compact.device), str(compact.dtype),
            compact.untyped_storage().data_ptr(), compact.storage_offset(),
            tuple(compact.shape), tuple(compact.stride()))


def _cpu(tensor):
    return compact_view(tensor).to(device="cpu", copy=True).contiguous()


def _same(a, b):
    import torch
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.reshape(-1).view(torch.uint8),
                        b.reshape(-1).view(torch.uint8)))


class UpdateObserver:
    """One update's CPU evidence, capped before copying at 128 MiB by default."""
    def __init__(self, output, step, *, detailed=True, max_bytes=128 * 1024**2):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.step = int(step)
        self.detailed = bool(detailed)
        self.max_bytes = int(max_bytes)
        self.bytes = 0
        self.snapshots = {}
        self.watched = {}
        self.saved_integer_calls = 0
        self.saved_integer_unique = 0
        self.unpacked_integer_calls = 0
        self.stage = "created"
        self.failure = None
        self.failure_pair = None
        self.marker("created", detailed=self.detailed)

    def marker(self, stage, **fields):
        self.stage = stage
        row = dict(step=self.step, stage=stage, time_unix=time.time(), **fields)
        with (self.output / "recovery_markers.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

    def snapshot(self, name, tensor):
        key = _key(tensor)
        if key in self.snapshots:
            record = self.snapshots[key]
            current = _cpu(tensor)
            if not _same(record["cpu"], current):
                self.changed(name, record, current, "snapshot_reuse")
            return key
        compact = compact_view(tensor)
        size = compact.numel() * compact.element_size()
        if self.bytes + size > self.max_bytes:
            self.failure = dict(kind="observation_budget_exceeded", name=name,
                                requested_bytes=size, captured_bytes=self.bytes,
                                limit_bytes=self.max_bytes)
            raise ObservationError("Integer/route observation budget exceeded; coverage is incomplete")
        record = dict(name=name, cpu=_cpu(tensor), metadata=_metadata(tensor))
        self.snapshots[key] = record
        self.bytes += size
        return key

    def watch(self, name, tensor):
        if self.detailed:
            self.watched[name] = (tensor, self.snapshot(name, tensor))

    def check_watched(self, stage):
        if not self.detailed:
            return
        for name, (tensor, key) in self.watched.items():
            record = self.snapshots[key]
            current = _cpu(tensor)
            if not _same(record["cpu"], current):
                self.changed(name, record, current, stage)
        self.marker(stage, immutable_inputs_unchanged=True)

    def changed(self, name, record, current, stage):
        import torch
        before = record["cpu"]
        if before.dtype == current.dtype and before.shape == current.shape:
            indices = (before.reshape(-1).view(torch.uint8) !=
                       current.reshape(-1).view(torch.uint8)).nonzero().reshape(-1)
            changed_bytes = int(indices.numel())
            sample = indices[:16].tolist()
        else:
            changed_bytes, sample = None, []
        self.failure = dict(kind="immutable_tensor_changed", name=name, stage=stage,
                            metadata=record["metadata"], changed_bytes=changed_bytes,
                            first_changed_byte_offsets=sample)
        self.failure_pair = (name, before, current)
        raise ObservationError("Immutable tensor changed at " + stage + ": " + name)

    def pack(self, tensor):
        import torch
        integer = tensor.dtype in (torch.int8, torch.uint8, torch.int16,
                                   torch.int32, torch.int64, torch.bool)
        key = None
        if integer:
            self.saved_integer_calls += 1
            old_count = len(self.snapshots)
            key = self.snapshot("autograd_saved_integer_" + str(self.saved_integer_calls), tensor)
            if len(self.snapshots) > old_count:
                self.saved_integer_unique += 1
        # detach avoids a reference cycle; this view shares the same version counter.
        # Explicit version checking preserves the usual saved-tensor mutation guard.
        return tensor.detach(), _version(tensor), key

    def unpack(self, packed):
        tensor, version, key = packed
        if _version(tensor) != version:
            self.failure = dict(kind="saved_tensor_version_changed", metadata=_metadata(tensor),
                                version_before=version, stage=self.stage)
            raise ObservationError("Autograd-saved tensor version changed")
        if key is not None:
            self.unpacked_integer_calls += 1
            current = _cpu(tensor)
            record = self.snapshots[key]
            if not _same(record["cpu"], current):
                self.changed(record["name"], record, current, "autograd_unpack")
        return tensor

    def saved_context(self):
        if not self.detailed:
            return nullcontext()
        import torch
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def finish(self):
        self.marker("observation_complete", captured_cpu_bytes=self.bytes,
                    saved_integer_calls=self.saved_integer_calls,
                    saved_integer_unique=self.saved_integer_unique,
                    unpacked_integer_calls=self.unpacked_integer_calls,
                    historical_failure_resolved=False)

    def record_failure(self, error):
        """Only use previously captured CPU data after an exception/CUDA assert."""
        report = dict(format="BRIGHT_READOUT_RECOVERY_OBSERVATION_V1", step=self.step,
                      stage=self.stage, error=type(error).__name__ + ": " + str(error),
                      traceback=traceback.format_exc(),
                      detailed=self.detailed, captured_cpu_bytes=self.bytes,
                      first_observation_failure=self.failure,
                      tensors_modified_by_observer=False,
                      historical_failure_resolved=False)
        directory = self.output / ("diagnostic_step_%06d" % self.step)
        directory.mkdir(exist_ok=True)
        try:
            import torch
            payload, used, limit = {}, 0, 32 * 1024**2
            if self.failure_pair is not None:
                name, before, after = self.failure_pair
                for key, value in (("before", before), ("after", after)):
                    size = value.numel() * value.element_size()
                    if used + size <= limit:
                        payload[key] = value
                        used += size
                report["changed_tensor_name"] = name
            else:
                # Keep canonical inputs first, not hundreds of saved graph intermediates.
                for name, (_, key) in self.watched.items():
                    value = self.snapshots[key]["cpu"]
                    size = value.numel() * value.element_size()
                    if used + size <= limit:
                        payload[name] = value
                        used += size
            if payload:
                torch.save(payload, directory / "cpu_evidence.pt")
            report["cpu_snapshot_bytes"] = used
            report["cpu_snapshot_keys"] = list(payload)
        except Exception as evidence_error:
            report["evidence_write_error"] = str(evidence_error)
        (directory / "failure.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")


def observed_update(experiment, model, optimizer, example, runtime, factory, state, slots,
                    *, step, read_weight, multiplier, output, detailed=True):
    """Original recipe verbatim, instrumented only during the short diagnostic window."""
    observer = UpdateObserver(output, step, detailed=detailed)
    try:
        if not detailed:
            observer.marker("update_begin")
            result = experiment.update(model, optimizer, example, runtime, factory, state, slots,
                                       step=step, read_weight=read_weight, multiplier=multiplier)
            observer.marker("update_end")
            return result
        import torch
        observer.marker("support_prepare_begin")
        support = factory.prepare(example, ("d",))["d"]
        observer.marker("support_prepare_end")
        for name in ("q", "ids"):
            observer.watch("common." + name, example.common[name])
        for name in ("route_q", "route_ids", "support_ids"):
            observer.watch("support." + name, support[name])
        observer.watch("target", example.target)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = state.lr_at_step(step, maximum=1e-4, horizon=5000, multiplier=multiplier)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autograd.detect_anomaly(check_nan=True), observer.saved_context():
            observer.marker("forward_begin")
            prediction = model(**example.common, **support)
            observer.check_watched("forward_end")
            observer.marker("original_loss_begin")
            original_loss = runtime.loss(prediction["probabilities"], example.target)
            observer.marker("original_loss_end")
            observer.marker("auxiliary_loss_begin")
            loss, read_stats = experiment.combined_loss(
                original_loss, prediction, example.target, support, example.common, slots,
                read_weight=read_weight)
            observer.marker("auxiliary_loss_end")
            if loss.ndim or not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite/non-scalar training loss")
            observer.check_watched("before_backward")
            observer.marker("backward_begin")
            loss.backward()
            observer.marker("backward_end")
        observer.check_watched("after_backward")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        label_gradient = sum(float(p.grad.detach().abs().sum())
                             for name, p in model.named_parameters()
                             if name.startswith("label_embedding") and p.grad is not None)
        reader_gradient = sum(float(p.grad.detach().abs().sum())
                              for name, p in model.named_parameters()
                              if name.startswith(("query_projection", "key_projection", "value_projection",
                                                  "label_embedding", "context_projection")) and p.grad is not None)
        observer.marker("optimizer_step_begin")
        optimizer.step()
        observer.marker("optimizer_step_end")
        counts = [int(value["step"]) for value in optimizer.state.values() if "step" in value]
        if not counts or any(count != step for count in counts):
            raise ValueError("Active optimizer parameter update count differs")
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("Nonfinite updated parameters")
        result = dict(original_loss=float(original_loss.detach()), training_loss=float(loss.detach()),
                      lr=lr, optimizer_step=step, grad_norm_preclip=float(norm),
                      label_grad_abs_sum=label_gradient, reader_grad_abs_sum=reader_gradient,
                      **read_stats)
        observer.finish()
        return result
    except Exception as error:
        # Never mask the training failure if the filesystem cannot write evidence.
        try:
            observer.record_failure(error)
        except Exception:
            pass
        raise
