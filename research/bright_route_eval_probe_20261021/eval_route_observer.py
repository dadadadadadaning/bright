"""Record the active forward and original route-validator outcomes, without repair."""
from __future__ import annotations

import contextlib
import copy
import sys

from route_trace import RouteTrace, _error


class SelectiveRouteTrace(RouteTrace):
    def __init__(self, torch, classifier_class, emit, position):
        super().__init__(torch, classifier_class, emit)
        self.position = position
        self.dense = False
        self.tensor_reads_blocked = False
        self.accepted = {}

    def _cpu_copy(self, tensor):
        if self.tensor_reads_blocked:
            raise RuntimeError("Further tensor observation disabled after runtime error")
        try:
            return super()._cpu_copy(tensor)
        except RuntimeError:
            self.tensor_reads_blocked = True
            raise

    def _gpu_recomputed(self, tensor):
        if self.tensor_reads_blocked:
            return {"performed": False, "reason": "tensor reads blocked"}
        try:
            return super()._gpu_recomputed(tensor)
        except RuntimeError:
            self.tensor_reads_blocked = True
            raise

    def summary(self):
        result = super().summary()
        # This selective probe compares per-validation snapshots. It does not
        # continuously watch tensors, so a zero watch counter would mislead.
        for key in ("watched_change_count", "watched_change_count_definition", "first_watch_change", "watched"):
            result.pop(key, None)
        result.update(continuous_tensor_watch=False, observation_scope="target validators and any earlier rejected validator",
                      tensor_reads_blocked=self.tensor_reads_blocked)
        return result

    @staticmethod
    def _content(record):
        return tuple(record.get(k, {}).get("cpu_sha256") for k in ("weights", "ids"))

    def _wrapper(self, name, original):
        def observed(*args, **kwargs):
            self.validation_call_count += 1
            sequence = self.validation_call_count
            values = {k: args[i] if i < len(args) else kwargs.get(k)
                      for i, k in enumerate(("q", "ids", "count", "batch", "pixels"))}
            frame = sys._getframe(1)
            sites = []
            for _ in range(2):
                if frame is None:
                    break
                sites.append(dict(path=frame.f_code.co_filename, line=frame.f_lineno,
                                  function=frame.f_code.co_name))
                frame = frame.f_back
            del frame
            header = dict(sequence=sequence, binding=name, callsites=sites,
                          position=dict(self.position), arm=self.position.get("active_arm"))
            self.set_context(self.position.get("stage", "unknown"), header["arm"])
            record, case = None, None
            if self.dense:
                record, case = self._capture(values["q"], values["ids"], "before_original_validator", values["count"])
                record.update(header)
                self._emit("route_before", observation=record)
                if self.tensor_reads_blocked:
                    raise RuntimeError("Route observation encountered a runtime error; original validator was not run")
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self.validator_failure_count += 1
                if isinstance(exc, RuntimeError):
                    self.tensor_reads_blocked = True
                # A rejected original check remains rejected. Only capture a CPU
                # case after ValueError; never read CUDA after RuntimeError.
                if isinstance(exc, ValueError) and not self.tensor_reads_blocked:
                    after, after_case = self._capture(values["q"], values["ids"], "after_original_rejection", values["count"])
                    if record is None:
                        record, case = after, after_case
                    else:
                        record["after_rejection"] = after
                        case["after_rejection"] = after_case
                        record["pre_post_bytes_equal"] = self._content(record) == self._content(after)
                if record is None:
                    record, case = {}, dict(q=None, ids=None)
                record.update(header, outcome=dict(status="raised", **_error(exc)))
                key = self._content(record)
                record["same_bytes_accepted_earlier"] = self.accepted.get(key) if key[0] else None
                case["metadata"] = copy.deepcopy(record)
                entry = self._validators[name]
                entry["latest"], entry["latest_case"] = record, case
                if entry["first_failure"] is None:
                    entry["first_failure"], entry["first_failure_case"] = record, case
                self._emit("route_rejected", observation=record)
                raise
            if record is not None:
                record["outcome"] = {"status": "returned"}
                key = self._content(record)
                if key[0]:
                    self.accepted[key] = header
                case["metadata"] = copy.deepcopy(record)
                self._validators[name].update(latest=record, latest_case=case)
                self._emit("route_accepted", observation=record)
            return result
        return observed


@contextlib.contextmanager
def track_evaluation(runtime, factory, evaluator, names, target_index, trace, emit):
    """State is updated BEFORE forward, rather than inferred from last validated probability."""
    position = trace.position
    bindings = [(runtime, "prepare"), (factory, "prepare"), (factory, "forward"),
                (evaluator, "_validate_probability")]
    saved = [(obj, key, key in vars(obj), vars(obj).get(key), getattr(obj, key))
             for obj, key in bindings]
    prepare_runtime, prepare_support, forward, validate = [x[4] for x in saved]

    def prepare(batch):
        index = position["index"] + 1
        if index >= len(names):
            raise ValueError("Evaluation exceeded diagnostic prefix")
        position.update(index=index, image=names[index], active_arm=None, stage="prepare_begin")
        trace.dense = index == target_index
        emit("image_begin", **position, dense=trace.dense)
        example = prepare_runtime(batch)
        if example.image != names[index]:
            raise ValueError("Evaluation image order changed")
        position["stage"] = "prepare_end"
        return example

    def supports(example, arms):
        position.update(active_arm=None, stage="support_prepare_begin")
        result = prepare_support(example, arms)
        position["stage"] = "support_prepare_end"
        return result

    def call(model, example, arm, support):
        position.update(active_arm=arm, stage="forward_begin")
        emit("forward_begin", **position)
        result = forward(model, example, arm, support)
        position["stage"] = "forward_end"
        emit("forward_end", **position)
        return result

    def probability(prob, target, path):
        position.update(active_arm=path, stage="probability_validation")
        result = validate(prob, target, path)
        position["stage"] = "probability_validated"
        return result

    runtime.prepare, factory.prepare, factory.forward = prepare, supports, call
    evaluator._validate_probability = probability
    try:
        with trace:
            yield position
    finally:
        for obj, key, existed, value, original in reversed(saved):
            if existed:
                setattr(obj, key, value)
            else:
                delattr(obj, key)
