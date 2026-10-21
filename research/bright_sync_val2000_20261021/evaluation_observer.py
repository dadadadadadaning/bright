"""CPU metadata only; preserve the first failure without reading poisoned CUDA state."""
from contextlib import contextmanager
from pathlib import Path
import sys
import traceback

from evaluate_checkpoint import write_json


class EvaluationObserver:
    def __init__(self, output, position):
        self.output = Path(output)
        self.position = position
        self.first_exception = None
        self.first_failure = None
        self.rng_restore_skipped = False
        self.evidence_error = None

    def record(self, exc):
        if self.first_exception is not None:
            return
        self.first_exception = exc
        self.first_failure = dict(position=dict(self.position),
            error=type(exc).__name__ + ": " + str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            tensor_values_modified=False, tensor_snapshots_taken=False)
        try:
            write_json(self.output / "evaluation_failure.json", self.first_failure)
        except Exception as save_error:
            self.evidence_error = str(save_error)


@contextmanager
def track_evaluation(runtime, factory, evaluator, names, state, observer):
    bindings = [(runtime, "prepare"), (factory, "prepare"), (factory, "forward"),
                (evaluator, "_validate_probability"), (state, "restore_rng")]
    saved = [(obj, key, key in vars(obj), vars(obj).get(key), getattr(obj, key))
             for obj, key in bindings]
    prepare_runtime, prepare_support, forward, validate, restore_rng = [x[4] for x in saved]
    position = observer.position

    def call(stage, function, *args, **kwargs):
        position["stage"] = stage
        try:
            result = function(*args, **kwargs)
            position["stage"] = stage + "_returned"
            return result
        except BaseException as exc:
            observer.record(exc)
            raise

    def prepare(batch):
        index = position["index"] + 1
        position.update(index=index, image=names[index] if index < len(names) else None,
                        active_arm=None, stage="prepare")
        try:
            if index >= len(names):
                raise ValueError("Evaluation exceeded requested Val450")
            example = call("prepare", prepare_runtime, batch)
            if example.image != names[index]:
                raise ValueError("Evaluation image order changed")
            return example
        except BaseException as exc:
            observer.record(exc)
            raise

    def supports(example, arms):
        position["active_arm"] = None
        return call("support_prepare", prepare_support, example, arms)

    def model_forward(model, example, arm, support):
        position["active_arm"] = arm
        return call("forward", forward, model, example, arm, support)

    def probability(prob, target, path):
        position["active_arm"] = path
        return call("probability_validation", validate, prob, target, path)

    def restore(*args, **kwargs):
        # This also covers failures outside wrapped stages (loss, metric, loader).
        # The evaluator calls restore_rng from its finally block while the
        # primary exception is active. Re-raise before its later GPU hashes.
        active_error = sys.exc_info()[1]
        if active_error is not None:
            observer.record(active_error)
        if observer.first_exception is not None:
            observer.rng_restore_skipped = True
            raise observer.first_exception
        return restore_rng(*args, **kwargs)

    runtime.prepare, factory.prepare, factory.forward = prepare, supports, model_forward
    evaluator._validate_probability, state.restore_rng = probability, restore
    try:
        yield position
    except BaseException as exc:
        observer.record(exc)
        raise
    finally:
        for obj, key, existed, value, original in reversed(saved):
            if existed:
                setattr(obj, key, value)
            else:
                delattr(obj, key)
