"""Strict restoration of an independently verified ablation checkpoint.

The caller verifies lineage, input files and sampling order before using this
helper. Restoration is in memory only. RNG restoration is deliberately separate
so that the caller can first construct its loader without advancing saved RNG.
"""
from __future__ import annotations

import copy
import math


FORMATS = {
    "BRIGHT_SLOT_MEMORY_ABLATION_V1",
    "BRIGHT_SLOT_MEMORY_DIAGNOSTIC_CHECKPOINT_V1",
}
MOMENTS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def _storage_key(tensor):
    if tensor.numel() == 0:
        return None
    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, storage.data_ptr())


def _assert_independent_storage(heads, optimizers):
    owners = {}

    def observe(arm, label, tensor):
        key = _storage_key(tensor)
        if key is None:
            return
        previous = owners.get(key)
        if previous is not None and previous[0] != arm:
            raise ValueError(
                "Cross-arm tensor storage alias: %s/%s and %s/%s"
                % (previous[0], previous[1], arm, label)
            )
        owners[key] = (arm, label)

    for arm, head in heads.items():
        for name, value in head.named_parameters():
            observe(arm, "parameter:" + name, value)
        for name, value in head.named_buffers():
            observe(arm, "buffer:" + name, value)
        for index, values in enumerate(optimizers[arm].state.values()):
            for name in MOMENTS:
                if name in values:
                    observe(arm, "moment:%s:%s" % (index, name), values[name])


def _step_value(torch, value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool:
            raise ValueError("Optimizer step must be a numeric scalar")
        number = float(value.detach().item())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    else:
        raise ValueError("Optimizer step must be a numeric scalar")
    if not math.isfinite(number) or number != int(number):
        raise ValueError("Optimizer step must be a finite integer")
    return int(number)


def _validate_optimizer(torch, head, optimizer, step, *, move_moments):
    model_parameters = list(head.parameters())
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if (len(parameters) != len(set(map(id, parameters)))
            or set(map(id, parameters)) != set(map(id, model_parameters))):
        raise ValueError("Optimizer parameters must match this head exactly once")
    if not parameters or set(optimizer.state) != set(parameters):
        raise ValueError("Checkpoint must contain AdamW state for every head parameter")
    counters = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            values = optimizer.state[parameter]
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad"):
                required.add("max_exp_avg_sq")
            if not isinstance(values, dict) or not required <= set(values):
                raise ValueError("Incomplete AdamW state")
            counter = _step_value(torch, values["step"])
            if counter != step:
                raise ValueError("Optimizer counter differs from checkpoint step")
            counters.append(counter)
            for name in MOMENTS:
                if name not in values:
                    continue
                moment = values[name]
                if not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape:
                    raise ValueError("Optimizer moment type/shape differs: " + name)
                if move_moments:
                    # load_state_dict controls scalar step placement; only the
                    # per-parameter moment tensors are normalized here.
                    moment = moment.to(device=parameter.device, dtype=parameter.dtype)
                    values[name] = moment
                if moment.device != parameter.device or moment.dtype != parameter.dtype:
                    raise ValueError("Optimizer moment device/dtype differs: " + name)
                if not bool(torch.isfinite(moment).all()):
                    raise ValueError("Nonfinite optimizer moment: " + name)
    return dict(
        parameter_count=len(parameters), state_count=len(optimizer.state),
        step_min=min(counters), step_max=max(counters),
    )


def restore_training_state(torch, heads, optimizers, saved, state):
    """Restore all arms exactly; leave RNG untouched and return JSON evidence.

    A rejected restoration must be discarded by the caller, not used for an
    update. This function does not provide transactional rollback on failure.
    """
    if not isinstance(saved, dict) or saved.get("format") not in FORMATS:
        raise ValueError("Unrecognized ablation checkpoint format")
    step = saved.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint step must be a positive integer")
    arms = tuple(heads)
    if not arms or tuple(optimizers) != arms:
        raise ValueError("Heads and optimizers must have the same ordered arms")
    for key in ("heads", "optimizers"):
        if not isinstance(saved.get(key), dict) or tuple(saved[key]) != arms:
            raise ValueError("Checkpoint arm identity/order differs: " + key)
    if "rng" not in saved:
        raise ValueError("Checkpoint RNG state is missing")
    if len(set(map(id, optimizers.values()))) != len(arms):
        raise ValueError("Arms share an optimizer instance")
    for arm, head in heads.items():
        optimizer = optimizers[arm]
        if not isinstance(optimizer, torch.optim.AdamW):
            raise ValueError("Expected the original AdamW optimizer: " + arm)
        actual = [p for g in optimizer.param_groups for p in g["params"]]
        if (len(actual) != len(set(map(id, actual)))
                or set(map(id, actual)) != set(map(id, head.parameters()))):
            raise ValueError("Optimizer is bound to a different head: " + arm)
    _assert_independent_storage(heads, optimizers)
    saved_hash = state.state_fingerprint(saved)
    rng_hash = state.state_fingerprint(state.capture_rng())
    evidence = {}
    for arm, head in heads.items():
        head.load_state_dict(saved["heads"][arm], strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in head.state_dict().values()):
            raise ValueError("Nonfinite restored head: " + arm)
        optimizers[arm].load_state_dict(copy.deepcopy(saved["optimizers"][arm]))
        counters = _validate_optimizer(
            torch, head, optimizers[arm], step, move_moments=True,
        )
        head_hash = state.state_fingerprint(head.state_dict())
        optimizer_hash = state.state_fingerprint(optimizers[arm].state_dict())
        if head_hash != state.state_fingerprint(saved["heads"][arm]):
            raise ValueError("Restored head fingerprint differs: " + arm)
        if optimizer_hash != state.state_fingerprint(saved["optimizers"][arm]):
            raise ValueError("Restored optimizer fingerprint differs: " + arm)
        evidence[arm] = dict(
            head_sha256=head_hash, optimizer_sha256=optimizer_hash,
            counters=counters,
        )
    _assert_independent_storage(heads, optimizers)
    if state.state_fingerprint(saved) != saved_hash:
        raise ValueError("Checkpoint payload was modified during restoration")
    if state.state_fingerprint(state.capture_rng()) != rng_hash:
        raise ValueError("Restoration advanced RNG before explicit restoration")
    return dict(
        step=step, arms=list(arms), fingerprints=evidence,
        saved_payload_sha256=saved_hash, saved_payload_unchanged=True,
        rng_unchanged=True, cross_arm_storage_independent=True,
    )


def validate_restored_rng(state, saved_rng, *, require_cuda=False):
    """Restore RNG once after loader construction and verify exact equality."""
    expected = state.state_fingerprint(saved_rng)
    state.restore_rng(copy.deepcopy(saved_rng), require_cuda=require_cuda)
    actual = state.state_fingerprint(state.capture_rng())
    if actual != expected:
        raise ValueError("Restored RNG fingerprint differs from checkpoint")
    if state.state_fingerprint(saved_rng) != expected:
        raise ValueError("Saved RNG payload was modified")
    return dict(rng_sha256=actual, restored=True, require_cuda=require_cuda)
