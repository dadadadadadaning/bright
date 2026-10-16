"""Explicit checkpoint-500 fork helpers for a new learning-rate experiment.

The historical trainer advertises no native resume entry point.  These helpers
restore its recorded state only after the new driver verifies the input ledger
and reconstructs the original, augmentation-free data order.  No historical
checkpoint, source module, bank, or completion flag is modified.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random

import numpy as np
import torch


BINDINGS = {
    "b_control": ("b", 1.0),
    "b_lr02": ("b", 0.2),
    "d_control": ("d", 1.0),
    "d_lr02": ("d", 0.2),
}


def state_fingerprint(value):
    """Device-independent exact digest of nested model/optimizer/RNG state."""
    checksum = hashlib.sha256()

    def add(obj):
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().contiguous()
            checksum.update(b"tensor:")
            checksum.update(str(t.dtype).encode())
            checksum.update(str(tuple(t.shape)).encode())
            checksum.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, np.ndarray):
            checksum.update(b"ndarray:")
            checksum.update(str(obj.dtype).encode())
            checksum.update(str(obj.shape).encode())
            checksum.update(obj.tobytes())
        elif isinstance(obj, dict):
            checksum.update(b"dict:")
            for key in sorted(obj, key=lambda item: (type(item).__name__, repr(item))):
                add(key)
                add(obj[key])
        elif isinstance(obj, (list, tuple)):
            checksum.update(type(obj).__name__.encode())
            checksum.update(str(len(obj)).encode())
            for item in obj:
                add(item)
        elif obj is None or isinstance(obj, (bool, str, int, float, np.generic)):
            checksum.update(type(obj).__name__.encode())
            checksum.update(repr(obj).encode())
        else:
            raise TypeError("Unsupported state entry: " + type(obj).__name__)
        checksum.update(b"\0")

    add(value)
    return checksum.hexdigest()


def lr_at_step(step, maximum=1e-4, horizon=5000, warmup=500,
               multiplier=1.0, branch_step=500):
    """The original 5000-step schedule, scaled only strictly after step 500."""
    if (isinstance(step, bool) or not isinstance(step, int)
            or not 1 <= step <= horizon or not 0 < warmup < horizon
            or not 0 <= branch_step < horizon
            or not math.isfinite(maximum) or maximum <= 0
            or not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError("Invalid learning-rate schedule arguments")
    if step <= warmup:
        value = float(maximum) * step / warmup
    else:
        progress = (step - warmup) / (horizon - warmup)
        value = float(maximum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return value * (float(multiplier) if step > branch_step else 1.0)


def capture_rng():
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([value.clone() for value in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []),
    }


def _validate_rng(state, require_cuda=False):
    if not isinstance(state, dict) or set(state) != {
            "python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Checkpoint must contain all four recorded RNG fields")
    cpu = state["torch_cpu"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("Invalid CPU Torch RNG state")
    cuda = state["torch_cuda"]
    if not isinstance(cuda, (list, tuple)):
        raise ValueError("CUDA RNG state must be a list")
    if require_cuda and not cuda:
        raise ValueError("CUDA continuation requires the recorded CUDA RNG state")
    for value in cuda:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("Invalid CUDA RNG state")
    # Validate Python and NumPy state without changing the process RNG.
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(state["numpy"])


def restore_rng(state, *, require_cuda=False):
    _validate_rng(state, require_cuda=require_cuda)
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("Recorded CUDA RNG cannot be restored without CUDA")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Visible CUDA device count differs from the checkpoint")
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def validate_checkpoint_payload(saved, *, require_cuda=False):
    """Check continuation-specific state; the driver checks provenance/hashes."""
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint payload must be a dictionary")
    if (saved.get("step") != 500 or saved.get("mode") != "train"
            or saved.get("smoke") is not False
            or saved.get("training_complete") is not False):
        raise ValueError("This study forks the non-smoke step-500 training checkpoint")
    required = ("heads", "optimizers", "rng", "config", "protocol_info",
                "source_hash", "input_file_sha256", "updates_per_arm")
    if any(name not in saved for name in required):
        raise ValueError("Checkpoint lacks required optimizer/RNG/provenance state")
    _validate_rng(saved["rng"], require_cuda=require_cuda)
    config = saved["config"]
    if config.get("steps") != 5000 or config.get("eval_every") != 500:
        raise ValueError("Source must use the original 5000-step/500-evaluation schedule")
    for base in ("b", "d"):
        if (saved["updates_per_arm"].get(base) != 500
                or base not in saved["heads"] or base not in saved["optimizers"]):
            raise ValueError("Missing completed step-500 B/D model/optimizer state")
        optimizer = saved["optimizers"][base]
        groups = optimizer.get("param_groups", [])
        state = optimizer.get("state", {})
        if len(groups) != 1 or not state:
            raise ValueError("Expected original AdamW parameter group and nonempty moments")
        group = groups[0]
        if (tuple(group.get("betas", ())) != (0.9, 0.999)
                or group.get("eps") != 1e-8 or group.get("amsgrad") is not False):
            raise ValueError("Optimizer does not match the original AdamW configuration")
        if float(group.get("weight_decay", -1)) != float(config.get("weight_decay", -2)):
            raise ValueError("Optimizer weight decay differs from recorded configuration")
        if not math.isclose(float(group.get("lr", -1)), float(config.get("lr", -2)),
                            rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("Step-500 learning rate differs from the warmup endpoint")
        for item in state.values():
            if not all(key in item for key in ("step", "exp_avg", "exp_avg_sq")):
                raise ValueError("AdamW moment state is incomplete")
            step = item["step"]
            if (float(step.item()) if isinstance(step, torch.Tensor) else float(step)) != 500:
                raise ValueError("AdamW step must equal 500 before forking")
            for key in ("exp_avg", "exp_avg_sq"):
                if not isinstance(item[key], torch.Tensor) or not bool(torch.isfinite(item[key]).all()):
                    raise ValueError("Invalid or non-finite AdamW moments")
    return True


def fork_heads_and_optimizers(saved, args, device, *, model_factory=None):
    """Independent B/D copies, including Adam moments; never a weights-only fork."""
    device = torch.device(device)
    validate_checkpoint_payload(saved, require_cuda=device.type == "cuda")
    if model_factory is None:
        # Driver first verifies/imports the exact historical package.
        from model import JointRegionMemoryClassifier
        model_factory = lambda: JointRegionMemoryClassifier(chunk_size=int(args.chunk_size))
    heads, optimizers, records = {}, {}, {}
    for name, (base, multiplier) in BINDINGS.items():
        head = model_factory().to(device)
        head.load_state_dict(copy.deepcopy(saved["heads"][base]), strict=True)
        if not all(bool(torch.isfinite(value).all()) for value in head.state_dict().values()):
            raise ValueError("Non-finite source head: " + base)
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(saved["config"]["lr"]),
                                      weight_decay=float(saved["config"]["weight_decay"]))
        optimizer.load_state_dict(copy.deepcopy(saved["optimizers"][base]))
        # PyTorch moves moments when loading.  Preserve non-capturable AdamW's
        # scalar step device exactly as its load_state_dict implementation does.
        for parameter, value in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in value:
                    value[key] = value[key].to(parameter.device).clone()
        head_hash = state_fingerprint(head.state_dict())
        optimizer_hash = state_fingerprint(optimizer.state_dict())
        if head_hash != state_fingerprint(saved["heads"][base]):
            raise ValueError("Head restoration mismatch: " + name)
        if optimizer_hash != state_fingerprint(saved["optimizers"][base]):
            raise ValueError("Optimizer restoration mismatch: " + name)
        heads[name], optimizers[name] = head, optimizer
        records[name] = {"base_arm": base, "post500_lr_multiplier": multiplier,
                         "head_sha256": head_hash, "optimizer_sha256": optimizer_hash}
    for base in ("b", "d"):
        left, right = base + "_control", base + "_lr02"
        if records[left]["head_sha256"] != records[right]["head_sha256"]:
            raise ValueError("Forked pair heads differ at initialization")
        if records[left]["optimizer_sha256"] != records[right]["optimizer_sha256"]:
            raise ValueError("Forked pair AdamW moments differ at initialization")
        for p, q in zip(heads[left].parameters(), heads[right].parameters()):
            if p.data_ptr() == q.data_ptr():
                raise ValueError("Forked model parameters share storage")
        left_state, right_state = optimizers[left].state, optimizers[right].state
        for p, q in zip(heads[left].parameters(), heads[right].parameters()):
            for key in ("exp_avg", "exp_avg_sq"):
                if key in left_state.get(p, {}) and left_state[p][key].data_ptr() == right_state[q][key].data_ptr():
                    raise ValueError("Forked optimizer moments share storage")
    return heads, optimizers, records


def raw_batch_identity(batch):
    """Read batch-one identity without moving images/targets to CUDA."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 5:
        raise ValueError("Expected original five-field BRIGHT batch")
    names = batch[4]
    if not isinstance(names, (tuple, list)) or len(names) != 1 or not isinstance(names[0], str):
        raise ValueError("Expected exactly one original image name")
    return names[0]


def train_matched_step(heads, optimizers, example, runtime, step, maximum_lr):
    """Four independent updates on one frozen example with identical RNG draws."""
    if step <= 500 or step > 1500:
        raise ValueError("Continuation updates must be within 501..1500")
    if set(heads) != set(BINDINGS) or set(optimizers) != set(BINDINGS):
        raise ValueError("All four learning-rate branches are required")
    before = capture_rng()
    after = None
    rows = {}
    for name, (base, multiplier) in BINDINGS.items():
        restore_rng(before)
        head, optimizer = heads[name], optimizers[name]
        head.train()
        lr = lr_at_step(step, maximum=maximum_lr, multiplier=multiplier)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        result = head(**example.common, **example.supports[base])
        loss = runtime.loss(result["probabilities"], example.target)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite or nonscalar training loss: " + name)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        reader_grad = sum(float(parameter.grad.detach().abs().sum())
                          for key, parameter in head.named_parameters()
                          if parameter.grad is not None and key.startswith((
                              "query_projection", "key_projection", "value_projection",
                              "label_embedding", "context_projection")))
        optimizer.step()
        actual_steps = [float(value['step'].item()) if isinstance(value['step'], torch.Tensor)
                        else float(value['step']) for value in optimizer.state.values() if 'step' in value]
        if not actual_steps or any(value != step for value in actual_steps):
            raise ValueError('AdamW actual update count differs from scheduled step: ' + name)
        if not all(bool(torch.isfinite(parameter).all()) for parameter in head.parameters()):
            raise ValueError("Non-finite updated parameter: " + name)
        rows[name] = {"base_arm": base, "lr": lr,
                      "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                      "reader_grad_abs_sum": reader_grad,
                      "optimizer_step": int(actual_steps[0]),
                      "all_optimizer_steps_verified": True}
        current_after = capture_rng()
        if after is None:
            after = current_after
        elif state_fingerprint(current_after) != state_fingerprint(after):
            raise ValueError("Branches consumed different random-number streams")
        del result, loss
    restore_rng(after)
    return rows
