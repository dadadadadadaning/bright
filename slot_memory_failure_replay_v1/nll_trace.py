"""Observe original NLL autograd targets without replacing loss inputs or math."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


def _cpu_copy(tensor):
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _number(value):
    value = value.item()
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def _label_details(torch, tensor, cpu):
    flat = cpu.reshape(-1)
    finite_mask = torch.isfinite(flat)
    values, counts = torch.unique(flat[finite_mask], sorted=True, return_counts=True)
    floating = flat.is_floating_point()
    finite = bool(finite_mask.all())
    integral = finite and (not floating or bool((flat == flat.round()).all()))
    histogram = [
        {"value": _number(v), "count": int(c)}
        for v, c in zip(values[:32], counts[:32])
    ]
    return dict(
        dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()),
        device=str(tensor.device), data_ptr=tensor.data_ptr(),
        storage_ptr=tensor.untyped_storage().data_ptr(), storage_offset=tensor.storage_offset(),
        version=_version(tensor), count=flat.numel(), finite=finite, integral=integral,
        minimum=_number(flat.min()) if flat.numel() else None,
        maximum=_number(flat.max()) if flat.numel() else None,
        histogram=histogram, unique_count=values.numel(), histogram_truncated=values.numel() > 32,
        histogram_finite_only=True, nonfinite_count=int((~finite_mask).sum()),
        sha256=hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    )


def label_snapshot(torch, tensor):
    """JSON-ready observed label metadata; device tensors are only read."""
    return _label_details(torch, tensor, _cpu_copy(tensor))


def _signed_ignore(value):
    value = int(value)
    return value - (1 << 64) if value >= 1 << 63 else value


class NLLTrace:
    """Scoped wrapper around the exact original ``deps.region_nll`` function.

    Snapshot copies and synchronization alter allocation/timing. Passing this
    diagnostic cannot establish that the original CUDA failure is fixed.
    """

    def __init__(self, torch_module, deps, emit):
        self.torch, self.deps, self.emit = torch_module, deps, emit
        self.original = deps.region_nll
        self.entries = []
        self.wrapper = None

    def __enter__(self):
        if self.wrapper is not None:
            raise RuntimeError("NLLTrace cannot be entered twice")

        def observed(*args, **kwargs):
            result = self.original(*args, **kwargs)
            if result.requires_grad and result.grad_fn is not None:
                self._capture(result.grad_fn)
                self.snapshot("immediately_after_nll")
            return result

        self.wrapper = observed
        self.deps.region_nll = observed
        return self

    def __exit__(self, *exception):
        # The diagnostic owns this scoped assignment, including exception paths.
        self.deps.region_nll = self.original
        return False

    def _capture(self, root):
        queue, seen = [root], set()
        captured = {entry["node"] for entry in self.entries}
        while queue:
            node = queue.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            name = type(node).__name__
            if name.startswith(("NllLossBackward", "NllLoss2DBackward")):
                if node not in captured:
                    values, target = node._saved_self, node._saved_target
                    if values.ndim < 1:
                        raise ValueError("NLL saved input has no class dimension")
                    classes = values.shape[0] if values.ndim == 1 else values.shape[1]
                    weight = getattr(node, "_saved_weight", None)
                    self.entries.append(dict(
                        node=node, node_type=name, values=values, target=target,
                        classes=int(classes),
                        ignore_index=_signed_ignore(getattr(node, "_saved_ignore_index", -100)),
                        reduction=int(getattr(node, "_saved_reduction", 1)),
                        weight=None if weight is None else _cpu_copy(weight),
                        baseline=None, cpu_case=None,
                    ))
            queue.extend(next_node for next_node, _ in node.next_functions if next_node is not None)

    def snapshot(self, stage):
        if not self.entries:
            self.emit("nll_saved_targets", stage=stage, count=0)
            return
        for index, entry in enumerate(self.entries):
            cpu = _cpu_copy(entry["target"])
            details = _label_details(self.torch, entry["target"], cpu)
            legal = ((cpu >= 0) & (cpu < entry["classes"])) | (cpu == entry["ignore_index"])
            invalid_count = int((~legal).sum())
            baseline = entry["baseline"]
            if baseline is None:
                entry["baseline"] = dict(sha256=details["sha256"], version=details["version"])
            changed = baseline is not None and baseline["sha256"] != details["sha256"]
            version_changed = baseline is not None and baseline["version"] != details["version"]
            # Retain CPU-only kernel inputs before backward, including on a
            # detected invalid/changed target. No device access is needed to save.
            if entry["cpu_case"] is None or stage == "before_backward" or changed:
                entry["cpu_case"] = dict(
                    input=_cpu_copy(entry["values"]), target=cpu,
                    weight=entry["weight"], reduction=entry["reduction"],
                    ignore_index=entry["ignore_index"], stage=stage,
                    node_type=entry["node_type"], classes=entry["classes"],
                )
            self.emit(
                "nll_saved_target", stage=stage, index=index, node_type=entry["node_type"],
                input_shape=list(entry["values"].shape), classes=entry["classes"],
                ignore_index=entry["ignore_index"], invalid_count=invalid_count,
                target_changed=changed, version_changed=version_changed, target=details,
            )
            if invalid_count or not details["integral"]:
                raise ValueError("Invalid NLL saved target at %s, node %s" % (stage, index))
            if changed or version_changed:
                raise ValueError("NLL saved target changed at %s, node %s" % (stage, index))

    def save_failure(self, path):
        """Save only previously captured CPU copies; safe after a CUDA failure."""
        cases = [entry["cpu_case"] for entry in self.entries if entry["cpu_case"] is not None]
        if not cases:
            return None
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        payload = dict(
            format="BRIGHT_NLL_DIAGNOSTIC_INPUTS_V1", cases=cases,
            original_kernel_inputs=True, exact_training_replay=False,
            note="Captured diagnostic inputs; saved CPU snapshots may predate the failure.",
        )
        self.torch.save(payload, path)
        return str(path)
