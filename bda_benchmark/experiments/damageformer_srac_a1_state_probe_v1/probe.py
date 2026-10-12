"""Identify the first observed frozen-A1 state mutation in Q training.

This wraps the *unchanged* Q factorial trainer and deliberately stops at its
final A1 guard, before it can certify a formal checkpoint. The output is
diagnostic evidence, never a completed model or a replacement for the E0 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_q_factorial_v1 import train_q
from bda_benchmark.experiments.damageformer_srac_q_factorial_v1.q_variants import (
    ARMS, QVariantFront, q_source_hash,
)


class FrozenStateChanged(RuntimeError):
    """A frozen A1 tensor changed during this diagnostic replay."""


class DiagnosticFinished(RuntimeError):
    """Final validation was traced; do not produce a formal checkpoint."""


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Byte-level equality also handles NaNs and bfloat16 tensors exactly."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_bytes(tensor).numpy().tobytes()).hexdigest()


def compare_state(reference: dict[str, torch.Tensor], module: torch.nn.Module) -> list[dict]:
    """Return changed state_dict keys with exact byte comparisons."""
    current = module.state_dict()
    parameter_names = set(dict(module.named_parameters()))
    buffer_names = set(dict(module.named_buffers()))
    changes = []
    for name in sorted(set(reference) | set(current)):
        before, now = reference.get(name), current.get(name)
        if before is None or now is None:
            changes.append({"key": name, "kind": "missing_or_added",
                            "reference_present": before is not None,
                            "current_present": now is not None})
            continue
        same_layout = before.dtype == now.dtype and before.shape == now.shape
        if same_layout and torch.equal(_bytes(before), _bytes(now)):
            continue
        record = {"key": name,
                  "kind": "parameter" if name in parameter_names else
                          "buffer" if name in buffer_names else "state",
                  "dtype_before": str(before.dtype), "dtype_after": str(now.dtype),
                  "shape_before": list(before.shape), "shape_after": list(now.shape),
                  "sha256_before": _tensor_digest(before),
                  "sha256_after": _tensor_digest(now)}
        if same_layout and before.is_floating_point():
            left, right = before.detach().cpu().float(), now.detach().cpu().float()
            discrepancy = (~torch.isclose(left, right, rtol=0, atol=0, equal_nan=True))
            record["changed_elements"] = int(discrepancy.sum().item())
            finite = discrepancy & torch.isfinite(left) & torch.isfinite(right)
            record["max_abs_difference"] = (
                float((left[finite] - right[finite]).abs().max().item())
                if finite.any() else None
            )
        changes.append(record)
    return changes


class StateTrace:
    def __init__(self, output: Path, full_every: int):
        self.output = output
        self.full_every = full_every
        self.module = None
        self.reference = None
        self.versions = {}
        self.steps = 0
        self.evaluations = 0
        self.base_fingerprint_calls = 0
        self.first_change = None
        self.final_validation = None

    def emit(self, row: dict) -> None:
        with (self.output / "state_trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def capture(self, module: torch.nn.Module) -> None:
        if self.reference is not None:
            raise RuntimeError("A1 reference was captured twice")
        self.module = module
        self.reference = {key: value.detach().cpu().clone()
                          for key, value in module.state_dict().items()}
        self.versions = {key: value._version for key, value in module.state_dict().items()}
        self.emit({"phase": "after_strict_A1_load", "state_keys": len(self.reference),
                   "parameters": len(dict(module.named_parameters())),
                   "buffers": len(dict(module.named_buffers()))})

    def poll(self, phase: str, force: bool = False) -> None:
        if self.module is None or self.reference is None:
            raise RuntimeError("A1 was not loaded before state probe")
        current = self.module.state_dict()
        versions = {key: value._version for key, value in current.items()}
        versions_changed = sorted(key for key in set(versions) | set(self.versions)
                                  if versions.get(key) != self.versions.get(key))
        if not force and not versions_changed:
            return
        changes = compare_state(self.reference, self.module)
        self.versions = versions
        row = {"phase": phase, "optimizer_steps": self.steps,
               "forced_full_scan": force, "version_changed_keys": versions_changed,
               "changed_state": changes}
        self.emit(row)
        if changes:
            self.first_change = row
            raise FrozenStateChanged(f"A1 state changed at {phase}: " +
                                     ", ".join(item["key"] for item in changes[:8]))


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--protocol", required=True,
                                 help="Passed legacy/native full Val450 E0 protocol")
    argument_parser.add_argument("--output", required=True, help="Fresh diagnostic directory")
    argument_parser.add_argument("--arm", choices=sorted(ARMS), default="building_reconstruction")
    argument_parser.add_argument("--full-every", type=int, default=20,
                                 help="Exact state scan interval during Train; version checks happen every step")
    argument_parser.add_argument("--formal-replay", action="store_true",
                                 help="Replay all 2000 Train updates and Val450; still diagnostic-only")
    argument_parser.add_argument("--workers", type=int, default=4)
    argument_parser.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    return argument_parser


def run(args: argparse.Namespace) -> dict:
    if args.full_every < 1 or args.workers < 0:
        raise ValueError("full-every must be positive and workers nonnegative")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    trace = StateTrace(output, args.full_every)
    q_arguments = ["--protocol", args.protocol, "--arm", args.arm,
                   "--output", str(output / "wrapped_train"),
                   "--workers", str(args.workers), "--amp", args.amp]
    if not args.formal_replay:
        q_arguments.extend(("--smoke", "--steps", "2", "--eval_every", "2"))
    train_args = train_q.parser().parse_args(q_arguments)
    manifest = {"format": "SRAC_A1_STATE_PROBE_V1", "diagnostic_only": True,
                "formal_replay": bool(args.formal_replay), "arm": args.arm,
                "expected_steps": 2000 if args.formal_replay else 2,
                "expected_val_images": 450 if args.formal_replay else 2,
                "protocol": str(Path(args.protocol).resolve()),
                "q_source_hash": q_source_hash(),
                "note": "Original Q trainer runs unchanged in-process; execution stops at its final A1 guard before a checkpoint is certified."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    original_a1_load = train_q.load_a1_strict
    original_adapter_load = QVariantFront.load_original_adapter
    original_evaluate = train_q.evaluate_front
    original_fingerprint = train_q.module_fingerprint
    original_step = torch.optim.AdamW.step

    def traced_a1_load(model, path):
        report = original_a1_load(model, path)
        trace.capture(model.base)
        return report

    def traced_adapter_load(model, state):
        result = original_adapter_load(model, state)
        trace.poll("after_front_adapter_load", force=True)
        return result

    def traced_evaluate(*call_args, **call_kwargs):
        trace.evaluations += 1
        part = "initial_val" if trace.evaluations == 1 else "final_val"
        trace.poll("before_" + part, force=True)
        try:
            result = original_evaluate(*call_args, **call_kwargs)
        except Exception:
            trace.poll("during_" + part + "_before_error", force=True)
            raise
        trace.poll("after_" + part, force=True)
        if trace.evaluations > 1:
            trace.final_validation = {
                "image_count": result.get("image_count"),
                "a1_valid_miou": result.get("metrics", {}).get("a1_pixel", {}).get("valid_miou"),
                "region_valid_miou": result.get("metrics", {}).get("front_region", {}).get("valid_miou"),
            }
            trace.emit({"phase": "final_val_observed", **trace.final_validation})
        return result

    def traced_fingerprint(module):
        if module is trace.module:
            trace.base_fingerprint_calls += 1
            if trace.base_fingerprint_calls == 3:
                trace.poll("before_final_A1_guard", force=True)
                raise DiagnosticFinished("Final A1 guard traced; formal checkpoint intentionally uncertified")
        return original_fingerprint(module)

    def traced_step(optimizer, *call_args, **call_kwargs):
        next_step = trace.steps + 1
        scan = next_step == 1 or next_step % trace.full_every == 0
        trace.poll(f"before_optimizer_step_{next_step}", force=scan)
        try:
            result = original_step(optimizer, *call_args, **call_kwargs)
        except Exception:
            trace.poll(f"during_optimizer_step_{next_step}_before_error", force=True)
            raise
        trace.steps = next_step
        trace.poll(f"after_optimizer_step_{next_step}", force=scan)
        return result

    train_q.load_a1_strict = traced_a1_load
    QVariantFront.load_original_adapter = traced_adapter_load
    train_q.evaluate_front = traced_evaluate
    train_q.module_fingerprint = traced_fingerprint
    torch.optim.AdamW.step = traced_step
    status, error = "unexpected_return", None
    try:
        train_q.run(train_args)
    except DiagnosticFinished:
        status = "finished_without_detected_A1_change"
    except FrozenStateChanged as exc:
        status, error = "A1_state_changed", str(exc)
    except Exception as exc:
        status, error = "trainer_error", f"{type(exc).__name__}: {exc}"
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if trace.module is not None and trace.first_change is None:
            try:
                trace.poll("after_trainer_error", force=True)
            except FrozenStateChanged as state_exc:
                status = "A1_state_changed"
                error = f"{state_exc}; original trainer error: {error}"
    finally:
        train_q.load_a1_strict = original_a1_load
        QVariantFront.load_original_adapter = original_adapter_load
        train_q.evaluate_front = original_evaluate
        train_q.module_fingerprint = original_fingerprint
        torch.optim.AdamW.step = original_step

    summary = {"format": manifest["format"], "diagnostic_only": True,
               "complete": status == "finished_without_detected_A1_change",
               "status": status, "error": error, "arm": args.arm,
               "formal_replay": bool(args.formal_replay),
               "optimizer_steps_observed": trace.steps,
               "evaluations_observed": trace.evaluations,
               "base_fingerprint_calls": trace.base_fingerprint_calls,
               "first_change": trace.first_change,
               "final_validation": trace.final_validation,
               "formal_checkpoint_produced": False,
               "test_accessed": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return summary


def main() -> None:
    result = run(parser().parse_args())
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
