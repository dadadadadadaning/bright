"""Bounded live probes and explicitly changed CPU-pooling continuation.

All original artifacts remain read-only. Probe modes restore and retain all
three readers and AdamW states. The live probe stops at the first 512-channel
statistics call; the CPU probe completes that first image and all three reader
forwards, checking their losses without backward or optimizer.step. Training
delegates to the independently tested continuation engine.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import random
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FORMAT = "SRAC_JOINT_LIVE_RECOVERY_V2"

def state_hash(state):
    result = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.view(__import__('torch').uint8).numpy().tobytes())
    return result.hexdigest()


def validate_loaded(saved, evidence):
    import torch
    import numpy as np
    arms = {'r0', 'rt', 'rf'}
    if (saved.get('format') != 'SRAC_REGION_MEMORY_JOINT_TRAIN_V1'
            or saved.get('mode') != 'train' or saved.get('smoke') is not False
            or saved.get('training_complete') is not False):
        raise ValueError('Checkpoint is not an incomplete formal joint checkpoint')
    if saved.get('config') != evidence['arguments']:
        raise ValueError('Checkpoint configuration differs from the original manifest')
    if (saved.get('source_hash') != evidence['original_source_hash']
            or saved.get('protocol_info') != evidence['protocol_info']):
        raise ValueError('Checkpoint source/bank/runtime binding differs')
    step = evidence['checkpoint_step']
    if saved.get('step') != step or saved.get('updates_per_arm') != {a: step for a in arms}:
        raise ValueError('Checkpoint updates differ across arms')
    if set(saved.get('heads', {})) != arms or set(saved.get('optimizers', {})) != arms:
        raise ValueError('Checkpoint is missing a head or optimizer')
    if saved.get('initial_head_hashes') != evidence['initial_head_hashes']:
        raise ValueError('Checkpoint initial-head binding differs')
    for arm in arms:
        if state_hash(saved['heads'][arm]) != evidence['sidecar']['head_hashes'][arm]:
            raise ValueError('Saved head checksum mismatch: ' + arm)
        if any(not bool(torch.isfinite(t).all()) for t in saved['heads'][arm].values()):
            raise ValueError('Nonfinite checkpoint head: ' + arm)
        optimizer = saved['optimizers'][arm]
        if not optimizer.get('state') or not optimizer.get('param_groups'):
            raise ValueError('Checkpoint optimizer state is empty: ' + arm)
        parameter_ids = []
        for group in optimizer['param_groups']:
            ids = group.get('params')
            if not isinstance(ids, list) or not ids or any(type(p) is not int for p in ids):
                raise ValueError('Invalid optimizer parameter IDs: ' + arm)
            parameter_ids.extend(ids)
        if len(set(parameter_ids)) != len(parameter_ids) or not set(optimizer['state']) <= set(parameter_ids):
            raise ValueError('Optimizer state/parameter mapping differs: ' + arm)
        for parameter in optimizer['state'].values():
            if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(parameter):
                raise ValueError('Incomplete AdamW state: ' + arm)
            if (not torch.is_tensor(parameter['exp_avg']) or not torch.is_tensor(parameter['exp_avg_sq'])
                    or parameter['exp_avg'].shape != parameter['exp_avg_sq'].shape):
                raise ValueError('Inconsistent AdamW moment shapes: ' + arm)
            count = parameter['step']
            count = float(count.item()) if torch.is_tensor(count) and count.numel() == 1 else float(count)
            if not 0 < count <= step or count != int(count):
                raise ValueError('Invalid AdamW update count: ' + arm)
            for value in parameter.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise ValueError('Nonfinite optimizer state: ' + arm)
    if set(saved.get('rng', {})) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
        raise ValueError('Checkpoint RNG is incomplete')
    rng = saved['rng']
    try:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Invalid checkpoint RNG state: ' + str(error)) from error
    if not isinstance(rng['torch_cuda'], list) or any(
            not torch.is_tensor(t) or t.device.type != 'cpu' or t.dtype != torch.uint8 or t.ndim != 1
            for t in rng['torch_cuda']):
        raise ValueError('Invalid CUDA RNG state schema')
    ledger = saved.get('input_file_sha256')
    if not isinstance(ledger, dict) or not ledger:
        raise ValueError('Checkpoint dataset fingerprint ledger is missing')
    if any(not isinstance(path, str) or not Path(path).is_absolute()
           or not isinstance(value, str) or len(value) != 64
           or any(c not in '0123456789abcdef' for c in value) for path, value in ledger.items()):
        raise ValueError('Invalid dataset fingerprint ledger schema')


def _latest_packet(runtime, *, channels=512):
    records = runtime.guard.report().get("captured_packets", [])
    # The bounded intervention is tied to the FIRST stats call, even if a full
    # CPU image forward later happens to execute another 512-channel pooling.
    return next((copy.deepcopy(row) for row in records
                 if row.get("content_channels") == channels), None)


def _failure_phase(output):
    path = Path(output) / "pool_debug" / "failure.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("phase")
        except (OSError, ValueError):
            pass
    return None


def probe(original_train, args, runtime_factory, saved, binding, evidence,
          probe_complete_class, *, mode):
    """Recreate checkpoint-revalidation allocations and inspect ONE Val batch."""
    import torch
    if mode not in ("live-probe", "cpu-probe"):
        raise ValueError("Probe requires live-probe or cpu-probe")
    output = original_train._new_output(args.output)
    backend = "live_contiguous_probe" if mode == "live-probe" else "cpu_fp32_reference"
    summary = {"format": FORMAT, "phase": "initializing_probe", "mode": mode,
               "complete": False, "execution_complete": False,
               "full_validation_evidence": False, "training": False, "training_complete": False,
               "probe_only": True, "test_accessed": False, "exact_historical_resume_claim": False,
               "checkpoint_step": saved["step"], "backend": backend, "recovery_binding": binding,
               "original_updates_per_arm": saved["updates_per_arm"],
               "new_updates_per_arm": {a: 0 for a in original_train.ARMS},
               "optimizer_steps_called": 0, "capture_ready": False,
               "gate_passed": False, "strict_cpu_gate_passed": False,
               "full_single_image_forward_complete": False,
               "whole_val450_processed": False, "failure": None}
    original_train.write_json(output / "summary.json", summary)
    runtime = None
    packet = None
    diagnostic_error = None
    try:
        # This is intentionally the same seed placement as continue_training
        # before checkpoint_reval. Saved RNG is NOT restored until after the
        # full revalidation/reconstructed iterator in actual training mode.
        original_train._seed(int(args.seed))
        with runtime_factory(args, output) as runtime:
            original_train._seed(int(args.seed))
            _, val_names = original_train._names(runtime, args.mode == "smoke")
            if not val_names or evidence.get("probe_image") != val_names[0]:
                raise ValueError("Bounded probe requires the captured failure image to be first in Val order")
            heads, _ = original_train._new_heads(args, runtime.device)
            optimizers = {a: torch.optim.AdamW(h.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                          for a, h in heads.items()}
            for arm in original_train.ARMS:
                heads[arm].load_state_dict(saved["heads"][arm], strict=True)
                optimizers[arm].load_state_dict(saved["optimizers"][arm])
                heads[arm].eval()
            summary.update(phase="bounded_first_val_stats", image=val_names[0],
                           expected_call=evidence.get("probe_call"),
                           full_val_loader=True, heads_and_optimizers_restored=True,
                           checkpoint_rng_restored=False)
            original_train.write_json(output / "summary.json", summary)
            iterator = iter(runtime.batches(val_names, train=False, seed=args.seed))
            try:
                batch = next(iterator)
            except StopIteration:
                raise ValueError("Val loader returned no first batch")
            try:
                with torch.no_grad():
                    example = runtime.prepare(batch)
                    if mode == "cpu-probe":
                        original_train._assert_example(example)
                        if example.image != val_names[0]:
                            raise ValueError("Single-image CPU probe loader identity changed")
                        checks = {}
                        for arm in original_train.ARMS:
                            decoded = heads[arm](**example.common, **example.supports[arm])
                            probability = decoded["probabilities"]
                            loss = runtime.loss(probability, example.target)
                            if (not bool(torch.isfinite(probability).all()) or loss.ndim != 0
                                    or not bool(torch.isfinite(loss))):
                                raise ValueError("Nonfinite single-image CPU probe head/loss: " + arm)
                            checks[arm] = {"finite_probabilities": True, "finite_loss": True,
                                           "loss": float(loss), "optimizer_updates": 0}
                        summary.update(full_single_image_forward_complete=True,
                                       per_head_forward_checks=checks)
                        packet = _latest_packet(runtime)
            except probe_complete_class as stopped:
                if mode != "live-probe":
                    raise RuntimeError("CPU probe stopped before complete single-image forward") from stopped
                packet = copy.deepcopy(stopped.report)
            except ValueError as error:
                # Only a documented original-reference mismatch with an actual
                # 512-channel capture is a completed diagnostic probe. Other
                # exceptions, including kernel asserts, fail normally.
                phase = _failure_phase(output)
                packet = _latest_packet(runtime)
                if mode == "cpu-probe" or phase != "cpu_reference_gate" or packet is None:
                    raise
                diagnostic_error = str(error)
                summary["failure"] = {"phase": phase, "error": str(error),
                                      "diagnostic_reference_mismatch": True}
            else:
                if mode == "live-probe":
                    raise RuntimeError("Probe did not stop at the first 512-channel statistics call")
            if packet is None:
                packet = _latest_packet(runtime)
            if packet is None or packet.get("content_channels") != 512:
                raise ValueError("Probe failed to retain a 512-channel actual-output packet")
            packet_path = Path(packet.get("path") or packet.get("packet", "")).resolve()
            if output.resolve() not in packet_path.parents:
                raise ValueError("Captured live packet is outside this new worker result")
            if not packet_path.is_file() or original_train.digest(packet_path) != packet.get("sha256"):
                raise ValueError("Captured live pooling packet checksum mismatch")
            if packet.get("image") != evidence["probe_image"] or packet.get("backend") != backend:
                raise ValueError("Captured packet image/backend differs from the bounded probe")
            if packet.get("call") != evidence.get("probe_call"):
                raise ValueError("Captured packet call differs from the failed live context")
            target = output / "capture.pth"
            shutil.copyfile(packet_path, target)
            if original_train.digest(target) != packet["sha256"]:
                raise ValueError("Copied pooling packet checksum mismatch")
            gate = packet.get("strict_cpu_gate_passed") is True and diagnostic_error is None
            summary.update(capture_ready=True, capture=str(target), capture_sha256=packet["sha256"],
                           packet=str(packet_path), image=packet["image"], call=packet.get("call"),
                           gate_passed=gate, strict_cpu_gate_passed=gate,
                           strict_original_cpu32_gate=packet.get("strict_original_cpu32_gate"),
                           independent_fp64_analysis_pending=True,
                           diagnostic_stop=True, training_started=False)
            if gate:
                summary["integrity"] = original_train._checked_verify(runtime)
            # Keep heads and optimizers alive through the exact probe call.
            del heads, optimizers
        summary["guard_report"] = original_train._restored_guard(runtime)
        summary.update(phase="probe_complete" if diagnostic_error is None else "probe_reference_mismatch",
                       complete=True, execution_complete=True)
        original_train.write_json(output / "summary.json", summary)
        return summary
    except BaseException as error:
        # CUDA may now be poisoned: inspect file/Python metadata only, never
        # retry a kernel, inspect GPU parameters or clear the CUDA cache.
        summary.update(phase="probe_failed", complete=False, execution_complete=False,
                       failure={"phase": _failure_phase(output) or summary["phase"],
                                "error": type(error).__name__ + ": " + str(error)})
        if runtime is not None and getattr(runtime, "guard", None) is not None:
            try:
                summary["guard_report"] = runtime.guard.report()
            except Exception as report_error:
                summary["guard_report_error"] = repr(report_error)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        original_train.write_json(output / "summary.json", summary)
        raise


def run(args):
    from recovery import (check_preserved, digest, read, source_hash, verify_release,
                          inspect_failed, check_train_gates)
    if args.mode not in ("live-probe", "cpu-probe", "train"):
        raise ValueError("Unexpected worker mode")
    if Path(args.output).expanduser().exists():
        raise FileExistsError("New worker output required: " + str(args.output))
    verify_release(HERE)
    evidence = read(args.evidence)
    train_gates = None
    if args.mode == "train":
        # A direct worker invocation must not bypass the live-context and
        # independently checked CPU-FP64 evidence required by the coordinator.
        train_gates = check_train_gates(args.evidence)
    check_preserved(evidence)
    current = inspect_failed(evidence["failed_run"], evidence["tool"])
    for field in ("arguments", "protocol_info", "checkpoint", "checkpoint_step", "original_source_hash"):
        if current[field] != evidence[field]:
            raise ValueError("Original failed-run binding changed before worker: " + field)
    original_preserved = evidence.get("original_preserved_sha256", current["preserved_sha256"])
    if current["preserved_sha256"] != original_preserved:
        raise ValueError("Original preserved evidence binding changed")
    if source_hash(HERE) != evidence["repair_source_hash"]:
        raise ValueError("Live recovery package source binding changed")
    import torch
    saved = torch.load(evidence["checkpoint"], map_location="cpu", weights_only=False)
    validate_loaded(saved, evidence)
    tool, repo = Path(evidence["tool"]).resolve(), Path(evidence["repo"]).resolve()
    sys.path.insert(0, str(tool))
    sys.path.insert(1, str(repo))
    original_train = importlib.import_module("train")
    original_server = importlib.import_module("server_runtime")
    for name in ("train", "server_runtime", "common", "model"):
        if Path(sys.modules[name].__file__).resolve().parent != tool:
            raise ValueError("Unexpected original package module: " + name)
    if original_train.source_hash() != evidence["original_source_hash"]:
        raise ValueError("Loaded original training source differs")
    pool = importlib.import_module("bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool")
    from pool_runtime import patched_guard, ProbeComplete
    from recover_training import continue_training
    for name in ("pool_runtime", "recover_training"):
        if Path(sys.modules[name].__file__).resolve().parent != HERE:
            raise ValueError("Unexpected live recovery module: " + name)
    backend = "live_contiguous_probe" if args.mode == "live-probe" else "cpu_fp32_reference"
    binding = dict(format=FORMAT, backend=backend,
                   feature_pool_device_intervention="cpu" if backend == "cpu_fp32_reference" else "cuda",
                   repair_source_hash=evidence["repair_source_hash"],
                   repair_release_sha256=evidence["repair_release_sha256"],
                   pool_runtime_sha256=digest(HERE / "pool_runtime.py"),
                   original_checkpoint_sha256=evidence["sidecar"]["sha256"],
                   original_history_sha256=digest(evidence["history"]),
                   latest_failed_recovery=evidence.get("latest_failed_recovery"),
                   exact_historical_resume_claim=False, automatic_fallback=False,
                   fixed_gpu_extracted_v3_bank=True,
                   runtime_feature_pool_backend_changed=backend == "cpu_fp32_reference",
                   uninterrupted_5000_update_equivalence_claim=False,
                   fixed_reference_tolerance={"atol": 2e-5, "rtol": 2e-5},
                   cuda_launch_blocking="1")
    if train_gates is not None:
        binding["validated_preflight_binding"] = train_gates

    class RecoveryRuntime(original_server.Runtime):
        def __enter__(self):
            try:
                super().__enter__()
                if self.protocol_info != saved["protocol_info"]:
                    raise ValueError("Current frozen source/bank/data differs from checkpoint")
                ledger = saved["input_file_sha256"]
                required = set()
                for name in self.train_names + self.val_names:
                    required.update(str(Path(p).resolve()) for p in self.deps.sample_paths(
                        self.bundle["args"].dataset, name).values())
                if not required <= {str(Path(p).resolve()) for p in ledger}:
                    raise ValueError("Saved ledger does not cover complete Train2918 and Val450")
                for index, (path, expected) in enumerate(ledger.items(), 1):
                    if digest(path) != expected:
                        raise ValueError("Saved dataset fingerprint changed: " + path)
                    self.input_ledger[Path(path).resolve()] = expected
                    if index % 1000 == 0 or index == len(ledger):
                        print("VERIFY_SAVED_INPUT_FILES %d/%d" % (index, len(ledger)), flush=True)
                return self
            except BaseException:
                self.close(*sys.exc_info())
                raise

    config = copy.deepcopy(evidence["arguments"])
    config["output"] = str(Path(args.output).resolve())
    history = [json.loads(line) for line in Path(evidence["history"]).read_text(encoding="utf-8").splitlines()]
    with patched_guard(pool, backend=backend, stop_after_stats=args.mode == "live-probe", binding=binding):
        if args.mode == "train":
            summary = continue_training(original_train, SimpleNamespace(**config), RecoveryRuntime,
                                        saved, history, binding)
        else:
            summary = probe(original_train, SimpleNamespace(**config), RecoveryRuntime, saved,
                            binding, evidence, ProbeComplete, mode=args.mode)
    check_preserved(evidence)
    if source_hash(HERE) != evidence["repair_source_hash"]:
        raise ValueError("Live recovery code changed during execution")
    return summary


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("live-probe", "cpu-probe", "train"), required=True)
    value.add_argument("--evidence", required=True)
    value.add_argument("--output", required=True)
    return value


if __name__ == "__main__":
    parsed = parser().parse_args()
    try:
        result = run(parsed)
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
        # Preserve an existing run when rejected at entry. For this NEW worker
        # result, record failures even when binding checks stop before Runtime.
        if not isinstance(error, FileExistsError):
            folder = Path(parsed.output).expanduser().resolve()
            folder.mkdir(parents=True, exist_ok=True)
            summary_file = folder / "summary.json"
            if summary_file.is_file():
                try:
                    record = json.loads(summary_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    record = {}
            else:
                record = {}
            record.update(format=FORMAT, complete=False, execution_complete=False,
                          full_validation_evidence=False, mode=parsed.mode,
                          worker_failure=type(error).__name__ + ": " + str(error))
            if parsed.mode != "train":
                record.update(probe_only=True, new_updates_per_arm={a:0 for a in ("r0","rt","rf")})
            temporary = summary_file.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                                 encoding="utf-8")
            temporary.replace(summary_file)
        raise
    if result.get("execution_complete") is not True or (
            parsed.mode == "cpu-probe" and result.get("strict_cpu_gate_passed") is not True):
        raise SystemExit(1)
