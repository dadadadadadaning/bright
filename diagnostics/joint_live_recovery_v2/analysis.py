"""CPU-only analysis of exact live pool inputs, outputs and CPU reference.

This module never initializes CUDA, changes a production backend, starts training,
relaxes tolerance or declares a root cause. A live GPU mismatch is an observation.
CPU continuation eligibility is assessed separately using its declared live CPU
packet and independent FP64 controls at the unchanged 2e-5 bound.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import torch

from numeric_core import OUTPUT_KEYS, compare, cpu_reference64, pool_literal, validate_snapshot
from pool_ops import validate_layout


FORMAT = "JOINT_LIVE_POOL_ANALYSIS_V2"
BACKENDS = {"live": "live_contiguous_probe", "cpu": "cpu_fp32_reference"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


@contextmanager
def cpu_threads(count):
    previous = torch.get_num_threads()
    if previous != count:
        torch.set_num_threads(count)
    try:
        yield
    finally:
        if torch.get_num_threads() != previous:
            torch.set_num_threads(previous)


def validate_packet(packet, mode, *, expected_image=None, expected_call=None,
                    expected_backend=None, evidence=None):
    if mode not in BACKENDS:
        raise ValueError("Analysis mode must be live or cpu")
    if not isinstance(packet, dict) or packet.get("format") != "LIVE_POOL_CAPTURE_V2":
        raise ValueError("Expected LIVE_POOL_CAPTURE_V2 packet")
    snapshot, meta = packet.get("snapshot"), packet.get("meta")
    check = validate_snapshot(snapshot)
    if not isinstance(meta, dict):
        raise ValueError("Missing live capture metadata")
    image, call = snapshot.get("image"), snapshot.get("call")
    backend = meta.get("backend")
    if (not isinstance(image, str) or not image or type(call) is not int or call < 1
            or meta.get("image") != image or meta.get("call") != call
            or snapshot.get("backend") != backend or backend != BACKENDS[mode]):
        raise ValueError("Snapshot/meta identity or declared backend mismatch")
    for value, expected, field in ((image, expected_image, "image"),
                                    (call, expected_call, "call"),
                                    (backend, expected_backend, "backend")):
        if expected is not None and value != expected:
            raise ValueError("Capture differs from parent binding: " + field)
    if meta.get("captured_before_original_comparison") is not True:
        raise ValueError("Packet did not prove capture before the original comparison")
    threads = meta.get("original_num_threads")
    if type(threads) is not int or threads < 1:
        raise ValueError("Missing positive source CPU thread count")
    if meta.get("cpu_execution_num_threads", threads) != threads:
        raise ValueError("CPU reference thread count differs from captured source policy")
    x = snapshot["content"]
    b, c, h, w = x.shape
    count = snapshot["token_count"]
    if c != 512 or meta.get("content_channels") != c:
        raise ValueError("This gate requires the captured 512-channel stats call")
    layout = snapshot.get("layout")
    if not isinstance(layout, dict) or any(name not in layout for name in ("content", "weights", "ids")):
        raise ValueError("Captured input layout is missing")
    for name in ("content", "weights", "ids"):
        validate_layout(snapshot[name], layout[name])
    source, source_sha = meta.get("original_source"), meta.get("original_sha256")
    if (not isinstance(source, str) or not Path(source).is_absolute()
            or not isinstance(source_sha, str) or not re.fullmatch("[0-9a-f]{64}", source_sha)):
        raise ValueError("Original pool source binding is missing")
    if evidence is not None:
        ledger = evidence.get("preserved_sha256")
        if not isinstance(ledger, dict) or ledger.get(str(Path(source).resolve())) != source_sha:
            raise ValueError("Captured original source differs from parent evidence")
    shapes = {"tokens": (b, count, c), "mass": (b, count), "reconstruction": (b, c, h, w)}
    for kind in ("actual", "expected"):
        outputs = packet.get(kind)
        if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_KEYS):
            raise ValueError("Missing exact live output fields: " + kind)
        for name, value in outputs.items():
            if (not torch.is_tensor(value) or value.device.type != "cpu"
                    or value.dtype != torch.float32 or tuple(value.shape) != shapes[name]
                    or value.requires_grad or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid/nonfinite captured CPU output: " + kind + "." + name)
    return check


@torch.no_grad()
def analyze_packet(packet, mode, *, expected_image=None, expected_call=None,
                   expected_backend=None, evidence=None, emit=print):
    check = validate_packet(packet, mode, expected_image=expected_image, expected_call=expected_call,
                            expected_backend=expected_backend, evidence=evidence)
    snapshot, meta = packet["snapshot"], packet["meta"]
    x, q, ids, count = (snapshot[key] for key in ("content", "weights", "ids", "token_count"))
    comparisons, timings = {}, {}
    started = time.monotonic()

    def timed(name, function):
        emit("CPU ANALYSIS " + name, flush=True)
        begin = time.monotonic()
        value = function()
        timings[name] = time.monotonic() - begin
        return value

    def comparison(name, actual, reference):
        with cpu_threads(4):
            comparisons[name] = timed(name, lambda: compare(actual, reference))

    comparison("actual_vs_exact_live_expected", packet["actual"], packet["expected"])
    with cpu_threads(meta["original_num_threads"]):
        source32 = timed("cpu32_source_threads", lambda: pool_literal(x, q, ids, count))
    comparison("exact_live_expected_vs_cpu32_source_threads", packet["expected"], source32)
    comparison("live_actual_vs_cpu32_source_threads", packet["actual"], source32)
    source_exact = all(torch.equal(packet["expected"][key], source32[key]) for key in OUTPUT_KEYS)
    with cpu_threads(4):
        four32 = timed("cpu32_four_threads", lambda: pool_literal(x, q, ids, count))
    comparison("cpu32_four_threads_vs_source_threads", four32, source32)
    comparison("exact_live_expected_vs_cpu32_four_threads", packet["expected"], four32)
    comparison("live_actual_vs_cpu32_four_threads", packet["actual"], four32)
    with cpu_threads(4):
        scatter64 = timed("cpu64_literal_scatter", lambda: pool_literal(x, q, ids, count, precision="fp64"))
        grouped64 = timed("cpu64_independent_grouped", lambda: cpu_reference64(snapshot))
    comparison("cpu64_scatter_vs_grouped", scatter64, grouped64)
    del scatter64
    comparison("live_actual_vs_cpu64", packet["actual"], grouped64)
    comparison("exact_live_expected_vs_cpu64", packet["expected"], grouped64)
    comparison("cpu32_source_threads_vs_cpu64", source32, grouped64)
    comparison("cpu32_four_threads_vs_cpu64", four32, grouped64)
    del source32, four32, grouped64
    reproduced = comparisons["exact_live_expected_vs_cpu32_source_threads"]["passed"]
    independent = comparisons["cpu64_scatter_vs_grouped"]["passed"]
    cpu_f64 = mode == "cpu" and comparisons["live_actual_vs_cpu64"]["passed"]
    cpu_checks = mode == "cpu" and all((reproduced, independent, cpu_f64,
                 comparisons["actual_vs_exact_live_expected"]["passed"],
                 comparisons["exact_live_expected_vs_cpu64"]["passed"]))
    return {"format": FORMAT, "mode": mode, "backend": meta["backend"],
            "image": snapshot["image"], "call": snapshot["call"], "input_check": check,
            "execution_complete": True, "diagnostic_complete": True, "adequate_capture": True,
            "cpu64_reference_agreement": independent, "cpu32_reference_reproduced": reproduced,
            "live_expected_exactly_reproduced": source_exact,
            "CPU_backend_vs_FP64_passed": bool(cpu_f64),
            "cpu_continuation_checks_passed": bool(cpu_checks),
            "cpu_thread_difference_present": not comparisons["cpu32_four_threads_vs_source_threads"]["passed"],
            "source_cpu_threads": meta["original_num_threads"], "comparison_cpu_threads": 4,
            "timings_seconds": timings, "elapsed_seconds": time.monotonic()-started,
            "comparisons": comparisons, "capture_meta": meta,
            "training": False, "training_allowed": False, "test_accessed": False,
            "full_validation_evidence": False, "original_gate_relaxed": False,
            "root_cause_confirmed": False, "failure": None,
            "interpretation": ("CPU backend checks concern this declared CPU packet only. "
                "Thread4 differences and live GPU discrepancies are observations; they do not change a gate "
                "or prove full-training stability. Full source-bound Val revalidation is still required.")}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "--capture", dest="input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=tuple(BACKENDS), required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--expected-image", required=True)
    p.add_argument("--expected-call", type=int, required=True)
    p.add_argument("--expected-backend", required=True)
    p.add_argument("--evidence")
    return p


def run(args):
    from source_binding import verify_release
    verify_release(Path(__file__).resolve().parent)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"format": FORMAT, "execution_complete": False, "diagnostic_complete": False,
               "adequate_capture": False, "cpu64_reference_agreement": False,
               "cpu32_reference_reproduced": False, "CPU_backend_vs_FP64_passed": False,
               "cpu_continuation_checks_passed": False, "training": False, "training_allowed": False,
               "test_accessed": False, "full_validation_evidence": False, "failure": None}
    write(output / "summary.json", summary)
    try:
        if not re.fullmatch("[0-9a-f]{64}", args.expected_sha256) or digest(args.input) != args.expected_sha256:
            raise ValueError("Capture checksum differs from parent binding")
        # All tensors and JSON-friendly metadata were captured by the guarded
        # worker. Safe loading occurs only after the packet checksum is bound.
        packet = torch.load(args.input, map_location="cpu", weights_only=True)
        evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8")) if args.evidence else None
        summary = analyze_packet(packet, args.mode, expected_image=args.expected_image,
                                 expected_call=args.expected_call, expected_backend=args.expected_backend,
                                 evidence=evidence)
        summary.update(capture=str(Path(args.input).resolve()), capture_sha256=args.expected_sha256,
                       input_sha256=args.expected_sha256,
                       evidence_sha256=digest(args.evidence) if args.evidence else None)
        write(output / "summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("adequate_capture", "cpu32_reference_reproduced",
                         "cpu64_reference_agreement", "CPU_backend_vs_FP64_passed",
                         "cpu_continuation_checks_passed", "elapsed_seconds")}, ensure_ascii=False), flush=True)
        return summary
    except Exception as error:
        summary["failure"] = {"type": type(error).__name__, "message": str(error)}
        write(output / "summary.json", summary)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    run(parser().parse_args())
