"""Observe NVIDIA GPU availability without importing torch or creating a CUDA context."""
from __future__ import annotations

import csv
import io
import os
import subprocess
import time


def _query(fields, *, gpu=None, applications=False):
    command = ["nvidia-smi", "--query-compute-apps=" + fields if applications
               else "--query-gpu=" + fields, "--format=csv,noheader,nounits"]
    if gpu is not None:
        command += ["--id=" + gpu]
    result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=15)
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(result.stdout))
            if row and any(value.strip() for value in row)]


def resolve_gpu(device_index):
    if not isinstance(device_index, int) or device_index < 0:
        raise ValueError("Expected a nonnegative CUDA device index")
    cards = _query("index,uuid,pci.bus_id")
    if not cards or any(len(row) != 3 for row in cards):
        raise RuntimeError("Cannot identify GPUs through nvidia-smi")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [v.strip() for v in visible.split(",")]
        if device_index >= len(tokens):
            raise ValueError("Device is outside CUDA_VISIBLE_DEVICES")
        target = tokens[device_index]
        if target.startswith("GPU-"):
            matches = [row for row in cards if row[1].startswith(target)]
        elif target.isdigit():
            if len(cards) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                raise ValueError("Multiple GPUs: use UUIDs in CUDA_VISIBLE_DEVICES to avoid ambiguous ordering")
            ordered = sorted(cards, key=lambda row: row[2])
            matches = ordered[int(target):int(target) + 1]
        else:
            raise ValueError("Empty, disabled or MIG GPU selection is unsupported by this queue")
    else:
        if len(cards) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise ValueError("Multiple GPUs: set CUDA_VISIBLE_DEVICES to the intended GPU UUID")
        ordered = sorted(cards, key=lambda row: row[2])
        matches = ordered[device_index:device_index + 1]
    if len(matches) != 1:
        raise ValueError("GPU selection does not identify exactly one device")
    return matches[0][1]


def observation(uuid):
    applications = _query("gpu_uuid,pid", applications=True)
    if any(len(row) != 2 for row in applications):
        raise RuntimeError("Cannot parse active CUDA process list")
    pids = [int(row[1]) for row in applications if row[0] == uuid]
    metrics = _query("utilization.gpu,memory.free", gpu=uuid)
    if len(metrics) != 1 or len(metrics[0]) != 2:
        raise RuntimeError("Cannot read GPU load and available memory")
    utilization, free_mib = (int(value) for value in metrics[0])
    return dict(uuid=uuid, compute_pids=pids, utilization_percent=utilization,
                free_memory_mib=free_mib, idle=not pids and utilization <= 5)


def wait_for_idle(device_index, log=None, poll_seconds=30, consecutive=2):
    """Wait for two idle observations. Availability is observed, not reserved."""
    if not 1 <= poll_seconds <= 60 or consecutive < 2:
        raise ValueError("Use 1..60 second polls and at least two observations")
    emit = log or print
    uuid = resolve_gpu(device_index)
    started = time.time()
    streak = 0
    while True:
        sample = observation(uuid)
        streak = streak + 1 if sample["idle"] else 0
        emit("GPU_QUEUE " + str(dict(sample, consecutive_idle=streak)))
        if streak >= consecutive:
            return dict(sample, waited_seconds=time.time() - started,
                        cuda_context_created=False, exclusive_reservation=False)
        time.sleep(poll_seconds)
