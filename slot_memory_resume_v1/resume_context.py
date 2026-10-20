"""Read-only preflight for formal continuations of the recorded six-arm run."""
from __future__ import annotations

import contextlib
import errno
import fcntl
import os
from pathlib import Path
import random
import stat

from inspect_failure import (
    DEFAULT_RUN, checkpoint_info, read_journal, read_object, sha256, validate_identity,
)
from probe import read_bridge


FORMAT = "BRIGHT_SLOT_MEMORY_RESUME_V1"
CHECKPOINT_FORMAT = "BRIGHT_SLOT_MEMORY_RESUME_CHECKPOINT_V1"
ARMS = ("e", "f", "b", "d", "r", "l")
ORIGINAL_STEP = 1000


@contextlib.contextmanager
def inactive_parent(root):
    """Hold an existing parent's worker lock without changing file contents.

    O_RDWR is needed for exclusive flock on NFS. There is no create, truncate,
    or write operation. Call this after prepare(), not around prepare(), because
    separate flock descriptors in the same process can conflict with each other.
    """
    if root is None:
        yield None
        return
    root = Path(root).expanduser().resolve()
    lock = root / "worker.lock"
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock, flags)
    except OSError as exc:
        raise ValueError("Cannot open the existing resume-parent worker lock: " + str(lock)) from exc
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Resume-parent worker.lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError("Resume parent is still running or locked: " + str(root)) from exc
            raise
        acquired = True
        yield root
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _check_splits(plan, manifest, seed):
    train, val = manifest.get("train_names"), manifest.get("val_names")
    for label, names, count in (("Train", train, 2918), ("Val", val, 450)):
        if (not isinstance(names, list) or len(names) != count
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != count):
            raise ValueError("Require the complete unique %s%d split" % (label, count))
    if set(train) & set(val):
        raise ValueError("Train and Val must be disjoint")
    endpoint = plan.get("endpoint_step")
    if type(endpoint) is not int or not ORIGINAL_STEP < endpoint <= len(train):
        raise ValueError("Resume endpoint must follow step1000 within the first Train epoch")
    count = plan.get("fixed_train_count")
    if type(count) is not int or not 1 <= count <= len(train):
        raise ValueError("Invalid declared fixed Train sample count")
    expected = sorted(random.Random(seed + 191).sample(train, count))
    if manifest.get("fixed_train_names") != expected:
        raise ValueError("Fixed Train names differ from the declared seed+191 sample")
    return endpoint


def _verified_original(train):
    choices = [
        item for item in checkpoint_info(train / "checkpoints", ARMS)
        if item["verified"] and item["step"] == ORIGINAL_STEP
    ]
    if len(choices) != 1:
        raise ValueError("Require the byte-verified original checkpoint1000 and sidecar")
    item = choices[0]
    return dict(kind="original", path=train / "checkpoints" / item["name"],
                step=ORIGINAL_STEP, sha256=item["sha256"])


def _resume_parent(root, lineage, original_manifest):
    """Select metadata-verified bytes; runtime still verifies payload and journal."""
    manifest_path, summary_path = root / "manifest.json", root / "summary.json"
    manifest, summary = read_object(manifest_path), read_object(summary_path)
    if manifest.get("format") != FORMAT:
        raise ValueError("Parent must be a formal resume output, never a diagnostic run")
    if (manifest.get("lineage") != lineage
            or manifest.get("original_manifest") != original_manifest):
        raise ValueError("Resume-parent lineage or original manifest differs")
    if summary.get("complete") is True:
        raise ValueError("Resume parent is already complete")
    journal = root / "train.jsonl"
    if not journal.is_file():
        raise ValueError("Resume-parent training journal is missing")
    choices = []
    for item in checkpoint_info(root / "checkpoints", ARMS):
        if not item["verified"] or not ORIGINAL_STEP <= item["step"] <= lineage["endpoint_step"]:
            continue
        path = root / "checkpoints" / item["name"]
        metadata = read_object(path.with_suffix(".pt.json"))
        if (metadata.get("format") != CHECKPOINT_FORMAT
                or metadata.get("lineage") != lineage
                or metadata.get("updates_per_arm") != {arm: item["step"] for arm in ARMS}):
            continue
        choices.append(item)
    if not choices:
        raise ValueError("No verified formal resume checkpoint matches the lineage")
    chosen = max(choices, key=lambda item: item["step"])
    path = root / "checkpoints" / chosen["name"]
    parent = dict(kind="resume", path=path, step=chosen["step"], sha256=chosen["sha256"])
    evidence = (manifest_path, summary_path, journal, path, path.with_suffix(".pt.json"))
    return parent, journal, evidence


def prepare(args):
    """Validate immutable inputs and select the requested continuation parent."""
    root = Path(getattr(args, "original_run", DEFAULT_RUN)).expanduser().resolve()
    seed = getattr(args, "seed", 3407)
    every = getattr(args, "checkpoint_every", 50)
    if type(seed) is not int or type(every) is not int or every < 1:
        raise ValueError("Seed must be an integer and checkpoint_every a positive integer")
    train = root / ("seed_%s/train" % seed)
    original_paths = [
        root / "plan.json", train / "manifest.json",
        train / "summary.json", train / "train.jsonl",
    ]
    evidence = {str(path): sha256(path) for path in original_paths}
    plan, manifest = validate_identity(root, seed)
    if tuple(plan["arms"]) != ARMS:
        raise ValueError("Require the original six-arm order e,f,b,d,r,l")
    endpoint = _check_splits(plan, manifest, seed)
    status = read_object(train / "summary.json")
    journal = read_journal(train / "train.jsonl", ARMS)
    last = journal["contiguous_prefix_step"]
    if (status.get("phase") != "failed" or status.get("complete") is not False
            or journal["issue_count"] or journal["trailing_incomplete"]
            or journal["stable_during_read"] is not True
            or not ORIGINAL_STEP <= last <= endpoint
            or status.get("updates_per_arm") != {arm: last for arm in ARMS}):
        raise ValueError("Original run must have a stable complete failed six-arm journal")
    original_parent = _verified_original(train)
    lineage = dict(
        original_run=str(root), seed=seed,
        original_manifest_sha256=evidence[str(train / "manifest.json")],
        original_journal_sha256=evidence[str(train / "train.jsonl")],
        original_parent_step=ORIGINAL_STEP,
        original_parent_checkpoint_sha256=original_parent["sha256"],
        endpoint_step=endpoint,
    )
    source = plan["source"]
    addon_arg = getattr(args, "addon_dir", None)
    addon = (
        Path(addon_arg) if addon_arg else
        Path(source["config"]["repo"]) / "New/slot_memory_ablation_v1"
    ).expanduser().resolve()
    bridge = read_bridge(addon)
    if bridge.check_package(addon) != plan["addon_release_sha256"]:
        raise ValueError("Original addon differs from the failed run")
    bridge.verify_ledgers(source)
    evidence[str(original_parent["path"])] = original_parent["sha256"]
    evidence[str(original_parent["path"].with_suffix(".pt.json"))] = sha256(
        original_parent["path"].with_suffix(".pt.json"))
    parent, parent_journal = original_parent, train / "train.jsonl"
    parent_arg = getattr(args, "parent_root", None)
    parent_root = Path(parent_arg).expanduser().resolve() if parent_arg else None
    if parent_root is not None:
        with inactive_parent(parent_root):
            parent, parent_journal, parent_paths = _resume_parent(parent_root, lineage, manifest)
            evidence.update({str(path): sha256(path) for path in parent_paths})
            if evidence[str(parent["path"])] != parent["sha256"]:
                raise ValueError("Resume-parent checkpoint changed during preflight")
    for path, digest in evidence.items():
        if sha256(path) != digest:
            raise ValueError("Input evidence changed during preflight: " + path)
    return dict(
        root=root, train=train, plan=plan, manifest=manifest, addon=addon, bridge=bridge,
        seed=seed, lineage=lineage, parent=parent, parent_journal=parent_journal,
        parent_root=parent_root, evidence_files=evidence, checkpoint_every=every,
    )


def protected_paths(context):
    """Directories that a new output must never occupy or be placed inside."""
    source = context["plan"]["source"]
    config = source["config"]
    paths = [
        context["root"], context["addon"], source["package"], source["source_run"],
        config["repo"], config["bank_root"], config["values"],
    ]
    if config.get("membership_root"):
        paths.append(config["membership_root"])
    if context.get("parent_root") is not None:
        paths.append(context["parent_root"])
    return list(dict.fromkeys(Path(path).expanduser().resolve() for path in paths))
