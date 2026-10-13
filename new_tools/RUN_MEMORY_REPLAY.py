#!/usr/bin/env python3
"""Offline, path-independent launcher for the unchanged Memory V2 replay probe.

Default: print a verified plan without a model forward or an output directory.
Explicit --run: delegate to the same-package, hash-pinned original probe.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True

PROBE_NAME = "AUDIT_MEMORY_REPLAY_V2.py"
PROBE_SHA256 = "5c36a67f41c9c671fb1bd887d6031f2a4877ac1c788e0138195bb364a9064545"
EVIDENCE_SHA256 = "d05ca254bbdff51e0c7cc08c6dbce1aec1fac8b55a4dadbd8f6d0f7506306213"
REPO = Path("/root/shared-nvme/BRIGHT-master")
CHECKPOINTS = Path("/root/shared-nvme/BRIGHT-checkpoints")
DEFAULT_PROTOCOL = Path("SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json")
INVENTORY = ("BRIGHT-memory-debug_esfI2YOk", "BRIGHT-prt-sequence_GgXF1Hmw",
             "BRIGHT-prt-debug_s67wQ0Di", "BRIGHT-prt-debug_EDXqo29z",
             "BRIGHT-region-night-unpack_HB4jnNVw")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return data


def unique_paths(paths):
    return sorted({Path(path).resolve() for path in paths}, key=str)


def _summary_evidence(summary_path, explicit=None):
    record = read_json(summary_path)
    saved = record.get("last_replay_evidence")
    if not isinstance(saved, dict) or saved.get("sha256") != EVIDENCE_SHA256:
        return None
    if record.get("format") != "SRAC_MEMORY_PAIRED_KEYS_V2":
        raise ValueError("Matching SHA appeared in a non-V2 summary: " + str(summary_path))
    relative = saved.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Invalid relative replay evidence path: " + str(summary_path))
    stage = summary_path.parent.resolve()
    evidence = (stage / relative).resolve()
    if stage not in evidence.parents:
        raise ValueError("Evidence path escapes its stage directory: " + str(summary_path))
    if explicit is not None and evidence != explicit:
        raise ValueError("Explicit --evidence does not match its keys_smoke summary: " + str(explicit))
    if not evidence.is_file() or digest(evidence) != EVIDENCE_SHA256:
        raise ValueError("Saved replay evidence is absent or has changed: " + str(evidence))
    return evidence


def find_evidence(roots, explicit=None):
    """Find the exact failed artifact, never a newest-run approximation."""
    if explicit is not None:
        evidence = Path(explicit).resolve()
        stages = [parent for parent in evidence.parents if parent.name == "keys_smoke"]
        if not stages:
            raise ValueError("Explicit --evidence must belong to a keys_smoke stage: " + str(evidence))
        summary_path = stages[0] / "summary.json"
        if not summary_path.is_file() or _summary_evidence(summary_path, evidence) is None:
            raise ValueError("Explicit --evidence lacks a matching keys_smoke summary: " + str(evidence))
        return evidence, [summary_path.resolve()]
    matches = {}
    for root in unique_paths(roots):
        if not root.is_dir():
            continue
        for summary_path in root.rglob("summary.json"):
            if summary_path.parent.name != "keys_smoke":
                continue
            try:
                evidence = _summary_evidence(summary_path)
            except (OSError, ValueError):
                # Malformed unrelated summaries are not selection candidates;
                # a record naming the required SHA must not be silently ignored.
                try:
                    saved = read_json(summary_path).get("last_replay_evidence")
                except (OSError, ValueError):
                    continue
                if isinstance(saved, dict) and saved.get("sha256") == EVIDENCE_SHA256:
                    raise
                continue
            if evidence is None:
                continue
            matches.setdefault(evidence, set()).add(summary_path.resolve())
    if not matches:
        raise ValueError("No keys_smoke summary and evidence match the required SHA256 under: "
                         + ", ".join(map(str, unique_paths(roots))))
    if len(matches) != 1:
        raise ValueError("Multiple distinct matching evidence files; refusing to select a run: "
                         + ", ".join(map(str, sorted(matches, key=str))))
    evidence, summaries = next(iter(matches.items()))
    return evidence, sorted(summaries, key=str)


def load_evidence_identity(path):
    # Only deserialize the fixed, already verified artifact, onto CPU.
    import torch
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != "SRAC_MEMORY_KEY_REPLAY_EVIDENCE_V2":
        raise ValueError("Unexpected failed evidence format")
    for key in ("protocol_sha256", "source_sha256"):
        if not isinstance(data.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise ValueError("Evidence lacks a valid " + key)
    return data


def find_protocol(checkpoints, tools_root, protocol, expected_sha):
    if protocol is not None:
        path = Path(protocol).resolve()
        if path.is_dir():
            path /= "protocol.json"
        if not path.is_file() or digest(path) != expected_sha:
            raise ValueError("Explicit --protocol does not match the failed evidence: " + str(path))
        return path
    historical = checkpoints / DEFAULT_PROTOCOL
    if historical.is_file() and digest(historical) == expected_sha:
        return historical.resolve()
    matches = set()
    for root in unique_paths((checkpoints, tools_root)):
        if root.is_dir():
            for path in root.rglob("protocol.json"):
                if path.is_file() and digest(path) == expected_sha:
                    matches.add(path.resolve())
    if len(matches) != 1:
        raise ValueError("Expected exactly one protocol matching evidence SHA256; found "
                         + str(len(matches)) + ": " + ", ".join(map(str, sorted(matches, key=str)))
                         + ". Pass --protocol with the intended exact original file.")
    return next(iter(matches))


def check_v2_source(repo, expected_sha):
    path = (repo / "bda_benchmark/experiments/damageformer_srac_memory_key_compare_v2/shared.py")
    if not path.is_file():
        raise ValueError("Installed V2 source is missing in --repo: " + str(path))
    spec = importlib.util.spec_from_file_location("_memory_replay_launcher_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.source_hash()
    if actual != expected_sha:
        raise ValueError("Installed V2 source differs from the failed evidence; launcher does not modify it")
    return actual


def make_plan(cli):
    repo = Path(cli.repo).resolve()
    tools_root = Path(cli.tools_root).resolve() if cli.tools_root else repo / "New"
    checkpoints = Path(cli.checkpoints).resolve()
    probe = Path(__file__).resolve().parent / PROBE_NAME
    if not probe.is_file() or digest(probe) != PROBE_SHA256:
        raise ValueError("Place the unmodified hash-pinned " + PROBE_NAME + " beside this launcher: " + str(probe))
    if not (repo / "bda_benchmark").is_dir():
        raise ValueError("--repo must identify the original BRIGHT repository: " + str(repo))
    evidence, summaries = find_evidence((checkpoints, tools_root), cli.evidence)
    identity = load_evidence_identity(evidence)
    protocol = find_protocol(checkpoints, tools_root, cli.protocol, identity["protocol_sha256"])
    installed_source = check_v2_source(repo, identity["source_sha256"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = (Path(cli.output).resolve() if cli.output else
              checkpoints / "SRAC_MEMORY_REPLAY_AUDIT_V2" / ("run_" + stamp + "_" + str(os.getpid())))
    if output.exists():
        raise ValueError("Output must be a new directory; refusing an existing path: " + str(output))
    command = [sys.executable, str(probe), "--repo", str(repo), "--protocol", str(protocol),
               "--evidence", str(evidence), "--evidence-sha256", EVIDENCE_SHA256,
               "--output", str(output)]
    if cli.cpu_only:
        command.append("--cpu-only")
    plan = dict(format="SRAC_MEMORY_NEW_LAYOUT_PLAN_V1", action="run" if cli.run else "plan_only",
                offline=True, repo=str(repo), tools_root=str(tools_root), checkpoints=str(checkpoints),
                launcher=str(Path(__file__).resolve()), probe=str(probe), probe_sha256=PROBE_SHA256,
                evidence=str(evidence), evidence_sha256=EVIDENCE_SHA256,
                evidence_selection="explicit" if cli.evidence else "unique_sha256_match",
                matching_summaries=list(map(str, summaries)), image=identity.get("image"),
                protocol=str(protocol), protocol_sha256=identity["protocol_sha256"],
                installed_v2_source_sha256=installed_source, output=str(output),
                training=False, test_accessed=False, gpu_ran=False,
                requested_gpu_forward=bool(cli.run and not cli.cpu_only),
                model_forward_count=0,
                requested_model_forward_count=0 if cli.cpu_only or not cli.run else 6,
                cpu_only=cli.cpu_only, command=command,
                known_tool_directories=[dict(path=str(tools_root / name), exists=(tools_root / name).is_dir())
                                        for name in INVENTORY],
                artifact_policy="Original protocol contents and their recorded input paths are unchanged; no extraction, relocation, or PRT execution.")
    return plan, command


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true", help="Verify and print paths; default, no model forward")
    mode.add_argument("--run", action="store_true", help="Run the original read-only probe after the same checks")
    p.add_argument("--cpu-only", action="store_true", help="With --run: archive identity checks only, no GPU")
    p.add_argument("--repo", default=str(REPO), help="Existing BRIGHT repository, not an unpacked helper directory")
    p.add_argument("--tools-root", help="Helper inventory/search root; defaults to --repo/New")
    p.add_argument("--checkpoints", default=str(CHECKPOINTS), help="Existing artifacts root; defaults remain outside New")
    p.add_argument("--protocol", help="Exact original protocol.json or its containing directory; SHA256 must match evidence")
    p.add_argument("--evidence", help="Explicit failed .pt if there are copied results; must match its keys_smoke summary and the fixed SHA256")
    p.add_argument("--output", help="Fresh report directory; never overwrite an existing directory")
    return p


def main(argv=None):
    cli = parser().parse_args(argv)
    plan, command = make_plan(cli)
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if not cli.run:
        return 0
    return subprocess.call(command, cwd=str(Path(cli.repo).resolve()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as exc:
        print("Memory replay launcher stopped: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
