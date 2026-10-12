"""Head-only prediction operators and immutable experiment provenance.

P and R are independent copies of the SAME 256-to-4 A1 affine head. T is
an auxiliary 128-to-4 token-head adaptation, not a matched-capacity P/R arm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
    mean_pool, resize_probability, unpool,
)

FORMAT = "SRAC_REGION_HEAD_ADAPT_V1"
TRAIN_ARMS = ("p", "r", "t")
FORMAL_STEPS = 5000
SMOKE_STEPS = 20


def source_hash():
    """Bind executable new files plus directly used, unchanged dependencies."""
    root = Path(__file__).parent
    experiments = root.parent
    files = list(root.glob("*.py")) + list(root.glob("*.sh"))
    files = [p for p in files if not p.name.startswith("test_")]
    for package, name in (
        ("damageformer_srac_stage_diagnostics_v1", "feature_core.py"),
        ("damageformer_srac_stage_diagnostics_v1", "feature_audit.py"),
        ("damageformer_srac_stage_diagnostics_v1", "grouped_pool.py"),
        ("damageformer_srac_stage_diagnostics_v1", "shared.py"),
        ("damageformer_srac_region_test_v1", "core.py"),
        ("damageformer_srac_region_val_formal_v1", "core.py"),
        ("damageformer_srac_fourstage_v2", "losses.py"),
        ("damageformer_srac_joint123_v1", "train.py"),
    ):
        files.append(experiments / package / name)
    checksum = hashlib.sha256()
    for path in sorted(set(files)):
        checksum.update(path.relative_to(experiments).as_posix().encode())
        checksum.update(path.read_bytes())
    # The established Lovasz implementation lives outside experiments.
    loss_file = experiments.parent / "script/standard_ML/train_S5_S8_GPU.py"
    checksum.update(b"script/standard_ML/train_S5_S8_GPU.py")
    checksum.update(loss_file.read_bytes())
    return checksum.hexdigest()


def load_source(protocol, output):
    from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_protocol
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import load_verified_bundle

    path, record = checked_protocol(protocol)
    args = argparse.Namespace(**record["arguments"])
    args.protocol, args.output = str(path.resolve()), str(Path(output).resolve())
    # Always load the full, passed source. A short new experiment does not
    # relabel the original source E0 as smoke.
    args.smoke = False
    bundle = load_verified_bundle(args)
    if (len(bundle["train_names"]) != 2918 or len(bundle["val_names"]) != 450
            or bundle["val_names"] != record["selected_images"]
            or set(bundle["train_names"]) & set(bundle["val_names"])):
        raise ValueError("Expected disjoint E0 Train2918/Val450 lists in their original order")
    bundle["front"].requires_grad_(False).eval()
    for module in bundle["decoders"].values():
        module.requires_grad_(False).eval()
    bundle["metric"].requires_grad_(False).eval()
    bundle["args"] = args
    return bundle


def make_heads(front):
    parent = front.base.clf_2
    if (not isinstance(parent, nn.Conv2d) or parent.in_channels != 256
            or parent.out_channels != 4 or parent.kernel_size != (1, 1)
            or parent.stride != (1, 1) or parent.padding != (0, 0)
            or parent.groups != 1):
        raise ValueError("P/R require the original affine 256-to-4 A1 1x1 head")
    region = front.region_classifier
    if (not isinstance(region, nn.Linear) or region.in_features != 128
            or region.out_features != 4):
        raise ValueError("T diagnostic requires the original 128-to-4 region head")
    heads = nn.ModuleDict({"p": copy.deepcopy(parent), "r": copy.deepcopy(parent),
                           "t": copy.deepcopy(region)})
    heads.requires_grad_(True)
    if any(p.dtype != torch.float32 for p in heads.parameters()):
        raise ValueError("This head-fitting protocol requires FP32 heads")
    return heads


def initial_parameter_checks(front, heads):
    def same(a, b):
        x, y = a.state_dict(), b.state_dict()
        return set(x) == set(y) and all(torch.equal(x[k], y[k]) for k in x)
    result = dict(p_matches_a1=same(heads["p"], front.base.clf_2),
                  r_matches_a1=same(heads["r"], front.base.clf_2),
                  p_r_equal=same(heads["p"], heads["r"]),
                  t_matches_region=same(heads["t"], front.region_classifier))
    result["passed"] = all(result.values())
    return result


@torch.no_grad()
def prepare_inputs(feature, result):
    required = ("q", "ids", "count", "shape", "tokens")
    if any(k not in result for k in required):
        raise ValueError("Incomplete frozen front output")
    feature = feature.detach().float()
    q, ids = result["q"].detach(), result["ids"].detach()
    shape = tuple(result["shape"])
    if (feature.ndim != 4 or feature.shape[1] != 256
            or tuple(feature.shape[-2:]) != shape):
        raise ValueError("Expected frozen 256-dimensional F on the original Q grid")
    mean, mass = mean_pool(feature, q, ids, result["count"])
    tokens = result["tokens"].detach().float()
    if tokens.shape != (feature.shape[0], result["count"], 128):
        raise ValueError("Original T tokens must have 128 channels on the same Q")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("Nonfinite original region tokens")
    return dict(feature=feature, mean_tokens=mean.detach(), tokens=tokens,
                mass=mass.detach(), q=q, ids=ids, count=result["count"], shape=shape)


def head_probability(arm, head, inputs, output_size):
    """Differentiable in head parameters only; the feature/Q input is frozen."""
    if arm == "p":
        grid = head(inputs["feature"]).float().softmax(1)
    elif arm in ("r", "t"):
        if arm == "r":
            logits = F.linear(inputs["mean_tokens"], head.weight[:, :, 0, 0], head.bias)
        else:
            logits = head(inputs["tokens"])
        grid = unpool(logits.float().softmax(-1), inputs["q"], inputs["ids"], inputs["shape"])
    else:
        raise ValueError("Unknown head arm: " + str(arm))
    return resize_probability(grid, output_size)


def assert_formal_report(report, smoke=False):
    checks = report.get("immutable")
    if (report.get("complete") is not True or not isinstance(checks, dict)
            or not checks or not all(v is True for v in checks.values())
            or report.get("native_a1_replay_ok") is not True):
        raise ValueError("Incomplete evaluation or failed immutable/native A1 checks")
    if smoke:
        if not (0 < report.get("image_count", 0) < 450) or report.get("full_validation_evidence") is not False:
            raise ValueError("Smoke must remain an explicitly nonformal Val prefix")
    elif (report.get("image_count") != 450 or report.get("full_validation_evidence") is not True
          or report.get("reproduction", {}).get("passed") is not True):
        raise ValueError("Formal fitting requires full Val450 and the unchanged 1e-5 reproduction gate")


def load_fit(fit_dir, bundle, allow_smoke=False):
    """Load only a completed, hash-bound last-step head-fitting experiment."""
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, safe_child
    root = Path(fit_dir)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("format") != FORMAT or summary.get("complete") is not True
            or summary.get("test_accessed") is not False
            or type(summary.get("smoke")) is not bool
            or (summary["smoke"] and not allow_smoke)):
        raise ValueError("Fit is incomplete, non-Train-only, or smoke was supplied for formal evaluation")
    if (summary.get("protocol_sha256") != bundle["protocol_sha256"]
            or summary.get("source_sha256") != source_hash()):
        raise ValueError("Fit protocol or executable source differs from current evaluation")
    steps = SMOKE_STEPS if summary["smoke"] else FORMAL_STEPS
    if (summary.get("steps") != steps
            or summary.get("updates_per_arm") != {arm: steps for arm in TRAIN_ARMS}):
        raise ValueError("Fit does not contain the preregistered equal update budget")
    assert_formal_report(summary["initial"], smoke=summary["smoke"])
    assert_formal_report(summary["final"], smoke=summary["smoke"])
    if summary.get("checkpoint_file") != "heads_last.pth":
        raise ValueError("Only equal-budget last-step heads are accepted")
    path = safe_child(root, summary["checkpoint_file"])
    if digest(path) != summary.get("checkpoint_sha256"):
        raise ValueError("Head checkpoint hash differs from completed fit")
    state = torch.load(path, map_location=bundle["device"], weights_only=False)
    for key, expected in dict(format=FORMAT, complete=True, completed_steps=steps,
                              smoke=summary["smoke"], protocol_sha256=bundle["protocol_sha256"],
                              source_sha256=source_hash()).items():
        if state.get(key) != expected:
            raise ValueError("Head checkpoint provenance differs: " + key)
    heads = make_heads(bundle["front"])
    heads.load_state_dict(state["heads"], strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in heads.parameters()):
        raise ValueError("Nonfinite fitted head parameters")
    metadata = dict(fit_dir=str(root.resolve()), summary_sha256=digest(summary_path),
                    checkpoint_sha256=digest(path), smoke=summary["smoke"], steps=steps)
    return heads, metadata
