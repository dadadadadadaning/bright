"""Recheck one failed INITIAL Val image without training or modifying evidence.

This is a fresh diagnostic execution with initialized P/R/T heads. It retains
the E0 numerical/source contract, but explicitly records that the diagnostic
package can differ from the old failed executable. It is never formal Val450.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failed", required=True,
                   help="Old fit/initial_val directory or its failed summary.json")
    p.add_argument("--protocol", required=True, help="The identical passed E0 protocol")
    p.add_argument("--output", required=True,
                   help="Fresh diagnostic directory outside the old fit directory")
    return p


def _object(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return record


def checked_failure(failed, protocol, output):
    """Validate the failed artifact and its E0 binding before loading models."""
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
    from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import checked_protocol

    summary_path = Path(failed).resolve()
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    if summary_path.name != "summary.json" or summary_path.parent.name != "initial_val":
        raise ValueError("Recheck requires an original fit/initial_val/summary.json")
    summary = _object(summary_path)
    if (summary.get("format") != "SRAC_REGION_HEAD_ADAPT_EVAL_V1"
            or summary.get("complete") is not False
            or summary.get("full_validation_evidence") is not False
            or summary.get("initial") is not True):
        raise ValueError("Requires a failed INITIAL head evaluation, not completed/fitted results")
    image = summary.get("image")
    error = summary.get("error")
    if (not isinstance(image, str) or not image.strip()
            or not isinstance(error, str)
            or not ("Probabilities are not normalized" in error
                    or "DiagnosticProbabilityError" in error)):
        raise ValueError("Failed record needs an image and a probability-normalization diagnostic error")

    fit_dir = summary_path.parent.parent
    manifest_path = fit_dir / "manifest.json"
    manifest = _object(manifest_path)
    if manifest.get("format") != "SRAC_REGION_HEAD_ADAPT_V1":
        raise ValueError("Unexpected fitting manifest format")
    old_source = manifest.get("source_sha256")
    if not isinstance(old_source, str) or len(old_source) != 64:
        raise ValueError("Old manifest has no valid executable source SHA256")
    if summary.get("source_sha256") not in (None, old_source):
        raise ValueError("Failed summary and fitting manifest executable hashes disagree")
    protocol_path, e0 = checked_protocol(protocol)
    protocol_path = protocol_path.resolve()
    protocol_sha = digest(protocol_path)
    if manifest.get("protocol_sha256") != protocol_sha:
        raise ValueError("Provided E0 SHA256 differs from the failed fitting manifest")
    if summary.get("protocol_sha256") not in (None, protocol_sha):
        raise ValueError("Failed summary and provided E0 hashes disagree")
    if image not in e0["selected_images"]:
        raise ValueError("Failed image is not in the passed E0 Val450 list")

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing diagnostic output: " + str(output))
    if output == fit_dir or fit_dir in output.parents:
        raise ValueError("Diagnostic output must be outside the old fitting evidence directory")
    return dict(summary_path=summary_path, manifest_path=manifest_path,
                protocol_path=protocol_path, protocol_sha256=protocol_sha,
                summary_sha256=digest(summary_path), manifest_sha256=digest(manifest_path),
                old_source_sha256=old_source, image=image, old_error=error,
                old_completed_images=summary.get("image_count"), output=output)


def run(cli):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest, write_json
    from .core import load_source, make_heads, source_hash
    from .evaluate import evaluate_heads

    origin = checked_failure(cli.failed, cli.protocol, cli.output)
    bundle = load_source(origin["protocol_path"], origin["output"])
    if bundle["protocol_sha256"] != origin["protocol_sha256"]:
        raise ValueError("E0 changed between diagnostic validation and model loading")
    if origin["image"] not in bundle["val_names"]:
        raise ValueError("Failed image is not in the loaded source Val450 list")
    current_source = source_hash()
    metadata = dict(
        format="SRAC_REGION_HEAD_PROBABILITY_RECHECK_V1", diagnostic_only=True,
        full_validation_evidence=False, training=False, test_accessed=False,
        image=origin["image"], failed_summary=str(origin["summary_path"]),
        failed_summary_sha256=origin["summary_sha256"],
        failed_manifest=str(origin["manifest_path"]),
        failed_manifest_sha256=origin["manifest_sha256"],
        protocol_sha256=origin["protocol_sha256"],
        old_source_sha256=origin["old_source_sha256"],
        current_source_sha256=current_source,
        diagnostic_source_changed=origin["old_source_sha256"] != current_source,
        old_error=origin["old_error"], old_completed_images=origin["old_completed_images"],
        scope="Fresh one-image initialized-head diagnostic; not an exact replay of prior process state or formal reproduction",
        numerical_rule="Inherited legacy/native E0 settings; no probability renormalization or threshold relaxation")
    bundle["probability_recheck"] = metadata
    for stem in ("summary", "manifest"):
        key = "probability_recheck_failed_" + stem
        bundle["input_paths"][key] = origin[stem + "_path"]
        bundle["inputs_before"][key] = origin[stem + "_sha256"]
    heads = make_heads(bundle["front"]).to(bundle["device"])
    try:
        report = evaluate_heads(bundle, heads, output=origin["output"],
                                limit=0, image=origin["image"], initial=True)
        if (report.get("full_validation_evidence") is not False
                or report.get("smoke") is not True or report.get("image_count") != 1):
            raise ValueError("A one-image recheck must remain explicitly nonformal")
        return report
    finally:
        unchanged = {
            stem: digest(origin[stem + "_path"]) == origin[stem + "_sha256"]
            for stem in ("summary", "manifest")}
        metadata["old_evidence_unchanged"] = unchanged
        if origin["output"].is_dir():
            write_json(origin["output"] / "recheck_origin.json", metadata)
        if not all(unchanged.values()):
            raise RuntimeError("Old failed evidence changed during one-image diagnostic")


if __name__ == "__main__":
    run(parser().parse_args())
