"""Isolated bridge from verified SAR/baseline Q sources to original route heads.

Run only in a dedicated worker process. This changes module aliases in memory
for that worker; original files, numerical operators, E0 gates, and Q loaders
remain unchanged. The context restores every alias, including after failure.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path


FORMAT = "SRAC_SAR_SOFT_ROUTE_V1"
ARMS = ("unsplit", "capacity_control", "hard", "soft")
Q_ARMS = ("baseline", "sar_cue")
ROUTE_SEED_OFFSET = 31
ROUTE_MODULE = "bda_benchmark.experiments.damageformer_srac_building_route_v1"
Q_MODULE = "bda_benchmark.experiments.damageformer_srac_q_factorial_v1"
HERE = Path(__file__).resolve().parent
_ACTIVE = False
_Q_SETTINGS = ("steps", "eval_every", "batch", "accum", "workers", "lr",
               "aux_weight", "cue_gain", "amp")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def _route_modules():
    return {name: importlib.import_module(ROUTE_MODULE + "." + name)
            for name in ("core", "train", "evaluate", "report")}


def source_hash():
    """Bind the unmodified route/Q implementations and this external bridge."""
    core = importlib.import_module(ROUTE_MODULE + ".core")
    variants = importlib.import_module(Q_MODULE + ".q_variants")
    if core.source_hash is source_hash:
        raise RuntimeError("Original core source_hash was replaced; recursive binding refused")
    value = hashlib.sha256()
    value.update(FORMAT.encode("ascii"))
    value.update(b"\0original_route\0" + core.source_hash().encode("ascii"))
    value.update(b"\0original_q\0" + variants.q_source_hash().encode("ascii"))
    for path in sorted(HERE.iterdir()):
        if (path.is_file() and path.suffix in (".py", ".sh")
                and not path.name.startswith("test_")):
            value.update(b"\0" + path.name.encode("utf-8") + b"\0")
            value.update(path.read_bytes())
    return value.hexdigest()


def _all_true(value):
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def load_route_source(bundle, q_checkpoint=None, smoke=False):
    """Load a *formal* 2000-step baseline/SAR Q even for a short route smoke.

    ``smoke`` controls the new route experiment only. It never grants permission
    to load an old smoke Q or the untouched E0 Q as the matched baseline.
    """
    del smoke
    if not q_checkpoint:
        raise ValueError("SAR-soft crossing requires a completed baseline or sar_cue last.pth")
    path = Path(q_checkpoint).resolve()
    if path.name != "last.pth" or not path.is_file():
        raise ValueError("Q source must be an existing formal last.pth")
    variants = importlib.import_module(Q_MODULE + ".q_variants")
    evaluation = importlib.import_module(Q_MODULE + ".evaluate_q")
    common = importlib.import_module(
        "bda_benchmark.experiments.damageformer_srac_transformer_v1.common")
    train_front = importlib.import_module(
        "bda_benchmark.experiments.damageformer_srac_transformer_v1.train_front")
    checkpoint_sha = digest(path)
    model, payload = variants.load_q_checkpoint(
        path, bundle["device"], a1_override=bundle["args"].a1_checkpoint,
        allow_smoke=False)
    # Preserve the original full protocol/source/frozen-model checkpoint gate.
    evaluation._checked_checkpoint(payload, bundle, checkpoint_sha, False)
    if (payload.get("arm") not in Q_ARMS or payload.get("smoke") is not False
            or payload.get("completed_steps") != 2000 or payload.get("step") != 2000
            or payload.get("test_accessed") is not False):
        raise ValueError("Only formal matched baseline/sar_cue Q sources are accepted")
    settings = payload.get("arguments", {})
    if (not isinstance(settings, dict) or any(key not in settings for key in _Q_SETTINGS)
            or settings.get("steps") != 2000):
        raise ValueError("Q source has incomplete training configuration")
    for name in ("base", "adapter", "embedding", "region_classifier"):
        if not evaluation._same_base(getattr(bundle["front"], name), getattr(model, name)):
            raise ValueError("Q source changed frozen A1/downstream component: " + name)
    if payload.get("protocol_sha256") != bundle["protocol_sha256"]:
        raise ValueError("Q source uses a different E0 protocol")
    seed = bundle["args"].seed
    if type(seed) is not int or seed < 0:
        raise ValueError("E0 seed must be a nonnegative integer")
    route_seed = seed + ROUTE_SEED_OFFSET
    # Q constructors consume different random counts; reset both branches to
    # the same sequence after load. This changes no numeric-runtime settings.
    common.seed_everything(route_seed)
    model.requires_grad_(False).eval()
    if digest(path) != checkpoint_sha:
        raise ValueError("Q source checkpoint changed while loading")
    bundle["input_paths"]["q_checkpoint"] = path
    bundle["inputs_before"]["q_checkpoint"] = checkpoint_sha
    metadata = dict(
        kind="completed_" + payload["arm"] + "_q", arm=payload["arm"],
        checkpoint=str(path), sha256=checkpoint_sha, steps=2000,
        protocol_sha256=bundle["protocol_sha256"],
        train_source_hash=payload["source_hash"],
        inherited_source_hash=payload["inherited_source_hash"],
        cue_gain=payload["cue_gain"],
        training_hyperparameters={key: settings[key] for key in _Q_SETTINGS},
        route_seed=route_seed,
        initial_readout_fingerprints={
            name: train_front.module_fingerprint(getattr(model, name))
            for name in ("embedding", "region_classifier")},
    )
    return model, metadata


@contextmanager
def adapted_modules():
    """Temporarily bind a new format/source loader in the dedicated worker.

    Only consumer aliases are replaced. Original core and Q modules remain
    intact, so their own source hashes and strict checkpoint gates still mean
    exactly what they meant when the source experiments were produced.
    """
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("Nested route adapters are not supported")
    modules = _route_modules()
    core = modules["core"]
    if tuple(core.ARMS) != ARMS or core.FORMAT != "SRAC_BUILDING_ROUTE_V1":
        raise ValueError("Installed original route schema is incompatible")
    original_core_hash = core.source_hash
    original_core_loader = core.load_route_source
    originals = []
    replacements = {"FORMAT": FORMAT, "source_hash": source_hash,
                    "load_route_source": load_route_source}
    _ACTIVE = True
    try:
        for name in ("train", "evaluate", "report"):
            module = modules[name]
            if module.FORMAT != core.FORMAT:
                raise ValueError("Route consumer was already adapted: " + name)
            for attribute, replacement in replacements.items():
                if hasattr(module, attribute):
                    originals.append((module, attribute, getattr(module, attribute)))
                    setattr(module, attribute, replacement)
        if core.source_hash is not original_core_hash or core.load_route_source is not original_core_loader:
            raise RuntimeError("Original route core was unexpectedly changed")
        yield {name: modules[name] for name in ("train", "evaluate", "report")}
    finally:
        for module, attribute, original in reversed(originals):
            setattr(module, attribute, original)
        _ACTIVE = False


def _check_q_metadata(metadata, protocol_sha):
    if (not isinstance(metadata, dict) or metadata.get("arm") not in Q_ARMS
            or metadata.get("kind") != "completed_" + str(metadata.get("arm")) + "_q"
            or metadata.get("steps") != 2000 or metadata.get("protocol_sha256") != protocol_sha
            or type(metadata.get("route_seed")) is not int):
        raise ValueError("Route fit has invalid matched Q provenance")
    fingerprints = metadata.get("initial_readout_fingerprints")
    if (not isinstance(fingerprints, dict)
            or set(fingerprints) != {"embedding", "region_classifier"}
            or any(not isinstance(value, str) or len(value) != 64 for value in fingerprints.values())):
        raise ValueError("Initial readout fingerprints are missing")
    checkpoint = Path(metadata.get("checkpoint", ""))
    if checkpoint.name != "last.pth" or digest(checkpoint) != metadata.get("sha256"):
        raise ValueError("Original Q checkpoint hash changed since route training")


def _check_saved_audit(root, *, smoke, expected_source, protocol_sha, q_source):
    summary = read_json(root / "summary.json")
    if (summary.get("format") != FORMAT + "_EVAL" or summary.get("complete") is not True
            or summary.get("smoke") is not smoke or summary.get("test_accessed") is not False
            or summary.get("image_count") != (2 if smoke else 450)
            or summary.get("full_validation_evidence") is not (not smoke)
            or summary.get("result_file") != "result.json"
            or summary.get("native_a1_replay_ok") is not True
            or not _all_true(summary.get("immutable"))):
        raise ValueError("Stored route initialization/final audit is incomplete: " + str(root))
    if not smoke and any(summary.get(key) is not True for key in (
            "reproduction_passed", "e0_protocol_comparison_passed", "q_a1_reproduction_passed")):
        raise ValueError("Stored route audit summary lacks a formal reproduction gate: " + str(root))
    path = root / "result.json"
    if digest(path) != summary.get("result_sha256"):
        raise ValueError("Stored route audit result hash changed: " + str(root))
    report = read_json(path)
    if (report.get("format") != FORMAT + "_EVAL" or report.get("smoke") is not smoke
            or report.get("test_accessed") is not False
            or report.get("protocol_sha256") != protocol_sha
            or report.get("source_sha256") != expected_source or report.get("q_source") != q_source
            or report.get("native_a1_replay_ok") is not True
            or not _all_true(report.get("immutable"))
            or report.get("immutable") != summary.get("immutable")):
        raise ValueError("Stored audit provenance/immutability mismatch: " + str(root))
    original_train = importlib.import_module(ROUTE_MODULE + ".train")
    original_train.assert_completed_audit(report, smoke)
    return report


def precheck_fit(fit, protocol, *, smoke=False):
    """Check the complete saved final-step fit before independent evaluation.

    This supplements the original evaluator; it does not replace its model,
    shape, runtime, native-replay, or historical reproduction checks.
    """
    import torch

    if type(smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    root = Path(fit).resolve()
    source_protocol = Path(protocol).resolve()
    if source_protocol.is_dir():
        source_protocol = source_protocol / "protocol.json"
    protocol_sha = digest(source_protocol)
    source = source_hash()
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "manifest.json")
    steps = 2 if smoke else 2000
    if (summary.get("format") != FORMAT or summary.get("complete") is not True
            or summary.get("smoke") is not smoke or summary.get("test_accessed") is not False
            or summary.get("training") is not False
            or summary.get("full_validation_evidence") is not (not smoke)
            or type(summary.get("steps")) is not int or summary["steps"] != steps
            or summary.get("updates_per_arm") != {arm: steps for arm in ARMS}
            or not _all_true(summary.get("immutable"))
            or summary.get("checkpoint") != "heads_last.pth"):
        raise ValueError("Route fit is not a complete matched final-step run")
    for name, record in (("summary", summary), ("manifest", manifest)):
        if (record.get("format") != FORMAT or record.get("protocol_sha256") != protocol_sha
                or record.get("source_sha256") != source
                or record.get("test_accessed") is not False or record.get("smoke") is not smoke):
            raise ValueError("Route " + name + " belongs to a different protocol/code/scope")
    q_source = summary.get("q_source")
    if manifest.get("q_source") != q_source:
        raise ValueError("Fit summary/manifest Q source differs")
    _check_q_metadata(q_source, protocol_sha)
    arguments = manifest.get("arguments")
    if (not isinstance(arguments, dict) or arguments.get("steps") != steps
            or arguments.get("smoke") is not smoke
            or arguments.get("threshold") != summary.get("threshold")
            or arguments.get("soft_floor") != summary.get("soft_floor")
            or arguments.get("lr") != 1e-4 or arguments.get("threshold") != 0.5
            or arguments.get("soft_floor") != 0.0
            or type(arguments.get("eval_every")) is not int
            or arguments["eval_every"] <= 0 or steps % arguments["eval_every"] != 0):
        raise ValueError("Route fit arguments differ from the matched experiment")
    if Path(arguments.get("q_checkpoint", "")).resolve() != Path(q_source["checkpoint"]).resolve():
        raise ValueError("Route arguments and Q provenance point to different checkpoints")
    parameters = summary.get("trainable_parameters")
    if (not isinstance(parameters, dict) or set(parameters) != set(ARMS)
            or any(type(value) is not int or value <= 0 for value in parameters.values())
            or len(set(parameters.values())) != 1 or manifest.get("trainable_parameters") != parameters):
        raise ValueError("Route readouts did not have equal parameter budgets")
    checkpoint = root / "heads_last.pth"
    checkpoint_sha = digest(checkpoint)
    if checkpoint_sha != summary.get("checkpoint_sha256"):
        raise ValueError("Route last checkpoint hash changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Route checkpoint must be a dictionary")
    expected = dict(format=FORMAT, complete=True, completed_steps=steps,
                    protocol_sha256=protocol_sha, source_sha256=source, smoke=smoke,
                    test_accessed=False, q_source=q_source, threshold=summary["threshold"],
                    soft_floor=summary["soft_floor"], arguments=arguments)
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError("Route checkpoint metadata differs: " + key)
    for field in ("heads", "optimizer_states"):
        if not isinstance(state.get(field), dict) or set(state[field]) != set(ARMS):
            raise ValueError("Route checkpoint lacks all four " + field)
    for arm in ARMS:
        head = state["heads"][arm]
        if not isinstance(head, dict) or set(head) != {"embedding", "region_classifier"}:
            raise ValueError("Route checkpoint head structure differs: " + arm)
        for values in head.values():
            if (not isinstance(values, dict) or not values
                    or any(not torch.is_tensor(value) or not bool(torch.isfinite(value).all())
                           for value in values.values())):
                raise ValueError("Invalid/nonfinite route head state: " + arm)
    journal = root / "training.jsonl"
    if digest(journal) != summary.get("training_journal_sha256"):
        raise ValueError("Training journal hash changed")
    stream = hashlib.sha256()
    with journal.open(encoding="utf-8") as reader:
        count = 0
        for count, line in enumerate(reader, 1):
            row = json.loads(line)
            if row.get("step") != count or not isinstance(row.get("image"), str):
                raise ValueError("Route training journal order differs")
            stream.update((str(count) + "\t" + row["image"] + "\n").encode("utf-8"))
    if count != steps or stream.hexdigest() != summary.get("train_stream_sha256"):
        raise ValueError("Route fit does not certify the declared shared image stream")
    for name in ("initial_val", "final_val"):
        report = _check_saved_audit(root / name, smoke=smoke, expected_source=source,
                                    protocol_sha=protocol_sha, q_source=q_source)
        if report.get("initial") is not (name == "initial_val"):
            raise ValueError("Initial/final audit role differs: " + name)
        saved = summary.get("initial" if name == "initial_val" else "final")
        if not isinstance(saved, dict) or any(saved.get(key) != report.get(key) for key in (
                "complete", "full_validation_evidence", "reproduction", "e0_metric_comparison",
                "q_a1_reproduction", "metrics")):
            raise ValueError("Fit summary and saved audit disagree: " + name)
    return dict(fit_root=str(root), checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                summary_sha256=digest(root / "summary.json"),
                manifest_sha256=digest(root / "manifest.json"),
                training_journal_sha256=digest(journal), summary=summary, manifest=manifest)
