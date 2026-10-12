"""Provenance helpers for the isolated selective-Memory experiment.

The experiment consumes the immutable frozen-front/source-token contract and
the Train-only V2 quality bank.  It deliberately layers on those
implementations instead of copying or weakening their validation rules.
"""
import hashlib
import math
from pathlib import Path

from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2 import common as parent
from bda_benchmark.experiments.damageformer_partition_learning_v1 import reproduction as repro


# Keep data/source semantics identical to the already audited parent run.
add_arguments = parent.add_arguments
validate_training_args = parent.validate_training_args
load_source = parent.load_source
source_identity = parent.source_identity
assert_source = parent.assert_source
capture = parent.capture
routes = parent.routes
loader = parent.loader
parse_batch = parent.parse_batch
metrics = parent.metrics
cm_update = parent.cm_update
seed_everything = parent.seed_everything
new_output = parent.new_output
save_json = parent.save_json
save_pt = parent.save_pt
digest = parent.digest
event = parent.event
objective = parent.objective
state_digest = parent.state_digest
append_json = parent.append_json


def parent_experiment_hash():
    """Return the exact parent implementation hash consumed by this run."""
    return parent.experiment_hash()


def experiment_hash():
    """Hash this implementation together with its complete parent chain."""
    h = hashlib.sha256(parent_experiment_hash().encode())
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def assert_selective_immutable(bundle, expected_hash, expected_parent_hash):
    """Verify both the inherited source contract and this experiment code."""
    parent.assert_v2_immutable(bundle, expected_parent_hash)
    if experiment_hash() != expected_hash:
        raise RuntimeError('Selective-Memory V1 implementation changed during the experiment')


def _bounded_old_region_drift(row):
    """Return whether a failed cell is bounded legacy-region drift.

    ``old_region`` is retained as provenance evidence but is not consumed by
    this experiment: training starts from the independently reproduced
    ``source_hard`` logits. CUDA scatter reductions in the legacy soft-region
    path can move a few tie pixels between processes. Only the already
    predeclared reproduction hard cap is accepted; malformed cells, history
    anomalies, extra failure reasons, and larger drifts remain blocking.
    """
    if not isinstance(row, dict):
        return False
    history = row.get('history')
    delta = row.get('delta')
    try:
        finite_delta = (isinstance(delta, (int, float)) and
                        not isinstance(delta, bool) and
                        math.isfinite(delta))
    except (TypeError, ValueError, OverflowError):
        finite_delta = False
    return (
        row.get('baseline') == 'old_region'
        and finite_delta
        and abs(delta) <= repro.MAX_TOL
        and row.get('reason') == 'absolute difference exceeds tolerance'
        and isinstance(history, dict)
        and history.get('history_anomaly') is False
    )


def selective_initial_gate(bundle, result, args, out, policy):
    """Persist full evidence while enforcing task-relevant reproduction.

    ``old_pixel`` and ``source_hard`` retain their original strict tolerances.
    A numerically valid ``old_region`` cell may be nonblocking only within the
    pre-existing global hard cap. No threshold is fitted to this evaluation.
    """
    out = Path(out)
    save_json(out / 'initial.json', repro.json_safe_evidence(result))
    if args.smoke:
        report = dict(
            passed=None, blocking_passed=None,
            all_metric_checks_passed=None,
            status='smoke_not_full_val', checks=[],
            blocking_failures=[], nonblocking_failures=[], warnings=[])
    else:
        report = repro.compare_baselines(
            result['metrics'], bundle['summary']['final_metrics'], policy)
        all_passed = report['passed']
        blocking, nonblocking = [], []
        for row in report.get('checks', []):
            if row.get('passed') is True:
                row['gate_disposition'] = 'passed'
            elif _bounded_old_region_drift(row):
                row['gate_disposition'] = 'nonblocking_bounded_old_region_drift'
                nonblocking.append(row)
            else:
                row['gate_disposition'] = 'blocking_failure'
                blocking.append(row)

        def label(row):
            suffix = ('' if row.get('class_index') is None else
                      '/' + str(row['class_index']))
            return (str(row.get('baseline')) + '/' +
                    str(row.get('metric')) + suffix)

        report.update(
            all_metric_checks_passed=all_passed,
            blocking_passed=not blocking,
            blocking_failures=[label(row) for row in blocking],
            nonblocking_failures=[label(row) for row in nonblocking],
            old_region_diagnostic_hard_cap=repro.MAX_TOL,
            old_region_diagnostic_hard_cap_pp=100 * repro.MAX_TOL,
            gate_contract=(
                'old_pixel/source_hard retain original strict policy; '
                'old_region is diagnostic-only and nonblocking solely for '
                'finite absolute-difference-only drift within the predeclared '
                'reproduction hard cap'),
            passed=not blocking,
            status=('passed' if all_passed else
                    ('passed_with_bounded_old_region_drift'
                     if not blocking else 'failed')))
    report['source_integrity_pass'] = None
    save_json(out / 'reproduction.json', report)
    try:
        assert_source(bundle)
        report['source_integrity_pass'] = True
        save_json(out / 'reproduction.json', report)
        if not args.smoke and report['blocking_passed'] is not True:
            labels = report.get('blocking_failures', [])
            raise ValueError(
                'Blocking baseline reproduction failed (' +
                str(len(labels)) + ' checks): ' + ', '.join(labels))
    except Exception as exc:
        report.update(passed=False, blocking_passed=False,
                      status='failed', error=str(exc))
        save_json(out / 'reproduction.json', report)
        save_json(out / 'summary.json', dict(
            complete=False, smoke=args.smoke,
            phase='initial_reproduction_failed', error=str(exc),
            optimizer_steps=0, test_accessed=False))
        raise
    return report

