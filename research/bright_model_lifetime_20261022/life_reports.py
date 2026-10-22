"""Diagnostic prefix acceptance; never promote partial Test metrics."""
import copy
import numpy as np
from evaluate_checkpoint import write_json, ARMS, REQUIRED_CHECKS
from grid_probe import read_json, sha256
from sweep_scoring import rebuild
from life_context import COUNT, STEP, SELECTION
from life_observers import validate_evidence


def finish_report(path, report, *, output, names, checks, provenance, full, evidence):
    if (full or len(names) != COUNT or len(set(names)) != COUNT
            or report.get('step') != STEP or report.get('image_count') != COUNT
            or report.get('checkpoint_selection') != SELECTION
            or report.get('split') != 'test' or report.get('optimizer_updates') != 0
            or report.get('arms') != list(ARMS) or read_json(path) != report):
        raise ValueError('Wrong diagnostic prefix/identity')
    if set(checks) != set(REQUIRED_CHECKS) or any(v is not True for v in checks.values()):
        raise ValueError('Diagnostic integrity checks incomplete')
    manifest = read_json(output / 'manifest.json')
    if manifest['names'] != names or manifest['canonical_test_names'][:COUNT] != names:
        raise ValueError('Diagnostic prefix order differs')
    if report['journal_sha256'] != sha256(path.with_name('images.jsonl')):
        raise ValueError('Diagnostic journal changed')
    pairs = {k:tuple(v['axes'][1:]) for k,v in report['transitions'].items()}
    cm, _, _, _, counts = rebuild(path.with_name('images.jsonl'), names, ('grid_probq',)+tuple(ARMS), pairs)
    if report['per_event_image_count'] != counts or any(
            not np.array_equal(a, np.asarray(report['metrics'][k]['confusion'])) for k,a in cm.items()):
        raise ValueError('Diagnostic aggregates differ from journal')
    validate_evidence(output,names,provenance['case'])
    final = copy.deepcopy(report)
    final.update(format='BRIGHT_ONE_IMAGE_MODEL_LIFETIME_V1', phase='diagnostic_complete',
        complete=False, diagnostic_complete=True, full_test_evidence=False, full_validation_evidence=False,
        final_integrity_pending=False, standalone_integrity_checks=checks,
        standalone_provenance=provenance, historical_failure_resolved=False, sanitizer_check_pending=True)
    write_json(path, final)
    path.with_name('result.json.sha256').write_text(sha256(path)+'  result.json\n')
    return final
