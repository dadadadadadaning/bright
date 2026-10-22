"""Compare recovered read_aux against original, explicitly retaining instrumentation differences.

Only complete step2500 Val450 reports are accepted. No inference or original-file writes.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

from recovery import CHECKPOINT_SHA, protocol_delta, require, sha

ANALYSIS_FILES = {
    'compare_runs.py': 'a6981da821bfcf781f5a8fd68bab0d27fc38c60a94fa9a03b4fbbcd24b9e57b0',
    'analyze_results.py': '0712bb7c85adb49ace38278a43483d5e8eeb74cc953e783831ab6db5ae09aabd',
}


def load_analysis(package):
    package = Path(package).resolve()
    for name, digest in ANALYSIS_FILES.items():
        require(sha(package / name) == digest, 'Unsupported analysis source: '+name)

    def module(name, tag):
        spec = importlib.util.spec_from_file_location(tag, package / name)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    analysis = module('analyze_results.py', '_verified_recovery_analysis')
    previous = sys.modules.get('analyze_results')
    sys.modules['analyze_results'] = analysis
    try:
        return module('compare_runs.py', '_verified_recovery_comparison')
    finally:
        if previous is None:
            sys.modules.pop('analyze_results', None)
        else:
            sys.modules['analyze_results'] = previous


def checked_protocols(before, after):
    """The single permitted protocol difference is the recorded blocking setting."""
    return protocol_delta(before, after)


def compare_data(helper, a, ma, ra, b, mb, rb):
    for key in ('split', 'step', 'seed', 'class_names', 'requested_names'):
        require(a.get(key) == b.get(key), 'Unmatched report field: '+key)
    require(a.get('split') == 'val' and a.get('step') == 2500 and a.get('seed') == 3407
            and a.get('image_count') == b.get('image_count') == 450,
            'Require matched seed3407 step2500 Val450')
    require(a.get('variant') == 'original' and b.get('variant') == 'read_aux',
            'Expected original before and recovered read_aux after')
    for key in ('initial_head_sha256', 'train_names', 'val_names', 'actual_training_names',
                'fixed_train_names', 'parameter_count', 'support_arm'):
        require(key in ma and key in mb and ma[key] == mb[key], 'Unmatched manifest field: '+key)
    delta = checked_protocols(ma['protocol_info'], mb['protocol_info'])
    for key in ('source', 'addon_sha256', 'extension_files', 'post500_lr_multiplier', 'scheduler_horizon',
                'warmup_steps', 'maximum_lr', 'weight_decay', 'grad_clip', 'optimizer', 'endpoint_step'):
        require(key in ma['plan'] and ma['plan'][key] == mb['plan'].get(key), 'Unmatched plan field: '+key)
    require(ma['plan'].get('variant') == 'original' and ma['plan'].get('read_weight') == 0
            and mb['plan'].get('variant') == 'read_aux' and mb['plan'].get('read_weight') == .01,
            'Comparison variants or auxiliary weights differ')
    ac, bc = dict(ma['runtime_config']), dict(mb['runtime_config'])
    ac.pop('output', None); bc.pop('output', None)
    require(ac == bc, 'Runtime configuration differs beyond output')
    lineage = mb.get('recovery', {})
    require(lineage.get('restored_step') == 800 and lineage.get('checkpoint_sha256') == CHECKPOINT_SHA
            and lineage.get('inherited_training_rows') == 800 and lineage.get('new_training_starts_at') == 801
            and lineage.get('exact_historical_replay') is False
            and lineage.get('historical_failure_resolved') is False
            and b.get('recovery') == lineage, 'Missing or inconsistent recovery provenance')
    merged = dict(a, metrics={'before': a['metrics'][a['variant']], 'after': b['metrics'][b['variant']]})
    result = helper.analyze(merged, 'before', 'after')
    per_image, events = [], {}
    for left, right in zip(ra, rb):
        require(left['image'] == right['image'] and left['event'] == right['event'], 'Image/event mismatch')
        lc, rc = left['confusion'][a['variant']], right['confusion'][b['variant']]
        lm, rm = helper.metrics(lc), helper.metrics(rc)
        require(lm['ground_truth_counts'] == rm['ground_truth_counts'], 'Per-image GT counts differ')
        per_image.append(dict(image=left['image'], event=left['event'],
                              before_miou4_pct=lm['miou4_pct'], after_miou4_pct=rm['miou4_pct'],
                              miou4_difference_pp=rm['miou4_pct']-lm['miou4_pct']))
        pair = events.setdefault(left['event'], [[[0]*4 for _ in range(4)] for _ in range(2)])
        for k, cm in enumerate((lc, rc)):
            for r in range(4):
                for c in range(4):
                    pair[k][r][c] += cm[r][c]
    result.update(before_variant='original', after_variant='read_aux', seed=3407,
                  comparison_kind='checkpoint_continuation_with_explicit_instrumentation_difference',
                  matched_recipe_except_reported_execution_settings=True,
                  exact_execution_protocol_match=ma['protocol_info'] == mb['protocol_info'],
                  exact_historical_trajectory_match=False, historical_failure_resolved=False,
                  execution_protocol_delta=delta,
                  before_protocol=copy.deepcopy(ma['protocol_info']), after_protocol=copy.deepcopy(mb['protocol_info']),
                  recovery=copy.deepcopy(lineage),
                  per_event={e: {'before': helper.metrics(p[0]), 'after': helper.metrics(p[1])} for e,p in events.items()},
                  grid_confusion_identical=a['metrics']['grid_probq']['confusion'] == b['metrics']['grid_probq']['confusion'])
    result['caveats'].append('The recovered run uses synchronous CUDA execution and observed updates801-804; this is not an identical-execution or exact historical trajectory claim.')
    if not result['grid_confusion_identical']:
        result['caveats'].append('Frozen Grid differs across executions; inspect this before interpreting tiny gains.')
    return result, per_image


def compare(package, baseline, output):
    """Write only a new comparison directory inside the completed recovery root."""
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    before_path = baseline / 'val_step_002500/result.json' if baseline.is_dir() else baseline
    after_path = output / 'val_step_002500/result.json'
    helper = load_analysis(package)
    summary = helper.read(output / 'summary.json')
    require(summary.get('phase') == 'complete' and summary.get('complete') is True
            and summary.get('training_complete') is True and summary.get('full_validation_evidence') is True
            and summary.get('last_complete_step') == 2500 and summary.get('restored_step') == 800
            and summary.get('new_updates') == 1700 and summary.get('test_accessed') is False,
            'Recovery is not complete through step2500 and Val450')
    a, ma, ra = helper.load(before_path)
    b, mb, rb = helper.load(after_path)
    result, rows = compare_data(helper, a, ma, ra, b, mb, rb)
    result['source_reports'] = {str(p): sha(p) for p in (before_path, after_path)}
    directory = output / 'comparison_original_to_read_aux'
    directory.mkdir(exist_ok=False)
    (directory / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (directory / 'paired_images.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result, directory
