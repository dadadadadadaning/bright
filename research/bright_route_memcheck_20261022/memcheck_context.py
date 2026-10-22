"""Pinned step1500 R-index incident; a diagnostic prefix, never Test878 evidence."""
from pathlib import Path
from grid_probe import read_json, sha256
from resume_train import verify_files
import sweep_context

TARGET = 'la_palma-volcano_00000812'
INDEX = 355
COUNT = INDEX + 1
STEP = 1500
CASE = 'prefix1500_memcheck'
FORMAT = 'BRIGHT_PARENT_IDS_MEMCHECK_V1'
SELECTION = 'diagnostic_fixed1500_canonical_test_prefix356'
FAILED = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_TEST_SWEEP_V1/test_sweep_hzld12ds/test1500')
BASE = FAILED.parent.parent.parent / 'SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS'
SWEEP_SHA = 'cb4dc39c75ab29971a9f1fe5d313ec7c1fd4b11dae3aaed5f74b561e9f7a9b70'


def check_incident(folder, canonical):
    if len(canonical) != 878 or canonical[INDEX] != TARGET:
        raise ValueError('Canonical target/order differs')
    summary = read_json(folder / 'summary.json')
    window = read_json(folder / 'operator_window.json')
    failure = summary.get('first_failure') or {}
    p = failure.get('position', {})
    if (summary.get('checkpoint_step') != STEP or summary.get('phase') != 'failed'
            or summary.get('completed_images') != INDEX or p.get('index') != INDEX
            or p.get('image') != TARGET or p.get('active_arm') != 'r'
            or 'parent.ids' not in failure.get('error', '')):
        raise ValueError('Expected recorded R parent.ids incident')
    records = []
    for side, stage in [('previous', 'forward_before'), ('current', 'forward_after')]:
        row = window[side]
        pos = row['position']
        if (pos.get('index') != INDEX or pos.get('image') != TARGET
                or pos.get('active_arm') != 'r' or pos.get('watch_stage') != stage):
            raise ValueError('Incident observation window differs')
        groups = [g for g in row['groups'] if 'parent.ids' in g['keys']]
        if len(groups) != 1:
            raise ValueError('Missing parent.ids metadata')
        records.append(groups[0])
    before, after = records
    if (before['metadata'] != after['metadata']
            or before['metadata']['dtype'] != 'torch.int64'
            or before['metadata']['shape'] != [9, 65536]
            or before['statistics']['sha256'] == after['statistics']['sha256']):
        raise ValueError('Recorded index object/storage/content evidence differs')


def preflight():
    if sha256(FAILED.parent / 'runner.pyz') != SWEEP_SHA:
        raise ValueError('Incident sweep archive changed')
    c = sweep_context.preflight()
    check_incident(FAILED, c['canonical_test_names'])
    manifest = read_json(FAILED / 'manifest.json')
    if manifest['provenance']['checkpoint_sha256'] != c['checkpoints'][str(STEP)]['sha256']:
        raise ValueError('Incident checkpoint differs')
    paths = [FAILED.parent / 'runner.pyz', FAILED.parent / 'request.json']
    paths += [FAILED / n for n in ('summary.json', 'manifest.json', 'evaluation_failure.json',
        'operator_window.json', 'operator_window_cpu.pt', 'route_change.json', 'route_change_cpu.pt',
        'test_step_001500/images.jsonl')]
    c['evidence_files'].update({str(p): sha256(p) for p in paths})
    verify_files(c['evidence_files'])
    return sweep_context.for_step(c, STEP)
