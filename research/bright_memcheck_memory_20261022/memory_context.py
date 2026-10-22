"""Fixed two-image resource diagnostic, independent of performance evaluation."""
from pathlib import Path
from memcheck_context import BASE, preflight as original_preflight
from memcheck_context import TARGET  # Keep the old observation schedule for these images.
from grid_probe import read_json, sha256

COUNT = 2
STEP = 1500
PREFIX_LAST = 'bata-explosion_00000079'
SELECTION = 'diagnostic_fixed1500_canonical_prefix2_memory'
FORMAT = 'BRIGHT_MEMCHECK_MEMORY_PAIR_V1'
CASES = ('plain', 'memcheck')
FAILED = BASE / 'memcheck1500_bv0vugq4'


def preflight():
    c = original_preflight()
    if c['canonical_test_names'][COUNT - 1] != PREFIX_LAST:
        raise ValueError('Recorded second image differs from canonical order')
    s = read_json(FAILED / 'summary.json')
    if (s.get('job_finished') is not True or s.get('phase') != 'diagnostic_failed'
            or s.get('completed_images') != 1 or 'OutOfMemoryError' not in s.get('case_error', '')):
        raise ValueError('Expected completed two-image OOM incident; no concurrent retry')
    for name in ('summary.json', 'request.json', 'runner.pyz',
                 'prefix1500_memcheck/summary.json', 'prefix1500_memcheck/sanitizer.log'):
        p = FAILED / name
        c['evidence_files'][str(p)] = sha256(p)
    return c
