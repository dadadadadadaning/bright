"""One fixed image and two memcheck runs; no additional optimizer updates."""
from memory_context import preflight as previous_preflight, BASE, STEP, TARGET
from grid_probe import read_json, sha256

COUNT = 1
CASES = ('light', 'watched')
FORMAT = 'BRIGHT_ROUTE_OBSERVER_AB_V1'
SELECTION = 'diagnostic_fixed1500_first_image_observer_ab'
FAILED = BASE / 'memory1500_bfzr43fv'


def preflight():
    c = previous_preflight()
    s = read_json(FAILED / 'summary.json')
    plain, mem = (s.get('cases', {}).get(k, {}) for k in ('plain', 'memcheck'))
    if (s.get('job_finished') is not True or plain.get('accepted') is not True
            or plain.get('completed_images') != 2 or mem.get('completed_images') != 1
            or 'OutOfMemoryError' not in mem.get('error', '')):
        raise ValueError('Expected completed memory1500 plain-pass/memcheck-OOM pair')
    for name in ('request.json', 'summary.json', 'runner.pyz', 'plain/memory.jsonl',
                 'memcheck/memory.jsonl', 'memcheck/sanitizer.log'):
        p = FAILED / name
        c['evidence_files'][str(p)] = sha256(p)
    return c
