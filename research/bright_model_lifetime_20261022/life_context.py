"""Same first image/checkpoint; isolate shared diagnostic wrappers."""
from ab_context import preflight as previous_preflight, BASE, STEP, TARGET
from grid_probe import read_json, sha256

COUNT = 1
CASES = ('minimal', 'instrumented')
FORMAT = 'BRIGHT_REAL_MODEL_LIFETIME_V1'
SELECTION = 'diagnostic_fixed1500_first_image_model_lifetime'
PREVIOUS = BASE / 'observer_ab1500_co1ab40o'


def preflight():
    c = previous_preflight()
    s = read_json(PREVIOUS / 'summary.json')
    if s.get('phase') != 'observer_pair_complete' or s.get('job_finished') is not True:
        raise ValueError('Expected finished observer A/B pair')
    for case in ('light', 'watched'):
        r = s.get('cases', {}).get(case, {})
        if (r.get('accepted') is not True or r.get('completed_images') != 1
                or r.get('sanitizer', {}).get('error_counts') != [0]):
            raise ValueError('Previous observer case did not pass: ' + case)
    for name in ('request.json', 'summary.json', 'runner.pyz',
                 'light/memory.jsonl', 'watched/memory.jsonl',
                 'light/sanitizer.log', 'watched/sanitizer.log'):
        p = PREVIOUS / name
        c['evidence_files'][str(p)] = sha256(p)
    return c
