"""New experiment orchestration; all previous model/experiment files stay unchanged."""
import hashlib
import json
from pathlib import Path

from bda_benchmark.experiments.damageformer_roi_followup_v1 import common as source
from bda_benchmark.experiments.damageformer_partition_learning_v1 import reproduction as repro

FORMAT = 'REGION_MEMORY_RESIDUAL_V2'


def add_arguments(parser, training=False):
    for key in ('checkpoint','run_root','dataset','train_list','val_list','output'):
        parser.add_argument('--'+key, required=True)
    for key in ('source_head','pool_mode','read_mode'):
        parser.add_argument('--'+key, choices=('hard',), default='hard')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    if training:
        for key, default in (('steps',5000),('eval_every',500),('batch',2),('accum',8),('seed',3407)):
            parser.add_argument('--'+key, type=int, default=default)
        parser.add_argument('--lr', type=float, default=1e-4)
    return parser


def implementation_hash():
    h = hashlib.sha256(source.experiment_hash().encode())
    paths = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'run.sh', Path(repro.__file__)]
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def require_hard(args):
    if (args.source_head, args.pool_mode, args.read_mode) != ('hard', 'hard', 'hard'):
        raise ValueError('This experiment fixes source, pool and read to hard; exchange/Q changes are separate experiments')


def baseline_policy(bundle, args):
    if args.smoke:
        return None
    rows, warnings, sha = repro.load_history(args.run_root, bundle['summary']['final_step'])
    path = str(Path(args.run_root) / 'validation.jsonl')
    if sha is not None:
        bundle['provenance']['source_file_sha256'][path] = sha
    result = repro.build_policy(bundle['summary']['final_metrics'], rows, warnings)
    result.update(history_file=path, history_sha256=sha)
    return result


def initial_gate(bundle, result, args, out, policy):
    """Always persist evidence, including when a baseline fails. No threshold fitting."""
    source.save_json(out / 'initial.json', repro.json_safe_evidence(result))
    if args.smoke:
        report = dict(passed=None, status='smoke_not_full_val', checks=[], warnings=[])
    else:
        report = repro.compare_baselines(result['metrics'], bundle['summary']['final_metrics'], policy)
        report['status'] = 'passed' if report['passed'] else 'failed'
    report['source_integrity_pass'] = None
    source.save_json(out / 'reproduction.json', report)
    try:
        source.assert_source(bundle)
        report['source_integrity_pass'] = True
        source.save_json(out / 'reproduction.json', report)
        if not args.smoke:
            repro.enforce(report)
    except Exception as exc:
        report.update(passed=False, status='failed', error=str(exc))
        source.save_json(out / 'reproduction.json', report)
        source.save_json(out / 'summary.json', dict(complete=False, smoke=args.smoke,
            phase='initial_reproduction_failed', error=str(exc), optimizer_steps=0, test_accessed=False))
        raise
    return report


def assert_unchanged(bundle, code_hash, bank_files=None):
    source.assert_source(bundle)
    if implementation_hash() != code_hash:
        raise RuntimeError('Memory V2 code changed during this run; use a new isolated run')
    for path, sha in (bank_files or {}).items():
        if source.digest(path) != sha:
            raise RuntimeError('Memory bank changed during training: ' + str(path))


class OutputLock:
    """Cross-platform exclusive lock; never remove another process's lock."""
    def __init__(self, output):
        self.output = Path(output)
        self.path = self.output.with_name(self.output.name + '.lock')
        self.owned = False

    def __enter__(self):
        import os
        if self.output.exists():
            raise FileExistsError('Preserve existing output and choose a new path: ' + str(self.output))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x', encoding='utf-8') as stream:
            self.owned = True
            json.dump(dict(pid=os.getpid(), output=str(self.output)), stream)
        return self

    def __exit__(self, *exc):
        if self.owned:
            self.path.unlink()
            self.owned = False
