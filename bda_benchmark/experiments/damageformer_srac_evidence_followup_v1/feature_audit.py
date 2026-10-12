"""Run unchanged A/B/C diagnostics with an explicit guarded native pool.

Can resolve the front from the completed J manifest to keep the feature and
Memory experiments on exactly the same snapshot. No historical file is edited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1 import feature_audit as parent
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import digest
from .native_pool import PoolGuard
from .provenance import code_hash, read_object


def parser():
    p = parent.parser()
    for action in p._actions:
        if action.dest == 'checkpoint':
            action.required = False
    p.add_argument('--joint_run', help='Resolve/verify the exact completed J front, not a newer snapshot')
    p.add_argument('--pool_mode', choices=('native', 'grouped'), default='native')
    p.add_argument('--pool_capture', choices=('failure', 'all'), default='failure')
    p.add_argument('--pool_check_reference', action='store_true')
    return p


def resolve_front(args):
    if args.joint_run:
        from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import validate_joint_header
        root = Path(args.joint_run)
        manifest = read_object(root/'manifest.json')
        validate_joint_header(manifest, read_object(root/'summary.json'))
        args.checkpoint = args.checkpoint or manifest['arguments']['checkpoint']
        if digest(args.checkpoint) != manifest['front_sha256']:
            raise ValueError('Feature and J front differ; do not mix snapshots')
        if digest(args.a1_checkpoint) != manifest['a1_sha256']:
            raise ValueError('Feature and J A1 differ')
    if not args.checkpoint:
        raise ValueError('Supply --joint_run or an exact --checkpoint')
    return args


def run(args):
    args = resolve_front(args)
    if args.pool_capture == 'all' and not (args.smoke or args.max_images in (1, 2)):
        raise ValueError('Capture-all is reserved for one/two-image diagnostics; use failure for full/prefix runs')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve old output and choose a new directory')
    before = code_hash()
    old_output, old_parse = parent.new_output, parent.parse_batch
    guard = None
    failed = None

    def guarded_output(path):
        nonlocal guard
        out = old_output(path)
        guard = PoolGuard(out/'pool_debug', mode=args.pool_mode,
                          capture=args.pool_capture, check_reference=args.pool_check_reference)
        guard.__enter__()
        return out

    def guarded_parse(*values, **kwargs):
        result = old_parse(*values, **kwargs)
        if guard is not None:
            guard.set_image(str(result[-1][0]))
        return result

    parent.new_output, parent.parse_batch = guarded_output, guarded_parse
    try:
        parent.run(args)
        if code_hash() != before:
            raise RuntimeError('Follow-up diagnostic source changed during execution')
    except BaseException as error:
        failed = error
        raise
    finally:
        parent.new_output, parent.parse_batch = old_output, old_parse
        if guard is not None:
            guard.__exit__(None, None, None)
        restore_failed = guard is not None and not guard.report()['restored']
        if output.is_dir():
            extra = dict(followup_source_sha256=before,
                         followup_source_unchanged=code_hash() == before,
                         pool_mode=args.pool_mode,
                         pool_guard=guard.report() if guard is not None else None)
            for name in ('summary.json', 'manifest.json', 'audit.json'):
                path = output/name
                if not path.is_file():
                    continue
                value = read_object(path)
                value.update(extra)
                if (failed is not None or restore_failed) and name != 'manifest.json':
                    value.update(complete=False, gates_passed=False, full_validation_evidence=False)
                parent.save_json(path, value)
        if restore_failed and failed is None:
            raise RuntimeError('Pool aliases were not restored; audit invalid')


if __name__ == '__main__':
    run(parser().parse_args())
