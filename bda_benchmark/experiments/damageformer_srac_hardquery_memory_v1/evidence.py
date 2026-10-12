"""E1: original Top-8 label evidence versus original trained Readers, without fitting."""
from __future__ import annotations

import argparse

from .io_utils import (add_source_arguments, load_verified_bundle, new_output, artifact_identity,
                       verify_unchanged, write_json, fail_summary, finite_positive)
from .runtime import evaluate


def parser():
    p = add_source_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument('--protocol', required=True)
    p.add_argument('--temperature', type=float, default=.1)
    return p


def run(args):
    from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.train import full_image_loader
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    finite_positive(args.temperature, 'temperature')
    output, capture = new_output(args.output), None
    try:
        bundle = load_verified_bundle(args)
        capture = FrozenFrontCapture(bundle['front'])
        names = bundle['val_names'][:2] if args.smoke else bundle['val_names']
        data = full_image_loader(args.dataset, names, False, args.workers, args.seed)
        with PoolGuard(output / 'pool_debug', mode=args.pool_mode) as guard:
            result = evaluate(bundle, capture, data, guard, temperature=args.temperature,
                              image_log=output / 'images.jsonl', smoke=args.smoke)
        immutable = verify_unchanged(bundle)
        passed = result['reproduction']['passed'] is not False
        record = dict(format='SRAC_HARDQUERY_EVIDENCE_V1', **artifact_identity(bundle), **result,
                      complete=passed, smoke=args.smoke, immutable=immutable,
                      full_validation_evidence=passed and not args.smoke and result['image_count'] == 450)
        write_json(output / 'evidence.json', record)
        write_json(output / 'summary.json', record)
        if not passed:
            raise RuntimeError('Full-Val historical reproduction failed; evidence retained, no training')
        return record
    except Exception as error:
        fail_summary(output, error)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == '__main__':
    run(parser().parse_args())
