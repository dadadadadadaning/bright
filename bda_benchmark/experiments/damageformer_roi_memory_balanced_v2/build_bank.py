"""Build independent all-support and hard-support Train reservoirs."""
import argparse
import time

import torch

from . import common
from .memory import (BANK_FORMAT, DualQualityReservoir,
                     target_distribution_and_quality, validate_bank)
from bda_benchmark.experiments.damageformer_roi_followup_v1.build_bank import smoke_images
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    OutputLock, require_hard)


def run(args):
    require_hard(args)
    if args.per_bucket < 1:
        raise ValueError('Positive per-event/class capacity required')
    begin = time.monotonic()
    common.seed_everything(args.seed)
    bundle = common.load_source(args)
    identity = common.source_identity(bundle, args)
    front, head = bundle['front'], bundle['heads'][args.source_head]
    device = next(front.parameters()).device
    images = smoke_images(bundle['train']) if args.smoke else bundle['train']
    events = sorted({common.event(name) for name in images})
    event_ids = {name: i for i, name in enumerate(events)}
    reservoir = DualQualityReservoir(args.per_bucket, args.seed)
    out = common.new_output(args.output)
    code_hash = common.experiment_hash()
    manifest = dict(
        format=BANK_FORMAT, complete=False, split='train', smoke=args.smoke,
        test_accessed=False, identity=identity, arguments=vars(args),
        image_count=len(images), event_names=events, image_names=list(images),
        per_bucket=args.per_bucket, source_hash=code_hash,
        extraction='FP32 full-image1024; frozen front and frozen chosen readout embedding',
        all_candidate_policy='V1-compatible labelled_mass>0',
        hard_candidate_policy='labelled_mass>0 AND hard_count>0; selected before cap',
        fair_capacity='separate event x dominant-class cap for all and hard banks; shared candidate priorities',
        hard_count='winner=q.argmax(1); slot=ids.gather(1,winner[:,None]); scatter_add one native-grid pixel',
        quality='labelled_mass, max class-fraction purity, categorical entropy',
        bank_limit_per_policy=len(events) * 4 * args.per_bucket)
    common.save_json(out / 'manifest.json', manifest)
    data = common.loader(args.dataset, images, workers=args.workers,
                         seed=args.seed + 2)
    all_candidates = hard_candidates = 0
    with torch.no_grad():
        for image_id, batch in enumerate(data):
            pre, post, _, gt, names = common.parse_batch(batch, device)
            if len(names) != 1 or names[0] != images[image_id]:
                raise RuntimeError('Bank extraction order changed')
            reference = common.capture(front, pre, post)
            route, _ = common.routes(reference, bundle['manifest'],
                                     args.pool_mode, args.read_mode)
            encoded = encode_regions(head, reference['features']['roi'], route)
            distribution, labelled_mass, hard_count, purity, entropy = (
                target_distribution_and_quality(gt, route))
            if not torch.isfinite(encoded['tokens']).all():
                raise RuntimeError('Nonfinite frozen token')
            reservoir.add(encoded['tokens'][0], distribution[0], labelled_mass[0],
                          hard_count[0], purity[0], entropy[0],
                          event_ids[common.event(names[0])], image_id)
            all_candidates += int((labelled_mass > 0).sum())
            hard_candidates += int(((labelled_mass > 0) & (hard_count > 0)).sum())
            if image_id == 0 or (image_id + 1) % 25 == 0 or image_id + 1 == len(images):
                print(f'quality bank {image_id + 1}/{len(images)} all={all_candidates} hard={hard_candidates}', flush=True)
    common.assert_v2_immutable(bundle, code_hash)
    banks = {policy: reservoir.finish(policy, images, events, identity, args.smoke)
             for policy in ('all', 'hard')}
    for policy, bank in banks.items():
        validate_bank(bank, identity, args.smoke, policy, bundle['train'])
        common.save_pt(out / f'bank_{policy}.pt', bank)
    manifest.update(
        complete=True,
        all_sha256=common.digest(out / 'bank_all.pt'),
        hard_sha256=common.digest(out / 'bank_hard.pt'),
        all_regions=len(banks['all']['tokens']),
        hard_regions=len(banks['hard']['tokens']),
        dim=banks['all']['tokens'].shape[1],
        all_candidate_regions=all_candidates,
        hard_candidate_regions=hard_candidates,
        all_buckets=reservoir.counts('all'),
        hard_buckets=reservoir.counts('hard'),
        elapsed_seconds=time.monotonic() - begin)
    common.save_json(out / 'manifest.json', manifest)
    common.save_json(out / 'summary.json', dict(
        complete=True, smoke=args.smoke, test_accessed=False,
        all_regions=manifest['all_regions'], hard_regions=manifest['hard_regions'],
        image_count=len(images), frozen_source_verified=True,
        all_sha256=manifest['all_sha256'], hard_sha256=manifest['hard_sha256']))
    print('V2 DUAL QUALITY BANK COMPLETE. Train only; no Val/Test supports.', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    common.add_arguments(parser)
    parser.add_argument('--per_bucket', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
