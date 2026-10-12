"""Bounded Train-only, frozen HARD region bank. No dense cache or checkpoint copy."""
import argparse
import time
import torch

from .common import source, add_arguments, require_hard, implementation_hash, assert_unchanged, OutputLock
from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import (
    BANK_FORMAT, BalancedReservoir, target_distribution, validate_bank)
from bda_benchmark.experiments.damageformer_roi_followup_v1.build_bank import smoke_images
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions


def run(args):
    require_hard(args)
    source.seed_everything(args.seed)
    code_hash = implementation_hash()
    bundle = source.load_source(args)
    identity = source.source_identity(bundle, args)
    images = smoke_images(bundle['train']) if args.smoke else bundle['train']
    events = sorted({source.event(name) for name in images})
    event_ids = {name: index for index, name in enumerate(events)}
    reservoir = BalancedReservoir(args.per_bucket, args.seed)
    out = source.new_output(args.output)
    begin = time.monotonic()
    manifest = dict(format=BANK_FORMAT, builder='region_memory_v2', complete=False,
        split='train', smoke=args.smoke, test_accessed=False, identity=identity,
        arguments=vars(args), image_count=len(images), event_names=events, image_names=images,
        per_bucket=args.per_bucket, source_hash=source.experiment_hash(), builder_hash=code_hash,
        bank_limit=len(events)*4*args.per_bucket,
        extraction='FP32 full Train images at1024; fixed hard embedding, Q and B; no augmentation',
        query_scale='Train random512 crops, Val1024; same frozen embedding, context/scale mismatch remains',
        bucket_policy='uniform random-priority reservoir per event and majority GT class; background included',
        support_labels='four-class Train soft fractions; area downsampling and exact HARD pool route; ignore255 excluded',
        candidate_policy='all positive labelled mass; no purity/size filter to avoid changing baseline bank policy')
    source.save_json(out/'manifest.json', manifest)
    source.save_json(out/'summary.json', dict(complete=False, smoke=args.smoke, test_accessed=False))
    counts, tiny, purity_sum = 0, 0, 0.
    device = bundle['device']
    data = source.loader(args.dataset, images, workers=args.workers, seed=args.seed+2)
    with torch.no_grad():
        for image_id, batch in enumerate(data):
            pre, post, _, gt, names = source.parse_batch(batch, device)
            if names != [images[image_id]] and list(names) != [images[image_id]]:
                raise RuntimeError('Bank extraction order/batch must be fixed, batch=1')
            reference = source.capture(bundle['front'], pre, post)
            route, _ = source.routes(reference, bundle['manifest'], 'hard', 'hard')
            encoded = encode_regions(bundle['heads']['hard'], reference['features']['roi'], route)
            distribution, mass = target_distribution(gt, route)
            reservoir.add(encoded['tokens'][0], distribution[0], mass[0],
                          event_ids[source.event(names[0])], image_id)
            valid = mass > 0
            counts += int(valid.sum())
            tiny += int((valid & (mass < 1)).sum())
            purity_sum += float(distribution.max(-1).values[valid].sum())
            if image_id == 0 or (image_id+1) % 25 == 0 or image_id+1 == len(images):
                print(f'bank {image_id+1}/{len(images)} candidates={counts}', flush=True)
    assert_unchanged(bundle, code_hash)
    bank = reservoir.finish(images, events, identity, args.smoke)
    validate_bank(bank, identity, args.smoke, bundle['train'])
    if not args.smoke and set(bank['image_names']) != set(bundle['train']):
        raise ValueError('Formal bank must visit all Train2918 images')
    source.save_pt(out/'bank.pt', bank)
    manifest.update(complete=True, bank_sha256=source.digest(out/'bank.pt'),
        bank_regions=len(bank['tokens']), dim=bank['tokens'].shape[1], buckets=reservoir.counts(),
        candidate_regions=counts, tiny_candidate_mass_lt1=tiny,
        candidate_slot_mean_gt_purity=purity_sum/max(1, counts),
        retained_slot_mean_gt_purity=float(bank['distribution'].max(-1).values.mean()),
        bank_bytes=(out/'bank.pt').stat().st_size, elapsed_seconds=time.monotonic()-begin)
    source.save_json(out/'manifest.json', manifest)
    source.save_json(out/'summary.json', dict(complete=True, smoke=args.smoke, test_accessed=False,
        bank_regions=len(bank['tokens']), image_count=len(images), bank_bytes=manifest['bank_bytes'],
        frozen_source_verified=True))
    print('TRAIN BANK COMPLETE. No Val/Test support; no dense cache.', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    add_arguments(parser)
    parser.add_argument('--per_bucket', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    if args.per_bucket < 1:
        parser.error('per_bucket must be positive')
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
