"""Build a bounded Train-only bank with the chosen readout embedding frozen."""
import argparse
import time

import torch

from . import common
from .memory import BANK_FORMAT, BalancedReservoir, target_distribution, validate_bank
from .routing import encode_regions


def smoke_images(train):
    """Prefer two different events so conservative retrieval has a valid probe."""
    first = train[0]
    second = next((name for name in train[1:] if common.event(name) != common.event(first)), train[1])
    return [first, second]


def main():
    parser = argparse.ArgumentParser(__doc__)
    common.add_arguments(parser)
    parser.add_argument('--per_bucket', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    if args.per_bucket < 1:
        parser.error('Positive per-event/class capacity required')
    begin = time.monotonic()
    common.seed_everything(args.seed)
    bundle = common.load_source(args)
    identity = common.source_identity(bundle, args)
    front = bundle['front']
    head = bundle['heads'][args.source_head]
    device = next(front.parameters()).device
    images = smoke_images(bundle['train']) if args.smoke else bundle['train']
    events = sorted({common.event(name) for name in images})
    event_ids = {name: index for index, name in enumerate(events)}
    reservoir = BalancedReservoir(args.per_bucket, args.seed)
    out = common.new_output(args.output)
    manifest = dict(format=BANK_FORMAT, complete=False, split='train', smoke=args.smoke,
                    test_accessed=False, identity=identity, arguments=vars(args),
                    image_count=len(images), event_names=events, image_names=list(images),
                    per_bucket=args.per_bucket, source_hash=common.experiment_hash(),
                    extraction='FP32 deterministic full-image1024; frozen front AND chosen readout embedding',
                    contents='bounded region vectors, four-class soft labels and Train source IDs only; no dense/Q cache',
                    support_labels='area-downsampled Train labels pooled with the exact chosen pool routing; ignore255 excluded',
                    bucket_policy='uniform random-priority reservoir per event and dominant class; background included',
                    bank_limit=len(events) * 4 * args.per_bucket)
    common.save_json(out / 'manifest.json', manifest)
    data = common.loader(args.dataset, images, workers=args.workers, seed=args.seed + 2)
    candidates = 0
    with torch.no_grad():
        for image_id, batch in enumerate(data):
            pre, post, _, gt, names = common.parse_batch(batch, device)
            if len(names) != 1 or names[0] != images[image_id]:
                raise RuntimeError('Bank extraction must preserve deterministic Train image order')
            reference = common.capture(front, pre, post)
            route, _ = common.routes(reference, bundle['manifest'], args.pool_mode, args.read_mode)
            encoded = encode_regions(head, reference['features']['roi'], route)
            distribution, labelled_mass = target_distribution(gt, route)
            if not torch.isfinite(encoded['tokens']).all():
                raise RuntimeError('Nonfinite frozen region representation')
            reservoir.add(encoded['tokens'][0], distribution[0], labelled_mass[0],
                          event_ids[common.event(names[0])], image_id)
            candidates += int((labelled_mass > 0).sum())
            if image_id == 0 or (image_id + 1) % 25 == 0 or image_id + 1 == len(images):
                print(f'bank {image_id + 1}/{len(images)} labelled_regions={candidates}', flush=True)
    common.assert_source(bundle)
    bank = reservoir.finish(images, events, identity, args.smoke)
    validate_bank(bank, identity, args.smoke, bundle['train'])
    common.save_pt(out / 'bank.pt', bank)
    manifest.update(complete=True, bank_sha256=common.digest(out / 'bank.pt'),
                    bank_regions=len(bank['tokens']), dim=bank['tokens'].shape[1],
                    candidate_regions=candidates, buckets=reservoir.counts(),
                    elapsed_seconds=time.monotonic() - begin)
    common.save_json(out / 'manifest.json', manifest)
    common.save_json(out / 'summary.json', dict(complete=True, smoke=args.smoke, test_accessed=False,
                    bank_regions=len(bank['tokens']), image_count=len(images),
                    frozen_source_verified=True, bank_sha256=manifest['bank_sha256']))
    print('TRAIN MEMORY BANK COMPLETE. Frozen embedding; no Val/Test supports.', flush=True)


if __name__ == '__main__':
    main()
