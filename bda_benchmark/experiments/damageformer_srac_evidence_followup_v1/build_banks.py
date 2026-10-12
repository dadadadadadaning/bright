"""Create M0/M1 only from verified, existing Train shards; never extract or overwrite."""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
    load_bundle, inventory, input_fingerprints, diagnostic_code_fingerprint, quality_inventory,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import distribution_inventory
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.metadata import disaster_id
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import digest, save_json, save_pt
from .quota import add_arguments, evidence_gate, partial_quotas, priority, source_hash


FORMAT = 'SRAC_EVENT_QUOTA_BANK_PAIR_V1'


def parser():
    p = add_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument('--floor', type=int, default=16)
    return p


def selected_payload(records, bundle):
    """Read selected rows verbatim; labels/features/metadata are not recomputed."""
    rows = []
    root, cache = bundle['cache_root'], bundle['cache']
    grouped = {}
    for event_id, class_id, image_id, region_id in records:
        grouped.setdefault(image_id, []).append((event_id, class_id, region_id))
    for image_id, choices in sorted(grouped.items()):
        record = cache['shards'][image_id]
        path = root / record['path']
        if digest(path) != record['sha256']:
            raise ValueError('Source shard changed during selection')
        shard = torch.load(path, map_location='cpu', weights_only=False)
        for event_id, class_id, region_id in choices:
            d = shard['distribution'][region_id].clone()
            if int(d.argmax()) != class_id or int(shard['event']) != event_id:
                raise ValueError('Selected row identity/class changed')
            if (not torch.isfinite(d).all() or (d < 0).any()
                    or not torch.isclose(d.sum(), d.new_tensor(1.), atol=1e-5, rtol=0)
                    or not torch.isfinite(shard['tokens'][region_id]).all()
                    or not torch.isfinite(shard['geometry'][region_id]).all()
                    or not bool(shard['weights'][region_id] > 0)
                    or not bool(shard['mass'][region_id] > 1e-6)
                    or not bool(d.max() >= cache['min_purity'])):
                raise ValueError('Selected row is invalid/nonfinite/ineligible')
            rows.append((event_id, class_id, image_id, region_id, shard['tokens'][region_id].clone(),
                         d, shard['geometry'][region_id].clone()))
    rows.sort(key=lambda row: row[:4])
    if not rows:
        raise ValueError('Empty quota bank')
    events = bundle['bank_cpu']['event_names']
    return dict(tokens=torch.stack([r[4] for r in rows]),
                distribution=torch.stack([r[5] for r in rows]),
                geometry=torch.stack([r[6] for r in rows]),
                events=torch.tensor([r[0] for r in rows], dtype=torch.long),
                images=torch.tensor([r[2] for r in rows], dtype=torch.long),
                regions=torch.tensor([r[3] for r in rows], dtype=torch.long),
                disasters=torch.tensor([disaster_id(events[r[0]]) for r in rows], dtype=torch.long),
                event_names=events, checkpoint_sha256=bundle['hashes']['front_sha256'])


def run(args):
    if args.floor < 0:
        raise ValueError('floor must be nonnegative')
    evidence_gate(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    status = dict(complete=False, smoke=args.smoke, test_accessed=False, training=False)
    save_json(output / 'summary.json', status)
    try:
        code_before = source_hash()
        # Loading on CPU validates the exact frozen J parent; no image forward occurs.
        bundle = load_bundle(args, torch.device('cpu'))
        gate = evidence_gate(args, bundle)
        inv = inventory(bundle, True)
        save_json(output / 'inventory_original.json', inv)
        events = bundle['bank_cpu']['event_names']
        capacity = [inv['all_train_by_event'][name]['eligible_regions']['dominant_counts'] for name in events]
        quotas0 = [inv['by_event'][name]['dominant_counts'] for name in events]
        totals = inv['bank']['dominant_counts']
        quotas1 = [[0] * 4 for _ in events]
        floors = []
        for c in range(4):
            assigned, feasible_floor = partial_quotas([row[c] for row in capacity], totals[c], args.floor)
            floors.append(feasible_floor)
            for e, value in enumerate(assigned):
                quotas1[e][c] = value
        limits = {(e, c): max(quotas0[e][c], quotas1[e][c]) for e in range(len(events)) for c in range(4)}
        heaps = {key: [] for key, value in limits.items() if value}
        # Save only bounded row references, never all ~3M token payloads in RAM.
        for index, record in enumerate(bundle['cache']['shards']):
            path = bundle['cache_root'] / record['path']
            if digest(path) != record['sha256']:
                raise ValueError('Train shard hash changed')
            shard = torch.load(path, map_location='cpu', weights_only=False)
            e = int(shard['event'])
            d = shard['distribution'].float()
            purity, dominant = d.max(-1)
            eligible = (shard['weights'] > 0) & (shard['mass'] > 1e-6) & (purity >= bundle['cache']['min_purity'])
            for c in range(4):
                key = (e, c)
                limit = limits[key]
                if not limit:
                    continue
                for region_id in (eligible & (dominant == c)).nonzero().flatten().tolist():
                    item = (priority(args.seed, shard['name'], region_id), index, region_id)
                    heap = heaps[key]
                    if len(heap) < limit:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
            if (index + 1) % 250 == 0:
                print(f'quota candidates {index + 1}/{len(bundle["train_names"])}', flush=True)
        bank_hashes, row_hashes, bank_diagnostics = {}, {}, {}
        for arm, quotas in (('m0', quotas0), ('m1', quotas1)):
            records = []
            for (e, c), heap in sorted(heaps.items()):
                keep = sorted(heap, reverse=True)[:quotas[e][c]]
                if len(keep) != quotas[e][c]:
                    raise ValueError('Insufficient eligible rows for quota')
                records.extend((e, c, index, region_id) for _, index, region_id in keep)
            bank = selected_payload(records, bundle)
            if torch.bincount(bank['distribution'].argmax(-1), minlength=4).tolist() != totals:
                raise ValueError('Class budget mismatch')
            save_pt(output / f'{arm}_bank.pt', bank)
            save_json(output / f'{arm}_rows.json', records)
            bank_hashes[arm] = digest(output / f'{arm}_bank.pt')
            row_hashes[arm] = digest(output / f'{arm}_rows.json')
            bank_diagnostics[arm] = dict(distribution=distribution_inventory(bank['distribution']),
                                         quality=quality_inventory(bank['distribution']))
        if bundle['inputs_before'] != input_fingerprints(bundle['input_paths']):
            raise ValueError('Original input files changed during bank construction')
        if code_before != source_hash() or bundle['code_before'] != diagnostic_code_fingerprint(bundle['source_manifest']['format']):
            raise ValueError('Source code changed during bank construction')
        if evidence_gate(args, bundle) != gate:
            raise ValueError('Evidence reports changed during bank construction')
        manifest = dict(format=FORMAT, complete=True, arguments=vars(args), source_sha256=code_before,
                        source_hashes=bundle['hashes'], source_head_hashes=bundle['head_hashes'],
                        reference_hashes=bundle['reference_hashes'], resolved_paths=bundle['paths'],
                        original_input_paths={k: str(v) for k, v in bundle['input_paths'].items()},
                        original_input_sha256=bundle['inputs_before'],
                        event_names=events, class_totals=totals, capacities=capacity,
                        quotas=dict(m0=quotas0, m1=quotas1), feasible_floor=floors,
                        bank_sha256=bank_hashes, selected_rows_sha256=row_hashes,
                        bank_diagnostics=bank_diagnostics,
                        evidence_gate=gate, smoke=args.smoke, test_accessed=False,
                        train_only=True, extraction=False, training=False,
                        shared_priority='SHA256(seed,image_name,region_id); same nested order in each event/class bucket',
                        m0='Reconstructed M0: historical event/class counts, NOT historical selected rows',
                        m1='Same class totals; feasible event floor plus sqrt(capacity); may exceed 64/event/class',
                        labels='Original Train soft labels copied verbatim; no Val/Test used to select rows',
                        cache='Existing full Train shards referenced by path and SHA256, never duplicated')
        save_json(output / 'manifest.json', manifest)
        save_json(output / 'summary.json', dict(status, complete=True, class_totals=totals,
                                               original_inputs_unchanged=True, evidence_gate=gate))
        print(json.dumps(dict(class_totals=totals, quotas=manifest['quotas'], smoke=args.smoke)), flush=True)
        print('QUOTA BANK PAIR COMPLETE. No extraction/training/Test. M0 is reconstructed.', flush=True)
    except Exception as error:
        save_json(output / 'summary.json', dict(status, failure=dict(type=type(error).__name__, message=str(error))))
        raise


if __name__ == '__main__':
    run(parser().parse_args())
