"""Evaluate the already selected final selective-Memory heads once on Test.

All checkpoint choices and retrieval settings come from the completed Train/Val
run.  Test labels are read only after those choices have been validated.
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from bda_benchmark.experiments.damageformer_roi_memory_selective_v1 import common
from bda_benchmark.experiments.damageformer_roi_memory_selective_v1 import train as trained
from bda_benchmark.experiments.damageformer_roi_memory_selective_v1.memory import (
    BANK_FORMAT, HEAD_FORMAT, POSTERIOR_MODES, SelectivePosteriorHead,
    assert_no_query_target_interfaces, build_quality_view,
    classify_posterior_regions, device_bank, read_banks,
)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import decode_regions, encode_regions


FORMAT = 'ROI_MEMORY_SELECTIVE_TEST_V1'
PREDICTIONS = ('source_hard',) + POSTERIOR_MODES
PAIRS = tuple(('query_only', name) for name in POSTERIOR_MODES[1:])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def split_names(path, count):
    names = [row.strip() for row in Path(path).read_text(encoding='utf-8').splitlines() if row.strip()]
    if (len(names) != count or len(set(names)) != count or
            any(not name or '/' in name or '\\' in name or name in ('.', '..') for name in names)):
        raise ValueError(f'Expected {count} unique plain image IDs: {path}')
    return names


def resolve_test_names(train_list, val_list):
    train_list, val_list = Path(train_list), Path(val_list)
    if (train_list.name != 'train_set.txt' or val_list.name != 'val_set.txt' or
            train_list.parent.resolve() != val_list.parent.resolve() or
            train_list.parent.name != 'standard_ML'):
        raise ValueError('Require matching standard_ML Train/Val split files')
    test_list = train_list.parent / 'test_set.txt'
    train = split_names(train_list, 2918)
    val = split_names(val_list, 450)
    test = split_names(test_list, 878)
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError('Train, Val and Test image IDs must be disjoint')
    return test_list, test


def check_test_files(dataset, names):
    dataset = Path(dataset)
    missing = []
    for name in names:
        for subdir, suffix in (('pre-event', '_pre_disaster.tif'),
                               ('post-event', '_post_disaster.tif'),
                               ('target', '_building_damage.tif')):
            path = dataset / subdir / (name + suffix)
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f'Test needs all RGB/SAR/GT files for scored evaluation; '
            f'{len(missing)} files missing, first: {missing[0]}. '
            'No Test mIoU was computed.')


def validate_run(run_root):
    run_root = Path(run_root).resolve(strict=True)
    manifest = read_json(run_root / 'manifest.json')
    summary = read_json(run_root / 'summary.json')
    if (manifest.get('format') != HEAD_FORMAT or manifest.get('stage') != 'posterior' or
            manifest.get('complete') is not True or summary.get('complete') is not True or
            manifest.get('smoke') is not False or summary.get('smoke') is not False or
            manifest.get('test_accessed') is not False or summary.get('test_accessed') is not False or
            summary.get('stage') != 'posterior' or
            manifest.get('code_hash') != common.experiment_hash() or
            manifest.get('parent_code_hash') != common.parent_experiment_hash()):
        raise ValueError('Require complete, unchanged formal posterior Train/Val run')
    args = manifest.get('arguments')
    if not isinstance(args, dict) or args.get('stage') != 'posterior' or args.get('smoke') is not False:
        raise ValueError('Missing formal posterior training arguments')
    step = summary.get('final_step')
    if (not isinstance(step, int) or step < 1 or step != args.get('steps') or
            summary.get('optimizer_steps') != step):
        raise ValueError('Final optimizer step differs from trained protocol')
    files = {}
    for name in POSTERIOR_MODES:
        path = run_root / (name + '_last.pth')
        if not path.is_file():
            raise FileNotFoundError(f'Missing final head checkpoint: {path}')
        files[name] = path
    return run_root, manifest, summary, SimpleNamespace(**args), files


def load_heads(paths, manifest, summary, dim, device):
    heads, hashes = {}, {}
    for name, path in paths.items():
        before = common.digest(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        if common.digest(path) != before:
            raise RuntimeError('Checkpoint changed while loading: ' + str(path))
        if (state.get('format') != HEAD_FORMAT or state.get('stage') != 'posterior' or
                state.get('name') != name or state.get('step') != summary['final_step'] or
                state.get('code_hash') != manifest['code_hash'] or
                state.get('source_identity') != manifest['identity'] or
                state.get('bank_file_sha256') != manifest['bank_file_sha256'] or
                state.get('metrics') != summary['final_metrics'][name]):
            raise ValueError('Final checkpoint identity/metrics mismatch: ' + name)
        weights = state.get('head')
        if not isinstance(weights, dict) or not weights or any(
                not torch.is_tensor(value) or not torch.isfinite(value).all()
                for value in weights.values()):
            raise ValueError('Invalid final head tensors: ' + name)
        head = SelectivePosteriorHead(dim).to(device)
        head.load_state_dict(weights, strict=True)
        head.requires_grad_(False).eval()
        if (sum(p.numel() for p in head.parameters()) !=
                manifest['head_parameters'][name]):
            raise ValueError('Final head parameter count mismatch: ' + name)
        heads[name] = head
        hashes[name] = before
    return heads, hashes


def source_and_bank(args, manifest):
    bundle = common.load_source(args)
    if common.source_identity(bundle, args) != manifest['identity']:
        raise ValueError('Front/hard head source identity differs from training')
    bank_root = Path(args.bank)
    expected = manifest['bank_file_sha256']
    for filename in ('bank_all.pt', 'bank_hard.pt', 'manifest.json'):
        path = bank_root / filename
        if expected.get(str(path)) != common.digest(path):
            raise ValueError('Train bank file hash mismatch: ' + str(path))
    cpu_banks, bank_manifest = read_banks(
        args.bank, manifest['identity'], False, common.digest, bundle['train'])
    if (bank_manifest.get('format') != BANK_FORMAT or
            set(cpu_banks['hard']['image_names']) != set(bundle['train']) or
            set(cpu_banks['all']['image_names']) != set(bundle['train'])):
        raise ValueError('Formal Train2918 Memory bank mismatch')
    bank_cpu = build_quality_view(cpu_banks['hard'], args.quality_quantile)
    if ([dict(row) for row in bank_cpu['quality_thresholds']] !=
            manifest['quality_thresholds'] or
            int(bank_cpu['quality_eligible'].sum()) !=
            manifest['quality_eligible_regions']):
        raise ValueError('Stored Train-only quality policy changed')
    return bundle, device_bank(bank_cpu, bundle['device'])


def compare_val_replay(actual, expected, miou_tolerance=5e-5,
                       confusion_fraction_tolerance=1e-5):
    """Fixed, predeclared drift bounds for final trained-head reproduction.

    Tolerances are in [0,1] metric units and fraction of evaluated pixels.
    The old soft-region branch is diagnostic only; source_hard and all four
    selected heads must pass before Test loading.
    """
    required = PREDICTIONS
    checks = {}
    for name in required:
        observed = actual.get(name)
        reference = expected.get(name)
        if not isinstance(observed, dict) or not isinstance(reference, dict):
            raise ValueError('Missing final Val metric: ' + name)
        a, b = observed.get('valid_miou'), reference.get('valid_miou')
        cm_a, cm_b = observed.get('confusion'), reference.get('confusion')
        if not all(isinstance(x, (float, int)) and math.isfinite(x)
                   for x in (a, b)):
            raise ValueError('Invalid final Val mIoU: ' + name)
        if (not isinstance(cm_a, list) or not isinstance(cm_b, list) or
                len(cm_a) != 4 or len(cm_b) != 4 or
                any(not isinstance(row, list) or len(row) != 4
                    for row in cm_a + cm_b)):
            raise ValueError('Missing 4x4 final Val confusion: ' + name)
        values = [*sum(cm_a, []), *sum(cm_b, [])]
        if any(not isinstance(x, int) or x < 0 for x in values):
            raise ValueError('Invalid final Val confusion: ' + name)
        total_a, total_b = sum(sum(row) for row in cm_a), sum(sum(row) for row in cm_b)
        l1 = sum(abs(cm_a[i][j] - cm_b[i][j])
                 for i in range(4) for j in range(4))
        fraction = l1 / max(1, total_b)
        passed = (total_a == total_b and total_a > 0 and
                  abs(a - b) <= miou_tolerance and
                  fraction <= confusion_fraction_tolerance)
        checks[name] = dict(
            passed=passed, observed_valid_miou=a, trained_valid_miou=b,
            absolute_miou_difference=abs(a-b), confusion_l1=l1,
            confusion_l1_fraction=fraction, evaluated_pixels=total_a)
    result = dict(passed=all(row['passed'] for row in checks.values()),
                  image_count=450, test_accessed=False,
                  miou_tolerance=miou_tolerance,
                  confusion_fraction_tolerance=confusion_fraction_tolerance,
                  checks=checks)
    return result


@torch.no_grad()
def evaluate(bundle, bank, heads, args, test_names):
    for model in (bundle['front'], bundle['heads']['hard'], *heads.values()):
        model.eval()
    data = common.loader(args.dataset, test_names, workers=args.workers,
                         seed=args.seed + 3)
    diagnostic = DiagnosticAccumulator(
        threshold=bundle['manifest']['arguments']['threshold'], pairs=PAIRS)
    count = valid_pixels = foreground_pixels = 0
    for batch in data:
        pre, post, _, gt, names = common.parse_batch(batch, bundle['device'])
        if ((gt < 0) | ((gt > 3) & (gt != 255))).any():
            raise ValueError('Unexpected Test GT labels; scored evaluation stopped')
        valid_pixels += int(((gt >= 0) & (gt < 4)).sum())
        foreground_pixels += int(((gt > 0) & (gt < 4)).sum())
        reference = common.capture(bundle['front'], pre, post)
        pool_route, read_route = common.routes(
            reference, bundle['manifest'], 'hard', 'hard')
        encoded = encode_regions(
            bundle['heads']['hard'], reference['features']['roi'], pool_route)
        tokens, mass = encoded['tokens'], encoded['mass']
        base_logits = bundle['heads']['hard'].classifier(tokens)
        shape = gt.shape[-2:]
        base = decode_regions(base_logits, read_route, mass, shape)
        legacy = bundle['heads']['hard'](
            reference['features']['roi'], pool_route, shape)
        torch.testing.assert_close(base, legacy, atol=2e-5, rtol=2e-4)
        predictions = {'source_hard': base.argmax(1)}
        event_names = [common.event(name) for name in names]
        for name in POSTERIOR_MODES:
            logits, _ = classify_posterior_regions(
                heads[name], tokens, mass, base_logits, bank, names, event_names,
                name, retrieval_policy=args.posterior_retrieval,
                topk=args.topk, preselect=args.preselect,
                per_class_cap=args.per_class_cap, min_cosine=args.min_cosine,
                query_chunk=args.query_chunk,
                exclude_same_event=not args.allow_same_event,
                temperature=args.temperature)
            probability = decode_regions(logits, read_route, mass, shape)
            if not torch.isfinite(probability).all():
                raise RuntimeError('Nonfinite Test probability: ' + name)
            predictions[name] = probability.argmax(1)
        diagnostic.update(gt, predictions, reference['building'], names)
        count += len(names)
        if count % 25 == 0:
            print(f'Test evaluation {count}/878', flush=True)
    if count != 878:
        raise RuntimeError(f'Expected Test878, evaluated {count}')
    if valid_pixels == 0 or foreground_pixels == 0:
        raise ValueError('Test targets contain no valid foreground labels; no mIoU reported')
    result = diagnostic.finalize()
    if result['image_count'] != 878:
        raise RuntimeError('Diagnostic image count differs from Test878')
    result['test_accessed'] = True
    return result


def run(run_root, output, device, workers):
    if os.environ.get('ALLOW_TEST_ACCESS') != 'YES':
        raise PermissionError('Set ALLOW_TEST_ACCESS=YES for the one-time Test evaluation')
    assert_no_query_target_interfaces()
    root, manifest, summary, args, head_paths = validate_run(run_root)
    args.device = device
    args.workers = workers
    test_list, names = resolve_test_names(args.train_list, args.val_list)
    output = Path(output).resolve()
    protected = (root, Path(args.bank).resolve(), Path(args.dataset).resolve(),
                 Path(args.checkpoint).resolve().parent)
    if output.exists() or any(output == path or path in output.parents
                              for path in protected):
        raise FileExistsError('Choose a fresh output outside source, bank and dataset')
    common.seed_everything(args.seed)
    bundle, bank = source_and_bank(args, manifest)
    heads, head_hashes = load_heads(
        head_paths, manifest, summary, bank['tokens'].shape[1], bundle['device'])
    head_states = {name: common.state_digest(head) for name, head in heads.items()}
    common.assert_selective_immutable(
        bundle, manifest['code_hash'], manifest['parent_code_hash'])
    validation = common.loader(
        args.dataset, bundle['val'], workers=args.workers,
        seed=args.seed + 3)
    val_result = trained.evaluate(
        bundle, heads, bank, validation, args, detailed=False)
    gate = compare_val_replay(
        val_result['metrics'], summary['final_metrics'])
    common.assert_selective_immutable(
        bundle, manifest['code_hash'], manifest['parent_code_hash'])
    if any(common.state_digest(heads[name]) != head_states[name]
           for name in POSTERIOR_MODES):
        raise RuntimeError('Final head changed during Val reproduction')
    hashes = {str(path): common.digest(path) for path in
              (Path(args.checkpoint), Path(args.train_list), Path(args.val_list),
               test_list, root / 'manifest.json', root / 'summary.json')}
    if common.digest(args.train_list) != manifest['identity']['train_sha256'] or (
            common.digest(args.val_list) != manifest['identity']['val_sha256']):
        raise ValueError('Train/Val list hash changed')
    if output.parent.exists() is False:
        output.parent.mkdir(parents=True, exist_ok=True)
    out = common.new_output(output)
    common.save_json(out / 'val_reproduction.json', gate)
    if not gate['passed']:
        common.save_json(out / 'summary.json', dict(
            complete=False, test_accessed=False,
            phase='Val_reproduction_failed', val_reproduction_file='val_reproduction.json'))
        raise ValueError('Final trained heads failed Val reproduction; no Test loaded')
    # The scored protocol requires every Test target. Absence stops here.
    check_test_files(args.dataset, names)
    meta = dict(format=FORMAT, complete=False, test_accessed=True,
                split='standard_ML/test_set.txt', image_count_expected=878,
                source_run=str(root), source_step=summary['final_step'],
                checkpoint_policy='fixed last checkpoint only, no Test selection',
                test_list_sha256=common.digest(test_list),
                train_val_source_file_sha256=hashes,
                posterior_head_file_sha256=head_hashes,
                bank_file_sha256=manifest['bank_file_sha256'],
                selective_code_hash=manifest['code_hash'],
                retrieval_settings={key: getattr(args, key) for key in
                                    ('topk', 'preselect', 'per_class_cap',
                                     'quality_quantile', 'min_cosine',
                                     'temperature', 'query_chunk',
                                     'posterior_retrieval', 'allow_same_event')},
                evaluation='fixed Train/Val choices; no training or Test model selection',
                test_gt_present=True, val_reproduction_file='val_reproduction.json')
    common.save_json(out / 'manifest.json', meta)
    common.save_json(out / 'summary.json', dict(
        complete=False, test_accessed=True, phase='evaluating_test',
        source_step=summary['final_step'], image_count_expected=878))
    result = evaluate(bundle, bank, heads, args, names)
    common.assert_selective_immutable(
        bundle, manifest['code_hash'], manifest['parent_code_hash'])
    for name, path in head_paths.items():
        if common.digest(path) != head_hashes[name]:
            raise RuntimeError('Final head changed during Test evaluation: ' + name)
        if common.state_digest(heads[name]) != head_states[name]:
            raise RuntimeError('Loaded head state changed during Test evaluation: ' + name)
    for path, original in manifest['bank_file_sha256'].items():
        if common.digest(path) != original:
            raise RuntimeError('Train bank changed during Test evaluation')
    for path, original in hashes.items():
        if common.digest(path) != original:
            raise RuntimeError('Source file changed during Test evaluation: ' + path)
    # Keep per-image confusion for a future paired, image-level CI. No masks or
    # pixel predictions are retained.
    common.save_json(out / 'audit.json', result)
    all_group = result['groups']['all']
    summary_out = dict(
        complete=True, test_accessed=True, source_step=summary['final_step'],
        image_count=result['image_count'],
        metrics=all_group['metrics'], comparisons=all_group['comparisons'],
        per_event={name: {'metrics': row['metrics'],
                          'comparisons': row['comparisons']}
                   for name, row in result['per_event'].items()},
        audit_file='audit.json', checkpoint_policy=meta['checkpoint_policy'])
    common.save_json(out / 'summary.json', summary_out)
    meta['complete'] = True
    common.save_json(out / 'manifest.json', meta)
    print({name: 100 * row['valid_miou']
           for name, row in summary_out['metrics'].items()}, flush=True)
    print('COMPLETE Test878 scored evaluation. No weights updated.', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    options = parser.parse_args()
    if options.workers < 0:
        parser.error('workers must be nonnegative')
    run(options.run_root, options.output, options.device, options.workers)


if __name__ == '__main__':
    main()
