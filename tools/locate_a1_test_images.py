"""Locate the recorded A1 Test cases, or export existing local presentation assets.

No training or inference. CSV paths distinguish expected paths from existing files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'bda_benchmark/dataset')
    p.add_argument('--predictions', type=Path,
                   default=Path('/root/shared-nvme/BRIGHT-test-sixmodels-v2/predictions'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--sample-id', help='Optional one exact sample ID from the recorded selection')
    p.add_argument('--record', type=Path,
                   default=ROOT/'research_reports/test_comparison_ppt/build/data.json')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--export-local-assets', action='store_true')
    a = p.parse_args()
    record = json.loads(a.record.read_text(encoding='utf-8-sig'))
    model = next(r for r in record['records'] if r['name'] == 'Ours_A1')
    expected = model['sha256']
    print('Recorded architecture:', model['factory'])
    print('Recorded SHA256:', expected)
    if a.checkpoint:
        h = hashlib.sha256()
        with a.checkpoint.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        print('Current SHA256: ', h.hexdigest())
        if h.hexdigest() != expected:
            raise SystemExit('STOP: checkpoint differs from the recorded A1. Do not reuse its predictions as this checkpoint output.')
    valid = set((ROOT/'bda_benchmark/dataset/splitname/standard_ML/test_set.txt').read_text().split())
    rows = [r for r in record['rows'] if not a.sample_id or r['sample_id'] == a.sample_id]
    if not rows:
        raise SystemExit('Sample ID absent from the recorded presentation selection.')
    if any(r['sample_id'] not in valid for r in rows):
        raise SystemExit('Selection contains non-Test IDs.')
    a.output.mkdir(parents=True, exist_ok=False)
    # Reusable inputs for the existing strict inference entry point.
    (a.output/'selection_test.json').write_text(json.dumps(
        {'split': 'test', 'samples': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    if a.checkpoint:
        model['checkpoint'] = str(a.checkpoint.resolve())
    model['expected_sha256'] = expected
    (a.output/'a1_models.json').write_text(json.dumps([model], indent=2), encoding='utf-8')
    entries = []
    for r in rows:
        sid = r['sample_id']
        paths = {
            'rgb': a.data/'pre-event'/f'{sid}_pre_disaster.tif',
            'sar': a.data/'post-event'/f'{sid}_post_disaster.tif',
            'gt': a.data/'target'/f'{sid}_building_damage.tif',
            'prediction_color': a.predictions/'Ours_A1'/f'{sid}_color.png',
            'prediction_labels': a.predictions/'Ours_A1'/f'{sid}_labels.png',
        }
        row = dict(sample_id=sid, split='test', crop_xywh=json.dumps(r['xywh']))
        row['exported_ppt_a1_crop'] = str(a.output/'ppt_crops'/f'{sid}_Ours_A1.png') if a.export_local_assets else ''
        row['exported_full_a1_color'] = str(a.output/'full_predictions'/f'{sid}_color.png') if a.export_local_assets else ''
        for name, path in paths.items():
            row[name] = str(path)
            row[name+'_exists'] = path.is_file()
        entries.append(row)
        print(sid, 'crop=', r['xywh'], 'prediction_exists=', paths['prediction_color'].is_file())
    with (a.output/'test_image_index.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    if a.export_local_assets:
        crops = a.output/'ppt_crops'
        full = a.output/'full_predictions'
        crops.mkdir()
        full.mkdir()
        with zipfile.ZipFile(ROOT/'BRIGHT_test_sixmodels_images.zip') as z:
            records = json.loads(z.read('predictions/run_manifest.json'))
            source = next(r for r in records if r['name'] == 'Ours_A1')
            if source['sha256'] != expected or source['status'] != 'complete' or source['split'] != 'test':
                raise RuntimeError('Archive provenance mismatch')
            for r in rows:
                sid = r['sample_id']
                for suffix in ('color', 'labels'):
                    name = f'{sid}_{suffix}.png'
                    (full/name).write_bytes(z.read(f'predictions/Ours_A1/{name}'))
                for kind in ('RGB', 'SAR', 'GT', 'DamageFormer', 'Ours_A1'):
                    name = f'{sid}_{kind}.png'
                    shutil.copy2(ROOT/'research_reports/test_comparison_ppt/build/assets'/name, crops/name)
    print('Output:', a.output.resolve())
    print('This is a selected Test-case lookup, not full-Test evaluation.')


if __name__ == '__main__':
    main()
