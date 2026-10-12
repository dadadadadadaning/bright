"""Read completed/partial Memory V2 results; never launch inference or training."""
import argparse
import json
from pathlib import Path
from .common import FORMAT
from bda_benchmark.experiments.damageformer_roi_followup_v1.report import render


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('bootstrap must be nonnegative')
    root = Path(args.root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if manifest.get('format') != FORMAT:
        raise ValueError('This reporter expects REGION_MEMORY_RESIDUAL_V2, not an older run')
    print('Phase:', summary.get('phase'), 'optimizer steps:', summary.get('optimizer_steps', 0))
    print('Initial reproduction:', summary.get('baseline_reproduction_pass'))
    print(render(root, pair='nomemory__memory', bootstrap=args.bootstrap))
    print('Changed valid pixel fraction vs frozen hard:', summary.get('changed_pixel_fraction_vs_source'))
    print('Training verification:', summary.get('training_verification'))
    if (root/'retrieval_quality.json').is_file():
        quality = json.loads((root/'retrieval_quality.json').read_text(encoding='utf-8'))
        print('Offline neighbor quality (region counts, NOT pixel mIoU):')
        for name, row in quality['by_class'].items():
            print(name, json.dumps(row, allow_nan=False))
        print(quality['caution'])
    print('Equal parameters are NOT equal support counts/FLOPs: nomemory4 learned slots+self; memory topK+self.')
    print('Residual identity is guaranteed only at initialization, not after learning. off is NOT source_hard.')


if __name__ == '__main__':
    main()
