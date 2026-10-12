"""Assemble recorded, unmodified A1 Test crops into comparison figures."""
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'research_reports/A1_test_lookup_20261009_v2'
OUT = ROOT/'research_reports/A1_test_assembled_20261009_v2'
KINDS = ['RGB', 'SAR', 'GT', 'DamageFormer', 'Ours_A1']
TITLES = ['Pre-event RGB', 'Post-event SAR', 'Ground truth', 'DamageFormer', 'A1']
COLORS = ['#ffffff', '#46b579', '#e4bd8b', '#b64645']


def make_figure(rows, title):
    n = len(rows)
    width, height = 13.2, 2.0 + 2.65*n
    fig = plt.figure(figsize=(width, height), facecolor='white')
    fig.text(.035, 1-.32/height, title, fontsize=17, weight='bold', color='#223344')
    fig.text(.035, 1-.62/height,
             'Recorded Test predictions | Same spatial crop within each row',
             fontsize=10, color='#526171')
    left, gap, side = .035, .012, .176
    image_h = side*width/height
    for col, name in enumerate(TITLES):
        fig.text(left+col*(side+gap)+side/2, 1-.99/height, name,
                 fontsize=12, ha='center', color='#223344')
    for row_idx, row in enumerate(rows):
        sid = row['sample_id']
        x, y, w, h = row['xywh']
        top = 1-(1.4+2.65*row_idx)/height
        fig.text(left, top+.07/height,
                 f'{sid}   |   crop: {w} x {h} px; origin: ({x}, {y})',
                 fontsize=10, color='#526171')
        for col, kind in enumerate(KINDS):
            ax = fig.add_axes([left+col*(side+gap), top-image_h, side, image_h])
            with Image.open(SOURCE/'ppt_crops'/f'{sid}_{kind}.png') as im:
                if im.size != (w, h):
                    raise ValueError(f'Crop dimensions mismatch: {sid}, {kind}, {im.size}')
                ax.imshow(im.copy(), interpolation='nearest', cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#c8cdd2')
                spine.set_linewidth(.45)
    fig.legend([Patch(facecolor=c, edgecolor='#888888', linewidth=.5) for c in COLORS],
               ['Background', 'Intact', 'Damaged', 'Destroyed'],
               loc='lower center', bbox_to_anchor=(.5,.28/height), ncol=4,
               frameon=False, fontsize=10)
    fig.text(.035,.13/height,
             'Selected crops, not full-Test metrics. A1 is the recorded checkpoint, not the proposed region-memory model.',
             fontsize=8, color='#526171')
    return fig


def main():
    OUT.mkdir(exist_ok=False)
    rows = json.loads((SOURCE/'selection_test.json').read_text(encoding='utf-8'))['samples']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    with PdfPages(OUT/'A1_Test_comparison_6pages.pdf') as pdf:
        for i in range(0, len(rows), 3):
            fig = make_figure(rows[i:i+3], f'BRIGHT Test comparison | {i//3+1:02d}')
            fig.savefig(OUT/f'A1_Test_comparison_{i//3+1:02d}.png', dpi=180)
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
    fig = make_figure(rows, 'BRIGHT Test comparison | 18 recorded cases')
    fig.savefig(OUT/'A1_Test_all18_overview.png', dpi=130)
    plt.close(fig)
    (OUT/'provenance.json').write_text(json.dumps({
        'source': str(SOURCE), 'sample_count': len(rows),
        'sample_ids': [r['sample_id'] for r in rows],
        'columns': KINDS, 'crop_pixels_modified': False,
        'note': 'Existing Test outputs assembled; no training or inference performed.'
    }, indent=2), encoding='utf-8')
    archive = OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f.name)
    print(archive)


if __name__ == '__main__':
    main()
