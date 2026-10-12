"""Render the most decision-relevant retrieval-audit results."""
import argparse
import json
from pathlib import Path


def pct(value):
    return 'N/A' if value is None else f'{100 * value:.4f}%'


def transition(row, source, target):
    return row['per_class'][source]['posterior_to_class'][target]


def render(root):
    root = Path(root)
    data = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    lines = [
        (f"complete: {data['complete']} smoke: {data['smoke']} "
         f"pipeline_smoke: {data.get('pipeline_smoke', False)} "
         f"limited_images: {data.get('limited_images', 0)} "
         f"images: {data['image_count']}"),
        'These are retrieval diagnostics, not pixel mIoU.',
        'Posterior accuracy/transitions are conditional on any valid support; '
        'same@K counts unsupported queries as misses.',
        '',
        ('case                                 coverage  post-acc  same@K   D same@K  '
         'X same@K  D->I      X->D      X->I      same-group'),
    ]
    for name, case in data['cases'].items():
        row = case['overall']
        damaged = row['per_class']['Damaged']
        destroyed = row['per_class']['Destroyed']
        lines.append(
            f'{name:36s} '
            f'{pct(row["support_coverage"]):>9s} '
            f'{pct(row["posterior_accuracy"]):>9s} '
            f'{pct(row["same_class_hit_at_k"]):>9s} '
            f'{pct(damaged["same_class_hit_at_k"]):>10s} '
            f'{pct(destroyed["same_class_hit_at_k"]):>10s} '
            f'{pct(transition(row, "Damaged", "Intact")):>9s} '
            f'{pct(transition(row, "Destroyed", "Damaged")):>9s} '
            f'{pct(transition(row, "Destroyed", "Intact")):>9s} '
            f'{pct(row["same_predicted_group_neighbor_fraction"]):>11s}')
    lines += ['', '=== GT-purity >= 0.95 diagnostic ===']
    for name, case in data['cases'].items():
        row = case['purity_ge_095']
        lines.append(
            f'{name:36s} post-acc={pct(row["posterior_accuracy"])} '
            f'D same@K={pct(row["per_class"]["Damaged"]["same_class_hit_at_k"])} '
            f'D->I={pct(transition(row, "Damaged", "Intact"))}')
    for case_name in ('adaptive_exclude_event',
                      'adaptive_same_predicted_group'):
        primary = data['cases'][case_name]
        lines += ['', f'=== {case_name} by predicted group ===']
        for name, row in primary['by_predicted_group'].items():
            lines.append(
                f'{name}: queries={row["queries"]} '
                f'coverage={pct(row["support_coverage"])} '
                f'post-acc={pct(row["posterior_accuracy"])} '
                f'same@K={pct(row["same_class_hit_at_k"])} '
                f'D same@K={pct(row["per_class"]["Damaged"]["same_class_hit_at_k"])}')
    primary = data['cases']['adaptive_exclude_event']
    lines += ['', '=== adaptive_exclude_event per event ===']
    for name, row in primary['per_event'].items():
        lines.append(
            f'{name:24s}: post-acc={pct(row["posterior_accuracy"])} '
            f'D same@K={pct(row["per_class"]["Damaged"]["same_class_hit_at_k"])} '
            f'D->I={pct(transition(row, "Damaged", "Intact"))}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(render(args.root))


if __name__ == '__main__':
    main()

