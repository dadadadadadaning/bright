"""Read-only quality-bank audit and residual-training report."""
import argparse
import json
from pathlib import Path

from bda_benchmark.experiments.damageformer_roi_followup_v1.report import (
    paired_bootstrap)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.report import contrast


def pct(value):
    return 'N/A' if value is None else f'{100 * value:.4f}%'


def render(root, pair=None, bootstrap=0):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    lines = [f'format: {manifest.get("format")}',
             f'complete: {summary.get("complete")}; smoke: {summary.get("smoke")}; Test: {summary.get("test_accessed")}']
    if manifest.get('format') == 'ROI_MEMORY_SUPPORT_AUDIT_V2':
        for name, row in summary['cases'].items():
            lines.append(
                f'{name}: support={pct(row["support_coverage"])}; '
                f'neighbors/query={row["mean_valid_neighbors"]:.4f}; '
                f'cosine={row["mean_query_neighbor_cosine"]:.5f}; '
                f'purity={row["selected_quality_means"]["purity"]:.5f}; '
                f'entropy={row["selected_quality_means"]["entropy"]:.5f}; '
                f'kNN acc={pct(row["knn_dominant_accuracy"])}; '
                f'kNN softCE={row["knn_soft_cross_entropy"]:.6f}; '
                f'mass-weighted acc/CE={pct(row["knn_mass_weighted_accuracy"])}/'
                f'{row["knn_mass_weighted_soft_cross_entropy"]:.6f}; '
                f'kNN supported/hard-labelled={row["knn_supported_queries"]}/'
                f'{row["query_labelled_hard"]} '
                f'({pct(row["knn_hard_query_coverage"])}); '
                f'all labelled queries={row["query_labelled_all"]}')
            lines.append(name + ' class neighbors/coverage: ' + json.dumps(
                row['by_class'], ensure_ascii=False, allow_nan=False))
        normal = summary['cases']['balanced_hard']
        for action in ('balanced_hard_random', 'balanced_hard_shuffle_labels'):
            other = summary['cases'][action]
            lines.append(
                f'balanced_hard versus {action}: '
                f'kNN accuracy {100 * (normal["knn_dominant_accuracy"] - other["knn_dominant_accuracy"]):+.4f} pp; '
                f'softCE {normal["knn_soft_cross_entropy"] - other["knn_soft_cross_entropy"]:+.6f}')
        lines += [
            'Gate: if normal neighbors do not beat random/shuffled labels on predeclared label-only evidence, stop before residual training.',
            'This audit is retrieval screening, not final pixel mIoU; Val GT is used only after feature-only ranking.',
        ]
        return '\n'.join(lines)
    metrics = summary.get('final_metrics', {})
    if (root / 'audit_final.json').is_file():
        audit = json.loads((root / 'audit_final.json').read_text(encoding='utf-8'))
        detailed_metrics = audit['groups']['all']['metrics']
    else:
        audit, detailed_metrics = None, metrics
    for name, row in detailed_metrics.items():
        lines.append(
            f'{name}: FG mIoU={pct(row["valid_miou"])}; '
            f'D P/R/F1={pct(row["precision"][2])}/{pct(row["recall"][2])}/{pct(row["f1"][2])}; '
            f'background false positives={row["background_to_building"]}')
    for name, row in summary.get('support', {}).items():
        lines.append(name + ' support quality/counts: ' + json.dumps(
            dict(coverage=row['support_coverage'],
                 mean_neighbors=row['mean_valid_neighbors'],
                 by_class=row['by_class'],
                 selected_quality_means=row['selected_quality_means']),
            ensure_ascii=False, allow_nan=False))
    lines += [
        f'QUALITY global_hard-global_all: {100 * summary.get("final_global_hard_minus_global_all", 0):+.4f} pp',
        f'BALANCE balanced_hard-global_hard: {100 * summary.get("final_balanced_hard_minus_global_hard", 0):+.4f} pp',
        f'MEMORY global_all-nomemory: {100 * summary.get("final_global_all_minus_nomemory", 0):+.4f} pp',
    ]
    if audit:
        pairs = audit['groups']['all']['comparisons']
        if pair is None:
            pair = next((candidate for candidate in
                ('global_all__global_hard', 'global_hard__balanced_hard',
                 'nomemory__global_all', 'nomemory__balanced_hard')
                if candidate in pairs), None)
        lines.append('Available pairs: ' + ', '.join(pairs))
        if pair not in pairs:
            raise ValueError('Choose an available pair with --pair')
        first, second = pair.split('__')
        lines.append(f'PRIMARY {second} minus {first}: '
                     f'{100 * (detailed_metrics[second]["valid_miou"] - detailed_metrics[first]["valid_miou"]):+.4f} pp')
        for label, row in contrast(detailed_metrics[first], detailed_metrics[second]).items():
            lines.append(label + ': ' + json.dumps(row, allow_nan=False))
        if bootstrap:
            if summary.get('complete') is not True or summary.get('smoke') or audit['image_count'] != 450:
                raise ValueError('Bootstrap requires complete formal Val450')
            lines.append('Paired validation uncertainty: ' + json.dumps(
                paired_bootstrap(audit['per_image'], pair, bootstrap)))
    lines += [
        'M0/M1/M2 are same-step residual controls; best checkpoints are secondary.',
        'hard_count>0 means non-ghost occupancy, NOT high purity or a building instance.',
        'balanced_hard is label-aware Train-support stratification; query GT never enters retrieval.',
        'random/shuffle/off are inference interventions, not independently trained baselines; off is not source_hard after training.',
        'No Test used.',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pair')
    parser.add_argument('--bootstrap', type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('bootstrap must be nonnegative')
    print(render(args.root, args.pair, args.bootstrap))


if __name__ == '__main__':
    main()
