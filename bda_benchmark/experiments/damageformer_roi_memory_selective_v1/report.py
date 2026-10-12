"""Read-only report for selective retrieval and posterior-gating experiments."""
import argparse
import json
from pathlib import Path


def pct(value):
    return 'N/A' if value is None else f'{100 * value:.4f}%'


def _metric_line(name, row):
    return (
        f'{name}: FG mIoU={pct(row["valid_miou"])}; '
        f'D P/R/F1={pct(row["precision"][2])}/'
        f'{pct(row["recall"][2])}/{pct(row["f1"][2])}; '
        f'background false positives={row["background_to_building"]}')


def _support_line(name, row):
    compact = dict(
        coverage=row.get('support_coverage'),
        mean_neighbors=row.get('mean_valid_neighbors'),
        selected_quality=row.get('selected_quality_means'),
        by_class=row.get('by_class'))
    for key in ('posterior_confidence_mean', 'posterior_entropy_mean',
                'gate_mean', 'strength_mean',
                'positive_gate_fraction', 'posterior_class_change_rate',
                'region_prediction_change_rate'):
        if key in row:
            compact[key] = row[key]
    return name + ' support: ' + json.dumps(
        compact, ensure_ascii=False, allow_nan=False)


def render(root):
    root = Path(root)
    summary = json.loads(
        (root / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads(
        (root / 'manifest.json').read_text(encoding='utf-8'))
    lines = [
        f'format: {manifest.get("format")}',
        f'complete: {summary.get("complete")}; '
        f'smoke: {summary.get("smoke")}; '
        f'Test accessed: {summary.get("test_accessed")}',
    ]
    if manifest.get('format') == 'ROI_MEMORY_SELECTIVE_AUDIT_V1':
        for name, row in summary['cases'].items():
            lines.append(
                f'{name}: support={pct(row["support_coverage"])}; '
                f'neighbors/query={row["mean_valid_neighbors"]:.4f}; '
                f'cosine={row["mean_query_neighbor_cosine"]:.5f}; '
                f'purity={row["selected_quality_means"]["purity"]:.5f}; '
                f'entropy={row["selected_quality_means"]["entropy"]:.5f}; '
                f'posterior confidence/entropy='
                f'{row["mean_posterior_confidence"]:.5f}/'
                f'{row["mean_posterior_entropy"]:.5f}; '
                f'kNN acc={pct(row["knn_dominant_accuracy"])}; '
                f'softCE={row["knn_soft_cross_entropy"]:.6f}; '
                f'supported/hard-labelled={row["knn_supported_queries"]}/'
                f'{row["query_labelled_hard"]}')
            lines.append(name + ' class coverage: ' + json.dumps(
                row['by_class'], ensure_ascii=False, allow_nan=False))
        normal = summary['cases']['quality_adaptive']
        for action in ('quality_adaptive_random',
                       'quality_adaptive_shuffle_labels'):
            other = summary['cases'][action]
            lines.append(
                f'quality_adaptive minus {action}: kNN accuracy '
                f'{100 * (normal["knn_dominant_accuracy"] - other["knn_dominant_accuracy"]):+.4f} pp; '
                f'softCE '
                f'{normal["knn_soft_cross_entropy"] - other["knn_soft_cross_entropy"]:+.6f}')
        lines += [
            'Interpretation gate: semantic retrieval should beat random and '
            'shuffled-label controls before posterior training is promoted.',
            'This is retrieval evidence, not final pixel mIoU.',
            'No Test used.',
        ]
        return '\n'.join(lines)
    metrics = summary.get('final_metrics', {})
    if (root / 'audit_final.json').is_file():
        audit = json.loads(
            (root / 'audit_final.json').read_text(encoding='utf-8'))
        detailed = audit['groups']['all']['metrics']
    else:
        detailed = metrics
    lines.append('stage: ' + str(summary.get('stage', manifest.get('stage'))))
    if 'baseline_reproduction_pass' in summary:
        lines.append(
            'blocking baseline reproduction passed: ' +
            str(summary.get(
                'blocking_baseline_reproduction_pass',
                summary.get('baseline_reproduction_pass'))) +
            '; gate status: ' +
            str(summary.get('baseline_gate_status', '<legacy summary>')) +
            '; all metric cells passed: ' +
            str(summary.get('all_baseline_metric_cells_pass')) +
            '; recorded nonblocking drift: ' +
            json.dumps(summary.get('nonblocking_baseline_drift', []),
                       ensure_ascii=False, allow_nan=False))
    for name, row in detailed.items():
        lines.append(_metric_line(name, row))
    for name, row in summary.get('support', {}).items():
        lines.append(_support_line(name, row))
    gains = summary.get('same_step_gains_vs_control', {})
    if gains:
        control = ('nomemory' if summary.get('stage') == 'retrieval'
                   else 'query_only')
        lines.append('Same-step gains versus ' + control + ': ' + json.dumps(
            {name: 100 * value for name, value in gains.items()},
            ensure_ascii=False, allow_nan=False) + ' pp')
    lines += [
        'Retrieval stage isolates support policy with matched residual heads.',
        'Posterior stage isolates explicit labels and reliability features '
        'with matched heads; query_only is the no-retrieval capacity control.',
        'random/shuffle_labels/off are inference interventions, not '
        'independently trained baselines.',
        'Best checkpoints are secondary; compare final same-step results.',
        'No Test used.',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(render(args.root))


if __name__ == '__main__':
    main()
