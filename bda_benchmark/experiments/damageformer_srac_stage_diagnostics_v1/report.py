"""Read existing diagnostic results without GPU or fresh inference."""
import argparse
import json
from pathlib import Path


def run(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    complete = summary.get('complete') is True and summary.get('gates_passed') is True
    path = root/'audit.json'
    if not path.exists():
        print('No completed audit.json. Inspect failure/partial output; do not interpret intervention gains.')
        return
    audit = json.loads(path.read_text(encoding='utf-8'))
    if not complete:
        print('FAILED/INCOMPLETE gates: values below are debugging evidence, NOT valid method comparisons.')
    elif not summary.get('full_validation_evidence'):
        print('SMOKE/SUBSET ONLY: not full Val and not formal reproduction evidence.')
    metrics = audit.get('metrics', {})
    print('\nMetrics (percent; compare matched paths, not different snapshots):')
    for name, value in metrics.items():
        score = value.get('valid_miou')
        prf = [value.get(key, [None]*4)[2] for key in ('precision', 'recall', 'f1')]
        display = ['null' if x is None else f'{100*x:.4f}' for x in prf]
        print(f'{name}: FG mIoU={100*score:.4f}; Damaged P/R/F1='+ '/'.join(display))
    for key, row in audit.get('class_transitions', {}).items():
        c = row['per_class']['Damaged']
        print(f"{key}: Damaged rescue={c['rescue']} harm={c['harm']} netTP={c['net_tp']} deltaFP={c['delta_fp']}")
    margin = audit.get('damaged_margin_sums')
    if margin:
        mass = margin['damaged_area_feature_cells']
        print('\nGT-area-weighted D-minus-I margin (feature grid; not independent objects):')
        print(json.dumps({**margin,
              'before_mean': margin['before_margin_sum']/mass if mass else None,
              'after_mean': margin['after_margin_sum']/mass if mass else None}, indent=2))
    for key, row in audit.get('transitions', {}).items():
        c = row['classes'][2]
        print(f"{key}: Damaged rescue={c['rescue']} harm={c['harm']} "
              f"netTP={c['net_correct']} deltaFP={c.get('delta_false_positive', 'see confusion matrices')}")
    cohorts = audit.get('difficult_cohorts', {})
    for arm, rows in cohorts.items():
        row = rows['damaged_j1_wrong']
        count = row['count']
        supported = row['any_supported_pixels']
        positive = row['evidence_damaged']
        print(f'\n{arm}: fixed GT-Damaged/J1-wrong cohort={count}')
        print('  Any support / cohort:', supported/count if count else None)
        print('  Mean supported Q mass:', row['support_mass_sum']/count if count else None)
        print('  Evidence predicts Damaged / supported:', positive/supported if supported else None)
        print('  Final rescue / cohort:', row['final_damaged']/count if count else None)
        print('  Final rescue / evidence predicts Damaged:',
              row['evidence_damaged_and_final_damaged']/positive if positive else None)
        print('  Intact/Destroyed damage tradeoffs are in all-class transitions and confusion matrices.')
    for key in ('cohorts', 'memory_cohorts', 'retrieval'):
        if key in audit:
            print('\n'+key+':')
            print(json.dumps(audit[key], indent=2, ensure_ascii=False))
    reproduction = audit.get('reproduction')
    if reproduction:
        print('\nReproduction passed:', reproduction.get('passed'))
        checks = reproduction.get('checks', [])
        if isinstance(checks, dict):
            checks = [dict(name=name, **value) for name, value in checks.items()]
        for check in checks:
            if check.get('passed') is False:
                print('  FAILED:', check)
    inventory_path = root/'inventory.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
        print('\nBank inventory:', json.dumps(inventory.get('bank'), ensure_ascii=False))
        print('All-Train shards scanned:', inventory.get('train_shards_scanned'))
        print('Read inventory.json for selection/eligibility details; this is not pixel kNN accuracy.')
    if cohorts:
        print('\nPer-event metrics/cohorts/retrieval and all-class paired counts: audit.json.')
        print('Per-image paired observations: images.jsonl. Memory audit does not add ROI/boundary strata.')
    else:
        print('\nPer-event, ROI-error, boundary/interior, size-proxy and paired counts: audit.json.')
        print('Per-image paired observations: per_image.jsonl.')
    print('GT is diagnostic only. Inference sensitivity does not prove retrained benefit. No Test selection.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    run(p.parse_args().root)


if __name__ == '__main__':
    main()
