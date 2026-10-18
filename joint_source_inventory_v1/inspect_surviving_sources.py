#!/usr/bin/env python3
"""Inventory surviving BRIGHT sources without loading weights or changing runs."""
import argparse
import json
import os
from pathlib import Path
import tempfile

DEFAULT_BASE = Path('/root/shared-nvme/BRIGHT-checkpoints')
DEFAULT_NEW = Path('/root/shared-nvme/BRIGHT-master/New')
SKIP = {'.git', '__pycache__', 'conda-envs', 'pfllib-master', 'pfllip-master',
        'dataset', 'datasets', 'pool_debug'}
MAX_JSON = 4 * 1024 * 1024


def status(path):
    p = Path(path)
    try:
        if p.is_file():
            return {'path': str(p), 'status': 'file', 'bytes': p.stat().st_size}
        return {'path': str(p), 'status': 'directory' if p.is_dir() else 'missing'}
    except OSError as exc:
        return {'path': str(p), 'status': 'unreadable', 'error': str(exc)}


def read_record(path, warnings):
    try:
        if path.stat().st_size > MAX_JSON:
            warnings.append('JSON too large; skipped: ' + str(path))
            return {}
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError) as exc:
        warnings.append(str(path) + ': ' + str(exc))
        return {}


def reference(raw):
    if not isinstance(raw, str) or not raw:
        return {'path': None, 'status': 'not_recorded'}
    if not Path(raw).is_absolute():
        return {'path': raw, 'status': 'relative_path_unresolved'}
    return status(raw)


def collect(base, new):
    report = dict(format='BRIGHT_SURVIVING_SOURCE_INVENTORY_V1', training=False,
                  cuda_used=False, weights_loaded=False, source_verified=False,
                  scan_roots=[status(base), status(new)], formal_q_candidates=[],
                  source_protocol_candidates=[], split_metadata_candidates=[],
                  bank_candidates=[], a1_name_candidates=[], weight_candidates=[], warnings=[])
    visited = set()
    for root in (Path(base), Path(new)):
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False,
                                            onerror=lambda e: report['warnings'].append(str(e))):
            dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP
                             and not (Path(directory) / d).is_symlink())
            parent = Path(directory)
            for name in sorted(files):
                path = parent / name
                if str(path) in visited or path.is_symlink():
                    continue
                visited.add(str(path))
                if path.suffix.lower() in ('.pth', '.pt', '.ckpt'):
                    info = status(path)
                    if name in ('paired_bank.pt', 'bank.pt', 'slot_bank.pt'):
                        info['sidecars'] = {n: status(parent / n) for n in
                                            ('manifest.json', 'summary.json', 'protocol.json',
                                             'images.jsonl', 'selected_rows.json')}
                        meta = read_record(parent / 'manifest.json', report['warnings']) if (parent / 'manifest.json').is_file() else {}
                        info['reported_format'] = meta.get('format')
                        info['reported_complete'] = meta.get('complete')
                        report['bank_candidates'].append(info)
                    else:
                        report['weight_candidates'].append(info)
                        if name == 'best_val_miou.pth' or name.lower().startswith('ckpt_damageformer'):
                            report['a1_name_candidates'].append(info)
                    if name == 'last.pth' and parent.name == 'train_baseline' and parent.parent.name == 'formal':
                        summary = read_record(parent / 'summary.json', report['warnings']) if (parent / 'summary.json').is_file() else {}
                        row = dict(info)
                        row['summary_fields'] = {k: summary.get(k) for k in
                            ('format', 'complete', 'arm', 'steps', 'smoke',
                             'full_validation_evidence', 'a1_unchanged', 'downstream_unchanged')}
                        row['evidence_files'] = [status(parent / n) for n in
                                                 ('summary.json', 'manifest.json', 'validation.jsonl')]
                        row['evidence_files'] += [status(parent.parent / 'val_baseline' / n)
                                                  for n in ('summary.json', 'result.json')]
                        report['formal_q_candidates'].append(row)
                is_metadata = name.endswith('.json') and ('source' in name.lower() or 'protocol' in name.lower())
                if not is_metadata:
                    continue
                value = read_record(path, report['warnings'])
                if value.get('format') == 'SOFT_MEMORY_Q_SOURCE_V1':
                    row = {'path': str(path), 'references': {k: reference(value.get(k)) for k in
                           ('q_checkpoint', 'a1_checkpoint', 'e0_metadata_protocol')}}
                    report['source_protocol_candidates'].append(row)
                elif value.get('format') == 'SRAC_HARDQUERY_PROTOCOL_V1':
                    args = value.get('arguments') or {}
                    inputs = value.get('input_paths') or {}
                    if not isinstance(args, dict) or not isinstance(inputs, dict):
                        report['warnings'].append('Invalid metadata fields: ' + str(path))
                        continue
                    row = dict(path=str(path), scope=value.get('scope'),
                               dataset=reference(args.get('dataset')),
                               train_list=reference(inputs.get('train_list')),
                               val_list=reference(inputs.get('val_list')))
                    report['split_metadata_candidates'].append(row)
    report['weight_candidates'].sort(key=lambda r: (-r.get('bytes', 0), r['path']))
    report['notes'] = [
        '候选路径和报告状态不等于来源、权重内容、哈希或CUDA验证通过。',
        '扫描只查找文件/读取小型JSON；不加载权重、不读取图像、不恢复旧实验。',
        '缺少旧成员库可改为Train重新选成员，但现有chain_v1不支持此模式。',
        'A1单列仅按常见文件名筛选；元数据仅识别名称含source或protocol的JSON，不解包备份。',
        '没有找到文件仅表示本次扫描范围内未发现，不代表所有备份都不存在。',
    ]
    return report


def render(report):
    lines = ['BRIGHT 基础文件盘点：未训练、未使用CUDA，所有结果仅为候选。']
    for row in report['scan_roots']:
        lines.append('扫描目录: ' + row['path'] + ' [' + row['status'] + ']')
    sections = (
        ('正式Q候选及旁边的证据文件', 'formal_q_candidates'),
        ('常见A1文件名候选（尚未核对Q记录的A1哈希）', 'a1_name_candidates'),
        ('Q来源协议候选及其指向的文件', 'source_protocol_candidates'),
        ('数据/Train/Val划分元数据候选', 'split_metadata_candidates'),
        ('成员库/父库/双槽库候选', 'bank_candidates'),
        ('其他权重候选（按大小，最多显示40个；含A1候选）', 'weight_candidates'),
    )
    for title, key in sections:
        rows = report[key]
        lines.append('\n' + title + ': ' + str(len(rows)))
        for row in rows[:40]:
            lines.append(json.dumps(row, ensure_ascii=False))
        if len(rows) > 40:
            lines.append('其余见 inventory.json。')
    lines.append('\n扫描警告: ' + str(len(report['warnings'])))
    lines.extend(report['warnings'][:10])
    lines.extend(report['notes'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--new-root', type=Path, default=DEFAULT_NEW)
    args = parser.parse_args()
    # Write only a new inventory directory; all old artifacts remain untouched.
    if not args.new_root.is_dir():
        parser.error('--new-root must be an existing directory')
    report = collect(args.checkpoint_root, args.new_root)
    output = Path(tempfile.mkdtemp(prefix='source_inventory_', dir=args.new_root))
    (output / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    text = render(report)
    (output / 'REPORT.txt').write_text(text, encoding='utf-8')
    print(text)
    print('REPORT:', output / 'REPORT.txt')
    print('JSON:', output / 'inventory.json')


if __name__ == '__main__':
    main()
