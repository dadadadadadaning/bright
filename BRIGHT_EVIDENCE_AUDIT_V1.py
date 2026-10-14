#!/usr/bin/env python3
"""Read BRIGHT JSON evidence without Torch, CUDA, checkpoint loading or training.

This is a report inventory, not an independent re-certification of experiments.
Only new audit files are written. Python 3.10+, standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

VERSION = "BRIGHT_EVIDENCE_AUDIT_V1"
CN = timezone(timedelta(hours=8))
BASE = Path('/root/shared-nvme/BRIGHT-checkpoints')
OUTPUT = Path('/root/shared-nvme/BRIGHT-master/New/evidence_records')
NAMES = {'summary.json', 'result.json', 'protocol.json', 'workflow_state.json',
         'pipeline_state.json', 'chain_state.json', 'sar_verification.json', 'report.json'}
SKIP_DIRS = {'predictions', 'checkpoints', 'shards', '__pycache__', '.git',
             'pool_debug', 'captures', 'cache', 'banks', 'tensor_dumps'}
JOINT = {'SRAC_REGION_MEMORY_JOINT_TRAIN_V1': ('r0', 'rt', 'rf'),
         'SRAC_SOFT_MEMORY_JOINT_V1': ('a', 'b', 'c', 'd')}
STATUS = {
 'failed': '失败记录', 'smoke': 'smoke／小样本检查',
 'formal': '报告声明正式 Val450 完成',
 'intermediate': '完整 Val450 中间评估', 'initial': '训练前完整 Val450 检查',
 'revalidation': '完整 Val450 断点重新验证',
 'diagnostic': '完整 Val450 诊断（非 Reader 训练收益）',
 'train_done': '训练及终检完成；独立验证另查',
 'stage_done': '阶段完成（不等于正式成绩）',
 'pipeline_done': '流水线声明完成（各组另列）',
 'partial': '部分／进度记录；存活状态未检测',
 'review': '证据不足或矛盾，需核对',
 'other_split': 'Test／非 Val450，单独记录',
 'unreadable': '无法稳定读取，需稍后重试',
}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def json_bytes(raw):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    obj = json.loads(raw, parse_constant=reject)
    if not isinstance(obj, dict):
        raise ValueError('JSON root is not an object')
    return obj


def stable_read(path, max_bytes):
    """Capture small source metadata and a digest; never read model weights."""
    before = path.stat()
    if before.st_size > max_bytes:
        raise ValueError('report exceeds configured size limit')
    with path.open('rb') as f:
        raw = f.read(max_bytes + 1)
    after = path.stat()
    if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
        raise ValueError('report changed during read')
    if len(raw) > max_bytes:
        raise ValueError('report exceeds configured size limit')
    return json_bytes(raw), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw), modified_at=datetime.fromtimestamp(after.st_mtime, CN).isoformat())


def tail_json(path, max_bytes=1024*1024):
    """Read the last complete journal row, ignoring an in-flight partial line."""
    try:
        with path.open('rb') as f:
            size = f.seek(0, 2)
            start = max(0, size-max_bytes)
            f.seek(start)
            raw = f.read(max_bytes)
        lines = raw.splitlines(keepends=True)
        if start and lines:
            lines = lines[1:]
        for line in reversed(lines):
            if not line.endswith((b'\n', b'\r')):
                continue
            try:
                return json_bytes(line), {'path': str(path), 'tail_only': True,
                    'modified_at':datetime.fromtimestamp(path.stat().st_mtime,CN).isoformat(),
                    'row_sha256': hashlib.sha256(line).hexdigest()}
            except (ValueError, UnicodeError):
                continue
    except OSError:
        pass
    return None, None


def scan(base, max_depth, max_files):
    warnings, files = [], []
    if not base.is_dir():
        return [], ['结果根目录不存在；这不证明实验未执行：' + str(base)]
    for folder, dirs, names in os.walk(base, followlinks=False, onerror=lambda e: warnings.append(str(e))):
        p = Path(folder)
        depth = len(p.relative_to(base).parts)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (p/d).is_symlink())
        if depth == 0 and base.name == 'BRIGHT-checkpoints':
            dirs[:] = [d for d in dirs if 'SRAC' in d.upper() or 'JOINT' in d.upper()]
        if depth >= max_depth:
            if dirs:
                warnings.append('扫描深度上限：' + str(p))
            dirs[:] = []
        for name in sorted(set(names) & NAMES):
            path = p/name
            if path.is_symlink():
                warnings.append('未跟随报告符号链接：' + str(path))
                continue
            files.append(path)
            if len(files) >= max_files:
                warnings.append('达到报告数量上限；本次扫描不完整')
                return files, warnings
    return files, warnings


def family(path, data):
    fmt = str(data.get('format', ''))
    for word, label in (('SOFT_MEMORY_JOINT','soft＋Memory 新四组'),
        ('SAR_SOFT','SAR-Q × 路由'), ('Q_FACTORIAL','SAR／Q 匹配实验'),
        ('BUILDING_ROUTE','建筑 soft／hard 路由'), ('REGION_HEAD','P／R／T 区域头'),
        ('MEMORY_REBUILD','V3 FP32 建库与检索诊断'), ('REGION_VAL_FORMAL','区域读出评估'),
        ('HARDQUERY_PROTOCOL','E0／冻结基准')):
        if word in fmt:
            return label
    text = str(path).upper() + ' ' + str(data.get('format', '')).upper()
    for word, label in (
        ('SOFT_MEMORY_JOINT', 'soft＋Memory 新四组'),
        ('SAR_SOFT', 'SAR-Q × 路由'), ('SAR_FULL_CHAIN', 'SAR 完整流水线'),
        ('Q_FACTORIAL', 'SAR／Q 匹配实验'), ('SAR_Q_MATCHED', 'SAR／Q 匹配实验'),
        ('REGION_HEAD', 'P／R／T 区域头'), ('PRT_', 'P／R／T 区域头'),
        ('BUILDING_ROUTE', '建筑 soft／hard 路由'),
        ('MEMORY_REBUILD', 'V3 FP32 建库与检索诊断'),
        ('MEMORY_KEY', '旧 Memory key 比较'), ('MEMORY_REPLAY', 'Memory 数值诊断'),
        ('JOINT_LIVE', 'R0／RT／RF 恢复续训'), ('JOINT_POOL', 'R0／RT／RF 恢复诊断'),
        ('REGION_MEMORY_JOINT', 'R0／RT／RF 联合训练'),
        ('SNAPSHOT_JOINT', '旧 J1／J2／J3'), ('REGION_TEST', '区域读出评估'),
        ('E0', 'E0／冻结基准'), ('STAGE_DIAGNOSTICS', '阶段诊断'),
    ):
        if word in text:
            return label
    return '其他 SRAC 记录'


def bad_checks(data):
    errors = []
    def visit(obj, label):
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if value is False:
                errors.append(label + '.' + key)
            elif isinstance(value, dict):
                visit(value, label + '.' + key)
    for key in ('immutable', 'integrity', 'final_integrity', 'validation'):
        visit(data.get(key), key)
    for key in ('reproduction', 'e0_metric_comparison', 'initial_parameters', 'initial_a1_reproduction', 'final_a1_reproduction'):
        if isinstance(data.get(key), dict) and data[key].get('passed') is False:
            errors.append(key + '.passed')
    guard = data.get('guard_report', {})
    if isinstance(guard, dict) and (guard.get('restored') is False or guard.get('failures')):
        errors.append('guard_report')
    return errors


def all_true(value):
    return isinstance(value, dict) and bool(value) and all(v is True for v in value.values())


def passed(data, key):
    return isinstance(data.get(key), dict) and data[key].get('passed') is True


def final_joint_ok(owner, final, arms):
    if not isinstance(owner, dict) or not isinstance(final, dict):
        return False
    updates = owner.get('updates_per_arm', {})
    steps_ok = (isinstance(updates, dict) and all(updates.get(a) == 5000 for a in arms)) or owner.get('checkpoint_step') == 5000
    guard = owner.get('guard_report', {})
    final_guard = final.get('guard_report', {})
    def guard_ok(g):
        return isinstance(g, dict) and g.get('restored') is True and g.get('failures') == [] and not g.get('restore_conflicts')
    return bool(steps_ok and owner.get('complete') is True and owner.get('execution_complete') is True
        and owner.get('full_validation_evidence') is True and owner.get('smoke') is False and owner.get('test_accessed') is False
        and not owner.get('error') and not owner.get('failure') and not bad_checks(owner)
        and all_true(owner.get('integrity')) and guard_ok(guard)
        and final.get('image_count') == 450 and final.get('execution_complete') is True
        and final.get('full_validation_evidence') is True and final.get('primary_endpoint') is True
        and final.get('final_integrity_pending') is False and final.get('saved_predictions_reopened') is True
        and final.get('smoke') is False and final.get('test_accessed') is False
        and all_true(final.get('final_integrity')) and guard_ok(final_guard)
        and isinstance(final.get('metrics'),dict) and set(arms) <= final['metrics'].keys()
        and owner.get('metrics') == final.get('metrics'))


def normalize_metric(row, *, percent=False):
    if not isinstance(row, dict):
        return None, None
    key = next((k for k in ('foreground_miou', 'valid_miou', 'foreground_miou_pct') if k in row), None)
    if key is None:
        return None, None
    score = row[key]
    scale = 1.0 if percent or key.endswith('_pct') else 100.0
    if not finite(score) or not 0 <= score <= (100 if scale == 1 else 1):
        return None, '前景指标非有限值或单位不明确'
    iou = row.get('iou', row.get('class_iou', row.get('class_iou_pct')))
    out = {'foreground_miou_pct': score*scale, 'metric_key': key,
           'class_iou_pct': None}
    if iou is not None:
        if not isinstance(iou, list) or len(iou) != 4 or any(x is not None and (not finite(x) or not 0 <= x <= (100 if scale == 1 else 1)) for x in iou):
            return None, '四类 IoU 格式或单位不明确'
        out['class_iou_pct'] = [None if x is None else x*scale for x in iou]
        average = sum(0 if x is None else x for x in iou[1:])/3*scale
        if abs(average-out['foreground_miou_pct']) > 0.001:
            return None, '声明的前景 mIoU 与三类 IoU 均值不一致'
    cm = row.get('confusion')
    if cm is not None:
        if (not isinstance(cm, list) or len(cm) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in cm)
                or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for r in cm for x in r)):
            return None, '混淆矩阵格式错误'
        recomputed = []
        for c in range(4):
            union = sum(cm[c]) + sum(r[c] for r in cm) - cm[c][c]
            recomputed.append(None if not union else 100*cm[c][c]/union)
        avg = sum(x or 0 for x in recomputed[1:])/3
        if abs(avg-out['foreground_miou_pct']) > 0.001:
            return None, '混淆矩阵与前景 mIoU 不一致'
        out['confusion_recomputed'] = True
        out['class_iou_pct'] = recomputed
        out['confusion'] = cm
    for k in ('precision', 'recall'):
        v = row.get(k)
        if isinstance(v, list) and len(v) == 4 and all(x is None or (finite(x) and 0 <= x <= 1) for x in v):
            out[k+'_pct'] = [None if x is None else 100*x for x in v]
    return out, None


def get_metrics(data):
    result, errors = {}, []
    metrics = data.get('metrics', {})
    if isinstance(metrics, dict):
        for name, row in metrics.items():
            out, error = normalize_metric(row)
            if out:
                result[name] = out
            if error:
                errors.append(str(name)+': '+error)
    # Comparison report explicitly stores percentages, unlike training reports.
    if data.get('format') == 'SRAC_SAR_SOFT_CROSS_COMPARISON_V1':
        for source, routes in data.get('cells', {}).items():
            for name, row in routes.items():
                out, error = normalize_metric(row, percent=True)
                if out:
                    result[source+'/'+name] = out
                if error:
                    errors.append(source+'/'+name+': '+error)
    return result, errors


def step_number(path, data):
    for key in ('checkpoint_step', 'checkpoint_steps', 'completed_steps', 'step'):
        if isinstance(data.get(key), int):
            return data[key]
    m = re.search(r'val_step_(\d+)', str(path))
    if m:
        return int(m.group(1))
    updates = data.get('updates_per_arm', {})
    if isinstance(updates, dict) and updates and all(isinstance(x, int) for x in updates.values()):
        return min(updates.values()) if len(set(updates.values())) == 1 else None
    if data.get('complete') is True and isinstance(data.get('steps'), int):
        return data['steps']
    return None


def classify(path, data, docs, metric_errors):
    notes = []
    fmt = str(data.get('format', ''))
    image_count = data.get('image_count')
    smoke = data.get('smoke') is True or 'smoke' in str(data.get('phase', '')).lower() or any('smoke' in p.lower() for p in path.parts)
    error = data.get('error') or data.get('failure')
    checks = bad_checks(data)
    if error:
        return 'failed', [str(error)]
    if checks or metric_errors:
        return 'review', ['检查未通过：'+', '.join(checks)] + metric_errors
    if smoke:
        return 'smoke', ['仅用于检查流程，不能证明方法收益']
    if image_count == 878 or str(data.get('split', '')).lower() == 'test' or data.get('test_accessed') is True:
        return 'other_split', ['不与 Val450 前景三类 mIoU 比较']
    if data.get('initial') is True or path.parent.name == 'initial_val' or data.get('phase') == 'initial_val':
        return ('initial' if image_count == 450 else 'partial'), ['训练前检查，不是训练后的方法收益']
    if fmt == 'SRAC_Q_FACTORIAL_V1':
        good = (data.get('complete') is True and data.get('steps') == 2000
                and data.get('a1_unchanged') is True and data.get('downstream_unchanged') is True
                and data.get('full_validation_evidence') is True and all_true(data.get('immutable'))
                and data.get('smoke') is False and data.get('test_accessed') is False)
        return ('train_done' if good else 'partial'), ['这是训练阶段；后续 formal/val_<arm> 才是独立评估']
    if fmt in JOINT:
        arms = JOINT[fmt]
        summary_path = path.parent/'summary.json'
        if path.parent.name in ('final_val', 'val', 'checkpoint_reval') or path.parent.name.startswith('val_step_'):
            summary_path = path.parent.parent/'summary.json'
        companion = docs.get(summary_path, {})
        if companion.get('error') or companion.get('failure'):
            notes.append('所属训练报告另有失败；这条历史评估不能充作最终成功')
        owner = data if path.name == 'summary.json' else companion
        final = data
        if path.name == 'summary.json':
            final = docs.get(path.parent/'final_val/result.json', docs.get(path.parent/'val/result.json', {}))
        if final_joint_ok(owner,final,arms):
            return 'formal', notes
        if image_count == 450 and data.get('execution_complete') is True:
            if data.get('full_validation_evidence') is True:
                return 'review', notes+['最终声明缺少匹配的5000步完成/完整性证据']
            kind = 'revalidation' if path.parent.name == 'checkpoint_reval' else 'intermediate'
            return kind, notes+['完整450图；非预定末步，不可分别择优当最终成绩']
        return 'partial', notes
    if fmt == 'SRAC_MEMORY_REBUILD_DIAGNOSE_V3':
        gates=('protocol','bank','split','labels','reference','native','pool','artifacts','immutable')
        validation=data.get('validation',{})
        if (data.get('complete') is True and data.get('full_validation_evidence') is True and image_count == 450
                and data.get('execution_complete') is True and isinstance(validation,dict)
                and all(validation.get(k) is True for k in gates)
                and data.get('training') is False and data.get('labelvote_is_untrained_diagnostic') is True
                and data.get('smoke') is False and data.get('paired_bank_smoke') is not True and data.get('test_accessed') is False):
            return 'diagnostic', ['未训练的 label vote；不能解释为 Reader 收益，也非旧FP16库复现']
        return 'partial', ['检索诊断尚缺最终完成证据']
    if fmt == 'SRAC_HARDQUERY_PROTOCOL_V1':
        gates=('input_files_unchanged','code_unchanged','frozen_evidence_unchanged','numerical_unchanged')
        if (data.get('complete') is True and data.get('full_validation_evidence') is True
            and data.get('inference_complete') is True and image_count==450 and data.get('scope')=='all'
            and data.get('smoke') is False and passed(data,'reproduction') and all(data.get(k) is True for k in gates)):
            return 'formal', ['冻结E0基准，不是新方法训练收益']
        return 'partial', ['E0正式范围或完整性字段尚未齐全']
    # Full pipeline certification is kept separate from an individual model score.
    if path.name in ('workflow_state.json', 'pipeline_state.json', 'chain_state.json', 'sar_verification.json'):
        if data.get('complete') is True:
            if data.get('reports_only') is True:
                return 'review', ['仅报告模式，未完成独立工件核验']
            if fmt=='BRIGHT_EXTERNAL_SAR_VALIDATION_V1':
                statuses=data.get('status',{})
                if (data.get('reports_only') is not False or data.get('match_errors')
                    or not isinstance(statuses,dict) or not all(passed(statuses,k) for k in ('baseline','sar','rgb','shifted'))):
                    return 'review', ['SAR四组独立匹配核验字段不齐全']
            return 'pipeline_done', ['这是流水线自己的完成声明，不代表所有子组都有正向收益']
        return 'partial', ['流水线状态与各子阶段状态分别记录']
    formal_families = ('SRAC_Q_FACTORIAL_EVAL_V1', 'SRAC_BUILDING_ROUTE_V1_EVAL', 'SRAC_REGION_VAL_FORMAL_V1',
        'SRAC_SAR_SOFT_ROUTE_V1_EVAL', 'SRAC_SAR_SOFT_CROSS_COMPARISON_V1',
        'SRAC_REGION_HEAD_ADAPT_EVAL_V1')
    if fmt in formal_families and data.get('complete') is True and data.get('full_validation_evidence') is True and image_count == 450:
        if fmt=='SRAC_Q_FACTORIAL_EVAL_V1':
            owner=docs.get(path.parent.parent/('train_'+str(data.get('arm')))/'summary.json',{})
            if owner.get('error') or owner.get('failure') or owner.get('complete') is False:
                return 'review', ['同级匹配训练报告失败或未完成；独立Val不能覆盖该矛盾']
        if fmt!='SRAC_SAR_SOFT_CROSS_COMPARISON_V1':
            good=(data.get('smoke') is False and data.get('test_accessed') is False
                  and all_true(data.get('immutable')) and data.get('native_a1_replay_ok') is True)
            # Some summary files hold boolean aliases; result carries full records.
            detail=docs.get(path.parent/'result.json',data) if path.name=='summary.json' else data
            good = good and passed(detail,'reproduction')
            if fmt in ('SRAC_Q_FACTORIAL_EVAL_V1','SRAC_BUILDING_ROUTE_V1_EVAL','SRAC_SAR_SOFT_ROUTE_V1_EVAL'):
                good=good and passed(detail,'q_a1_reproduction')
            if fmt=='SRAC_Q_FACTORIAL_EVAL_V1':
                good=good and data.get('checkpoint_steps')==2000
            if fmt=='SRAC_REGION_HEAD_ADAPT_EVAL_V1':
                owner=docs.get(path.parent.parent/'summary.json',{})
                good=good and owner.get('complete') is True and owner.get('full_validation_evidence') is True and all(owner.get('updates_per_arm',{}).get(a)==5000 for a in ('p','r','t'))
            if not good:
                return 'review', ['正式声明存在，但完整性／来源／训练末步字段不足']
        return 'formal', ['依据保存的完成报告；本工具未重跑CUDA或重新核验权重字节']
    if data.get('complete') is True and data.get('full_validation_evidence') is True and image_count == 450:
        return 'review', ['完整450图声明，但此format尚未专门识别；保留指标供人工核对']
    if data.get('complete') is True:
        return 'stage_done', ['complete只说明本阶段结束；不能自动视为训练/Val收益']
    if image_count == 450 and data.get('execution_complete') is True:
        return 'intermediate', ['已处理450图；最终完成/完整性声明不足']
    return 'partial', notes


def make_record(path, data, source, docs):
    metrics, errors = get_metrics(data)
    code, notes = classify(path, data, docs, errors)
    fmt = str(data.get('format', ''))
    binding = data.get('recovery_binding', {})
    if binding or 'JOINT_LIVE_RECOVERY' in str(path):
        notes.append('恢复实验：不能等同未中断的原GPU训练；以本次同后端三组比较')
    if code == 'other_split' or data.get('image_count') == 878 or str(data.get('split','')).lower() == 'test' or data.get('test_accessed') is True:
        metrics = {}  # Never re-label Test four-class means as foreground Val.
        notes.append('Test／非Val450记录：不进入前景Val指标表和差值比较')
    return dict(source=source, family=family(path, data), format=fmt, status=code,
        status_text=STATUS[code], phase=data.get('phase', data.get('current_stage')),
        arm=data.get('arm'), step=step_number(path,data), image_count=data.get('image_count'),
        updates_per_arm=data.get('updates_per_arm'), metrics=metrics, notes=notes,
        completion_flags={k:data[k] for k in ('complete','execution_complete','full_validation_evidence',
            'primary_endpoint','final_integrity_pending','checkpoint_revalidation_complete','test_accessed') if k in data},
        recovery={k:binding[k] for k in ('backend','exact_historical_resume_claim','prefix_updates') if k in binding},
        comparisons_pp=compare(metrics), artifact_reverified=False)


def compare(metrics):
    pairs = [('r0','grid_probq'),('rt','r0'),('rf','r0'),('rf','rt'),
             ('b','a'),('d','c'),('c','a'),('d','b'),
             ('grid_probq','pixel_grid'),('region','region_mean'),
             ('soft','unsplit'),('soft','capacity_control'),('hard','capacity_control')]
    return {a+' − '+b:metrics[a]['foreground_miou_pct']-metrics[b]['foreground_miou_pct']
            for a,b in pairs if a in metrics and b in metrics}


def row_progress(path, data):
    rows = []
    for name in ('train.jsonl', 'training.jsonl'):
        row, src = tail_json(path.parent/name)
        if row:
            rows.append(dict(source=src, kind='training_progress', step=row.get('step'),
                image=row.get('image'), epoch=row.get('epoch'), lr=row.get('lr'),
                heads=row.get('heads'), note='最后完整日志行，不证明进程仍存活'))
    return rows


def collect(base, max_depth=9, max_files=6000, max_bytes=16*1024*1024):
    paths, warnings = scan(base, max_depth, max_files)
    docs, sources, unreadable = {}, {}, []
    for path in paths:
        try:
            data, source = stable_read(path,max_bytes)
            # Most protocol files are configs, not completed evaluation reports.
            if path.name == 'protocol.json' and not isinstance(data.get('metrics'), dict):
                continue
            docs[path], sources[path] = data, source
        except (OSError, ValueError, UnicodeError) as exc:
            unreadable.append(dict(source={'path':str(path)}, family='读取异常', format='',
                status='unreadable', status_text=STATUS['unreadable'], metrics={},
                notes=[str(exc)], comparisons_pp={}, artifact_reverified=False))
    records, progress = [], []
    for path, data in docs.items():
        try:
            record = make_record(path,data,sources[path],docs)
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
            records.append(dict(source=sources[path],family='报告结构待核对',format=str(data.get('format','')),
                status='review',status_text=STATUS['review'],metrics={},comparisons_pp={},
                notes=['报告结构不符合已识别格式：'+str(exc)],artifact_reverified=False))
            continue
        # Check only small JSON bindings; do not load/hash large model weights.
        summary_path=path.parent/'summary.json'
        summary=docs.get(summary_path,{})
        result_path=path.parent/'result.json'
        expected=summary.get('result_sha256')
        if expected and path.name in ('summary.json','result.json'):
            actual=sources.get(result_path,{}).get('sha256')
            if expected!=actual:
                record['notes'].append('summary绑定的result SHA256缺失或不匹配；报告可能仍在写入')
                if record['status'] not in ('failed','smoke','other_split'):
                    record.update(status='review',status_text=STATUS['review'])
            else:
                record['notes'].append('小型result.json的SHA256绑定已复核；未核验权重/预测数组')
        if path.name=='result.json' and (summary.get('error') or summary.get('failure')):
            record['notes'].append('同目录summary报告失败：'+str(summary.get('error') or summary.get('failure')))
            if record['status'] in ('formal','diagnostic'):
                record.update(status='review',status_text=STATUS['review'])
        if data.get('format')=='SRAC_MEMORY_REBUILD_DIAGNOSE_V3' and record['status']=='diagnostic':
            other_path=result_path if path.name=='summary.json' else summary_path
            other=docs.get(other_path)
            fields=('image_count','complete','full_validation_evidence','execution_complete','smoke','validation')
            if other is not None and any(other.get(k)!=data.get(k) for k in fields):
                record.update(status='review',status_text=STATUS['review'])
                record['notes'].append('同目录summary/result诊断字段不一致')
        records.append(record)
        if path.name == 'summary.json':
            progress.extend(row_progress(path,data))
            # SAR train summary contains best_valid_miou, NOT last metrics.
            # Read the actual final journal row and inherit the training outcome.
            if data.get('format') == 'SRAC_Q_FACTORIAL_V1':
                row, src = tail_json(path.parent/'validation.jsonl', max_bytes=max_bytes)
                if row:
                    metric, errors = get_metrics(row)
                    derived = dict(record)
                    derived.update(source=src, metrics=metric, step=row.get('step'), image_count=row.get('image_count'),
                        notes=record['notes']+['训练内末次Val日志；不是独立重加载验证，不使用best_valid_miou'])
                    if derived['status']=='train_done' and (row.get('step')!=data.get('steps') or row.get('image_count')!=450):
                        derived.update(status='review',status_text=STATUS['review'])
                        derived['notes'].append('末次Val日志步数/图数与已完成训练不一致')
                    if errors and derived['status']!='failed':
                        derived.update(status='review',status_text=STATUS['review'])
                        derived['notes'] += errors
                    records.append(derived)
    records.extend(unreadable)
    records.sort(key=lambda r:(r['family'], r['source']['path']))
    return dict(format=VERSION, generated_at=datetime.now(CN).isoformat(), base=str(base),
        scope='读取服务器报告，不重跑验证、不加载权重、不访问数据集或CUDA',
        scan_limits=dict(max_depth=max_depth,max_files=max_files,max_report_bytes=max_bytes),
        warnings=warnings, documents_loaded=len(docs), records=records, progress=progress,
        status_counts=dict(Counter(r['status_text'] for r in records)))


def brief(report):
    lines=['# BRIGHT 本次证据简表','', '时间（UTC+8）：'+report['generated_at'],
        '', '读取保存报告；没有重新认证权重或执行GPU验证。完整历史、失败现场与文件路径见 REPORT.md。',
        '每类最多展示最近6条带指标记录；这不是最佳模型选择，不跨run相减。','']
    if report['warnings']:
        lines+=['扫描提醒：']+['- '+escape(w) for w in report['warnings']]+['']
    if report['progress']:
        lines+=['## 最近完整训练日志行（不证明仍在运行）','']
        for row in sorted(report['progress'],key=lambda x:x['source'].get('modified_at',''),reverse=True)[:5]:
            lines+=['- step='+str(row.get('step'))+'，image='+str(row.get('image'))+'；`'+row['source']['path']+'`']
        lines.append('')
    expected=('区域读出评估','建筑 soft／hard 路由','E0／冻结基准','P／R／T 区域头',
        'V3 FP32 建库与检索诊断','R0／RT／RF 联合训练','R0／RT／RF 恢复续训',
        'SAR／Q 匹配实验','SAR-Q × 路由','soft＋Memory 新四组')
    for fam in expected:
        rows=[r for r in report['records'] if r['family']==fam]
        lines+=['## '+fam,'']
        if not rows:
            lines+=['本次扫描未检出；不代表实验从未执行。','']
            continue
        counts=Counter(r['status_text'] for r in rows)
        lines.append('；'.join(k+' '+str(v)+'条' for k,v in counts.items()))
        candidates=sorted((r for r in rows if r['metrics']),key=lambda r:r['source'].get('modified_at',''),reverse=True)
        seen=set()
        for r in candidates:
            p=Path(r['source']['path'])
            identity=str(p.parent)
            if identity in seen: continue
            seen.add(identity)
            lines+=['', '- '+r['status_text']+'；step='+str(r.get('step'))+'；image_count='+str(r.get('image_count')),
                '  '+ '; '.join(name+f"={m['foreground_miou_pct']:.4f}%" for name,m in r['metrics'].items()),
                '  来源：`'+r['source']['path']+'`']
            for name,value in r['comparisons_pp'].items():
                lines.append('  '+name+f'：{value:+.4f} pp')
            if len(seen)>=6: break
        if not candidates:
            latest=max(rows,key=lambda r:r['source'].get('modified_at',''))
            lines+=['最近阶段：'+str(latest.get('phase'))+'；步数='+str(latest.get('updates_per_arm') or latest.get('step')),
                    '来源：`'+latest['source']['path']+'`']
        lines.append('')
    lines+=['## 记录边界','',
        '- 正式完成、完整中间Val、诊断、smoke、失败分别保留。',
        '- 单次正向差值不是多种子稳定收益；baseline完成不是SAR机制已证明。',
        '- 脚本不根据报告修改complete，不修复流水线，不启动或终止任何训练。','']
    return '\n'.join(lines)


def escape(x):
    return str(x).replace('|','\\|').replace('\n',' ')


def render(report):
    lines = ['# BRIGHT 实验状态与证据记录', '', '生成时间（UTC+8）：'+report['generated_at'],
        '', '**本文件是报告盘点，不是重新认证。** 成绩按原报告记录；权重、输入哈希和CUDA计算未重验。',
        '只读扫描：`'+report['base']+'`。没有找到报告只表示本次未检出，不能断言实验从未运行。',
        '同一实验的训练、评估、流程报告可能分别列出，记录数量不是独立实验数量。',
        '', '## 应怎样记录“已经证明”', '',
        '- 完整Val450中间评估可以分析趋势；不能作为5000步末步结果，也不能证明多种子稳定增益。',
        '- SAR baseline完成只建立匹配对照；新增SAR有效还需同批sar_cue、等容量RGB和错位SAR。',
        '- RT/RF超过R0只支持本次同检查点历史标注案例的增量；不单独证明检索排序或长期稳定性。',
        '- soft相对空间对照的增益，不自动证明实例内超像素、SAR＋soft或soft＋Memory组合有效。',
        '- 检索label vote、smoke、CPU测试、建库完成，都不能算Reader训练收益。',
        '- Test四类指标不会折算成Val前景三类；失败实验中留下的分数不能提升为正式成绩。',
        '', '## 汇总', '']
    for status, count in sorted(report['status_counts'].items()):
        lines.append('- '+status+'：'+str(count)+' 条报告记录')
    if report['warnings']:
        lines.extend(['', '扫描提醒：']+['- '+escape(w) for w in report['warnings']])
    lines += ['', '## 发现的阶段与成绩', '',
        '|记录|实验类型／阶段|状态|步数／图数|指标|', '|---|---|---|---|---|']
    for i,r in enumerate(report['records'],1):
        scores='；'.join(f"{name}={m['foreground_miou_pct']:.4f}%" for name,m in r['metrics'].items()) or '无可确认的前景指标'
        lines.append('|'+ '|'.join(escape(v) for v in (i,r['family']+' / '+str(r.get('phase') or r.get('arm') or Path(r['source']['path']).parent.name),
            r['status_text'],str(r.get('step'))+' / '+str(r.get('image_count')),scores))+'|')
    lines += ['', '## 逐条证据与可解释范围', '']
    for i,r in enumerate(report['records'],1):
        lines += [f"### {i}. {r['family']} — {r['status_text']}", '',
            '来源：`'+r['source']['path']+'`',
            '记录格式：`'+str(r.get('format',''))+'`；阶段：`'+str(r.get('phase'))+'`',
            '完成字段：`'+json.dumps(r.get('completion_flags',{}),ensure_ascii=False)+'`']
        if r.get('updates_per_arm'):
            lines.append('各组步数：`'+json.dumps(r['updates_per_arm'])+'`')
        if r.get('recovery'):
            lines.append('恢复配置：`'+json.dumps(r['recovery'],ensure_ascii=False)+'`')
        lines.extend('- '+escape(n) for n in r['notes'])
        if r['metrics']:
            lines += ['', '数值单位：百分比；类别顺序 Background / Intact / Damaged / Destroyed。', '',
                '|路径|前景三类mIoU|四类IoU|', '|---|---:|---|']
            for name,m in r['metrics'].items():
                ious=' / '.join('NA' if x is None else f'{x:.4f}' for x in (m['class_iou_pct'] or []))
                lines.append(f"|{escape(name)}|{m['foreground_miou_pct']:.4f}|{ious}|")
            lines += ['', '同一报告内的描述性差值（pp；不是统计显著性结论）：']
            lines.extend('- '+escape(k)+f'：{v:+.4f}' for k,v in r['comparisons_pp'].items())
            if {'r0','rt','rf'} <= r['metrics'].keys():
                a=r['metrics']; ds=[a[x]['foreground_miou_pct']-a['r0']['foreground_miou_pct'] for x in ('rt','rf')]
                lines.append('- '+('RT/RF相对R0均为正向点估计；仍需预定末步和重复种子。' if all(v>0 for v in ds)
                    else 'Memory分支尚非全部高于R0；按各组差值记录，不预设收益。'))
                if 'grid_probq' in a and a['rt']['class_iou_pct'] and a['grid_probq']['class_iou_pct']:
                    delta=[None if x is None or y is None else x-y for x,y in zip(a['rt']['class_iou_pct'],a['grid_probq']['class_iou_pct'])]
                    lines.append('- RT相对同次grid_probq各类IoU差：'+ ' / '.join('NA' if x is None else f'{x:+.4f}' for x in delta)+' pp。')
        lines.append('')
    lines += ['## 最近训练日志行（不表示进程存活）', '']
    for row in report['progress']:
        lines += ['- `'+row['source']['path']+'`：step='+str(row.get('step'))+'，image='+str(row.get('image'))]
    lines += ['', '## 仍不能自动证明的结论', '',
        '区域下降的唯一根因、Memory多种子稳定增益、SAR机制成立、组合收益、Test泛化，均需要对应匹配证据；本脚本不根据文件存在或单个complete字段代替这些结论。', '']
    return '\n'.join(lines)


def save(report, parent):
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(CN).strftime('%Y%m%d_%H%M%S_%f')
    folder = parent / ('record_'+stamp+'_'+str(os.getpid()))
    folder.mkdir(exist_ok=False)
    for name,text in (('evidence.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)),
                      ('SUMMARY.md',brief(report)),
                      ('REPORT.md',render(report))):
        with (folder/name).open('x',encoding='utf-8') as f:
            f.write(text+'\n')
    with (folder/'metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['source','family','status','step','image_count','arm','foreground_miou_pct','background_iou_pct','intact_iou_pct','damaged_iou_pct','destroyed_iou_pct'])
        for r in report['records']:
            for name,m in r['metrics'].items():
                writer.writerow([r['source']['path'],r['family'],r['status_text'],r.get('step'),r.get('image_count'),name,m['foreground_miou_pct'],*(m['class_iou_pct'] or [None]*4)])
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--output-parent',type=Path,default=OUTPUT)
    parser.add_argument('--max-depth',type=int,default=9)
    parser.add_argument('--max-files',type=int,default=6000)
    parser.add_argument('--max-report-mib',type=int,default=16)
    parser.add_argument('--stdout-only',action='store_true',help='只打印Markdown，不写报告')
    args=parser.parse_args(argv)
    if min(args.max_depth,args.max_files,args.max_report_mib)<1:
        parser.error('扫描上限必须为正数')
    base=args.base.expanduser().resolve()
    parent=args.output_parent.expanduser().resolve()
    if not args.stdout_only and (parent==base or base in parent.parents):
        parser.error('报告输出不能位于实验结果树内；请使用BRIGHT-master/New下的新目录')
    report=collect(base,args.max_depth,args.max_files,args.max_report_mib*1024*1024)
    if args.stdout_only:
        print(render(report))
        return 0
    folder=save(report,parent)
    print('本工具仅盘点报告，未重跑验证，未加载权重，未使用CUDA。')
    print('扫描结果根目录：',base)
    print('读取JSON报告：',report['documents_loaded'])
    for label,count in sorted(report['status_counts'].items()):
        print(label+': '+str(count)+' 条记录')
    print('中文记录：',folder/'REPORT.md')
    print('简明记录：',folder/'SUMMARY.md')
    print('指标表格：',folder/'metrics.csv')
    print('结构化证据：',folder/'evidence.json')
    print('完成情况以报告分级为准；没有发现文件不等于实验从未执行。')
    if report['warnings']:
        print('扫描提醒：',json.dumps(report['warnings'],ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
