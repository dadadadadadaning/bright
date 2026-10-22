"""Accept only complete878 journals; export every fixed step, never select by Test."""
import copy
import csv
import json
from pathlib import Path
import numpy as np
from grid_probe import read_json,sha256
from evaluate_checkpoint import write_json,ARMS,REQUIRED_CHECKS
from sweep_context import STEPS,SELECTION
from sweep_scoring import rebuild


def numeric(matrix):
    a=np.asarray(matrix)
    if a.shape!=(4,4) or a.dtype.kind not in 'iu' or (a<0).any():raise ValueError('Invalid confusion matrix')
    if not a.sum():
        return {k:None for k in ('miou4_pct','foreground_miou3_pct','background_iou_pct',
            'intact_iou_pct','damaged_iou_pct','destroyed_iou_pct','oa_pct','building_iou_pct',
            'damaged_precision_pct','damaged_recall_pct','destroyed_precision_pct','destroyed_recall_pct')}
    tp=a.diagonal().astype(float);gt=a.sum(1);pred=a.sum(0);union=gt+pred-tp
    iou=np.divide(tp,union,out=np.zeros(4),where=union!=0)
    precision=np.divide(tp,pred,out=np.zeros(4),where=pred!=0)
    recall=np.divide(tp,gt,out=np.zeros(4),where=gt!=0)
    building_tp=int(a[1:,1:].sum());building_fp=int(a[0,1:].sum());building_fn=int(a[1:,0].sum())
    denom=building_tp+building_fp+building_fn
    return dict(miou4_pct=float(iou.mean()*100),foreground_miou3_pct=float(iou[1:].mean()*100),
        background_iou_pct=float(iou[0]*100),intact_iou_pct=float(iou[1]*100),
        damaged_iou_pct=float(iou[2]*100),destroyed_iou_pct=float(iou[3]*100),
        oa_pct=float(tp.sum()/a.sum()*100),building_iou_pct=100*building_tp/denom if denom else 0,
        damaged_precision_pct=float(precision[2]*100),damaged_recall_pct=float(recall[2]*100),
        destroyed_precision_pct=float(precision[3]*100),destroyed_recall_pct=float(recall[3]*100))


def verify_report(path,report,names,step):
    path=Path(path)
    if (step not in STEPS or len(names)!=878 or len(set(names))!=878 or report.get('step')!=step
        or report.get('split')!='test' or report.get('image_count')!=878
        or report.get('class_names')!=['background','intact','damaged','destroyed']
        or report.get('arms')!=list(ARMS) or report.get('optimizer_updates')!=0
        or report.get('checkpoint_selection')!=SELECTION or report.get('test_accessed') is not True
        or report.get('test_used_for_selection') is not False):raise ValueError('Wrong Test identity, step or completeness')
    if report['journal_sha256']!=sha256(path.with_name('images.jsonl')):raise ValueError('Test journal changed')
    comparisons={k:tuple(v['axes'][1:]) for k,v in report['transitions'].items()}
    expected={b+'_minus_'+a:(a,b) for a,b in (('e','b'),('f','d'),('e','f'),('b','d'),('r','d'),('l','d'))}
    expected.update({a+'_minus_grid_probq':('grid_probq',a) for a in ARMS})
    if comparisons!=expected:raise ValueError('Missing or changed ablation comparisons')
    paths=('grid_probq',)+tuple(ARMS)
    if set(report['metrics'])!=set(paths):raise ValueError('Incomplete seven-path metrics')
    cm,pairs,events,event_pairs,counts=rebuild(path.with_name('images.jsonl'),names,paths,comparisons)
    if report['per_event_image_count']!=counts or set(report['per_event'])!=set(events):raise ValueError('Event count mismatch')
    for key,a in cm.items():
        if not a.sum():raise ValueError('Full Test has no valid pixels')
        m=report['metrics'][key]
        if not np.array_equal(a,np.asarray(m['confusion'])):raise ValueError('Aggregate confusion differs from journal')
        values=numeric(a)
        if abs(values['miou4_pct']-m['miou4']*100)>1e-10:raise ValueError('mIoU4 formula mismatch')
        if abs(values['foreground_miou3_pct']-m['foreground_miou']*100)>1e-10:raise ValueError('Foreground formula mismatch')
    for event,matrices in events.items():
        for key,a in matrices.items():
            if not np.array_equal(a,np.asarray(report['per_event'][event][key]['confusion'])):
                raise ValueError('Event confusion differs from journal')
    return cm


def finish_report(path,report,*,output,names,evidence,checks,provenance,full):
    step=provenance['checkpoint_step'];path=Path(path);output=Path(output)
    if path!=output/('test_step_%06d'%step)/'result.json' or not full:raise ValueError('Full Test path required')
    if set(checks)!=set(REQUIRED_CHECKS) or any(v is not True for v in checks.values()):raise ValueError('Integrity checks incomplete')
    if read_json(path)!=report:raise ValueError('Saved report differs')
    manifest=read_json(output/'manifest.json')
    if manifest['names']!=names or manifest['protocol_info']!=provenance['protocol_info']:raise ValueError('Manifest differs')
    operators=read_json(output/'operator_observer_summary.json')
    if operators.get('boundary_calls',0)<1 or operators.get('bindings_restored') is not True:
        raise ValueError('Missing operator observation closure')
    verify_report(path,report,names,step)
    with path.with_name('result.pending.json').open('xb') as stream:stream.write(path.read_bytes())
    finished=copy.deepcopy(report)
    finished.update(complete=True,phase='standalone_evaluation_complete',final_integrity_pending=False,
        full_test_evidence=True,full_validation_evidence=False,primary_endpoint=False,
        standalone_integrity_checks=checks,standalone_provenance=provenance,
        pending_report_sha256=sha256(path),historical_failure_resolved=False,exact_historical_replay=False)
    write_json(path,finished);path.with_name('result.json.sha256').write_text(sha256(path)+'  result.json\n')
    return finished


def accept_case(root,step,record,request):
    folder=Path(root)/('test%04d'%step);s=read_json(folder/'summary.json')
    if (record['returncode']!=0 or record.get('supervisor_reason') or s.get('complete') is not True
        or s.get('full_test_evidence') is not True or s.get('checkpoint_step')!=step
        or s.get('optimizer_updates')!=0 or s.get('image_count')!=878
        or any(s.get('integrity_checks',{}).get(k) is not True for k in REQUIRED_CHECKS)):
        raise ValueError('Test case did not pass completion/integrity checks')
    path=folder/('test_step_%06d'%step)/'result.json';report=read_json(path)
    if report.get('complete') is not True or report.get('full_test_evidence') is not True:raise ValueError('Report incomplete')
    if any(report.get('standalone_integrity_checks',{}).get(k) is not True for k in REQUIRED_CHECKS):
        raise ValueError('Final report integrity checks incomplete')
    if report.get('pending_report_sha256')!=sha256(path.with_name('result.pending.json')):
        raise ValueError('Pending report changed')
    if report['standalone_provenance']['checkpoint_sha256']!=request['checkpoints'][str(step)]['sha256']:
        raise ValueError('Report checkpoint identity differs')
    names=read_json(folder/'manifest.json')['names']
    if names!=request['canonical_test_names']:raise ValueError('Test order changed')
    verify_report(path,report,names,step)
    from night_assessment import validate_route_evidence
    validate_route_evidence(folder,names,'watched')
    return dict(result=str(path),result_sha256=sha256(path))


def export(root,status):
    rows=[];contrasts=[];grid={};per_event=[]
    for step in STEPS:
        record=status['cases'].get('test%04d'%step,{})
        if not record.get('accepted'):continue
        path=Path(record['result'])
        if sha256(path)!=record['result_sha256']:raise ValueError('Accepted report changed')
        report=read_json(path);metrics={a:numeric(m['confusion']) for a,m in report['metrics'].items()}
        grid[str(step)]=report['metrics']['grid_probq']['confusion']
        for arm in ('grid_probq',)+tuple(ARMS):rows.append(dict(step=step,arm=arm,**metrics[arm]))
        for name,t in report['transitions'].items():
            a,b=t['axes'][1:]
            contrasts.append(dict(step=step,comparison=name,miou4_delta_pp=metrics[b]['miou4_pct']-metrics[a]['miou4_pct'],
                damaged_iou_delta_pp=metrics[b]['damaged_iou_pct']-metrics[a]['damaged_iou_pct'],
                destroyed_iou_delta_pp=metrics[b]['destroyed_iou_pct']-metrics[a]['destroyed_iou_pct'],
                building_iou_delta_pp=metrics[b]['building_iou_pct']-metrics[a]['building_iou_pct']))
        for event,arms in report['per_event'].items():
            for arm,m in arms.items():per_event.append(dict(step=step,event=event,arm=arm,**numeric(m['confusion'])))
    root=Path(root)
    for filename,data in (('metrics.csv',rows),('contrasts.csv',contrasts),('per_event.csv',per_event)):
        with (root/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]) if data else ['step','status'])
            writer.writeheader();writer.writerows(data)
    summary=dict(requested_steps=list(STEPS),complete_steps=[int(k) for k in grid],
        incomplete_steps=[s for s in STEPS if str(s) not in grid],
        frozen_grid_confusions_identical=(all(v==next(iter(grid.values())) for v in grid.values()) if len(grid)>1 else None),
        exploratory_test=True,test_used_for_selection=False,metrics=rows,contrasts=contrasts)
    summary['comparability_review_required']=summary['frozen_grid_confusions_identical'] is False
    write_json(root/'comparison.json',summary)
    lines=['# 五个固定检查点的探索性 Test878 比较','',
        '仅完整878张且完整性检查通过的节点进入指标表。失败节点保留空缺，不插值、不补零、不选择最佳Test检查点。','',
        '|步数|分支|四类mIoU %|前景mIoU %|背景IoU %|完好IoU %|受损IoU %|毁坏IoU %|',
        '|---|---|---|---|---|---|---|---|']
    for r in rows:lines.append('|%s|%s|%.4f|%.4f|%.4f|%.4f|%.4f|%.4f|'%tuple(r[k] for k in
        ('step','arm','miou4_pct','foreground_miou3_pct','background_iou_pct','intact_iou_pct','damaged_iou_pct','destroyed_iou_pct')))
    keys=('d_minus_grid_probq','b_minus_e','d_minus_f','f_minus_e','d_minus_b','d_minus_r','d_minus_l')
    lines+=['','四类mIoU差值，单位为百分点：','',
        '|步数|D−Grid|B−E|D−F|F−E|D−B|D−R|D−L|','|---|---|---|---|---|---|---|---|']
    for step in summary['complete_steps']:
        deltas={r['comparison']:r['miou4_delta_pp'] for r in contrasts if r['step']==step}
        lines.append('|'+str(step)+'|'+'|'.join('%+.4f'%deltas[k] for k in keys)+'|')
    lines+=['','未完整完成的步数：'+str(summary['incomplete_steps']),
        '不同节点的冻结Grid混淆矩阵是否完全相同：'+str(summary['frozen_grid_confusions_identical']),
        '若Grid不一致，应先检查推理可重复性，再解释很小的跨步差异。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    return summary
