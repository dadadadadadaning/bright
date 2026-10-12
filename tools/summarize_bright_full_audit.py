"""Aggregate audit evidence without treating automated scans as human review."""
import collections,csv,json,re,sys
from pathlib import Path
out=Path(sys.argv[1])
def write(name,rows,fields):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
root=Path('D:/software/dowmload/BRIGHT-checkpoints')
runs=[]
for p in root.rglob('training_summary.json'):
    try:v=json.loads(p.read_text(encoding='utf-8-sig'))
    except Exception:continue
    config={}
    if (p.parent/'config.json').exists():
        try:config=json.loads((p.parent/'config.json').read_text(encoding='utf-8-sig'))
        except Exception:pass
    initial={}
    if (p.parent/'initial_validation.json').exists():
        try:initial=json.loads((p.parent/'initial_validation.json').read_text(encoding='utf-8-sig'))
        except Exception:pass
    m=v.get('best_metrics',{}) or {}; initial=initial.get('metrics',initial)
    row=dict(file=str(p),family=p.relative_to(root).parts[0],run=p.parent.name,
        variant=v.get('variant',config.get('variant','')),completed_step=v.get('completed_step',v.get('completed_steps','')),
        target_steps=v.get('target_optimizer_steps',config.get('optimizer_steps','')),
        best_step=v.get('best_step',''),best_valid_miou=v.get('best_valid_miou',m.get('valid_miou','')),
        initial_valid_miou=initial.get('valid_miou',''),damaged_f1=m.get('damaged_f1',''),
        event_count=m.get('event_count',''),processed_images=m.get('processed_images',''),
        crop=config.get('crop_size',''),batch=config.get('batch_size',''),accum=config.get('accumulation_steps',''),
        parent=config.get('parent_ckpt',''),seed=config.get('seed',''),
        event_balanced_sampling=v.get('event_balanced_sampling',''),damage_focused_crops=v.get('damage_focused_crops',''),
        scope='smoke_or_guard' if re.search('smoke|guard',p.parent.name,re.I) else 'requires_protocol_review')
    runs.append(row)
write('checkpoint_training_summaries.csv',runs,list(runs[0]))
inv=[]
with (out/'file_inventory.csv').open(encoding='utf-8-sig',newline='') as f:
    inv=list(csv.DictReader(f))
coverage=collections.Counter((r['root'],r['type'],r['status']) for r in inv)
write('coverage_by_type.csv',[dict(root=k[0],extension=k[1],status=k[2],files=v) for k,v in coverage.items()],['root','extension','status','files'])
families=collections.defaultdict(lambda:collections.Counter())
for r in inv:
    if r['root']=='BRIGHT-master':continue
    rel=Path(r['relative_path']);family=rel.parts[0]
    if family=='BRIGHT-eval-results':family='NESTED_COPY_TREE'
    families[family][r['root']]+=1
write('archive_family_coverage.csv',[{'family':k,**{r:v[r] for r in ['BRIGHT-eval-results','BRIGHT-train-logs','BRIGHT-checkpoints']}} for k,v in sorted(families.items())],['family','BRIGHT-eval-results','BRIGHT-train-logs','BRIGHT-checkpoints'])
logs=[]
with (out/'log_scan.jsonl').open(encoding='utf-8') as f:
    for line in f:
        v=json.loads(line);p=v['file'];hits=v['matches']
        vals=[(n,float(m.group(1))) for n,s in hits for m in [re.search(r'valid_miou\s*[:=]\s*([\d.]+)',s,re.I)] if m]
        err=[(n,s) for n,s in hits if re.search(r'Traceback|Error:|Terminated',s)]
        logs.append(dict(file=p,metric_points=len(vals),first_raw=vals[0][1] if vals else '',max_raw=max((x[1] for x in vals),default=''),last_raw=vals[-1][1] if vals else '',error_markers=len(err),last_error=str(err[-1]) if err else '',test_monitor_hits=sum('test monitor' in s.lower() for _,s in hits),completion_markers=sum('training complete' in s.lower() for _,s in hits),note='Raw units/protocols not harmonized; marker absence does not prove completion'))
write('log_status_index.csv',logs,list(logs[0]))
print(json.dumps({'training_summary_files':len(runs),'log_records':len(logs),'inventory_rows':len(inv),'families':len(families)}))
