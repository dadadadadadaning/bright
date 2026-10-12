"""Read-only source inventory and NumPy counterexamples, not PyTorch execution."""
import ast
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np

root=Path('D:/software/dowmload/BRIGHT-master/bda_benchmark')
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
rows=[]; errors=[]; definitions=[]
pattern=re.compile(r'superpixel|srac|memory|prototype|region|partition|cmr|iarm',re.I)
for folder,dirs,files in os.walk(root,onerror=lambda e:errors.append(str(e))):
    for name in files:
        path=Path(folder)/name
        if path.suffix not in {'.py','.sh'}: continue
        row={'file':str(path),'target_name_match':bool(pattern.search(str(path.relative_to(root))))}
        try:
            data=path.read_bytes(); content=data.decode('utf-8-sig',errors='replace')
            row.update(bytes=len(data),lines=len(content.splitlines()),sha256=hashlib.sha256(data).hexdigest(),
                status='full_text_machine_read_not_manual_review')
            if path.suffix=='.py':
                try:
                    tree=ast.parse(content)
                    if row['target_name_match']:
                        definitions.append({'file':str(path),'definitions':[
                            {'name':n.name,'line':n.lineno,'end':n.end_lineno} for n in ast.walk(tree)
                            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))],
                            'imports':[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]})
                except Exception as error: row['parse_error']=str(error)
        except Exception as error: row.update(status='read_error',error=str(error))
        rows.append(row)
fields=sorted(set().union(*(row.keys() for row in rows)))
with (out/'code_inventory.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
with (out/'target_code_structure.jsonl').open('w',encoding='utf-8') as stream:
    for row in definitions: stream.write(json.dumps(row,ensure_ascii=False)+'\n')

# Reproduce the arithmetic of the inspected functions, not their torch runtime.
def softmax(x):
    x=np.asarray(x,dtype=float); e=np.exp(x-x.max()); return e/e.sum()
results={}
original=np.array([.9,.8,.7]); same_event=np.array([True,False,True])
masked=np.where(same_event,-1e9,original)
indices=np.argsort(-masked)[:3]
reranked=.35*original[indices]+.65*np.array([.2,.9,.7])
results['excluded_candidate_reenters_rerank']={
    'coarse_indices':indices.tolist(),'masked_scores':masked[indices].tolist(),
    'global_mean_with_padding':float(masked[indices].mean()),
    'rerank_scores':reranked.tolist(),
    'returned_original_index':int(indices[reranked.argmax()]),
    'returned_same_event':bool(same_event[indices[reranked.argmax()]])}
results['all_neighbors_excluded_still_vote']={
    'scores':[-10000.,-10000.,-10000.],
    'weights':softmax(np.array([-10000.,-10000.,-10000.])/.1).tolist(),
    'explanation':'Exemplar V2 softmax does not abstain when every support is excluded.'}
base=np.array([2.,1.,0.,-1.]); updated=base.copy(); updated[2]+=3.
results['unchanged_bg_logit_not_unchanged_bg_prediction']={
    'base_logits':base.tolist(),'updated_logits':updated.tolist(),
    'base_prediction':int(base.argmax()),'updated_prediction':int(updated.argmax()),
    'base_bg_probability':float(softmax(base)[0]),'updated_bg_probability':float(softmax(updated)[0])}
records=[('event_a','image_first',i) for i in range(300)]+[('event_a','image_later',i) for i in range(300)]
kept=records[:256]
results['event_cap_prefix_bias']={'input_images':sorted(set(x[1] for x in records)),
    'selected_images':sorted(set(x[1] for x in kept)),'selected_count':len(kept)}
unique_scores=np.array([.90,.89]); unique_labels=np.array([1,2])
duplicated_scores=np.array([.90]+[.89]*7); duplicated_labels=np.array([1]+[2]*7)
results['padding_changes_class_vote']={
    'unique_damaged_probability':float(softmax(unique_scores/.1)[unique_labels==2].sum()),
    'padded_damaged_probability':float(softmax(duplicated_scores/.1)[duplicated_labels==2].sum()),
    'unique_neighbor_count':2,'padded_slots':8,
    'explanation':'Repeated last neighbor changes decision and exceeds the per-event cap in effective votes.'}
vectors=np.tile(np.array([[1.,0.]]),(5,1))
chosen=[0]; distance=1-vectors@vectors[0]
while len(chosen)<3:
    candidate=int(distance.argmax());chosen.append(candidate)
    distance=np.minimum(distance,1-vectors@vectors[candidate])
results['farthest_subset_can_repeat_same_record']={'indices':chosen,'unique_count':len(set(chosen)),
    'explanation':'Selected entries are not masked before argmax in factorized build_memory.farthest_subset.'}
(out/'counterexamples_numpy.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')

metric_rows=[]
evalroot=Path('D:/software/dowmload/BRIGHT-eval-results/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2')
for name in ['audit_val.json','audit_val_event.json','audit_train_event.json']:
    path=evalroot/name
    if not path.exists(): continue
    obj=json.loads(path.read_text(encoding='utf-8-sig'))
    metric_rows.append(dict(file=str(path),query_count=obj.get('query_count'),
        memory_macro_f1=obj.get('correct_memory',{}).get('macro_f1'),
        direct_macro_f1=obj.get('direct_metric_classifier',{}).get('macro_f1'),
        memory_minus_direct_pp=obj.get('memory_minus_direct_classifier_macro_f1_pp'),
        exclude_same_event=obj.get('exclude_same_event'),
        random_control=obj.get('random_label_controls',{}).get('method')))
(out/'exemplar_result_comparison.json').write_text(json.dumps(metric_rows,indent=2),encoding='utf-8')
summary=dict(total_code_files=len(rows),name_matched_files=sum(r['target_name_match'] for r in rows),
    read_errors=[r for r in rows if r['status']=='read_error'],walk_errors=errors,
    scope='Full text machine scan and selected semantic review; counterexamples are NumPy arithmetic, not production PyTorch tests.')
(out/'coverage.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False)); print(json.dumps(metric_rows,ensure_ascii=False));print(json.dumps(results,ensure_ascii=False))
