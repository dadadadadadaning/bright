"""Read-only Memory audit. NumPy core; PyTorch needed only for .pt/.pth or encoding.

Never imports training code, updates weights, tunes thresholds or reads Test.
JSON fixtures use the same schema as tensor caches. Run --help for inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array(value, dtype=None):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def forbid_test(obj, path=''):
    if re.search(r'(^|[\W_])test([\W_]|$)', Path(path).stem, re.I):
        raise ValueError('Test input forbidden: ' + str(path))
    def visit(x):
        if not isinstance(x, dict):
            return
        for key, value in x.items():
            if key == 'test_accessed' and bool(value):
                raise ValueError('Input declares test_accessed=True')
            if key == 'split' and str(value).lower() not in ('train', 'val', 'validation', 'cal', 'fit'):
                raise ValueError('Unsupported split: ' + str(value))
            if key in ('source_list', 'data_list', 'train_list', 'val_list') and isinstance(value, str):
                if re.search(r'(^|[\W_])test([\W_]|$)', Path(value).stem, re.I):
                    raise ValueError('Input provenance references Test')
            if isinstance(value, dict):
                visit(value)
    visit(obj)


def load(path, trusted=False):
    forbid_test({}, path)
    if Path(path).suffix.lower() == '.json':
        obj = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    else:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError('Use the existing BRIGHT PyTorch environment to read tensor caches.') from error
        try:
            obj = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as error:
            if not trusted:
                raise RuntimeError('Safe cache load failed. Only for your trusted local files, '
                                   'explicitly pass --trusted-pickle. No automatic unsafe fallback.') from error
            obj = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError('Expected a dictionary cache: ' + str(path))
    forbid_test(obj)
    return obj


def table(obj, label_base=1):
    labels = array(obj['labels'], np.int64)
    if labels.ndim != 1:
        raise ValueError('labels must be one-dimensional')
    # Negative values remain unknown. Zero-based I/D/X is explicit, never guessed.
    labels = np.where(labels >= 0, labels + (1-label_base), -1)
    if not np.isin(labels, [-1, 0, 1, 2, 3]).all():
        raise ValueError('Unexpected labels; check --label-base')
    images = obj.get('images', obj.get('names'))
    if images is None or 'events' not in obj:
        raise ValueError('Cache must have images/names and events')
    result = dict(labels=labels, images=np.asarray(images, dtype=str),
                  events=np.asarray(obj['events'], dtype=str))
    for name in ('keys', 'query_keys', 'source_indices', 'region_ids', 'pred_building_ids'):
        if name in obj:
            result[name] = array(obj[name])
    if 'keys' not in result and 'query_keys' in result:
        result['keys'] = result['query_keys']
    for name, value in result.items():
        if len(value) != len(labels):
            raise ValueError('Row count mismatch: ' + name)
    return result


def identities(bank):
    if 'source_indices' in bank:
        return [('source', int(x)) for x in bank['source_indices']], 'source_indices'
    key = 'region_ids' if 'region_ids' in bank else 'pred_building_ids'
    if key in bank:
        return [(str(e), str(i), int(r)) for e, i, r in
                zip(bank['events'], bank['images'], bank[key])], key
    return None, 'unavailable: different regions from one image are NOT treated as duplicates'


def subset(data, rows):
    return {key: value[rows] for key, value in data.items()}


def coverage(bank, candidates=None):
    ids, method = identities(bank)
    result = []
    buckets = set(zip(bank['events'], bank['labels']))
    if candidates is not None:
        buckets |= set(zip(candidates['events'], candidates['labels']))
    for event, cls in sorted(buckets):
        take = np.flatnonzero((bank['events'] == event) & (bank['labels'] == cls))
        counts = Counter(bank['images'][take]); ordered = sorted(counts.values(), reverse=True)
        row = dict(event=event, label=int(cls), entries=len(take), images=len(counts),
                   unique_cases=len({ids[i] for i in take}) if ids is not None else None,
                   max_image_share=ordered[0]/len(take) if len(take) else None,
                   top5_image_share=sum(ordered[:5])/len(take) if len(take) else None,
                   candidate_entries=None, candidate_images=None, image_coverage=None,
                   candidate_max_image_share=None, source_index_min=None, source_index_max=None)
        if 'source_indices' in bank and len(take):
            source = bank['source_indices'][take]
            row.update(source_index_min=int(source.min()), source_index_max=int(source.max()))
        if candidates is not None:
            sel = (candidates['events'] == event) & (candidates['labels'] == cls)
            pool = Counter(candidates['images'][sel]); total = int(sel.sum())
            row.update(candidate_entries=total, candidate_images=len(pool),
                       image_coverage=len(set(counts) & set(pool))/len(pool) if pool else None,
                       candidate_max_image_share=max(pool.values())/total if total else None)
        result.append(row)
    return result, method


class SupportCounts:
    """Exact counts in O(number of events/classes) per query, without Q x M matrix."""
    def __init__(self, bank):
        self.ec = Counter(zip(bank['events'], bank['labels']))
        self.eic = Counter(zip(bank['events'], bank['images'], bank['labels']))
        self.events = sorted(set(bank['events']))

    def per_class(self, event, image, exclude_event, exclude_image, cap):
        counts, capped = [], []
        for cls in (1, 2, 3):
            per_event = []
            for ev in self.events:
                n = 0 if exclude_event and ev == event else self.ec[(ev, cls)]
                if exclude_image and n:
                    n -= self.eic[(ev, image, cls)]
                per_event.append(n)
            counts.append(sum(per_event))
            capped.append(sum(min(n, cap) if cap else n for n in per_event))
        # Global event cap is shared across classes, not the sum of class caps.
        global_cap = 0
        for ev in self.events:
            n = 0 if exclude_event and ev == event else sum(self.ec[(ev,c)] for c in (1,2,3))
            if exclude_image and n:
                n -= sum(self.eic[(ev,image,c)] for c in (1,2,3))
            global_cap += min(n, cap) if cap else n
        return counts, capped, global_cap


def vote(indices, scores, bank_labels, temperature):
    if len(indices) == 0:
        return None
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        return None
    weight = np.exp((scores-scores.max())/temperature)
    weight /= weight.sum()
    posterior = np.bincount(bank_labels[indices], weights=weight, minlength=4)[1:4]
    return int(posterior.argmax()+1)


def inspect_neighbors(indices, scores, bank, event, image, exclude_event, exclude_image,
                      temperature, case_ids=None, declared_valid=None):
    indices = np.asarray(indices, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    if indices.ndim != 1 or scores.shape != indices.shape:
        raise ValueError('Neighbor indices/scores must be aligned vectors')
    valid = (indices >= 0) & (indices < len(bank['labels'])) & np.isfinite(scores)
    if declared_valid is not None:
        declared_valid = np.asarray(declared_valid, dtype=bool)
        if declared_valid.shape != indices.shape:
            raise ValueError('Neighbor valid-mask shape mismatch')
        valid &= declared_valid
    used = indices[valid]; used_scores = scores[valid]
    allowed = np.ones(len(used), dtype=bool)
    if exclude_event:
        allowed &= bank['events'][used] != event
    if exclude_image:
        allowed &= bank['images'][used] != image
    # Dedup-only comparison retains original source eligibility: do not mix two repairs.
    keep, seen = [], set()
    for pos, idx in enumerate(used):
        identity = case_ids[idx] if case_ids is not None else int(idx)
        if identity not in seen:
            seen.add(identity); keep.append(pos)
    unique_legal = {case_ids[idx] if case_ids is not None else int(idx) for idx in used[allowed]}
    return dict(returned_slots=len(indices), active_slots=len(used),
                duplicate_slot_count=len(used)-len(set(used.tolist())),
                duplicate_case_count=len(used)-len(seen) if case_ids is not None else None,
                unique_legal_neighbors=len(unique_legal), forbidden_active_slots=int((~allowed).sum()),
                legal_unique_to_slot_deficit=1-len(unique_legal)/len(indices) if len(indices) else None,
                original_vote=vote(used,used_scores,bank['labels'],temperature),
                dedup_vote=vote(used[keep],used_scores[keep],bank['labels'],temperature),
                dedup_identity='case' if case_ids is not None else 'slot_index_only')


def retrieve(scores, allowed, events, topk, cap, legacy=False):
    """Reference replays: NumPy tie ordering may differ from historical torch.topk."""
    masked = np.where(allowed, scores, -np.inf if cap else -1e4)
    order = np.argsort(-masked, kind='stable')
    if not cap:
        ids = order[:min(topk,len(order))] if legacy else order[allowed[order]][:topk]
        return ids, masked[ids]
    if legacy:
        order = order[:min(len(order), max(50*topk,topk))]
    selected=[]; counts=Counter()
    for idx in order:
        if not allowed[idx] or counts[events[idx]] >= cap:
            continue
        selected.append(int(idx)); counts[events[idx]] += 1
        if len(selected) == topk:
            break
    if legacy and selected:
        selected.extend([selected[-1]]*(topk-len(selected)))
    ids=np.asarray(selected,dtype=np.int64)
    return ids,scores[ids]


def metrics(target, prediction):
    target=np.asarray(target); prediction=np.asarray(prediction)
    valid=np.isin(target,[1,2,3]) & np.isin(prediction,[1,2,3])
    cm=np.zeros((3,3),dtype=np.int64)
    np.add.at(cm,(target[valid].astype(int)-1,prediction[valid].astype(int)-1),1)
    tp=cm.diagonal(); precision=np.divide(tp,cm.sum(0),out=np.zeros(3),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(3),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=precision+recall>0)
    return dict(confusion=cm.tolist(), precision=precision.tolist(),recall=recall.tolist(),
                f1=f1.tolist(), macro_f1=float(f1.mean()), evaluated=int(valid.sum()),
                abstained_or_invalid=int((np.isin(target,[1,2,3]) & ~valid).sum()),
                definition='Region/building only; NOT pixel mIoU. Absent-class F1=0.')


def write_csv(path, rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mode',choices=['bank','evidence'],default='bank')
    p.add_argument('--memory',help='Bank cache with labels/events/images; evidence mode uses --train-cache')
    p.add_argument('--query',required=True,help='Existing Val records, neighbor cache or Stage1 Val cache')
    p.add_argument('--candidates',help='Original Train candidate records for coverage comparison')
    p.add_argument('--train-cache',help='Evidence mode Stage1 Train cache')
    p.add_argument('--fit-scores',help='Evidence mode specialist Train scores containing fit_mask')
    p.add_argument('--neighbors',help='Optional trace: indices[N,K], scores[N,K], optional valid_mask; query_sha256 and memory_sha256 required')
    p.add_argument('--metric',help='Exemplar V2 MetricKeyEncoder checkpoint; used only with --replay')
    p.add_argument('--replay',action='store_true',help='Recompute global cosine kNN only; NOT attention or local reranker')
    p.add_argument('--allow-unverified-key-space',action='store_true',help='Explicitly permit diagnostic replay without matching encoder hashes')
    p.add_argument('--label-base',type=int,choices=[0,1],default=1)
    p.add_argument('--base-pred-field',help='Exact original prediction field; never substitute direct_logits silently')
    p.add_argument('--base-label-base',type=int,choices=[0,1],default=1)
    p.add_argument('--topk',type=int,default=8)
    p.add_argument('--max-per-event',type=int,default=0,help='0=no cap; factorized default is 2')
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--min-purity',type=float,default=.7)
    p.add_argument('--exclude-same-event',action='store_true')
    p.add_argument('--allow-same-image',action='store_true')
    p.add_argument('--trusted-pickle',action='store_true')
    p.add_argument('--output',required=True,help='Must be a NEW output directory')
    a=p.parse_args()
    if a.topk<1 or a.max_per_event<0 or a.temperature<=0 or not 0<=a.min_purity<=1:
        p.error('Invalid K/cap/temperature/purity')
    out=Path(a.output).resolve()
    if out.exists():
        p.error('Output already exists; choose a new directory. No overwrite.')
    paths={k:getattr(a,k) for k in ('memory','query','candidates','train_cache','fit_scores','neighbors','metric') if getattr(a,k)}
    for value in paths.values():
        forbid_test({},value)
    hashes={k:sha(v) for k,v in paths.items()}
    for value in paths.values():
        source=Path(value).resolve()
        if out==source or out in source.parents:
            p.error('Output must not contain any input file')
    query_obj=load(a.query,a.trusted_pickle); warnings=[]
    if query_obj.get('split')=='train':
        p.error('Query must be Val, not Train')
    if a.mode=='evidence':
        if not a.train_cache or not a.fit_scores or a.replay or a.neighbors:
            p.error('Evidence mode requires --train-cache --fit-scores and performs counts only (no replay/trace)')
        train_obj=load(a.train_cache,a.trusted_pickle); fit_obj=load(a.fit_scores,a.trusted_pickle)
        if train_obj.get('split') not in ('train',None):
            p.error('Evidence support must be Train')
        if train_obj.get('source_checkpoint_sha256') != query_obj.get('source_checkpoint_sha256'):
            p.error('Evidence Train/Val feature checkpoint mismatch')
        if not train_obj.get('source_checkpoint_sha256'):
            warnings.append('Evidence feature checkpoint provenance UNVERIFIED: hashes absent.')
        bank_all=table(train_obj,0); query=table(query_obj,0)
        fit=array(fit_obj['fit_mask'],bool)
        if fit.shape!=bank_all['labels'].shape: p.error('fit_mask row count mismatch')
        keep=fit & (array(train_obj['label_purity'])>=a.min_purity) & (bank_all['labels']>=1)
        bank=subset(bank_all,keep);bank['source_indices']=np.flatnonzero(keep)
        candidates=bank_all
        # Match history.py: query only buildings predicted Intact.
        rows=np.flatnonzero(array(query_obj['base_pred'])==0)
        query=subset(query,rows); base=array(query_obj['base_pred'],np.int64)[rows]+1
        a.exclude_same_event=True
        warnings.append('Evidence mode tests insufficient per-class support preconditions only; local reranker not executed.')
        warnings.append('fit_mask is linked by row order; original fit-cache hash unavailable unless separately recorded.')
    else:
        if not a.memory: p.error('Bank mode requires --memory')
        memory_obj=load(a.memory,a.trusted_pickle)
        if memory_obj.get('split') not in ('train',None): p.error('Memory must be Train')
        bank=table(memory_obj,a.label_base);query=table(query_obj,a.label_base)
        candidate_obj=load(a.candidates,a.trusted_pickle) if a.candidates else None
        if candidate_obj is not None and candidate_obj.get('split') not in ('train',None):
            p.error('Candidate pool must be Train')
        candidates=table(candidate_obj,a.label_base) if candidate_obj is not None else None
        if a.candidates and memory_obj.get('metadata',{}).get('records_sha256'):
            if memory_obj['metadata']['records_sha256']!=hashes['candidates']:
                p.error('Candidate records do not match Memory records hash')
        base=None;rows=np.arange(len(query['labels']))
        if a.base_pred_field:
            base=array(query_obj[a.base_pred_field],np.int64)+(1-a.base_label_base)
            if base.shape!=query['labels'].shape: p.error('Base prediction row count mismatch')
        else:
            warnings.append('Original prediction unspecified: Damaged->Intact subgroup unavailable.')
        if a.replay and 'keys' not in query:
            if not a.metric or 'descriptors' not in query_obj:
                p.error('Replay requires query keys/query_keys, or Exemplar descriptors plus --metric')
            import torch
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
            from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.core import MetricKeyEncoder
            ck=load(a.metric,a.trusted_pickle)
            model=MetricKeyEncoder(ck['input_dim'],ck['key_dim']).eval()
            model.load_state_dict(ck['model'],strict=True)
            with torch.inference_mode():
                query['keys']=np.concatenate([model(torch.as_tensor(query_obj['descriptors'][i:i+512]).float())[0].numpy()
                                               for i in range(0,len(query['labels']),512)])
        if a.replay:
            meta=memory_obj.get('metadata',{}); qmeta=query_obj.get('metadata',{})
            bank_hash=meta.get('encoder_sha256',meta.get('metric_sha256'))
            qhash=hashes.get('metric') or qmeta.get('encoder_sha256',qmeta.get('metric_sha256'))
            if bank_hash and qhash and bank_hash!=qhash: p.error('Key encoder hash mismatch')
            if not bank_hash or not qhash:
                if not a.allow_unverified_key_space: p.error('Key-space provenance unverified; use explicit override only for diagnostics')
                warnings.append('Key-space provenance UNVERIFIED; replay cannot certify historical results.')
            if 'keys' not in bank: p.error('Memory has no keys')
            for data in (bank,query):
                key=array(data['keys'],float)
                if key.ndim!=2 or not np.isfinite(key).all(): p.error('Invalid keys')
                data['keys']=key/np.maximum(np.linalg.norm(key,axis=1,keepdims=True),1e-12)
            if bank['keys'].shape[1]!=query['keys'].shape[1]: p.error('Key dimensions differ')
            warnings.append('Replay is NumPy cosine kNN, not historical torch tie-order or attention output.')
    if not len(bank['labels']) or not np.isin(bank['labels'],[1,2,3]).all():
        p.error('Support bank must be nonempty and contain only I/D/X labels')
    if not len(query['labels']): p.error('Empty query set')
    trace=None
    if a.neighbors:
        trace=load(a.neighbors,a.trusted_pickle);meta=trace.get('metadata',trace)
        if meta.get('query_sha256')!=hashes['query'] or meta.get('memory_sha256')!=hashes['memory']:
            p.error('Trace query_sha256/memory_sha256 mismatch or missing; do not infer row alignment')
        trace={k:array(trace[k]) for k in ('indices','scores','valid_mask') if k in trace}
        if 'indices' not in trace or 'scores' not in trace: p.error('Trace needs indices and scores')
        if not np.issubdtype(trace['indices'].dtype,np.integer):
            p.error('Trace indices must be integers')
        if trace['indices'].ndim!=2 or trace['indices'].shape!=trace['scores'].shape or len(trace['indices'])!=len(query['labels']):
            p.error('Trace shape mismatch')
        if 'valid_mask' in trace and trace['valid_mask'].shape!=trace['indices'].shape:
            p.error('Trace valid_mask shape mismatch')
    if trace and a.replay: p.error('Use separate runs for historical trace and recomputed replay')
    cov,identity_method=coverage(bank,candidates); case_ids,_=identities(bank)
    counter=SupportCounts(bank); results=[]
    if case_ids is not None:
        first={}
        for i,key in enumerate(case_ids): first.setdefault(key,i)
        unique_bank=subset(bank,np.asarray(list(first.values()),dtype=np.int64))
        unique_counter=SupportCounts(unique_bank)
    else:
        unique_counter=None
    for i,(event,image,label) in enumerate(zip(query['events'],query['images'],query['labels'])):
        legal,capped,global_cap=counter.per_class(event,image,a.exclude_same_event,not a.allow_same_image,a.max_per_event)
        row=dict(query_row=int(rows[i]),event=str(event),image=str(image),gt=int(label),
                 base_pred=int(base[i]) if base is not None else None,
                 damaged_to_intact=bool(label==2 and base[i]==1) if base is not None else None,
                 global_event_capped_support=global_cap)
        for j,cls in enumerate(('intact','damaged','destroyed')):
            row['legal_'+cls]=legal[j]; row['capped_'+cls]=capped[j]
            row['short_'+cls]=legal[j]<a.topk
        if unique_counter is not None:
            unique_counts,_,unique_cap=unique_counter.per_class(event,image,a.exclude_same_event,
                                                               not a.allow_same_image,a.max_per_event)
            for j,cls in enumerate(('intact','damaged','destroyed')):
                row['unique_legal_'+cls]=unique_counts[j]
            row['unique_global_event_capped_support']=unique_cap
        if trace:
            row.update(inspect_neighbors(trace['indices'][i],trace['scores'][i],bank,event,image,
                       a.exclude_same_event,not a.allow_same_image,a.temperature,case_ids,
                       trace['valid_mask'][i] if 'valid_mask' in trace else None))
        elif a.replay:
            scores=bank['keys']@query['keys'][i];allowed=np.ones(len(scores),bool)
            if a.exclude_same_event: allowed &= bank['events']!=event
            if not a.allow_same_image: allowed &= bank['images']!=image
            idx,vals=retrieve(scores,allowed,bank['events'],a.topk,a.max_per_event,True)
            row.update(inspect_neighbors(idx,vals,bank,event,image,a.exclude_same_event,
                       not a.allow_same_image,a.temperature,case_ids))
            fixed,fs=retrieve(scores,allowed,bank['events'],a.topk,a.max_per_event,False)
            row['full_legal_vote']=vote(fixed,fs,bank['labels'],a.temperature)
        results.append(row)
        if (i+1)%1000==0: print(f'audited {i+1}/{len(query["labels"])}',flush=True)
    def summarize(group):
        n=len(group); summary=dict(queries=n)
        for cls in ('intact','damaged','destroyed'):
            counts=[r['legal_'+cls] for r in group]
            summary[cls+'_support']=dict(zero=sum(x==0 for x in counts),
                positive_below_k=sum(0<x<a.topk for x in counts),at_least_k=sum(x>=a.topk for x in counts))
        for key in ('duplicate_slot_count','duplicate_case_count','forbidden_active_slots'):
            available=[r[key] for r in group if r.get(key) is not None]
            summary[key+'_affected_fraction']=sum(x>0 for x in available)/len(available) if available else None
        for mode in ('original_vote','dedup_vote','full_legal_vote'):
            if not any(mode in r for r in group): continue
            pred=[r.get(mode) if r.get(mode) is not None else -1 for r in group]
            summary[mode]=metrics([r['gt'] for r in group],pred)
            pairs=[r for r in group if r.get('original_vote') is not None and r.get(mode) is not None]
            summary[mode+'_paired_change']=dict(eligible=len(pairs),
                changed=sum(r[mode]!=r['original_vote'] for r in pairs),
                damaged_rescued=sum(r['gt']==2 and r['original_vote']!=2 and r[mode]==2 for r in pairs),
                intact_new_false_damage=sum(r['gt']==1 and r['original_vote']==1 and r[mode]==2 for r in pairs))
        return summary
    summary=dict(all=summarize(results),damaged=summarize([r for r in results if r['gt']==2]),
                 damaged_to_intact=summarize([r for r in results if r['damaged_to_intact']]) if base is not None else None,
                 per_event={ev:summarize([r for r in results if r['event']==ev]) for ev in sorted(set(query['events']))},
                 case_identity=identity_method,warnings=warnings,
                 neighbor_analysis='historical_trace' if trace else 'reference_replay' if a.replay else 'not_available_counts_only',
                 limitations=['No attention or local-region reranker replay.',
                              'legal_* counts bank rows; unique_legal_* counts cases only when identity exists.',
                              'No alternative-bank sampling experiment in this script.',
                              'Existing query selection is preserved; GT-filtered caches remain filtered.',
                              'Missing prediction is abstention, not a fabricated class. Compare coverage before metrics.'])
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'coverage.csv',cov);write_csv(out/'queries.csv',results)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    manifest=dict(arguments=vars(a),input_sha256=hashes,inputs=paths,script_sha256=sha(__file__),
                  training=False,test_accessed=False,original_files_modified=False)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved audit to '+str(out))


if __name__=='__main__':
    main()
