#!/usr/bin/env python3
"""Audit saved soft-slot / Memory evidence. No optimizer and no Test access."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D'
DEFAULT_TOOL = '/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2'
FORMAT = 'SRAC_SLOT_MEMORY_AUDIT_V1'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def release_hashes():
    from source_io import digest, read_json
    hashes = read_json(HERE / 'SHA256SUMS.json')
    for name, expected in hashes.items():
        p = HERE / name
        if p.is_symlink() or p.parent != HERE or not p.is_file() or digest(p) != expected:
            raise ValueError('Audit release file missing/changed: '+str(p))
    actual = {p.name for p in HERE.glob('*.py')}
    if not actual <= set(hashes):
        raise ValueError('Unexpected Python file in audit release')
    return hashes


def status_base(args):
    return dict(format=FORMAT, complete=False, execution_complete=False,
                full_validation_evidence=False, training=False, test_accessed=False,
                primary_endpoint=False, checkpoint_step=args.step, image_count=0,
                smoke=bool(getattr(args, 'smoke', False)), final_integrity_pending=True,
                arguments=vars(args), error=None)


def source_static(args):
    from source_io import validate_source
    return validate_source(args.train_root, args.tool_dir, args.step)


def validate_destination(path, static):
    """Reject overlapping output before creating a directory or a lock."""
    destination = Path(path).resolve()
    config = static['manifest']['arguments']
    protected = [static['root'], static['tool'], HERE]
    protected += [config[k] for k in ('bank_root','values') if config.get(k)]
    for source in protected:
        source = Path(source).resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError('Audit output overlaps protected source: '+str(source))


def validate_worker_result(folder, *, smoke, step):
    from source_io import read_json, digest
    folder = Path(folder)
    summary = read_json(folder/'summary.json')
    result = read_json(folder/'result.json')
    for row in (summary,result):
        if (row.get('format') != FORMAT or row.get('complete') is not True
                or row.get('execution_complete') is not True
                or row.get('final_integrity_pending') is not False
                or row.get('full_validation_evidence') is not (not smoke)
                or row.get('training') is not False or row.get('test_accessed') is not False
                or row.get('smoke') is not smoke or row.get('checkpoint_step') != step
                or row.get('image_count') != (2 if smoke else 450)):
            raise ValueError('Worker returned without required diagnostic evidence: '+str(folder))
    if summary.get('result_sha256') != digest(folder/'result.json'):
        raise ValueError('Worker result checksum differs from summary')
    if not smoke and result.get('replay',{}).get('passed') is not True:
        raise ValueError('Formal worker did not pass original Val450 replay')
    return True


def prepare_heads(modules, saved, device):
    heads = {}
    for arm in ('b', 'd'):
        head = modules.Classifier(chunk_size=saved['config']['chunk_size']).to(device)
        head.load_state_dict(saved['heads'][arm], strict=True)
        head.eval().requires_grad_(False)
        heads[arm] = head
    return heads


def analyze_example(example, data, heads, memory_images, memory_events, chunk_size):
    """Model predictions first; GT is only consulted in the post-hoc statistics."""
    import torch
    from slot_stats import slot_diagnostics
    from neighbor_stats import neighbor_diagnostics
    from reader_stats import reader_diagnostics, add_posthoc_label_scores
    with torch.no_grad():
        decoded = {arm: head(**example.common, **example.supports[arm]) for arm, head in heads.items()}
        for arm, result in decoded.items():
            probability = result['probabilities']
            if (not bool(torch.isfinite(probability).all()) or bool((probability < 0).any())
                    or not torch.allclose(probability.sum(1),torch.ones_like(probability[:,0]),atol=2e-5,rtol=0)):
                raise ValueError('Invalid classifier probability before argmax: '+arm)
        predictions = {'grid_probq': example.common['base_probability_full'].argmax(1)[0].cpu().numpy()}
        predictions.update({arm: result['probabilities'].argmax(1)[0].cpu().numpy()
                            for arm, result in decoded.items()})
        # Neither query GT nor any oracle statistic enters the above calls.
        slots = slot_diagnostics(data['feature'], data['result']['q'], data['result']['ids'],
                                 data['building'], example.target, data['slots']['soft'])
        # Candidate/Reader rows describe role1. Separate GT mass in that role
        # from spatial strata based on the whole parent region's GT mass.
        for cls, name in ((1,'intact'),(2,'damaged')):
            slots['weights']['role1_gt_'+name] = slots['query_label_mass'][:,1] * slots['query_labels'][:,1,cls]
        local, memory = example.supports['b'], example.supports['d']
        neighbors = neighbor_diagnostics(
            local['slot_f'][0], slots['query_labels'], local['slot_valid'][0],
            memory['support_ids'], memory['support_f'], memory['support_labels'], memory['support_valid'],
            memory_images, memory_events, local['support_ids'], local['support_labels'], local['support_valid'],
            chunk_size=chunk_size)
        rows = dict(slots['region_rows'])
        rows.update(neighbors['region_rows'])
        for arm in ('b', 'd'):
            observed = reader_diagnostics(heads[arm], decoded[arm], example.supports[arm], chunk_size=chunk_size)
            add_posthoc_label_scores(observed, slots['query_labels'], slots['query_valid'])
            rows.update({'reader_'+arm+'_'+name:value for name,value in observed.items()})
    return dict(slots=slots, rows=rows, predictions=predictions, neighbor_definitions=neighbors['definitions'])


def selected_cases(example, analysis, memory_images, memory_events):
    """At most four deterministic post-hoc examples per image; not unbiased samples."""
    import math
    import torch
    slots, rows = analysis['slots'], analysis['rows']
    label = slots['query_labels'][:, 1]
    valid = slots['query_valid'][:, 1]
    eligible = valid & (label[:, 1:].sum(-1) >= .2)
    rankings = [
        ('highest_damaged_label_mass', slots['query_label_mass'][:,1] * label[:,2]),
        ('largest_slot_label_difference', rows['label_total_variation']),
        ('intact_with_memory_damaged_exposure', label[:,1] * rows['reader_d_damaged_label_mass']),
        ('largest_memory_minus_local_damaged_evidence', rows['memory_minus_local_pred_vote_damaged']),
    ]
    selected, out = set(), []
    support = example.supports['d']
    for reason, score in rankings:
        ok = eligible & torch.isfinite(score)
        if reason == 'highest_damaged_label_mass':
            ok = ok & (score > 0)
        if reason == 'intact_with_memory_damaged_exposure':
            ok = ok & (label[:,1] >= .5) & (score > 0)
        for rid in selected:
            ok[rid] = False
        if not bool(ok.any()):
            continue
        rid = int(score.masked_fill(~ok, -float('inf')).argmax())
        selected.add(rid)
        candidates = []
        for mid in support['support_ids'][rid].cpu().tolist()[:8]:
            if mid < 0:
                continue
            candidates.append(dict(parent_id=mid, image=memory_images[mid], event=memory_events[mid],
                                   slot_labels=support['support_labels'][mid].cpu().tolist(),
                                   slot_valid=support['support_valid'][mid].cpu().tolist()))
        fields = {}
        for name in ('feature_cosine','label_total_variation','slot0_ess','slot1_ess',
                     'building_probability_variance','memory_unique_source_images',
                     'memory_unique_source_events','memory_damaged50_unique_images',
                     'memory_intact50_unique_images','reader_d_damaged_label_mass',
                     'reader_b_damaged_label_mass'):
            v = float(rows[name][rid])
            fields[name] = v if math.isfinite(v) else None
        out.append(dict(image=example.image,event=example.event,parent_region_id=rid,reason=reason,
                        selection='posthoc selected for inspection; not an unbiased success rate',
                        query_slot_labels=slots['query_labels'][rid].cpu().tolist(),
                        query_slot_valid=slots['query_valid'][rid].cpu().tolist(), metrics=fields,
                        displayed_candidates=candidates, candidate_display_limit=8,
                        actual_parent_budget=support['support_ids'].shape[1]))
    return out


def worker(args):
    from source_io import (import_runtime, load_saved, runtime_arguments, check_runtime_inputs,
                           check_source_unchanged, digest)
    output = Path(args.output).resolve()
    sealed = release_hashes()
    static = source_static(args)
    validate_destination(output,static)
    output.mkdir(parents=True, exist_ok=False)
    state = status_base(args)
    write_json(output/'summary.json', state)
    started = time.monotonic()
    try:
        modules = import_runtime(static)
        torch = modules.torch
        from slot_stats import RegionAccumulator
        from pixel_stats import PixelAccumulator, check_replay
        from reporting import write_tables
        from neighbor_stats import inventory
        saved = load_saved(static, modules)
        runtime_args = runtime_arguments(saved, output)

        class CaptureRuntime(modules.Runtime):
            def extract(self, batch):
                data = super().extract(batch)
                self.audit_data = data
                return data

        region = RegionAccumulator()
        pixel = PixelAccumulator()
        events = {}
        names_seen = []
        with CaptureRuntime(runtime_args, output) as runtime:
            modules.train._formal_runtime(runtime, False)
            if runtime.train_names != static['manifest']['train_names'] or runtime.val_names != static['manifest']['val_names']:
                raise ValueError('Runtime Train/Val order differs from original manifest')
            check_runtime_inputs(runtime, saved)
            names = runtime.smoke_names()[1] if args.smoke else list(runtime.val_names)
            if args.smoke and (len(names) != 2 or len({runtime.deps.event(n) for n in names}) != 2):
                raise ValueError('Smoke requires two distinct-event Val images')
            if any(n not in runtime.val_names for n in names):
                raise ValueError('Audit may only forward Val images')
            write_json(output/'sample_names.json', dict(split='Val', names=names,
                       order_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest()))
            heads = prepare_heads(modules, saved, runtime.device)
            head_before = {a:modules.common.module_hash(h) for a,h in heads.items()}
            if head_before != {a:static['meta']['head_hashes'][a] for a in heads}:
                raise ValueError('Loaded head tensor hash mismatch')
            memory_images = [runtime.train_names[i] for i in runtime.bank['images'].cpu().tolist()]
            memory_events = [runtime.bank['event_names'][i] for i in runtime.bank['events'].cpu().tolist()]
            table = runtime.values['rules']['soft']
            inv = inventory(table['slot_labels'], table['slot_valid'], memory_images, memory_events)
            state.update(phase='forward', planned_images=len(names), inventory=inv)
            write_json(output/'summary.json', state)
            coverage = modules.train._blank_coverage()
            with (output/'images.jsonl').open('w',encoding='utf8') as image_log, (output/'cases.jsonl').open('w',encoding='utf8') as case_log:
                for index,batch in enumerate(runtime.batches(names, train=False, seed=runtime_args.seed)):
                    with torch.no_grad():
                        example = runtime.prepare(batch)
                        modules.train._assert_example(example)
                        if index >= len(names) or example.image != names[index]:
                            raise ValueError('Val sample order changed')
                        modules.train._merge_coverage(coverage, modules.train._coverage(example))
                        data = runtime.audit_data
                        analysis = analyze_example(example,data,heads,memory_images,memory_events,runtime_args.chunk_size)
                    names_seen.append(example.image)
                    region.add(analysis['rows'], analysis['slots']['weights'])
                    masks = {k:v.cpu().numpy() for k,v in analysis['slots']['pixel_masks'].items()}
                    target = example.target[0].cpu().numpy()
                    pixel.add(target,analysis['predictions'],masks)
                    if example.event not in events:
                        events[example.event] = (RegionAccumulator(),PixelAccumulator())
                    er,ep = events[example.event]
                    er.add(analysis['rows'],analysis['slots']['weights'])
                    ep.add(target,analysis['predictions'],masks)
                    ip = PixelAccumulator()
                    ip.add(target,analysis['predictions'],masks)
                    image_row = dict(image=example.image,event=example.event, pixels=ip.report())
                    image_log.write(json.dumps(image_row,ensure_ascii=False,allow_nan=False)+'\n')
                    image_log.flush()
                    for row in selected_cases(example,analysis,memory_images,memory_events):
                        case_log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    case_log.flush()
                    state.update(image_count=len(names_seen), elapsed_seconds=time.monotonic()-started)
                    write_json(output/'summary.json',state)
                    if len(names_seen)%10 == 0 or args.smoke:
                        print(f'SLOT/MEMORY DIAGNOSTIC step={args.step} Val {len(names_seen)}/{len(names)}; final checks pending',flush=True)
                    # Do not retain decoded tensors / pixel arrays across images.
                    neighbor_definitions = analysis['neighbor_definitions']
                    del analysis, example, data, target, masks, ip
                    runtime.audit_data = None
            if names_seen != names:
                raise ValueError('Val loader stopped before all selected images')
            if coverage['d']['supported_parent_queries'] == 0:
                raise ValueError('No historical supports observed; this cannot validate Memory')
            check_runtime_inputs(runtime,saved)
            integrity = modules.train._checked_verify(runtime)
            if {a:modules.common.module_hash(h) for a,h in heads.items()} != head_before:
                raise ValueError('A diagnostic head changed during inference')
            input_hashes = {str(p):sha for p,sha in runtime.input_ledger.items()}
        guard = modules.train._restored_guard(runtime)
        check_source_unchanged(static)
        if release_hashes() != sealed:
            raise ValueError('Audit source changed during execution')
        pixels = pixel.report()
        replay = dict(passed=None, reason='Two-image smoke cannot reproduce full Val450')
        if not args.smoke:
            replay = check_replay(pixels['all_labeled']['metrics'],static['reference'],tolerance_pp=.001)
            if replay['passed'] is not True:
                write_json(output/'replay_failure.json',replay)
                raise ValueError('Original selected checkpoint Val450 replay failed; see replay_failure.json')
        result = dict(state, phase='complete_smoke' if args.smoke else 'complete',
                      complete=True, execution_complete=True, final_integrity_pending=False,
                      full_validation_evidence=not args.smoke, elapsed_seconds=time.monotonic()-started,
                      regions=region.report(), pixels=pixels, inventory=inv, replay=replay,
                      coverage=coverage, integrity=integrity, guard_restored=guard,
                      per_event={e:dict(regions=r.report(),pixels=p.report()) for e,(r,p) in events.items()},
                      neighbor_definitions=neighbor_definitions,
                      interpretation='Read-only diagnostics of saved B/D; not a trained ablation or stable Memory gain')
        write_json(output/'input_fingerprints.json',input_hashes)
        write_json(output/'source_fingerprints.json',dict(original=static['old_files'],audit=sealed))
        write_json(output/'result.json',result)
        write_tables(output,result)
        state.update({k:result[k] for k in ('complete','execution_complete','final_integrity_pending',
                     'full_validation_evidence','phase','elapsed_seconds')})
        state['result_sha256'] = digest(output/'result.json')
        write_json(output/'summary.json',state)
        print((output/'REPORT.txt').read_text(),flush=True)
        print('SLOT/MEMORY DIAGNOSTIC COMPLETE. No training. No Test.',flush=True)
    except BaseException as exc:
        state.update(phase='failed',error=f'{type(exc).__name__}: {exc}',complete=False,
                     execution_complete=False,full_validation_evidence=False)
        write_json(output/'summary.json',state)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise


def pipeline(args):
    output = Path(args.output).resolve()
    release_hashes()
    static = source_static(args)
    validate_destination(output,static)
    # Lock is per source/checkpoint, so another audit cannot silently duplicate this one.
    output.parent.mkdir(parents=True,exist_ok=True)
    key = hashlib.sha256((str(Path(args.train_root).resolve())+':'+str(args.step)).encode()).hexdigest()[:20]
    with (output.parent/('.audit_'+key+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An audit of this source/checkpoint is already running')
        output.mkdir(exist_ok=False)
        (output/'logs').mkdir()
        state=status_base(args)
        state.update(mode=args.command,stages={})
        write_json(output/'workflow_state.json',state)
        try:
            release_hashes()
            source_static(args)
            common=['--train-root',args.train_root,'--tool-dir',args.tool_dir,'--step',str(args.step)]
            commands=[('cpu_contracts',[sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-q'])]
            commands.append(('smoke',[sys.executable,'-u',str(HERE/'audit.py'),'worker',*common,
                                      '--smoke','--output',str(output/'smoke')]))
            if args.command=='full':
                commands.append(('diagnose',[sys.executable,'-u',str(HERE/'audit.py'),'worker',*common,
                                            '--output',str(output/'diagnose')]))
            for stage,cmd in commands:
                state['current_stage']=stage
                row=state['stages'][stage]=dict(status='running',command=cmd)
                write_json(output/'workflow_state.json',state)
                print('START '+stage,flush=True)
                with (output/'logs'/(stage+'.log')).open('w',encoding='utf8') as log:
                    child=subprocess.Popen(cmd,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                           text=True,bufsize=1,pass_fds=(lock.fileno(),),
                                           env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                    row['pid']=child.pid
                    write_json(output/'workflow_state.json',state)
                    for line in child.stdout:
                        log.write(line); log.flush(); print(line,end='',flush=True)
                    code=child.wait()
                row.update(status='returned' if code==0 else 'failed',returncode=code)
                write_json(output/'workflow_state.json',state)
                if code:
                    raise RuntimeError(f'{stage} failed with returncode {code}; see logs/{stage}.log')
                if stage != 'cpu_contracts':
                    validate_worker_result(output/stage,smoke=stage=='smoke',step=args.step)
                row['status']='complete'
                write_json(output/'workflow_state.json',state)
            state.update(complete=True,execution_complete=True,final_integrity_pending=False,
                         full_validation_evidence=args.command=='full',current_stage='complete')
            write_json(output/'workflow_state.json',state)
        except BaseException as exc:
            state.update(error=f'{type(exc).__name__}: {exc}')
            write_json(output/'workflow_state.json',state)
            (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
            raise


def report(root):
    from source_io import read_json
    root=Path(root)
    print('诊断目录：',root)
    if (root/'workflow_state.json').is_file():
        d=read_json(root/'workflow_state.json')
        print(json.dumps({k:d.get(k) for k in ('complete','full_validation_evidence','current_stage','error')},ensure_ascii=False,indent=2))
    for path in (root/'diagnose',root/'smoke',root):
        if (path/'REPORT.txt').is_file():
            print((path/'REPORT.txt').read_text(encoding='utf8'))
            return
        if (path/'summary.json').is_file():
            d=read_json(path/'summary.json')
            print(path.name,json.dumps({k:d.get(k) for k in ('phase','image_count','complete','error')},ensure_ascii=False))
    print('尚无完整诊断报告；以上状态不等于进程当前存活。')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('plan','check','full','worker','report'))
    parser.add_argument('--train-root',default=DEFAULT_ROOT)
    parser.add_argument('--tool-dir',default=DEFAULT_TOOL)
    parser.add_argument('--step',type=int,choices=(500,1500,5000),default=500)
    parser.add_argument('--output')
    parser.add_argument('--root')
    parser.add_argument('--smoke',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.command=='report':
        if not args.root: parser.error('report requires --root')
        return report(args.root)
    if args.command=='plan':
        release_hashes()
        static=source_static(args)
        print('来源和检查点文件核验通过；尚未进行真实 CUDA 前向。')
        print('检查点：',static['checkpoint'])
        print('计划：原 B/D；Val450；Train库只读；不训练、不访问Test。')
        return
    if not args.output: parser.error('--output required')
    if args.smoke and args.command!='worker': parser.error('--smoke is internal')
    return worker(args) if args.command=='worker' else pipeline(args)


if __name__=='__main__':
    main()
