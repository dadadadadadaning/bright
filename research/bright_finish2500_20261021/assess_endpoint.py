"""Fresh synchronized process per assessment, preserving original CUDA training settings."""
import copy
import json
import os
from pathlib import Path
import sys
import traceback

from grid_probe import read_json, sha256
from evaluate_checkpoint import write_json, ARMS, REQUIRED_CHECKS, concise_metrics
from resume_train import verify_files, require_scope
from probe import json_value
from sync_val2000 import validate_protocol

FORMAT = 'BRIGHT_FINISH2500_ASSESSMENT_V1'
SPECS = {'fixed2000':(2000,'fixed_train'), 'val2500':(2500,'val'),
         'fixed2500':(2500,'fixed_train'), 'test2500':(2500,'test')}


def finalize(path, report, *, output, names, evidence, checks, provenance, primary):
    step,split = provenance['checkpoint_step'],report['split']
    if Path(path).resolve() != Path(output).resolve() / ('%s_step_%06d' % (split,step)) / 'result.json':
        raise ValueError('Unexpected fresh report path')
    if set(checks) != set(REQUIRED_CHECKS) or any(v is not True for v in checks.values()):
        raise ValueError('Assessment integrity incomplete')
    if read_json(path) != report or report['step'] != step or report['image_count'] != len(names):
        raise ValueError('Saved assessment differs')
    m = read_json(Path(output)/'manifest.json')
    if m['protocol_info'] != provenance['protocol_info'] or m['names'] != names:
        raise ValueError('Assessment execution metadata differs')
    if split == 'test':
        if len(names) != 878 or step != 2500 or report['test_accessed'] is not True:
            raise ValueError('Test must be all878 at fixed2500')
        if report['journal_sha256'] != sha256(Path(path).with_name('images.jsonl')):
            raise ValueError('Test journal changed')
    else:
        evidence.validate_report(report,step=step,split=split,names=names,arms=list(ARMS),primary=primary)
        evidence.validate_saved_report(path,report,names,sha256,output)
    with Path(path).with_name('result.pending.json').open('xb') as stream:
        stream.write(Path(path).read_bytes())
    finished = copy.deepcopy(report)
    finished.update(complete=True,phase='standalone_evaluation_complete',final_integrity_pending=False,
        full_validation_evidence=split=='val', full_test_evidence=split=='test',
        primary_endpoint=bool(primary),optimizer_updates=0,protocol_info=provenance['protocol_info'],
        standalone_provenance=provenance,standalone_integrity_checks=checks,
        pending_report_sha256=sha256(path),historical_execution_replayed=False)
    write_json(path,finished)
    Path(path).with_name('result.json.sha256').write_text(sha256(path)+'  result.json\n')
    if split != 'test': evidence.validate_saved_report(path,finished,names,sha256,output)
    return finished


def run_assessment(context, output, assessment):
    step,split = SPECS[assessment]
    primary = split == 'val' and step == context['plan']['endpoint_step']
    parent_root = Path(context['parent_journal']).parent
    status = dict(format=FORMAT,phase='loading',complete=False,full_validation_evidence=False,
        full_test_evidence=False,training_performed=False,optimizer_updates=0,
        test_accessed=False,test_requested=split=='test',checkpoint_step=step,split=split,
        source_root=str(parent_root),output=str(output),cuda_launch_blocking='1',pid=os.getpid())
    position = dict(index=-1,image=None,active_arm=None,stage='loading')
    observer = None
    folder = output / ('%s_step_%06d' % (split,step))
    write_json(output/'summary.json',status)
    try:
        manifest,plan,bridge = (context[k] for k in ('manifest','plan','bridge'))
        source,addon = plan['source'],context['addon']
        os.environ['CUDA_LAUNCH_BLOCKING']='1'
        verify_files(context['evidence_files'])
        import torch
        import evidence
        import checkpoint as checkpoint_io
        from evaluation_observer import EvaluationObserver,track_evaluation
        from restore_state import restore_training_state,validate_restored_rng
        import test_endpoint
        policy = read_json(source['config']['protocol'])['numeric_policy']
        torch.set_num_threads(int(policy['num_threads']))
        if torch.get_num_interop_threads() != int(policy['num_interop_threads']):
            torch.set_num_interop_threads(int(policy['num_interop_threads']))
        if not torch.cuda.is_available(): raise ValueError('Original CUDA server required')
        bindings = bridge.load_runtime(source)
        state = bindings.legacy.state
        sys.modules['runtime_bridge']=bridge
        sys.path.insert(0,str(addon))
        runner = bridge.named(addon/'ablation_runner.py','endpoint_runner')
        support_module = bridge.named(addon/'supports.py','endpoint_supports')
        evaluator = bridge.named(addon/'evaluation.py','endpoint_evaluation')
        saved,cp = checkpoint_io.load(torch,context['parent']['path'],state,plan=plan,
            original_manifest=manifest,lineage=context['lineage'],journal_path=context['parent_journal'])
        if saved['step'] != step: raise ValueError('Checkpoint step differs')
        inherited = saved['input_file_sha256']
        print('ASSESS %s: verifying %s inherited inputs' % (assessment,len(inherited)),flush=True)
        verify_files(inherited)
        runtime_output = output/'runtime'
        runtime_output.mkdir()
        args = bridge.runtime_arguments(source,seed=context['seed'],output=runtime_output)
        with bindings.Runtime(args,runtime_output) as runtime:
            bindings.original_train._formal_runtime(runtime,False)
            train_names,val_names = bindings.original_train._names(runtime,False)
            if train_names != manifest['train_names'] or val_names != manifest['val_names']:
                raise ValueError('Original Train/Val split changed')
            override = validate_protocol(runtime.protocol_info,manifest['protocol_info'])
            eval_protocol = json_value(runtime.protocol_info)
            bindings.original_train._checked_verify(runtime)
            factory = support_module.SupportFactory(runtime,seed=context['seed'])
            heads,optimizers,_ = runner.model_copies(torch,bindings.original_train,state,ARMS,vars(args),runtime.device)
            status['restoration'] = restore_training_state(torch,heads,optimizers,saved,state)
            require_scope(runtime,inherited)
            extra_inputs,split_files = {},{}
            if split == 'test':
                status['test_accessed']=True
                write_json(output/'summary.json',status)
                names,extra_inputs,split_files = test_endpoint.declare_test(runtime,manifest,repo=source['config']['repo'])
            else:
                names = val_names if split == 'val' else manifest['fixed_train_names']
                if split == 'val' and len(names) != 450: raise ValueError('Full Val450 required')
            status['rng_restoration'] = validate_restored_rng(state,saved['rng'],require_cuda=True)
            del saved
            before_heads = {a:state.state_fingerprint(h.state_dict()) for a,h in heads.items()}
            before_opt = {a:state.state_fingerprint(o.state_dict()) for a,o in optimizers.items()}
            before_rng = state.state_fingerprint(state.capture_rng())
            provenance = dict(source_root=str(parent_root),checkpoint_step=step,checkpoint_sha256=cp['sha256'],
                trainer_sha256=sha256(parent_root/'trainer.pyz'),lineage=json_value(context['lineage']),
                original_protocol_info=manifest['protocol_info'], protocol_info=json_value(runtime.protocol_info),
                pre_test_protocol_info=eval_protocol, launch_blocking_override=override,
                evaluation_source_sha256=sha256(addon/'evaluation.py'),optimizer_updates=0,
                split=split,test_accessed=split=='test',test_adapter='test_endpoint.py' if split=='test' else None,
                checkpoint_selection='fixed2500_all_six_arms' if split=='test' else 'original_schedule',
                test_split_files=split_files)
            write_json(output/'manifest.json',dict(format=FORMAT,source_checkpoint=json_value(cp),
                names=names,train_names=train_names,val_names=val_names,arm_order=list(ARMS),
                protocol_info=provenance['protocol_info'],pre_test_protocol_info=eval_protocol,
                original_protocol_info=manifest['protocol_info'],launch_blocking_override=override,
                runtime_config=vars(args),split=split,test_accessed=split=='test',provenance=provenance))
            status.update(phase='evaluating',requested_images=len(names),split=split)
            write_json(output/'summary.json',status)
            observer = EvaluationObserver(output,position)
            with track_evaluation(runtime,factory,evaluator,names,state,observer):
                if split == 'test':
                    report = test_endpoint.evaluate_test(runtime,heads,names,folder,seed=context['seed'],
                        original_train=bindings.original_train,legacy=bindings.legacy,factory=factory,evaluator=evaluator,
                        batches=lambda:runtime.deps.full_image_loader(runtime.bundle['args'].dataset,names,False,args.workers,context['seed']))
                else:
                    report = evaluator.evaluate(runtime,heads,names,folder,step=step,split=split,
                        seed=context['seed'],original_train=bindings.original_train,legacy=bindings.legacy,
                        factory=factory,primary_endpoint=primary,smoke=False)
            checks = dict(runtime_verified=False,source_verified=False,guard_restored=False,
                checkpoint_and_parent_unchanged=False,evaluation_inputs_match_checkpoint=False,
                head_state_unchanged=before_heads=={a:state.state_fingerprint(h.state_dict()) for a,h in heads.items()},
                optimizer_state_unchanged=before_opt=={a:state.state_fingerprint(o.state_dict()) for a,o in optimizers.items()},
                rng_restored=before_rng==state.state_fingerprint(state.capture_rng()))
            if not all(checks[k] for k in ('head_state_unchanged','optimizer_state_unchanged','rng_restored')):
                raise ValueError('Evaluation changed model/optimizer/RNG')
            status['runtime_integrity']=json_value(bindings.original_train._checked_verify(runtime))
            checks['runtime_verified']=True
            if split == 'test': test_endpoint.verify_scope(runtime,inherited,extra_inputs)
            else: require_scope(runtime,inherited)
            verify_files({str(p):h for p,h in runtime.input_ledger.items()})
            write_json(output/'input_fingerprints.json',{str(p):h for p,h in runtime.input_ledger.items()})
            checks['evaluation_inputs_match_checkpoint']=True
            if split == 'test':
                status['evaluation_inputs_match_checkpoint_note']='Inherited inputs match checkpoint; separately declared canonical Test878 inputs verified'
        guard = bindings.original_train._restored_guard(runtime)
        if guard is None or guard.get('failures'): raise ValueError('Pool guard closure failed')
        checks['guard_restored']=True
        bridge.verify_ledgers(source)
        bindings.legacy_bridge.verify_dependencies(bindings.legacy)
        if bridge.check_package(addon) != plan['addon_release_sha256']: raise ValueError('Source changed')
        checks['source_verified']=True
        verify_files(context['evidence_files'])
        checks['checkpoint_and_parent_unchanged']=True
        final = finalize(folder/'result.json',report,output=output,names=names,evidence=evidence,
                         checks=checks,provenance=provenance,primary=primary)
        status.update(phase='standalone_evaluation_complete',complete=True,image_count=len(names),
            full_validation_evidence=split=='val',full_test_evidence=split=='test',result=str(folder/'result.json'),
            metrics=concise_metrics(final),integrity_checks=checks,error=None)
    except BaseException as exc:
        status.update(phase='failed',complete=False,error=type(exc).__name__+': '+str(exc))
        (output/'traceback.txt').write_text(traceback.format_exc())
    if observer is not None:
        status.update(first_failure=observer.first_failure,rng_restore_skipped_after_failure=observer.rng_restore_skipped)
    journal = folder/'images.jsonl'
    if journal.is_file():
        with journal.open() as stream: status['completed_images']=sum(bool(s.strip()) for s in stream)
    status['last_position']=position
    write_json(output/'summary.json',status)
    print('ASSESSMENT RESULT '+json.dumps({k:v for k,v in status.items() if k not in ('restoration','rng_restoration')},ensure_ascii=False),flush=True)
    return 0 if status['complete'] else 2
