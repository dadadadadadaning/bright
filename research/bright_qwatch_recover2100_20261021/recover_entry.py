"""Resume verified2100 with frozen-Q observations; then fixed2500 Val and exploratory Test878."""
import argparse
import contextlib
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

from grid_probe import read_json, sha256
from evaluate_checkpoint import write_json, ARMS, REQUIRED_CHECKS
from index_eval_probe import prepared as original_prepared, FAILURE_ROOT, FAILURE_TRAINER_SHA
from resume_context import prepare, inactive_parent
from resume_train import DEFAULT_RUN, DEFAULT_OUTPUT, verify_files
from sync_val2000 import validate_protocol

BASE = Path(DEFAULT_OUTPUT)
EVAL_BASE = BASE.parent / 'SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1'
CERTIFICATE = EVAL_BASE / 'sync_val2000_hkpnci6r'
CERTIFICATE_SHA = '14a3cb440ce49899da7d7b81061b9b2df66ab88526203919f28823bf8b3617ab'
FORMAT = 'BRIGHT_QWATCH_RECOVER2100_THEN_TEST878_V1'
from finish_entry import initial_context as certified_boundary
RECOVERY_PARENT = BASE / 'finish2500_a4x9h5ay'
PARENT_ARCHIVE_SHA = 'cb8a94d4235d2f0ce70f7c70c9d8e1c32285ba7a7a8faf58eceaf472b9e624e2'
PARENT_CP_SHA = 'da8474ca8ba741aa10a0b4a5f3278e0d69a05b803c67c7c6cd9cf910572027f1'
PARENT_PAYLOAD_SHA = '65a69c63e7f634ade3cfb29a97f718d2f0af3bed699c41f50e450e8e1e6a2033'
STAGES = ('train', 'fixed2000', 'val2500', 'fixed2500', 'test2500')


def initial_context():
    boundary = certified_boundary()
    c = prepare(SimpleNamespace(original_run=DEFAULT_RUN, seed=3407, checkpoint_every=10,
                               addon_dir=None, parent_root=str(RECOVERY_PARENT)))
    parent = c['parent']
    s = read_json(RECOVERY_PARENT / 'summary.json')
    meta = read_json(Path(parent['path']).with_suffix('.pt.json'))
    marker = s.get('last_marker', {})
    workflow = read_json(RECOVERY_PARENT / 'workflow_summary.json')
    if (parent['step'] != 2100 or parent['sha256'] != PARENT_CP_SHA
        or c['plan']['endpoint_step'] != 2500 or c['lineage'] != boundary['lineage']
        or sha256(RECOVERY_PARENT / 'trainer.pyz') != PARENT_ARCHIVE_SHA
        or meta.get('payload_fingerprint') != PARENT_PAYLOAD_SHA
        or meta.get('roundtrip_verified') is not True
        or meta.get('updates_per_arm') != {a:2100 for a in ARMS}
        or s.get('phase') != 'failed' or s.get('last_complete_step') != 2135
        or s.get('failure_phase') != 'update_2136'
        or marker.get('step') != 2136 or marker.get('arm') != 'f'
        or marker.get('image') != 'la_palma-volcano_00000900'
        or marker.get('phase') != 'forward_begin' or marker.get('completed_arms') != ['e']
        or s.get('optimizer_steps_returned_in_observed_step') != ['e']
        or workflow.get('phase') != 'failed' or workflow.get('failed_stage') != 'train'
        or workflow.get('test_complete') is not False):
        raise ValueError('Recovery requires the pinned paired2100 checkpoint and the observed partial2136 failure')
    paths = [RECOVERY_PARENT/n for n in ('trainer.pyz','markers.jsonl','traceback.txt',
                                        'workflow_summary.json','request.json')]
    c['evidence_files'].update(boundary['evidence_files'])
    c['evidence_files'].update({str(p):sha256(p) for p in paths})
    c['boundary_certificate'] = boundary['boundary_certificate']
    c['evaluation_boundary_step'] = 2000
    c['parent_q_observation'] = dict(enabled=True, observations_synchronize=True,
        route_values_modified=False, exact_historical_replay=False, historical_failure_resolved=False,
        acceptance_predicate_unchanged=True, restored_paired_step=2100,
        discarded_uncheckpointed_steps=list(range(2101,2136)), discarded_partial_step=2136)
    return c


def endpoint_context(root):
    c = prepare(SimpleNamespace(original_run=DEFAULT_RUN, seed=3407, checkpoint_every=10,
                               addon_dir=None, parent_root=str(root)))
    s = read_json(root / 'summary.json')
    if (c['parent']['step'] != 2500 or s.get('training_complete') is not True
        or s.get('phase') != 'training_complete_evaluation_pending'
        or s.get('new_updates_per_arm') != {a: 400 for a in ARMS}):
        raise ValueError('Need complete400 new paired updates and checkpoint2500')
    validate_observations(root,s)
    for name in ('parent_q_observer_summary.json','parent_q_observations.jsonl',
                 'route_observer_identity.json','route_observer_summary.json'):
        c['evidence_files'][str(root/name)] = sha256(root/name)
    return c


def read_request(root):
    root = Path(root).resolve()
    if root.parent != BASE or not root.name.startswith('recover2100_'):
        raise ValueError('Unexpected workflow directory')
    request = read_json(root / 'request.json')
    if request['artifact_sha256'] != sha256(root / 'trainer.pyz'):
        raise ValueError('Pinned workflow archive changed')
    verify_files(request['evidence_files'])
    return request


@contextlib.contextmanager
def leases():
    with contextlib.ExitStack() as stack:
        inherited = os.environ.pop('BRIGHT_FINISH_FDS', '')
        if inherited:
            streams = [stack.enter_context(os.fdopen(int(fd), 'a+')) for fd in inherited.split(',')]
        else:
            streams = []
            for path in (BASE / 'active.lock', EVAL_BASE / 'evaluation.lock'):
                path.parent.mkdir(parents=True, exist_ok=True)
                streams.append(stack.enter_context(path.open('a+')))
        for stream in streams:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield streams


def launch():
    artifact = Path(sys.argv[0]).resolve()
    if not artifact.is_file() or artifact.suffix != '.pyz':
        raise ValueError('Launch the pinned pyz archive')
    with leases() as streams:
        print('PREFLIGHT: paired checkpoint2100 + Val2000 evidence; original route checks + Q snapshots; fixed2500 Test', flush=True)
        c = initial_context()
        root = Path(tempfile.mkdtemp(prefix='recover2100_', dir=BASE))
        pinned = root / 'trainer.pyz'
        shutil.copyfile(artifact, pinned)
        write_json(root / 'request.json', dict(format=FORMAT, artifact_sha256=sha256(pinned),
            evidence_files=c['evidence_files'], checkpoint_step=2100, endpoint_step=2500,
            parent_q_observation=c['parent_q_observation'],
            stages=list(STAGES), test_checkpoint_selection='fixed2500_all_six_arms',
            test_used_for_selection=False, existing_test_exposure=True))
        write_json(root / 'workflow_summary.json', dict(format=FORMAT, complete=False, phase='queued',
            output=str(root), training_complete=False, test_complete=False))
        command = [sys.executable, '-B', '-u', str(pinned), 'worker', '--root', str(root)]
        fds = tuple(s.fileno() for s in streams)
        with (root / 'workflow.log').open('x') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=fds,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', BRIGHT_FINISH_FDS=','.join(map(str,fds))))
        write_json(root / 'launcher.json', dict(pid=proc.pid, command=command))
        (BASE / 'latest_recover2100.txt').write_text(str(root)+'\n')
        print('已启动：监测续训2101–2500（每10步保存）→ 固定Train诊断 → Val450 → Test878。', flush=True)
        print(f'PID={proc.pid}\nOUTPUT={root}\n日志={root / "workflow.log"}', flush=True)
        print(f'查看日志：{sys.executable} {pinned} logs', flush=True)
        print(f'查看结果：{sys.executable} {pinned} status', flush=True)
    return 0


def train(root):
    request = read_request(root)
    c = initial_context()
    if c['evidence_files'] != request['evidence_files']:
        raise ValueError('Source changed after launch')
    import recover_training
    from parent_q_observer import observe_bindings
    with (root / 'worker.lock').open('a+') as own, inactive_parent(RECOVERY_PARENT):
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with observe_bindings(c,root):
            return recover_training.run_training(c, root)


def assess(root, stage):
    read_request(root)
    if stage not in STAGES[1:]:
        raise ValueError('Unexpected assessment')
    c = certified_boundary() if stage == 'fixed2000' else endpoint_context(root)
    if stage == 'test2500':
        verify_before_test(root,c)
    parent_root = Path(FAILURE_ROOT) if stage == 'fixed2000' else root
    output = root / 'assessments' / stage
    output.mkdir(parents=True, exist_ok=False)
    from assess_endpoint import run_assessment
    with inactive_parent(parent_root):
        return run_assessment(c, output, stage)


def verify_before_test(root, context):
    import evidence
    for stage,split,names,primary in (
        ('val2500','val',context['manifest']['val_names'],True),
        ('fixed2500','fixed_train',context['manifest']['fixed_train_names'],False)):
        output=root/'assessments'/stage
        s=read_json(output/'summary.json')
        path=output/('%s_step_002500' % split)/'result.json'
        report=read_json(path)
        pending_path=path.with_name('result.pending.json')
        if (s.get('complete') is not True or report.get('complete') is not True
            or report.get('split') != split or report.get('primary_endpoint') is not primary
            or report.get('test_accessed') is not False
            or report.get('standalone_provenance',{}).get('checkpoint_sha256') != context['parent']['sha256']
            or report.get('pending_report_sha256') != sha256(pending_path)
            or any(report.get('standalone_integrity_checks',{}).get(k) is not True for k in REQUIRED_CHECKS)):
            raise ValueError('Complete matching Val/fixedTrain2500 required before Test')
        pending=read_json(pending_path)
        evidence.validate_report(pending,step=2500,split=split,names=names,arms=list(ARMS),primary=primary)
        evidence.validate_saved_report(path,report,names,sha256,output)
        if pending['metrics'] != report['metrics']:
            raise ValueError('Finalized evaluation metrics changed')


def validate_observations(root,status):
    if status.get('new_updates_per_arm') != {a:400 for a in ARMS}:
        raise ValueError('Require400 complete paired updates from2100')
    q = read_json(root/'parent_q_observer_summary.json')
    routes = read_json(root/'route_observer_summary.json')
    if (q.get('paired_updates_observed') != 400 or q.get('snapshots',0) < 400
        or q.get('content_changes') != 0 or q.get('failures') != 0 or q.get('evidence_error')
        or q.get('route_values_modified') is not False
        or routes.get('calls',0) == 0 or routes.get('failures') != 0
        or routes.get('evidence_error') or routes.get('bindings_restored') is not True):
        raise ValueError('Q/route observation evidence is incomplete or reports a failure')


def coordinate(root):
    with leases():
        read_request(root)
        summary = dict(format=FORMAT, phase='starting', complete=False, training_complete=False,
            validation_complete=False, test_complete=False, output=str(root), pid=os.getpid(),
            restored_step=2100, endpoint_step=2500,
            observation_synchronizes=True, historical_failure_resolved=False, stages={}, original_failed_runs_unchanged=True,
            test_checkpoint_selection='fixed2500_all_six_arms', existing_test_exposure=True)
        try:
            for stage in STAGES:
                summary['phase'] = stage
                write_json(root / 'workflow_summary.json', summary)
                print('\nWORKFLOW STAGE ' + stage, flush=True)
                command = [sys.executable, '-B', '-u', str(root/'trainer.pyz'),
                    'train' if stage == 'train' else 'assess', '--root', str(root), '--stage', stage]
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                if stage != 'train':
                    env['CUDA_LAUNCH_BLOCKING'] = '1'
                result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL)
                status_path = root/'summary.json' if stage == 'train' else root/'assessments'/stage/'summary.json'
                status = read_json(status_path) if status_path.is_file() else dict(error='Stage did not write summary')
                summary['stages'][stage] = dict(returncode=result.returncode, summary=str(status_path),
                    phase=status.get('phase'), error=status.get('error'))
                if result.returncode != 0:
                    raise RuntimeError(stage + ' failed: ' + str(status.get('error')))
                if stage == 'train':
                    if status.get('last_complete_step') != 2500 or status.get('training_complete') is not True:
                        raise ValueError('Training did not reach2500')
                    validate_observations(root,status)
                    summary['training_complete'] = True
                    summary['last_complete_step'] = 2500
                    summary['new_updates_per_arm'] = status['new_updates_per_arm']
                elif status.get('complete') is not True:
                    raise ValueError(stage + ' missing complete evidence')
                if stage == 'fixed2500':
                    summary['validation_complete'] = True
                if stage == 'test2500':
                    summary['test_complete'] = True
                write_json(root / 'workflow_summary.json', summary)
            verify_files(read_json(root/'request.json')['evidence_files'])
            write_comparison(root)
            summary.update(phase='complete', complete=True, metrics_csv=str(root/'metrics.csv'))
        except BaseException as exc:
            summary.update(phase='failed', complete=False, failed_stage=stage,
                           error=type(exc).__name__+': '+str(exc))
            (root/'workflow_traceback.txt').write_text(traceback.format_exc())
        write_json(root/'workflow_summary.json', summary)
        print('WORKFLOW RESULT '+json.dumps(summary,ensure_ascii=False),flush=True)
        return 0 if summary['complete'] else 2


def write_comparison(root):
    reports=[('val',2000,CERTIFICATE/'val_step_002000/result.json'),
             ('val',2500,root/'assessments/val2500/val_step_002500/result.json'),
             ('test',2500,root/'assessments/test2500/test_step_002500/result.json')]
    with (root/'metrics.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('split','step','arm','background_iou_pct','intact_iou_pct',
                         'damaged_iou_pct','destroyed_iou_pct','miou4_pct','foreground_miou3_pct',
                         'miou4_minus_grid_pp','checkpoint_sha256'))
        for split,step,path in reports:
            report=read_json(path)
            grid=sum(v or 0.0 for v in report['metrics']['grid_probq']['iou'])/4
            for arm,m in report['metrics'].items():
                miou=sum(v or 0.0 for v in m['iou'])/4
                writer.writerow((split,step,arm,*[None if v is None else 100*v for v in m['iou']],
                    100*miou,100*m['foreground_miou'],100*(miou-grid),
                    report['standalone_provenance']['checkpoint_sha256']))


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('start','worker','train','assess','logs','status'),nargs='?',default='start')
    parser.add_argument('--root')
    parser.add_argument('--stage')
    args = parser.parse_args()
    try:
        if args.command == 'start': return launch()
        own = Path(sys.argv[0]).resolve().parent
        root = (Path(args.root).resolve() if args.root else own if own.parent == BASE and own.name.startswith('recover2100_')
                else Path((BASE/'latest_recover2100.txt').read_text().strip()))
        if args.command == 'worker': return coordinate(root)
        if args.command == 'train': return train(root)
        if args.command == 'assess': return assess(root,args.stage)
        if args.command == 'logs':
            os.execvp('tail',['tail','-n','80','-F',str(root/'workflow.log')])
        print(json.dumps(read_json(root/'workflow_summary.json'),ensure_ascii=False,indent=2))
        for path in [root/'summary.json', *sorted((root/'assessments').glob('*/summary.json'))]:
            s=read_json(path)
            print(path.parent.name, json.dumps({k:s[k] for k in ('phase','complete','last_complete_step',
                'training_complete','image_count','metrics','error') if k in s},ensure_ascii=False))
        return 0
    except Exception as exc:
        print('STOP: '+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
