"""Explicit new source/member backend for the existing paired anchoring engine."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
from io_utils import HERE,read,digest,verify_release,old_modules,named
from new_source import FORMAT as SOURCE_FORMAT,EXPECTED
from new_members import FORMAT as BANK_FORMAT
from runtime_adapter import NewRuntime

ORIGINAL_SOURCE_SHA='b84e2867a4117911b8279e2f97ccf9c5fbd021a7dc539f59e6bbb06396546142'

def _paths():
    return dict(tool=HERE/'deps/original',study=HERE/'deps/study',helpers=HERE/'deps/helpers',
                audit=HERE/'deps/audit',chain=HERE/'deps/chain')

def check_inputs(args):
    if args.mode not in ('smoke','train'):
        raise ValueError('Study mode must be smoke or train')
    if args.topk!=64 or args.key_name!='t128' or args.chunk_size<1 or args.lr!=1e-4 or args.weight_decay!=.005:
        raise ValueError('Only Top64 T128 and fixed optimizer configuration supported')
    source=read(args.protocol)
    if source.get('format')!=SOURCE_FORMAT or source.get('release_sha256')!=verify_release():
        raise ValueError('New source protocol/code binding differs')
    for k in EXPECTED:
        path=Path(source['input_paths'][k])
        if digest(path)!=source['input_fingerprints'][k] or digest(path)!=EXPECTED[k]:
            raise ValueError('Audited Q/A1/split changed: '+k)
    if not (Path(args.repo)/'bda_benchmark').is_dir():raise FileNotFoundError('BRIGHT repo missing')
    for root,format in ((args.bank_root,BANK_FORMAT),(args.values,'SRAC_SOFT_MEMORY_VALUES_V1')):
        p=Path(root)
        if not p.is_dir() or not (p/'summary.json').is_file():raise FileNotFoundError(p)
        summary=read(p/'summary.json')
        if summary.get('complete') is not True:raise ValueError('Bank/values incomplete: '+str(p))
        if summary.get('format')!=format:raise ValueError('Wrong bank/value format: '+str(p))
        if summary.get('smoke') is not (args.mode=='smoke'):
            raise ValueError('Study mode and bank/value smoke status differ: '+str(p))
    return True

def dependencies(args):
    from io_utils import named
    paths=_paths()
    bridge=named(paths['study']/'legacy.py','anchor_legacy')
    legacy=bridge.load_legacy(paths['helpers'])
    config=read(args.config)
    required={'repo','protocol','membership_root','bank_root','values','device','workers','seed',
              'key_name','topk','chunk_size','lr','weight_decay','steps','eval_every'}
    if set(config)!=required:raise ValueError('Explicit training configuration fields differ')
    if (config['steps']!=5000 or config['eval_every']!=500 or config['device']!='cuda:0'
            or config['seed']<0 or config['workers']<0):raise ValueError('Study protocol invalid')
    smoke=args.mode=='smoke'
    config=dict(config,command=args.mode,mode=args.mode,smoke=smoke,baseline_policy='independent',
                baseline_root=None,baseline_evidence=None,allow_smoke_values=smoke)
    check_inputs(SimpleNamespace(**config))
    _,_,_,_,_,common=old_modules()
    if common.source_hash()!=ORIGINAL_SOURCE_SHA:raise ValueError('Original runtime code changed')
    static=dict(start='fresh',start_step=0,tool=str(paths['tool']),config=config,source_sha256=ORIGINAL_SOURCE_SHA,
                checkpoint=None,saved=None,old_files={},config_path=str(Path(args.config).resolve()),
                config_sha256=digest(args.config),original_trajectory_claim=False,strict_runtime_checks_pending=True,
                backend='surviving_Q_new_Train_members')
    return legacy,static

def import_runtime(static,legacy):
    import importlib
    _,_,_,_,train,common=old_modules()
    bridge=named(_paths()['study']/'legacy.py','anchor_legacy')
    bridge.verify_dependencies(legacy)
    bridge._package_files(_paths()['tool'],expected_source=ORIGINAL_SOURCE_SHA)
    original_run=importlib.import_module('run')
    if original_run.verify_release()['code_sha256']!=ORIGINAL_SOURCE_SHA:raise ValueError('Original code differs')
    import torch
    return torch,common,train,original_run,NewRuntime

def runtime_config(config,smoke):
    updated=dict(config)
    updated['mode']='smoke' if smoke else 'train'
    updated['command']=updated['mode']
    updated['smoke']=bool(smoke)
    updated['allow_smoke_values']=bool(smoke)
    return updated
