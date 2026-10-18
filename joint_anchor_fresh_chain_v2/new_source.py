"""New source protocol from surviving formal Q/A1 and explicit Train/Val lists.

Does not create or claim restoration of an E0 protocol or historical member bank.
Formal Q validators are reused; a new explicitly recorded numeric policy is used.
"""
from pathlib import Path
from types import SimpleNamespace
from io_utils import HERE,read,write,digest,verify_release,old_modules

FORMAT='BRIGHT_SURVIVING_Q_SOURCE_V2'
EXPECTED={
 'q_checkpoint':'f37ba4b5cda34bab0f2eec94d1766e163b378308b7edf61505d57089065f8264',
 'a1_checkpoint':'cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add',
 'train_list':'353e8cc4e082737c62d958ca9d1c892e1cc1c08905ff274d2ad56b28e283369d',
 'val_list':'c3343785744187e5f4bd54d26eb26ead3bf40f83cfabb5b9b2e7842cbfac065d'}

def validate(record):
    import torch
    fs,*_=old_modules()
    if record.get('format')!=FORMAT or record.get('test_accessed') is not False:
        raise ValueError('New explicit source protocol required')
    paths={k:Path(v) for k,v in record['input_paths'].items()}
    for k,p in paths.items():
        if digest(p)!=record['input_fingerprints'][k]:raise ValueError('Source changed: '+str(p))
        if k in EXPECTED and digest(p)!=EXPECTED[k]:raise ValueError('Not the audited source: '+k)
    q=paths['q_checkpoint']
    if q.name!='last.pth' or q.parent.name!='train_baseline' or q.parent.parent.name!='formal':
        raise ValueError('Expected formal baseline last.pth')
    payload=torch.load(q,map_location='cpu',weights_only=False)
    deps=fs._dependencies(Path(record['repo']))
    fs._validate_formal(payload,read(paths['q_train_summary']),read(paths['q_train_manifest']),
                        None,None,paths,deps,fs.TRAINING_EVIDENCE)
    names={s:fs._split_names(paths[s+'_list'],n) for s,n in [('train',2918),('val',450)]}
    if set(names['train'])&set(names['val']):raise ValueError('Train/Val overlap')
    for s in names:
        if payload[s+'_sha256']!=digest(paths[s+'_list']):raise ValueError('Q split mismatch')
    if payload['a1_sha256']!=digest(paths['a1_checkpoint']):raise ValueError('A1 mismatch')
    return payload,deps,names,paths

def create(args,output):
    paths={k:Path(getattr(args,k)).resolve() for k in EXPECTED}
    qdir=paths['q_checkpoint'].parent
    paths.update({k:qdir/n for k,n in [('q_train_summary','summary.json'),('q_train_manifest','manifest.json'),('q_train_validation','validation.jsonl')]})
    record=dict(format=FORMAT,repo=str(Path(args.repo).resolve()),dataset=str(Path(args.dataset).resolve()),
        device=args.device,workers=args.workers,seed=args.seed,
        input_paths={k:str(v) for k,v in paths.items()},input_fingerprints={k:digest(v) for k,v in paths.items()},
        evidence_level='training_final_val450',historical_e0_source_reverified=False,
        old_members_recovered=False,test_accessed=False,release_sha256=verify_release(),
        numeric_policy=dict(name='new_shared_fp32',matmul_tf32=False,cudnn_tf32=True,
                            deterministic=False,num_threads=8,num_interop_threads=8))
    payload,deps,names,_=validate(record)
    missing=[]
    for split in names:
        for name in names[split]:
            for folder,suffix in [('pre-event','_pre_disaster.tif'),('post-event','_post_disaster.tif'),('target','_building_damage.tif')]:
                p=Path(record['dataset'])/folder/(name+suffix)
                if not p.is_file():missing.append(str(p))
    if missing:raise FileNotFoundError('Missing data: '+repr(missing[:20]))
    record.update(historical_runtime_recorded=payload['e0_runtime'],
                  historical_protocol_sha256=payload['protocol_sha256'],
                  historical_initial_front_sha256=payload['front_sha256'])
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    write(output,record)
    print('SOURCE COMPLETE: audited Q/A1 and Train2918/Val450; new members required; no CUDA yet.',flush=True)
    return record

def load(protocol,output):
    import torch
    fs,*_=old_modules(); protocol=Path(protocol).resolve(); record=read(protocol)
    if record['release_sha256']!=verify_release():raise ValueError('Source release differs')
    policy=record['numeric_policy']
    torch.set_num_threads(policy['num_threads'])
    if torch.get_num_interop_threads()!=policy['num_interop_threads']:
        torch.set_num_interop_threads(policy['num_interop_threads'])
    payload,deps,names,paths=validate(record)
    # configure before any CUDA tensors; use same new policy in every process.
    deps.numeric.configure_runtime('legacy',matmul_tf32=policy['matmul_tf32'],cudnn_tf32=policy['cudnn_tf32'],deterministic=policy['deterministic'])
    device=torch.device(record['device'])
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Real source reconstruction requires CUDA')
    torch.cuda.set_device(device)
    front=deps.q.QVariantFront(arm='baseline',cue_gain=payload['cue_gain'],**payload['config']).to(device)
    deps.common.load_a1_strict(front,paths['a1_checkpoint'])
    fs._check_adapter(front,payload['adapter']);front.load_adapter_state_dict(payload['adapter'])
    front.requires_grad_(False).eval()
    paths['new_source_protocol']=protocol
    paths.update({'new_code_'+str(i):p for i,p in enumerate(sorted(HERE.rglob('*.py')))})
    fingerprints={k:digest(p) for k,p in paths.items()}
    for k,sha in record['input_fingerprints'].items():
        if fingerprints[k]!=sha:raise ValueError('Source changed during model load')
    provenance=dict(kind='completed_q_baseline_new_members',evidence_level='training_final_val450',
        historical_e0_source_reverified=False,historical_initial_front_file_reverified=False,
        current_q_checkpoint_reverified=True,current_split_files_reverified=True,
        q_numeric_profile_reproduced=False,numeric_policy=policy,old_members_recovered=False,
        historical_initial_front_sha256_recorded=payload['front_sha256'])
    return dict(front=front,device=device,args=SimpleNamespace(dataset=record['dataset'],pool_mode='native',
        seed=record['seed'],workers=record['workers'],device=str(device)),train_names=names['train'],val_names=names['val'],
        runtime_settings=deps.numeric.runtime_settings(),hashes=dict(a1_sha256=fingerprints['a1_checkpoint'],
        train_sha256=fingerprints['train_list'],val_sha256=fingerprints['val_list'],q_checkpoint_sha256=fingerprints['q_checkpoint']),
        head_hashes={},own_source_sha256=digest(__file__),input_paths={k:str(p) for k,p in paths.items()},
        input_fingerprints=fingerprints,inputs_before=fingerprints,source_protocol_sha256=digest(protocol),
        source_checkpoint_sha256=fingerprints['q_checkpoint'],original_front_sha256=payload['front_sha256'],
        a1_sha256=fingerprints['a1_checkpoint'],evidence_level='training_final_val450',source_provenance=provenance,
        source_kind='completed_q_baseline_new_members')
