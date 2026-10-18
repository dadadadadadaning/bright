"""Immutable code ledger and JSON helpers for the independent new experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    return h.hexdigest()

def read(path):
    d=json.loads(Path(path).read_text())
    if not isinstance(d,dict): raise ValueError('Expected JSON object: '+str(path))
    return d

def write(path,d):
    path=Path(path); part=path.with_name(path.name+'.tmp')
    part.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    part.replace(path)

def verify_release():
    expected=read(HERE/'SHA256SUMS.json')
    for name,sha in expected.items():
        p=HERE/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or digest(p)!=sha:
            raise ValueError('Release changed: '+name)
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*.py')}
    if not actual<=set(expected):raise ValueError('Unbound Python code in release')
    return digest(HERE/'SHA256SUMS.json')

def named(path,label):
    path=Path(path).resolve(); tag='_fresh_v2_'+label+'_'+hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if tag in sys.modules: return sys.modules[tag]
    spec=importlib.util.spec_from_file_location(tag,path); m=importlib.util.module_from_spec(spec)
    sys.modules[tag]=m
    try:spec.loader.exec_module(m)
    except BaseException:sys.modules.pop(tag,None);raise
    return m

def old_modules():
    """Isolated worker processes; reject an unrelated cached bare import."""
    verify_release(); root=HERE/'deps/original'
    for p in root.glob('*.py'):
        cached=sys.modules.get(p.stem)
        if cached and Path(getattr(cached,'__file__','/')).resolve()!=p.resolve():
            raise ImportError('Bare module collision: '+p.stem)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    import fresh_source, fresh_parent_bank, base_runtime, server_runtime, train, common
    return fresh_source,fresh_parent_bank,base_runtime,server_runtime,train,common
