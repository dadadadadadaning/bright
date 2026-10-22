"""Retain inherited evaluator modules byte-for-byte; add separate sweep modules."""
import hashlib
import json
from pathlib import Path
import zipfile
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'bright_test_route_night_20261022/bright_test_route_night_v1.pyz'
EXPECTED='4610a882f4f3b0936ff105d5a8a8a0375e169ef00ce88d1eb03781d707bd3d2f'


def build():
    if hashlib.sha256(BASE.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Inherited archive changed')
    with zipfile.ZipFile(BASE) as z:files={n:z.read(n) for n in z.namelist() if n!='__main__.py'}
    ledger={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    for name in ('sweep_context','sweep_routes','sweep_scoring','sweep_assessment','sweep_reports','sweep_entry'):
        files[name+'.py']=(HERE/(name+'.py')).read_bytes()
    files['SWEEP_BASE.json']=json.dumps(dict(sha256=EXPECTED,files=ledger),indent=2).encode()
    files['__main__.py']=b'from sweep_entry import main\nraise SystemExit(main())\n'
    output=HERE/'bright_test_sweep_v1.pyz'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,data)
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    (HERE/'SHA256SUMS').write_text(checksum+'  '+output.name+'\n');print(checksum,output)
    return output


if __name__=='__main__':build()
