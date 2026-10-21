"""Build deterministic archive; keep original helpers byte-identical."""
import hashlib
import json
from pathlib import Path
import zipfile
HERE=Path(__file__).resolve().parent
BASE_SHA='aec1a463e26ce8d9620863801ba44420fe951da91a6add9e222736d9a171197c'
base=HERE.parent/'bright_qwatch_recover2100_20261021/bright_qwatch_recover2100_v1.pyz'
if hashlib.sha256(base.read_bytes()).hexdigest()!=BASE_SHA: raise ValueError('Base archive changed')
with zipfile.ZipFile(base) as z: files={n:z.read(n) for n in z.namelist() if n!='__main__.py'}
ledger={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
for n in ('night_config.py','night_entry.py','night_assessment.py','night_routes.py'): files[n]=(HERE/n).read_bytes()
files['NIGHT_BASE.json']=json.dumps(dict(source_sha256=BASE_SHA,files=ledger),indent=2).encode()
files['__main__.py']=b'from night_entry import main\nraise SystemExit(main())\n'
out=HERE/'bright_test_route_night_v1.pyz'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in sorted(files.items()):
        i=zipfile.ZipInfo(n,(2026,10,22,0,0,0)); i.compress_type=zipfile.ZIP_DEFLATED; i.external_attr=0o644<<16
        z.writestr(i,b)
h=hashlib.sha256(out.read_bytes()).hexdigest()
out.with_suffix('.pyz.sha256').write_text(h+'  '+out.name+'\n')
print(h,out)
