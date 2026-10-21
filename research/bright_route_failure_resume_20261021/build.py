"""Bundle the pinned continuation archive plus scoped failure evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
BASE_SHA="994ee877c6edf332e4ebae70c48c81442877dcb6ab053e8cab4323ebb0075e3d"


def build(base):
    base=Path(base)
    if hashlib.sha256(base.read_bytes()).hexdigest()!=BASE_SHA:
        raise ValueError("Require the original pinned training-next archive")
    with zipfile.ZipFile(base) as archive:
        files={n:archive.read(n) for n in archive.namelist() if n!="__main__.py"}
    ledger={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    for name in ("passive_routes.py","failure_bindings.py","failure_entry.py"):
        files[name]=(HERE/name).read_bytes()
    files["route_trace.py"]=(HERE.parent/"bright_route_eval_probe_20261021/route_trace.py").read_bytes()
    files["INHERITED_TRAINER.json"]=json.dumps(dict(sha256=BASE_SHA,files=ledger),indent=2).encode()
    files["__main__.py"]=b"from failure_entry import main\nraise SystemExit(main())\n"
    output=HERE/"bright_route_failure_resume_v1.pyz"
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,21,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,data)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".pyz.sha256").write_text(digest+"  "+output.name+"\n")
    print(digest,output)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("base_archive")
    build(p.parse_args().base_archive)
