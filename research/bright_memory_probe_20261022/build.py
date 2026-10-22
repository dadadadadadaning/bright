"""Build a deterministic, inspectable Python zip application."""
from pathlib import Path
import hashlib
import zipfile


def build():
    root=Path(__file__).resolve().parent
    output=root/'bright_memory_probe_v1.pyz'
    sources={name:(root/name).read_bytes() for name in ('probe_core.py','probe_cli.py')}
    sources['__main__.py']=b'from probe_cli import main\nraise SystemExit(main())\n'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(sources.items()):
            entry=zipfile.ZipInfo(name,date_time=(2026,10,22,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=0o644 << 16
            archive.writestr(entry,data)
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    (root/'SHA256SUMS').write_text(checksum+'  '+output.name+'\n')
    print(checksum+'  '+str(output))
    return output


if __name__=='__main__':build()
