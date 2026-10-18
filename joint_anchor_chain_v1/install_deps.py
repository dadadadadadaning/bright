"""Install immutable code dependencies only; never fetch weights or edit sources."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent


from chain_io import digest, package_files

def verify_package(root, expected_manifest):
    package_files(root,expected_manifest)
    return digest(Path(root)/'SHA256SUMS.json')


def install_package(root, spec):
    root = Path(root).absolute()
    if root.exists() or root.is_symlink():
        verify_package(root, spec['manifest_sha256'])
        print('复用代码:', root, flush=True)
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='anchor_chain_download_', dir=root.parent) as temp:
        temp = Path(temp)
        archive = temp / 'code.zip'
        with urllib.request.urlopen(spec['url'], timeout=60) as source, archive.open('xb') as target:
            shutil.copyfileobj(source, target)
        if digest(archive) != spec['archive_sha256']:
            raise ValueError('Downloaded ZIP checksum mismatch')
        unpack = temp / 'unpack'
        unpack.mkdir()
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                path = Path(item.filename)
                if (not path.parts or path.parts[0] != spec['directory'] or path.is_absolute()
                        or '..' in path.parts or '\\' in item.filename
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError('Unsafe dependency archive path')
            z.extractall(unpack)
        candidate = unpack / spec['directory']
        verify_package(candidate, spec['manifest_sha256'])
        if root.exists():
            raise FileExistsError('Dependency appeared during install; preserved: ' + str(root))
        os.rename(candidate, root)
    print('已安装固定版本代码:', root, flush=True)


def install(anchor, audit, helpers):
    specs = json.loads((HERE / 'dependencies.json').read_text())
    install_package(anchor, specs['anchor'])
    install_package(audit, specs['audit'])
    # Reuse the already released helper installer instead of implementing a
    # second helper ZIP/source protocol.
    sys.path.insert(0, str(Path(anchor).resolve()))
    spec = importlib.util.spec_from_file_location('_chain_anchor_install_shared', Path(anchor) / 'install_shared.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install(helpers)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor-tool', required=True)
    p.add_argument('--audit-tool', required=True)
    p.add_argument('--helpers', required=True)
    a = p.parse_args()
    install(a.anchor_tool, a.audit_tool, a.helpers)
