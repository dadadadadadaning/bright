#!/usr/bin/env python3
"""Install one pinned shared helper package only if absent. Does not run training."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from legacy import DEPENDENCY_MANIFEST_SHA256

URL='https://raw.githubusercontent.com/dadadadadadaning/bright/616e3c068fbd01d9941699baf0f30194fcf90d0f/bright_joint_loss_after500_v1.zip'
ARCHIVE_SHA='dbd373aedd0cca3903e2081613b27b8ec05ecade7a9d4444682b5728e6e01e48'


def verify(root):
    manifest=root/'SHA256SUMS.json'
    if not manifest.is_file() or manifest.is_symlink() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=DEPENDENCY_MANIFEST_SHA256:
        raise ValueError('Shared dependency exists but is missing/changed; preserved: '+str(root))
    hashes=json.loads(manifest.read_text())
    for name,expected in hashes.items():
        p=root/name
        if Path(name).name!=name or p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Shared dependency file mismatch: '+str(p))
    if not {p.name for p in root.glob('*.py')}<=set(hashes): raise ValueError('Unbound shared Python file')


def install(destination):
    destination=Path(destination).expanduser().absolute()
    if destination.is_symlink(): raise ValueError('Refuse symlink dependency directory')
    if destination.exists():
        verify(destination);print('复用已有共享工具:',destination);return
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='anchor_shared_',dir=destination.parent) as temp:
        temp=Path(temp);archive=temp/'helpers.zip'
        with urllib.request.urlopen(URL,timeout=60) as source,archive.open('xb') as target:
            shutil.copyfileobj(source,target)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA:
            raise ValueError('Downloaded shared dependency checksum mismatch')
        unpack=temp/'unpack';unpack.mkdir()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                parts=Path(member.filename).parts
                if not parts or parts[0]!='joint_loss_after500_v1' or '..' in parts or Path(member.filename).is_absolute():
                    raise ValueError('Unexpected pinned archive path')
            z.extractall(unpack)
        candidate=unpack/'joint_loss_after500_v1';verify(candidate)
        if destination.exists(): raise FileExistsError('Dependency appeared during install; no overwrite')
        os.rename(candidate,destination)
    print('共享工具已安装（仅一次，无训练）:',destination)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',default='/root/shared-nvme/BRIGHT-master/New/joint_loss_after500_v1')
    install(parser.parse_args().destination)
