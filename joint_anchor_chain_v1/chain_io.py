"""Shared JSON/hash/package checks for the chain, installer and audit adapter."""
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def reject_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    result = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return result


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def package_files(root, expected_manifest=None):
    root = Path(root).expanduser().absolute()
    manifest = root / 'SHA256SUMS.json'
    if root.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError('Pinned package manifest missing/linked: ' + str(root))
    manifest_sha = digest(manifest)
    if expected_manifest is not None and manifest_sha != expected_manifest:
        raise ValueError('Pinned package manifest differs; original preserved: ' + str(root))
    hashes = read(manifest)
    if not hashes:
        raise ValueError('Empty package manifest: ' + str(root))
    files = {str(manifest): manifest_sha}
    for name, expected in hashes.items():
        path = root / name
        if (Path(name).name != name or name in ('.','..','SHA256SUMS.json')
                or '\\' in name or not isinstance(expected,str) or not re.fullmatch(r'[0-9a-f]{64}',expected)
                or path.is_symlink() or not path.is_file() or digest(path) != expected):
            raise ValueError('Pinned package file changed: ' + str(path))
        files[str(path)] = expected
    if not {p.name for p in root.glob('*.py')} <= set(hashes):
        raise ValueError('Unexpected Python file in package: ' + str(root))
    return files
