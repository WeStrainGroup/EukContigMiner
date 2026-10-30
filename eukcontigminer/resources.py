"""Single asset entry point: manifest, hashing, verification and preparation."""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metadata():
    return json.loads((Path(__file__).parent / 'release.json').read_text())


def verify_models(root):
    root = Path(root).resolve()
    meta = metadata()
    for name, item in meta['files'].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p) != item['sha256']:
            raise ValueError('Missing or changed model file: ' + name + '. Run eukcontigminer download-models.')
    return root, meta


def copy_checked(source, target, expected):
    """Copy -> verify -> publish without overwrite; retry-safe unique temporary."""
    source, target = Path(source), Path(target)
    if not source.is_file() or sha(source) != expected:
        raise ValueError('Model checksum mismatch: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError('Existing model differs: ' + str(target))
        return
    fd, temp_name = tempfile.mkstemp(prefix=target.name + '.', suffix='.partial', dir=target.parent)
    temp = Path(temp_name)
    try:
        with source.open('rb') as src, os.fdopen(fd, 'wb') as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
            dst.flush()
            os.fsync(dst.fileno())
        if sha(temp) != expected:
            raise ValueError('Copied model checksum mismatch')
        try:
            os.link(temp, target)
        except FileExistsError:
            if sha(target) != expected:
                raise ValueError('Existing model differs: ' + str(target))
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
