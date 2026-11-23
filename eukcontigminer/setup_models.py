"""Fetch pinned model assets once; prediction itself is offline."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import tarfile
from pathlib import PurePosixPath
import urllib.request

from .resources import copy_checked, copy_package_checked, metadata, sha, verify_models


def main(argv=None):
    p = argparse.ArgumentParser(description='Prepare v0.90 assets; downloads are checksum-verified.')
    p.add_argument('--output', default='models')
    p.add_argument('--nt-dir', help='Use an existing original NTv3_100M_pre directory')
    p.add_argument('--esm-weights', help='Use the original esmc_300m_2024_12_v0.pth')
    p.add_argument('--heads-archive', help='Use a local v0.90 models archive')
    args = p.parse_args(argv)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    package = Path(__file__).parent / 'assets'
    for name, item in meta['files'].items():
        if item['source'] == 'package':
            copy_package_checked(package/name, root/name, item['sha256'])
    head_files = {k: v for k, v in meta['files'].items() if v['source'] == 'heads'}
    if not all((root/n).is_file() and sha(root/n) == v['sha256'] for n, v in head_files.items()):
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            archive = Path(args.heads_archive) if args.heads_archive else Path(tmp)/'heads.tar.gz'
            if not args.heads_archive:
                urllib.request.urlretrieve(meta['heads_archive']['url'], archive)
            if sha(archive) != meta['heads_archive']['sha256']:
                raise ValueError('Model archive checksum mismatch')
            expected = set(head_files)
            allowed_dirs = {str(PurePosixPath(name).parent) for name in expected}
            seen = set()
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    name = member.name
                    parts = PurePosixPath(name).parts
                    if any(part.startswith('._') for part in parts):
                        continue
                    if member.isdir():
                        if name.rstrip('/') in allowed_dirs:
                            continue
                        raise ValueError('Unexpected model archive directory: ' + name)
                    if not member.isfile():
                        raise ValueError('Only regular model files allowed')
                    if PurePosixPath(name).is_absolute() or '..' in parts:
                        raise ValueError('Unsafe model archive path: ' + name)
                    if name not in expected:
                        raise ValueError('Unexpected model archive entry: ' + name)
                    if name in seen:
                        raise ValueError('Duplicate model archive entry: ' + name)
                    src = Path(tmp)/PurePosixPath(name)
                    src.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as stream, src.open('xb') as dest:
                        shutil.copyfileobj(stream, dest)
                    copy_checked(src, root/name, head_files[name]['sha256'])
                    seen.add(name)
            if seen != expected:
                raise ValueError('Model archive is missing entries')

    for name, item in meta['files'].items():
        if item['source'] != 'upstream':
            continue
        if (root/name).is_file() and sha(root/name) == item['sha256']:
            continue
        if name.startswith('encoders/nt/') and args.nt_dir:
            src = Path(args.nt_dir)/Path(name).name
        elif name.startswith('encoders/esm/') and args.esm_weights:
            src = Path(args.esm_weights)
        else:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(repo_id=item['repository'], filename=item['filename'], revision=item['revision'])
        copy_checked(src, root/name, item['sha256'])
    verify_models(root)
    print(json.dumps(dict(version='0.90', models=str(root), verified=True)))
