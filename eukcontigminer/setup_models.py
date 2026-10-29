"""Fetch pinned model assets once; prediction itself is offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

from .cli import metadata, sha, verify_models


def copy_checked(source, target, expected):
    if not Path(source).is_file() or sha(source) != expected:
        raise ValueError('Model checksum mismatch: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != expected:
            raise ValueError('Existing model differs: ' + str(target))
        return
    temp = target.with_name(target.name + '.partial')
    with Path(source).open('rb') as src, temp.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
    if sha(temp) != expected:
        raise ValueError('Copied model checksum mismatch')
    os.link(temp, target)
    temp.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description='Prepare v0.71 assets; downloads are checksum-verified.')
    p.add_argument('--output', default='models')
    p.add_argument('--nt-dir', help='Use an existing original NTv3_100M_pre directory')
    p.add_argument('--esm-weights', help='Use the original esmc_300m_2024_12_v0.pth')
    p.add_argument('--heads-archive', help='Use a local v0.7 models archive')
    args = p.parse_args(argv)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    package = Path(__file__).parent / 'assets'
    for name, item in meta['files'].items():
        if item['source'] == 'package':
            copy_checked(package/name, root/name, item['sha256'])
    head_files = {k: v for k, v in meta['files'].items() if v['source'] == 'heads'}
    if not all((root/n).is_file() and sha(root/n) == v['sha256'] for n, v in head_files.items()):
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            archive = Path(args.heads_archive) if args.heads_archive else Path(tmp)/'heads.tar.gz'
            if not args.heads_archive:
                urllib.request.urlretrieve(meta['heads_archive']['url'], archive)
            if sha(archive) != meta['heads_archive']['sha256']:
                raise ValueError('Model archive checksum mismatch')
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                if {m.name for m in members} != set(head_files) or len(members) != len(head_files):
                    raise ValueError('Unexpected model archive entries')
                for member in members:
                    if not member.isfile():
                        raise ValueError('Only regular model files allowed')
                    src = Path(tmp)/Path(member.name).name
                    with tar.extractfile(member) as stream, src.open('xb') as dest:
                        shutil.copyfileobj(stream, dest)
                    copy_checked(src, root/member.name, head_files[member.name]['sha256'])
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
    print(json.dumps(dict(version='0.71', models=str(root), verified=True)))
