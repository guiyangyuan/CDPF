#!/usr/bin/env python3
"""Prepare the verified UCF101 midframes archive and official OpenAI CLIP weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
CLIP_URL = f'https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/ViT-B-16.pt'
UCF_SHA256 = '2fe1bd40e64408d02ff173badf744da557033ae6c75dd4ea51519366551f3ae6'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_ucf(archive, data_root):
    if digest(archive) != UCF_SHA256:
        raise ValueError('UCF archive checksum does not match the verified author bundle')
    data_root.mkdir(parents=True, exist_ok=True)
    destination = data_root / 'ucf101'
    marker = destination / 'source_archive.json'
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text()).get('sha256') == UCF_SHA256:
            return destination
        raise FileExistsError(f'Existing dataset lacks the expected archive marker: {destination}')
    with tempfile.TemporaryDirectory(prefix='.ucf-extract-', dir=data_root) as temp:
        temp = Path(temp)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                path = Path(entry.filename)
                if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f'Unsafe archive entry: {entry.filename}')
            z.extractall(temp)
        source = temp / 'ucf101'
        if not (source / 'split_zhou_UCF101.json').is_file():
            raise ValueError('Archive does not contain the expected UCF101 split')
        (source / 'source_archive.json').write_text(json.dumps({'sha256': UCF_SHA256, 'archive': archive.name}, indent=2))
        source.rename(destination)
    return destination


def download_clip():
    destination = ROOT / 'clip/ViT-B-16.pt'
    if destination.exists():
        if digest(destination) != CLIP_SHA256:
            raise ValueError('Existing CLIP file has an unexpected checksum; it was not overwritten')
        return destination
    partial = destination.with_suffix('.pt.part')
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--connect-timeout', '30',
                    '--continue-at', '-', '--output', str(partial), CLIP_URL], check=True)
    if digest(partial) != CLIP_SHA256:
        raise ValueError('Downloaded CLIP checksum mismatch')
    partial.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucf-zip', type=Path)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--download-clip', action='store_true')
    args = parser.parse_args()
    if args.ucf_zip:
        print('Prepared UCF101:', extract_ucf(args.ucf_zip.resolve(), args.data_root.resolve()))
    if args.download_clip:
        print('Verified CLIP:', download_clip())

if __name__ == '__main__':
    main()
