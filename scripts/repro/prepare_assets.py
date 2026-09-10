#!/usr/bin/env python3
"""Safely prepare verified author dataset archives and official OpenAI CLIP weights."""
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
DATASETS = {
    'ucf101': {'sha256': UCF_SHA256, 'required': ('split_zhou_UCF101.json',)},
    'dtd': {'sha256': '5ba4d3000e60f5541ba15a082033ca0696eee68125e9c28edff3dbe22f04313b',
            'required': ('split_zhou_DescribableTextures.json',)},
    'fgvc_aircraft': {'sha256': '84ede9f7272e9942bb687d427dd1184dbacc9814a7303174746deb6fb0b56530',
                      'required': ('variants.txt', 'images_variant_train.txt', 'images_variant_val.txt',
                                   'images_variant_test.txt', 'images')},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_dataset(archive, data_root, dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset archive: {dataset}')
    specification = DATASETS[dataset]
    if digest(archive) != specification['sha256']:
        raise ValueError(f'{dataset} archive checksum does not match the verified author bundle')
    data_root.mkdir(parents=True, exist_ok=True)
    destination = data_root / dataset
    marker = destination / 'source_archive.json'
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text()).get('sha256') == specification['sha256']:
            return destination
        raise FileExistsError(f'Existing dataset lacks the expected archive marker: {destination}')
    with tempfile.TemporaryDirectory(prefix=f'.{dataset}-extract-', dir=data_root) as temp:
        temp = Path(temp)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                path = Path(entry.filename)
                if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f'Unsafe archive entry: {entry.filename}')
            z.extractall(temp)
        source = temp / dataset
        if not source.is_dir():
            raise ValueError(f'Archive does not contain the expected {dataset}/ root')
        missing = [name for name in specification['required'] if not (source / name).exists()]
        if missing:
            raise ValueError(f'Archive does not contain required {dataset} content: {missing[0]}')
        (source / 'source_archive.json').write_text(json.dumps(
            {'sha256': specification['sha256'], 'archive': archive.name}, indent=2))
        source.rename(destination)
    return destination


def extract_ucf(archive, data_root):
    """Backward-compatible UCF101 entry point."""
    return extract_dataset(archive, data_root, 'ucf101')


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
    parser.add_argument('--dtd-zip', type=Path)
    parser.add_argument('--aircraft-zip', type=Path)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--download-clip', action='store_true')
    args = parser.parse_args()
    if args.ucf_zip:
        print('Prepared UCF101:', extract_ucf(args.ucf_zip.resolve(), args.data_root.resolve()))
    if args.dtd_zip:
        print('Prepared DTD:', extract_dataset(args.dtd_zip.resolve(), args.data_root.resolve(), 'dtd'))
    if args.aircraft_zip:
        print('Prepared FGVC Aircraft:', extract_dataset(args.aircraft_zip.resolve(), args.data_root.resolve(), 'fgvc_aircraft'))
    if args.download_clip:
        print('Verified CLIP:', download_clip())

if __name__ == '__main__':
    main()
