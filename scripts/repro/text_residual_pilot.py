#!/usr/bin/env python3
"""Small fixed Aircraft pilot: three trainings, three reloads, no retries.

Requires an explicit `run` command and CUDA. `export` only loads a checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Dassl.pytorch'))

import numpy as np
import torch
from dassl.config import get_cfg_default
from dassl.data.datasets import build_dataset
from dassl.engine import build_trainer
from dassl.utils import set_random_seed
import train
from trainers.textconsistency_runtime import source_digest, sample_digest
from trainers.textconsistency_common import sha256_file

VARIANTS = ('classname', 'description', 'contrast')
BRANCHES = ('S', 'M', 'R', 'Z', 'F')
MODEL_SOURCE_SHA = 'b582a30a733c735126afccf6c77f64c0d1fdec3be9fd4b34d8a227b15e7c39bf'
CONFIG_HASHES = {'classname': 'd72b29115949af803f4c221a5d76c6630baed30727c54b36e91646dff42e130c', 'description': '3cec99139bdc6a3d9e00710fec261ac4bd18a0cbfc9f733028bd560fe016ea07', 'contrast': '9dd1a4e4a067cc9bf0409227cbc63faf5db5dc688ac0710c81ef54e18309da51'}
CLIP_SHA = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
TEMPLATE_SHA = 'd86833c2483c1d12907abeb6198412f1d0a020bd12999a3f170967a7f549e3b8'
SAMPLE_HASHES = {
    'train_samples_sha256': 'ae5851085ef5e74b7cb612cda7454fde300601ef14ad21b707bbbfbbd3c85e9a',
    'val_samples_sha256': '7ac44ab40ea24086902991743d3b51720e467204034efd2b5a98352416fd02bb',
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def collect_predictions(model, batches, device):
    arrays = {key: [] for key in ('labels', *BRANCHES)}
    paths = []
    model.eval()
    for batch in batches:
        outputs = model(batch['img'].to(device))
        labels = batch['label'].cpu().numpy()
        if labels.dtype != np.int64 or labels.ndim != 1:
            raise ValueError('Malformed labels')
        arrays['labels'].append(labels)
        for key in BRANCHES:
            value = outputs[key].cpu().numpy()
            if value.ndim != 2 or len(value) != len(labels) or not np.isfinite(value).all():
                raise ValueError(f'Malformed {key} logits')
            arrays[key].append(value)
        paths.extend(str(path) for path in batch['impath'])
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Empty or duplicate export rows')
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    np.testing.assert_array_equal(arrays['Z'], arrays['S'] + arrays['R'])
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    return arrays, paths


def export(args):
    model_dir, out = Path(args.model_dir).resolve(), Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Use a fresh export directory: {out}')
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.merge_from_file(str(model_dir / 'effective_config.yaml'))
    if (cfg.TRAINER.NAME != 'CoOp_TextResidual' or cfg.SEED != 1
            or cfg.DATASET.NAME != 'FGVCAircraft' or cfg.DATASET.SUBSAMPLE_CLASSES != 'base'
            or cfg.DATASET.NUM_SHOTS != 16 or cfg.OPTIM.MAX_EPOCH != 20):
        raise ValueError('This exporter is restricted to the fixed Aircraft pilot')
    cfg.OUTPUT_DIR = str(out / 'runtime')
    cfg.USE_CUDA = True; cfg.TRAINER.TEXTCONSISTENCY.DEVICE = 'cuda'
    cfg.freeze()
    set_random_seed(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    trainer = build_trainer(cfg)
    trainer.load_model(model_dir, epoch=20)
    for key, expected in SAMPLE_HASHES.items():
        if trainer.run_provenance[key] != expected:
            raise ValueError(f'Aircraft pilot split differs from fixed anchor: {key}')
    arrays, paths = collect_predictions(trainer.model, trainer.val_loader, trainer.device)
    if arrays['S'].shape != (200, 50):
        raise ValueError('Expected 200 Base validation images and 50 classes')
    rows = [{'sample_id': str(Path(path).resolve().relative_to(Path(cfg.DATASET.ROOT).resolve())),
             'label': int(label)} for path, label in zip(paths, arrays['labels'])]
    expected_rows = [{'sample_id': str(Path(item.impath).resolve().relative_to(Path(cfg.DATASET.ROOT).resolve())),
                      'label': int(item.label)} for item in trainer.dm.dataset.val]
    if rows != expected_rows:
        raise ValueError('Validation output order differs from the dataset')
    np.savez_compressed(out / 'logits.npz', **arrays)
    result = {'status': 'exported', 'split': 'base_val', 'rows': rows,
              'class_order': list(trainer.dm.dataset.classnames),
              'provenance': trainer.run_provenance, 'checkpoint': trainer.loaded_checkpoints,
              'logits_sha256': sha256_file(out / 'logits.npz'),
              'correct': {key: int((arrays[key].argmax(1) == arrays['labels']).sum())
                          for key in BRANCHES if key != 'R'},
              'neighbor_indices': trainer.model.neighbor_indices, 'branch_semantics': {'Z': 'S+clip_scale*image@text_delta.T', 'F': '.5*S+.5*M', 'R': 'not a standalone classifier'}}
    write_json(out / 'export.json', result)
    return result


def analyze(output):
    arrays, records = {}, {}
    for variant in VARIANTS:
        folder = output / variant / 'export'
        record = json.loads((folder / 'export.json').read_text())
        if sha256_file(folder / 'logits.npz') != record['logits_sha256']:
            raise ValueError('Export file changed')
        with np.load(folder / 'logits.npz', allow_pickle=False) as values:
            arrays[variant] = {key: values[key].copy() for key in values.files}
        records[variant] = record
    first = records['classname']
    for record in records.values():
        if record['rows'] != first['rows'] or record['class_order'] != first['class_order']:
            raise ValueError('Cannot compare different validation inputs')
        if record['provenance']['code_sha256'] != first['provenance']['code_sha256']:
            raise ValueError('Cannot compare different model source revisions')
    correct = lambda variant, branch: arrays[variant][branch].argmax(1) == arrays[variant]['labels']
    candidate = correct('contrast', 'Z')
    for variant in VARIANTS:
        np.testing.assert_array_equal(arrays[variant]['S'], arrays['classname']['S'])
    comparisons = {}
    for variant, branch in (('classname', 'S'), ('classname', 'F'), ('classname', 'Z'), ('description', 'Z')):
        base = correct(variant, branch)
        difference = int(candidate.sum()) - int(base.sum())
        comparisons[f'contrast.Z - {variant}.{branch}'] = {
            'gain_pp': difference / 2, 'corrected': int((candidate & ~base).sum()),
            'harmed': int((~candidate & base).sum()), 'passes_1pp': difference >= 2}
    return {'status': 'completed', 'scope': 'Aircraft seed1 Base validation development only',
            'correct': {variant: {branch: int(correct(variant, branch).sum())
                                 for branch in ('S', 'M', 'Z', 'F')} for variant in VARIANTS},
            'n': 200, 'comparisons': comparisons,
            'continuation_signal': all(value['passes_1pp'] for value in comparisons.values()),
            'automatic_expansion': False}


def run(args):
    data, weights, output = [Path(value).resolve() for value in (args.data_root, args.clip_path, args.output_root)]
    if output.exists():
        raise FileExistsError(f'No resume or overwrite: {output}')
    for protected in (ROOT, data, weights):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Keep pilot output separate from code, data and CLIP weights')
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot training requires an available CUDA GPU')
    if sha256_file(weights) != CLIP_SHA or sha256_file(ROOT / 'templates/airplane_prompts_full.json') != TEMPLATE_SHA:
        raise ValueError('CLIP or description asset differs from the existing pilot')
    if not (data / 'fgvc_aircraft/split_fewshot/shot_16-seed_1.pkl').is_file():
        raise FileNotFoundError('Reuse the prepared seed1 few-shot cache before training')
    configs = {name: ROOT / f'configs/repro/text_residual/{name}.yaml' for name in VARIANTS}
    if source_digest() != MODEL_SOURCE_SHA or any(sha256_file(path) != CONFIG_HASHES[name]
                                                 for name, path in configs.items()):
        raise ValueError('Frozen correction pilot code or configuration differs; do not silently change the protocol')
    # Read only the small split/cache metadata; do not decode or rehash all images.
    check = get_cfg_default(); train.extend_cfg(check)
    check.merge_from_file(str(ROOT / 'configs/datasets/fgvc_aircraft.yaml'))
    check.merge_from_file(str(configs['classname']))
    check.DATASET.ROOT = str(data); check.SEED = 1
    dataset = build_dataset(check)
    if len(dataset.train_x) != 800 or len(dataset.val) != 200 or len(dataset.classnames) != 50:
        raise ValueError('Unexpected Aircraft Base few-shot split sizes')
    for split, items in (('train', dataset.train_x), ('val', dataset.val)):
        if sample_digest(items, data) != SAMPLE_HASHES[split + '_samples_sha256']:
            raise ValueError(f'Unexpected few-shot {split} split; fix before spending GPU time')
    input_hashes = {str(path): sha256_file(path) for path in
                    (*configs.values(), ROOT / 'configs/datasets/fgvc_aircraft.yaml', Path(__file__).resolve(),
                     data / 'fgvc_aircraft/split_fewshot/shot_16-seed_1.pkl')}
    source = source_digest()
    output.mkdir(parents=True)
    stages = []
    for variant in VARIANTS:
        model, exported = output / variant / 'model', output / variant / 'export'
        command = [sys.executable, '-u', str(ROOT / 'train.py'), '--root', str(data), '--seed', '1',
                   '--dataset-config-file', str(ROOT / 'configs/datasets/fgvc_aircraft.yaml'),
                   '--config-file', str(configs[variant]), '--output-dir', str(model),
                   'USE_CUDA', 'True', 'TRAINER.TEXTCONSISTENCY.DEVICE', 'cuda',
                   'TRAINER.TEXTCONSISTENCY.CLIP_PATH', str(weights), 'TEST.NO_TEST', 'True']
        stages.extend([(variant + '-train', command), (variant + '-export',
                       [sys.executable, '-u', str(Path(__file__).resolve()), 'export',
                        '--model-dir', str(model), '--output-dir', str(exported)])])
    status = {'state': 'running', 'source_sha256': source, 'input_hashes': input_hashes,
              'gpu': torch.cuda.get_device_name(0), 'stages': [], 'planned_commands': stages,
              'max_training_runs': 3, 'wallclock_cap_seconds': 3600, 'minimum_gain_pp': 1.0}
    write_json(output / 'status.json', status)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'Dassl.pytorch'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    try:
        for name, command in stages:
            if source_digest() != source or any(sha256_file(Path(path)) != digest for path, digest in input_hashes.items()):
                raise ValueError('Source or fixed configuration changed during the pilot')
            remaining = 3600 - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Pilot wallclock cap reached')
            status['current_stage'] = name; write_json(output / 'status.json', status)
            print(f'Starting {name}; log: {output / (name + ".log")}', flush=True)
            with (output / (name + '.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=remaining)
            status['stages'].append({'name': name, 'status': 'passed'})
            write_json(output / 'status.json', status)
        result = analyze(output)
        write_json(output / 'analysis.json', result)
        status.update(state='completed', continuation_signal=result['continuation_signal'])
    except BaseException as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic() - start
        write_json(output / 'status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    launch = commands.add_parser('run')
    for option in ('data-root', 'clip-path', 'output-root'):
        launch.add_argument('--' + option, required=True)
    extract = commands.add_parser('export')
    extract.add_argument('--model-dir', required=True); extract.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args) if args.command == 'run' else export(args), indent=2, ensure_ascii=False))
