#!/usr/bin/env python3
"""Exercise the normal CLI with four real classes and one image per split/class.
Fixture results are deliberately 1-shot and must never enter the 16-shot paper table.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    source = args.data_root.resolve() / 'ucf101'
    original = json.loads((source / 'split_zhou_UCF101.json').read_text())
    labels = sorted({row[1] for row in original['train']})[:4]
    split = {}
    for key, rows in original.items():
        chosen = {}
        for row in rows:
            if row[1] in labels: chosen.setdefault(row[1], row)
        split[key] = [chosen[label] for label in labels]
    fixture = output / 'fixture/ucf101'; fixture.mkdir(parents=True)
    (fixture / 'split_zhou_UCF101.json').write_text(json.dumps(split))
    (fixture / 'UCF-101-midframes').symlink_to(source / 'UCF-101-midframes', target_is_directory=True)
    (output / 'SMOKE_ONLY.json').write_text(json.dumps({'smoke_only': True, 'classes': 4,
        'base_classes': 2, 'novel_classes': 2, 'images_per_class_per_split': 1, 'epochs': 1}, indent=2))
    common = [sys.executable, str(ROOT / 'train.py'), '--root', str(fixture.parent), '--seed', '1',
              '--trainer', 'CoOp_TextConsistency_V2_FG', '--dataset-config-file', str(ROOT / 'configs/datasets/ucf101.yaml'),
              '--config-file', str(ROOT / 'configs/repro/ucf101.yaml')]
    opts = ['TRAINER.TEXTCONSISTENCY.DEVICE','cpu','TRAINER.TEXTCONSISTENCY.N_VIEWS','5',
            'TRAINER.TEXTCONSISTENCY.N_GROUPS','2','TRAINER.TEXTCONSISTENCY.TEXT_BATCH_SIZE','4',
            'DATASET.NUM_SHOTS','1','DATALOADER.NUM_WORKERS','0','DATALOADER.TRAIN_X.BATCH_SIZE','2',
            'DATALOADER.TEST.BATCH_SIZE','2','OPTIM.MAX_EPOCH','1']
    train_dir = output / 'train'
    commands = [('train', common + ['--output-dir',str(train_dir)] + opts + ['DATASET.SUBSAMPLE_CLASSES','base'])]
    for split_name in ('base','new'):
        commands.append((f'eval_{split_name}', common + ['--output-dir',str(output / f'eval_{split_name}'),
                         '--model-dir',str(train_dir),'--load-epoch','1','--eval-only'] + opts + ['DATASET.SUBSAMPLE_CLASSES',split_name]))
    env = os.environ.copy(); env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT/'Dassl.pytorch')])
    env['OMP_NUM_THREADS'] = '4'; env['MKL_NUM_THREADS'] = '4'
    for stage, command in commands:
        print('Executing smoke stage:', stage, flush=True)
        with (output / f'{stage}.console.txt').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    base = json.loads((output/'eval_base/metrics.json').read_text())
    novel = json.loads((output/'eval_new/metrics.json').read_text())
    if base['checkpoints'] != novel['checkpoints']: raise AssertionError('Different checkpoint used across splits')
    for result in (base,novel):
        if result['state'] != 'loaded_checkpoint' or result['seed'] != 1: raise AssertionError('Wrong state or seed')
    summary = {'smoke_only':True,'train_cli_passed':True,'base_eval_cli_passed':True,'novel_eval_cli_passed':True,
               'same_checkpoint_for_base_and_novel':True,'code_sha256':base['code_sha256'],
               'checkpoint_sha256':base['checkpoints'][0]['sha256']}
    (output/'verification.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
