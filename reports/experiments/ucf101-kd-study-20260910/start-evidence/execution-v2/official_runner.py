"""Execute the already-reviewed matched20 plan; preserve upstream code unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metadata(command, upstream, source):
    sys.path[:0] = [str(upstream), str(source / 'Dassl.pytorch')]
    from train import setup_cfg
    from dassl.utils import set_random_seed
    from dassl.data.datasets import build_dataset
    def option(name):
        return command[command.index(name) + 1]
    cfg = setup_cfg(SimpleNamespace(
        root=option('--root'), output_dir=option('--output-dir'), seed=int(option('--seed')),
        trainer=option('--trainer'), dataset_config_file=option('--dataset-config-file'),
        config_file=option('--config-file'), opts=command[command.index('DATASET.NUM_SHOTS'):],
        resume='', source_domains=None, target_domains=None, transforms=None, backbone='', head=''))
    set_random_seed(cfg.SEED)
    dataset = build_dataset(cfg)
    def digest(items):
        rows = [(str(Path(x.impath).relative_to(Path(cfg.DATASET.ROOT).resolve())), int(x.label), x.classname)
                for x in items]
        return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
    return {'seed': cfg.SEED, 'dataset': cfg.DATASET.NAME, 'subsample': cfg.DATASET.SUBSAMPLE_CLASSES,
            'train_samples_sha256': digest(dataset.train_x), 'test_samples_sha256': digest(dataset.test),
            'train_count': len(dataset.train_x), 'test_count': len(dataset.test), 'effective_config': cfg.dump()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-checkpoint-sha', help='Resume only the documented first-checkpoint compatibility failure')
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    source = root / 'releases/cpu-prep-20260910'
    upstream = root / 'references/official-coop-20260910'
    import torch
    assert torch.cuda.is_available(), 'GPU required'
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source / 'Dassl.pytorch'), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')  # Only our own freshly trained upstream checkpoints.
    start = 0
    checkpoints = {}
    if args.resume_checkpoint_sha:
        previous = json.loads((output / 'status.json').read_text())
        assert previous['state'] == 'failed' and previous['completed_commands'] == 0
        assert previous['active_command'] == 0 and previous['error'].startswith('UnpicklingError')
        assert not (output / 'seed1/eval_base').exists()
        checkpoint = output / 'seed1/prompt_learner/model.pth.tar-20'
        assert sha(checkpoint) == args.resume_checkpoint_sha
        trained = (output / 'evidence/00-seed1-train.log').read_text()
        assert 'Finish training' in trained and 'epoch [20/20]' in trained
        assert torch.load(checkpoint, map_location='cpu', weights_only=False)['epoch'] == 20
        plan = json.loads((output / 'plan.json').read_text())
        assert plan['recipe'] == 'matched20' and plan['seeds'] == [1, 2, 3] and plan['epochs'] == 20
        assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip() == plan['upstream_commit']
        assert not subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'], text=True).strip()
        for name, digest in json.loads((output / 'provenance.json').read_text())['dassl_source_sha256'].items():
            assert sha(source / name) == digest
        save(output / 'checkpoint-load-first-failure.json', previous)
        save(output / 'recovery.json', {'resumed_unix': time.time(), 'kept_seed1_checkpoint_sha256': args.resume_checkpoint_sha,
                                     'reason': 'Own checkpoint contains legacy scheduler objects; use legacy loading only in upstream processes; no retraining or weight conversion.'})
        checkpoints[1] = args.resume_checkpoint_sha
        save(output / 'evidence/seed1-checkpoint.json', {'epoch': 20, 'sha256': checkpoints[1], 'path': str(checkpoint)})
        start = 1
    else:
        plan = json.loads(subprocess.check_output([
            sys.executable, str(source / 'scripts/repro/plan_official_coop.py'), '--upstream-root', str(upstream),
            '--data-root', str(root / 'data'), '--output-root', str(output), '--recipe', 'matched20'], env=env, text=True))
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'plan.json', plan)
    preflight = json.loads((source / 'reports/preparation/cpu-prep-20260910/ucf101-preflight.json').read_text())
    expected = {x['seed']: x['train_sha256'] for x in preflight['seeds']}
    provenance = {'upstream_commit': plan['upstream_commit'], 'project_commit': (source / 'DEPLOYED_COMMIT').read_text().strip(),
                  'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'clip_sha256': sha(root / 'assets/ViT-B-16.pt'),
                  'dassl_source_sha256': {str(f.relative_to(source)): sha(f) for f in sorted((source / 'Dassl.pytorch/dassl').rglob('*.py'))},
                  'execution_note': 'Unmodified upstream trainer/entry. FP32 and vendored Dassl adaptations. Upstream cudnn benchmark=True; current TC uses deterministic algorithms. Same numeric seed does not guarantee identical initialized prompt because backbone construction consumes RNG differently.'}
    provenance['checkpoint_loading'] = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 only in upstream child processes for self-generated checkpoint scheduler objects; SHA checked before evaluation.'
    save(output / ('provenance-after-compatibility-fix.json' if start else 'provenance.json'), provenance)
    status = {'state': 'running', 'started_unix': time.time(), 'completed_commands': start, 'total_commands': len(plan['commands'])}
    results = []
    try:
        for index, command in enumerate(plan['commands']):
            if index < start:
                continue
            seed = int(command[command.index('--seed') + 1])
            split = command[command.index('DATASET.SUBSAMPLE_CLASSES') + 1]
            evaluation = '--eval-only' in command
            name = f'{index:02d}-seed{seed}-' + (split if evaluation else 'train')
            info = metadata(command, upstream, source)
            if not evaluation:
                assert info['train_samples_sha256'] == expected[seed], 'Official/TC few-shot mismatch'
            save(output / 'evidence' / (name + '.json'), info)
            checkpoint = output / f'seed{seed}/prompt_learner/model.pth.tar-20'
            if evaluation:
                assert checkpoint.is_file() and sha(checkpoint) == checkpoints[seed]
            log_path = output / 'evidence' / (name + '.log')
            status.update(active_command=index, active_name=name)
            save(output / 'status.json', status)
            print(f'Running official {index+1}/{len(plan["commands"])}: {name}', flush=True)
            with log_path.open('w') as log:
                result = subprocess.run(command, cwd=upstream, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} exited {result.returncode}; inspect {log_path}')
            if not evaluation:
                assert torch.load(checkpoint, map_location='cpu', weights_only=False)['epoch'] == 20
                checkpoints[seed] = sha(checkpoint)
                save(output / 'evidence' / f'seed{seed}-checkpoint.json', {'epoch': 20, 'sha256': checkpoints[seed], 'path': str(checkpoint)})
            else:
                assert sha(checkpoint) == checkpoints[seed], 'Checkpoint changed during evaluation'
                log = log_path.read_text()
                assert str(checkpoint) in log and '(epoch = 20)' in log, 'Expected checkpoint load not logged'
                total = int(re.findall(r'\* total: ([\d,]+)', log)[-1].replace(',', ''))
                correct = int(re.findall(r'\* correct: ([\d,]+)', log)[-1].replace(',', ''))
                assert total == info['test_count'] and 0 <= correct <= total
                results.append({'seed': seed, 'split': split, 'correct': correct, 'total': total,
                                'accuracy': 100 * correct / total, 'checkpoint_sha256': checkpoints[seed]})
                save(output / 'results.json', results)
            status['completed_commands'] = index + 1
            save(output / 'status.json', status)
        means = {split: sum(r['accuracy'] for r in results if r['split'] == split) / 3 for split in ('base', 'new')}
        pairs = []
        for seed in (1, 2, 3):
            pair = {r['split']: r['accuracy'] for r in results if r['seed'] == seed}
            pairs.append({'seed': seed, **pair, 'hm': 2*pair['base']*pair['new']/(pair['base']+pair['new'])})
        save(output / 'summary.json', {'dataset': 'UCF101', 'recipe': 'matched20', 'seeds': pairs, 'means': means,
                                     'hm_of_means': 2*means['base']*means['new']/(means['base']+means['new']), 'results': results})
        status.update(state='completed', finished_unix=time.time())
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        save(output / 'status.json', status)


if __name__ == '__main__':
    main()
