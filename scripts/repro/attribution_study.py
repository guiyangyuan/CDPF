#!/usr/bin/env python3
"""Fixed attribution queue. Preparation uses isolated CPU metadata processes."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cross_dataset_study as common

DATASETS = ('dtd', 'fgvc_aircraft')
ALL_DATASETS = ('ucf101',) + DATASETS
SEEDS = (1, 2, 3)
MODEL_SHA256 = common.MODEL_SHA256
CLIP_SHA256 = common.CLIP_SHA256
UCF_SOURCE_SHA256 = 'd74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f'
BRANCHES = ('student', 'teacher', 'ensemble')
FUSION_BRANCHES = ('student', 'description_mean', 'simple_fusion', 'description_instance',
                   'mean_probability_fusion', 'adk_probability_fusion')
sha256 = common.sha256
atomic_json = common.atomic_json


def read_json(path):
    return json.loads(Path(path).read_text())


def value(command, flag):
    return command[command.index(flag) + 1]


def disjoint(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    return left != right and left not in right.parents and right not in left.parents


def validate_receipt(receipt, source_root):
    seen_paths = []
    for group, datasets, path_field in (('fusion_baselines', ALL_DATASETS, 'baseline_dir'),
                                       ('historical_kd1', DATASETS, 'model_dir'),
                                       ('ucf101_full', ('ucf101',), 'model_dir')):
        rows = receipt.get(group, [])
        expected = {(d, s) for d in datasets for s in SEEDS}
        actual = [(r.get('dataset'), r.get('seed')) for r in rows]
        if len(actual) != len(expected) or set(actual) != expected or any(type(r.get('seed')) is not int for r in rows):
            raise ValueError(f'Incomplete, duplicate or incorrect dataset/seed in {group}')
        for row in rows:
            path = Path(row[path_field])
            if not path.is_absolute() or path.name != f"seed{row['seed']}":
                raise ValueError('Input model path must be absolute and match seed')
            if any(not disjoint(path, old) for old in seen_paths):
                raise ValueError('Input model paths overlap or are reused')
            seen_paths.append(path)
            expected_source = UCF_SOURCE_SHA256 if group == 'fusion_baselines' and row['dataset'] == 'ucf101' else MODEL_SHA256
            if row.get('source_sha256') != expected_source:
                raise ValueError('Receipt source digest differs from fixed source')
            checkpoint = row.get('checkpoint_sha256', '')
            if len(checkpoint) != 64 or any(c not in '0123456789abcdef' for c in checkpoint):
                raise ValueError('Invalid checkpoint SHA in receipt')
            if group == 'fusion_baselines':
                if not Path(row['source_root']).is_absolute():
                    raise ValueError('Source root must be absolute')
                if row['dataset'] in DATASETS and Path(row['source_root']).resolve() != Path(source_root).resolve():
                    raise ValueError('DTD/Aircraft source root differs from requested source')
                if set(row.get('historical', {})) != {'base', 'new'}:
                    raise ValueError('Baseline receipt lacks both historical splits')
    for dataset in ALL_DATASETS:
        roots = {str(Path(r['source_root']).resolve()) for r in receipt['fusion_baselines'] if r['dataset'] == dataset}
        if len(roots) != 1:
            raise ValueError('Each dataset must use one frozen baseline source root')
    sources = {str(Path(source_root).resolve())} | {str(Path(r['source_root']).resolve()) for r in receipt['fusion_baselines']}
    for path in seen_paths:
        if any(not disjoint(path, source) for source in sources):
            raise ValueError('Input model overlaps source')


def row_for(receipt, group, dataset, seed):
    return next(r for r in receipt[group] if r['dataset'] == dataset and r['seed'] == seed)


def full_commands(source, data, output, seeds, epochs, kd, python, dataset):
    # Only the source-owned train/config paths use runner.ROOT; caller paths stay untouched.
    previous_path = list(sys.path)
    try:
        runner = common.load_file(HERE / 'run_pilot.py', 'attribution_run_pilot')
        runner.ROOT = Path(source).resolve()
        return runner.build_commands(data, output, seeds, epochs, ['hierarchical_adapter'],
                                     python=python, dataset=dataset, kd_weight=float(kd))
    finally:
        sys.path[:] = previous_path



def build_plan(source_root, data_root, output_root, input_receipt, python=sys.executable):
    source, data, output = (Path(p).resolve() for p in (source_root, data_root, output_root))
    receipt_path = Path(input_receipt).resolve()
    receipt = read_json(receipt_path)
    validate_receipt(receipt, source)
    forbidden = [source, data, receipt_path] + [Path(r['source_root']) for r in receipt['fusion_baselines']]
    forbidden += [Path(r.get('baseline_dir', r.get('model_dir'))) for g in ('fusion_baselines', 'historical_kd1', 'ucf101_full') for r in receipt[g]]
    if any(not disjoint(output, p) for p in forbidden):
        raise ValueError('New output must be isolated from every input')
    stages = []
    for kind in ('smoke', 'anchor', 'fusion', 'formal', 'historical'):
        for dataset in (ALL_DATASETS if kind == 'fusion' else DATASETS):
            seeds = [1] if kind in ('smoke', 'anchor') else list(SEEDS)
            kd = 0 if kind in ('smoke', 'formal') else 1
            name = f'{kind}_{dataset}' + ('' if kind == 'fusion' else f'_kd{kd}')
            root = output / name
            epochs = 1 if kind == 'smoke' else 20
            stage = dict(name=name, kind=kind, dataset=dataset, seeds=seeds, epochs=epochs,
                         kd_weight=kd, root=str(root), source_root=str(source))
            if kind == 'fusion':
                commands = []
                for seed in seeds:
                    row = row_for(receipt, 'fusion_baselines', dataset, seed)
                    stage['source_root'] = row['source_root']
                    for split in ('base', 'new'):
                        commands.append([python, str(HERE / 'evaluate_frozen_fusion.py'), '--source-root', row['source_root'],
                                         '--baseline-dir', row['baseline_dir'], '--dataset', dataset, '--seed', str(seed),
                                         '--split', split, '--data-root', str(data), '--output-dir', str(root / f'seed{seed}' / split),
                                         '--expected-source-sha256', row['source_sha256']])
            else:
                commands = full_commands(source, data, root, seeds, epochs, kd, python, dataset)
                if kind == 'historical':
                    commands = [c for c in commands if '--eval-only' in c]
                    for c in commands:
                        c[c.index('--model-dir') + 1] = row_for(receipt, 'historical_kd1', dataset, int(value(c, '--seed')))['model_dir']
            stage['commands'] = commands
            stages.append(stage)
    return dict(purpose='fixed_attribution_study', source_root=str(source), data_root=str(data), output_root=str(output),
                input_receipt=str(receipt_path), input_receipt_sha256=sha256(receipt_path), receipt=receipt,
                python=python, stages=stages, seeds=list(SEEDS), expected_model_sha256=MODEL_SHA256,
                expected_clip_sha256=CLIP_SHA256, fixed_settings=common.FIXED,
                command_count=sum(len(s['commands']) for s in stages), selection='all branches; no search or retries')


def operational_hashes():
    return {str(HERE / name): sha256(HERE / name) for name in (
        'attribution_study.py', 'attribution_audit.py', 'evaluate_frozen_fusion.py', 'cross_dataset_study.py',
        'cross_dataset_audit.py', 'run_pilot.py', 'structure_spec.py')}


def source_hashes(source):
    source = Path(source)
    paths = [source / 'train.py', source / 'clip/bpe_simple_vocab_16e6.txt.gz']
    for directory, pattern in (('trainers', '*.py'), ('datasets', '*.py'), ('clip', '*.py'),
                               ('Dassl.pytorch/dassl', '*.py'), ('configs', '*.yaml'), ('templates', '*.json'),
                               ('scripts/repro', '*.py')):
        paths.extend(sorted((source / directory).rglob(pattern)))
    return {str(path): sha256(path) for path in paths}


def child_env(source, threads):
    env = os.environ.copy()
    env.update({key: str(threads) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['PYTHONPATH'] = os.pathsep.join([str(source), str(Path(source) / 'Dassl.pytorch')])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
    return env


def isolated(source, payload, python=sys.executable):
    try:
        result = subprocess.run([python, str(Path(__file__).resolve()), '--metadata-worker'],
                                input=json.dumps(payload), text=True, capture_output=True, cwd=source,
                                env=child_env(source, 1), check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"CPU metadata {payload['action']} failed in {source}: {error.stderr}") from error
    return json.loads(result.stdout)


def token_unique_indices(rows, tokenize):
    """Mirror the source encoder's within-class token deduplication without encoding."""
    tokens = tokenize([text for row in rows for text in row], truncate=True)
    effective, offset = [], 0
    for row in rows:
        seen, indices = set(), []
        for index in range(len(row)):
            key = tuple(tokens[offset + index].tolist())
            if key not in seen:
                seen.add(key)
                indices.append(index)
        effective.append(indices)
        offset += len(row)
    return effective


def dataset_metadata(source, data, dataset):
    evidence = common.dataset_evidence(source, data, dataset)
    preflight = common.source_module(source, 'scripts/repro/preflight.py', 'attribution_preflight')
    import trainers.textconsistency_common as tc
    from clip import clip
    from clip.simple_tokenizer import default_bpe
    vocabulary = Path(default_bpe()).resolve()
    evidence['tokenizer'] = {'path': str(vocabulary), 'sha256': sha256(vocabulary), 'truncate': True}
    template = Path(source) / tc.TEMPLATE_FILES[preflight.DATASETS[dataset]['name']]
    templates = read_json(template)
    evidence['template'] = {'path': str(template), 'sha256': sha256(template)}
    for row in evidence['loader_evidence']:
        for split in row['splits'].values():
            descriptions = tc.description_rows(templates, split['classes'], 20)
            indices = token_unique_indices(descriptions, clip.tokenize)
            counts = [len(row) for row in indices]
            split['description_rows'] = descriptions
            split['effective_description_indices'] = indices
            split['effective_description_rows'] = [[row[i] for i in kept] for row, kept in zip(descriptions, indices)]
            split['description_counts'] = counts
            # This is the explicit contiguous order of the fixed protocol, including the old UCF source.
            order = {'version': 1, 'policy': 'contiguous', 'seed': 0, 'n_groups': 4,
                     'classes': [{'classname': n, 'count': c, 'order': list(range(c))} for n, c in zip(split['classes'], counts)]}
            import hashlib
            order['sha256'] = hashlib.sha256(json.dumps(order, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            split['view_order_manifest'] = order
    return evidence


def worker(payload):
    source, plan = payload['source'], payload['plan']
    sys.path[:0] = [source, str(Path(source) / 'Dassl.pytorch')]
    import torch
    torch.set_num_threads(1)
    if payload['action'] == 'datasets':
        return {d: dataset_metadata(source, plan['data_root'], d) for d in payload['datasets']}
    if payload['action'] == 'configs':
        result = {}
        fusion = common.load_file(HERE / 'evaluate_frozen_fusion.py', 'attribution_fusion')
        for stage in payload['stages']:
            configs = []
            for command in stage['commands']:
                if stage['kind'] == 'fusion':
                    _, _, train_cfg, cfg, _ = fusion.build_source_configs(Path(source), Path(plan['data_root']),
                        Path(value(command, '--output-dir')), stage['dataset'], int(value(command, '--seed')), value(command, '--split'))
                    metadata = fusion.validate_baseline_metadata(Path(value(command, '--baseline-dir')), stage['dataset'],
                        int(value(command, '--seed')), value(command, '--split'), value(command, '--expected-source-sha256'),
                        expected_train_config=train_cfg, expected_eval_config=cfg)
                    row = row_for(plan['receipt'], 'fusion_baselines', stage['dataset'], int(value(command, '--seed')))
                    historical = row['historical'][value(command, '--split')]
                    if (metadata['checkpoint_sha256'] != row['checkpoint_sha256'] or
                        sha256(Path(metadata['eval_dir']) / 'metrics.json') != historical['metrics_sha256'] or
                        metadata['metric']['test_samples_sha256'] != historical['test_sha256'] or
                        metadata['historical_student']['accuracy'] != historical['accuracy']):
                        raise ValueError('Historical baseline differs from verified receipt')
                else:
                    cfg = common.effective_config_for_command(source, command)
                configs.append(cfg)
            result[stage['name']] = configs
        return result
    if payload['action'] == 'historical':
        import attribution_audit as audit
        return [audit.inspect_historical(source, plan, row, payload['datasets'][row['dataset']]) for row in payload['rows']]
    raise ValueError('Unknown metadata worker action')


def input_hashes(plan):
    hashes = {plan['input_receipt']: sha256(plan['input_receipt'])}
    for group in ('fusion_baselines', 'historical_kd1', 'ucf101_full'):
        for row in plan['receipt'][group]:
            directory = Path(row.get('baseline_dir', row.get('model_dir')))
            cp = directory / 'textconsistency/model.pth.tar-20'
            if sha256(cp) != row['checkpoint_sha256'] or cp.with_name(cp.name + '.sha256').read_text().strip() != row['checkpoint_sha256']:
                raise ValueError('Input checkpoint/sidecar differs from verified receipt')
            for required in ('run_manifest.json', 'effective_config.yaml', 'log.txt'):
                if not (directory / required).is_file():
                    raise FileNotFoundError(directory / required)
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    hashes[str(path)] = sha256(path)
    return hashes


def prepare(plan, preparation_dir, clip_path=None):
    target = Path(preparation_dir).resolve()
    if target.exists():
        raise FileExistsError(f'Choose a fresh preparation directory: {target}')
    forbidden = [plan['output_root'], plan['source_root'], plan['data_root'], plan['input_receipt']]
    forbidden += [r.get('baseline_dir', r.get('model_dir')) for g in ('fusion_baselines', 'historical_kd1', 'ucf101_full') for r in plan['receipt'][g]]
    forbidden += [r['source_root'] for r in plan['receipt']['fusion_baselines']]
    if any(not disjoint(target, path) for path in forbidden):
        raise ValueError('Preparation directory must be isolated from inputs and study')
    clip = Path(clip_path or Path.home() / '.cache/clip/ViT-B-16.pt').resolve()
    if sha256(clip) != CLIP_SHA256:
        raise ValueError('Complete CLIP checkpoint SHA differs')
    sources = {plan['source_root']: MODEL_SHA256}
    sources.update({r['source_root']: r['source_sha256'] for r in plan['receipt']['fusion_baselines']})
    for source, expected in sources.items():
        if common.model_source_digest(source) != expected:
            raise ValueError('Frozen model source digest differs')
    frozen = operational_hashes()
    for source in sources:
        frozen.update(source_hashes(source))
    before = input_hashes(plan)
    datasets_by_source = {}
    for source in sources:
        ds = list(ALL_DATASETS) if source == plan['source_root'] else ['ucf101']
        datasets_by_source[source] = isolated(source, dict(action='datasets', source=source, plan=plan, datasets=ds), plan['python'])
    for source, datasets in datasets_by_source.items():
        if source != plan['source_root']:
            current = datasets_by_source[plan['source_root']]['ucf101']
            if datasets['ucf101']['loader_evidence'] != current['loader_evidence'] or datasets['ucf101']['template']['sha256'] != current['template']['sha256']:
                raise ValueError('UCF baseline/full source loaders or descriptions differ')
    configs = {}
    for source in sources:
        stages = [s for s in plan['stages'] if s['source_root'] == source]
        configs.update(isolated(source, dict(action='configs', source=source, plan=plan, stages=stages), plan['python']))
    histories = isolated(plan['source_root'], dict(action='historical', source=plan['source_root'], plan=plan,
        rows=plan['receipt']['historical_kd1'] + plan['receipt']['ucf101_full'],
        datasets=datasets_by_source[plan['source_root']]), plan['python'])
    after = input_hashes(plan)
    if before != after:
        raise ValueError('Historical inputs changed during preparation')
    evidence = dict(protocol=plan, files_sha256=frozen, input_files_sha256=before,
                    clip={'path': str(clip), 'sha256': CLIP_SHA256}, datasets_by_source=datasets_by_source,
                    datasets=datasets_by_source[plan['source_root']], expected_effective_configs=configs,
                    historical=histories, environment=common.cpu_environment(),
                    preparation_policy='CPU threads=1; no CLIP tensor load; split_fewshot is allowed derived loader output')
    verify_frozen(evidence, check_data=True)
    target.mkdir(parents=True)
    atomic_json(target / 'preparation.json', evidence)
    atomic_json(target / 'protocol.json', plan)
    return evidence


def verify_frozen(prepared, check_data=False):
    plan = prepared['protocol']
    if read_json(plan['input_receipt']) != plan['receipt']:
        raise ValueError('Receipt changed')
    for collection in ('files_sha256', 'input_files_sha256'):
        for path, expected in prepared[collection].items():
            if not Path(path).is_file() or sha256(path) != expected:
                raise ValueError(f'Frozen input or dependency changed: {path}')
    if input_hashes(plan) != prepared['input_files_sha256']:
        raise ValueError('Historical input inventory changed')
    if sha256(prepared['clip']['path']) != CLIP_SHA256 or prepared['clip']['sha256'] != CLIP_SHA256:
        raise ValueError('CLIP changed')
    sources = {plan['source_root']: MODEL_SHA256}
    sources.update({r['source_root']: r['source_sha256'] for r in plan['receipt']['fusion_baselines']})
    for source, expected in sources.items():
        if common.model_source_digest(source) != expected:
            raise ValueError('Model source changed')
    if check_data:
        for source, datasets in prepared['datasets_by_source'].items():
            actual = isolated(source, dict(action='datasets', source=source, plan=plan, datasets=list(datasets)), plan['python'])
            if actual != datasets:
                raise ValueError('Real loader, template or image inventory changed')


def execute(plan, preparation_dir):
    output = Path(plan['output_root'])
    if output.exists():
        raise FileExistsError(f'Choose a fresh output root: {output}')
    import torch
    common.assert_gpus_idle(torch)
    prepared = read_json(Path(preparation_dir) / 'preparation.json')
    if prepared['protocol'] != plan:
        raise ValueError('Prepared protocol differs')
    verify_frozen(prepared, check_data=True)
    output.mkdir(parents=True)
    atomic_json(output / 'protocol.json', plan)
    atomic_json(output / 'data_preflight.json', prepared['datasets_by_source'])
    status = dict(state='running', stage=None, pid=os.getpid(), commands=[], completed=[])
    atomic_json(output / 'status.json', status)
    try:
        for stage in plan['stages']:
            status['stage'] = stage['name']
            stage_root = Path(stage['root']); stage_root.mkdir(parents=True)
            atomic_json(stage_root / 'protocol.json', stage)
            atomic_json(stage_root / 'data_preflight.json', prepared['datasets_by_source'][stage['source_root']][stage['dataset']])
            for index, command in enumerate(stage['commands']):
                run_record(command, stage['name'], index, stage['source_root'], output, status)
            audit_command = [plan['python'], str(HERE / 'attribution_audit.py'), '--study-root', str(output),
                             '--preparation-dir', str(Path(preparation_dir).resolve()), '--stage', stage['name']]
            run_record(audit_command, stage['name'], 'audit', plan['source_root'], output, status)
            status['completed'].append(stage['name']); atomic_json(output / 'status.json', status)
        verify_frozen(prepared, check_data=True)
        status['stage'] = 'final_audit'
        run_record([plan['python'], str(HERE / 'attribution_audit.py'), '--study-root', str(output),
                    '--preparation-dir', str(Path(preparation_dir).resolve()), '--final'],
                   'final_audit', 'audit', plan['source_root'], output, status)
        status['completed'].append('final_audit')
        status.update(state='passed', stage=None)
    except BaseException as error:
        status.update(state='failed', error={'type': type(error).__name__, 'message': str(error)})
        atomic_json(output / 'status.json', status)
        raise
    atomic_json(output / 'status.json', status)


def run_record(command, stage, index, source, output, status):
    logs = output / 'logs' / stage; logs.mkdir(parents=True, exist_ok=True)
    name = f'{index:03d}' if isinstance(index, int) else str(index)
    record = dict(stage=stage, index=index, command=command, started=time.time(), ended=None, exit_code=None,
                  stdout_log=str(logs / f'{name}.stdout.log'), stderr_log=str(logs / f'{name}.stderr.log'))
    status['commands'].append(record); atomic_json(output / 'status.json', status)
    process = None
    try:
        with open(record['stdout_log'], 'w') as out, open(record['stderr_log'], 'w') as err:
            process = subprocess.Popen(command, cwd=source, env=child_env(source, 4), stdout=out, stderr=err)
            record['pid'] = process.pid; atomic_json(output / 'status.json', status)
            record['exit_code'] = process.wait()
        if record['exit_code']:
            raise subprocess.CalledProcessError(record['exit_code'], command)
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if record['exit_code'] is None:
            record['exit_code'] = 'launch_or_wait_error'
        record['error'] = str(error)
        raise
    finally:
        record['ended'] = time.time(); atomic_json(output / 'status.json', status)


def main(argv=None):
    if sys.argv[1:] == ['--metadata-worker']:
        payload = json.load(sys.stdin)
        with contextlib.redirect_stdout(io.StringIO()):
            result = worker(payload)
        print(json.dumps(result)); return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'data-root', 'output-root', 'input-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preparation-dir', type=Path)
    parser.add_argument('--clip-path', type=Path)
    parser.add_argument('--python', default=sys.executable)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('plan-only', 'prepare-only', 'execute', 'audit-only'):
        mode.add_argument('--' + name, action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args.source_root, args.data_root, args.output_root, args.input_receipt, args.python)
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    if not args.preparation_dir:
        parser.error('--preparation-dir is required outside plan mode')
    if args.prepare_only:
        prepare(plan, args.preparation_dir, args.clip_path)
    elif args.execute:
        execute(plan, args.preparation_dir)
    else:
        subprocess.run([args.python, str(HERE / 'attribution_audit.py'), '--study-root', str(args.output_root.resolve()),
                        '--preparation-dir', str(args.preparation_dir.resolve()), '--final'], check=True,
                       env=child_env(args.source_root.resolve(), 4))


if __name__ == '__main__':
    main()
