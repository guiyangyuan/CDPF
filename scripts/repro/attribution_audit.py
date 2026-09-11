#!/usr/bin/env python3
"""Independent fail-closed audit of the fixed attribution experiment."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attribution_study as study
import cross_dataset_audit as common

NAMES = {'ucf101': 'UCF101', 'dtd': 'DescribableTextures', 'fgvc_aircraft': 'FGVCAircraft'}
read_json = study.read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def loader(evidence, seed, split):
    rows = [r for r in evidence['loader_evidence'] if r['seed'] == seed]
    require(len(rows) == 1, 'Missing or duplicate loader seed')
    return rows[0]['splits'][split]


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def relocated(config):
    result = copy.deepcopy(config)
    result['OUTPUT_DIR'] = '<output>'
    result['DATASET']['ROOT'] = '<data>'
    return result


def compare_kd_configs(kd0, kd1):
    require(kd0['SEED'] == kd1['SEED'], 'KD pair seed differs')
    left, right = copy.deepcopy(kd0), copy.deepcopy(kd1)
    require(left['TRAINER']['TEXTCONSISTENCY']['KD_WEIGHT'] == 0.0 and
            right['TRAINER']['TEXTCONSISTENCY']['KD_WEIGHT'] == 1.0, 'KD direction must be KD1 minus KD0')
    for cfg in (left, right):
        cfg.pop('OUTPUT_DIR', None)
        cfg['TRAINER']['TEXTCONSISTENCY'].pop('KD_WEIGHT')
    require(left == right, 'KD training configs differ outside KD_WEIGHT and output')


def accuracy_counts(accuracy, correct, total, expected_total):
    require(type(correct) is int and type(total) is int and total > 0 and 0 <= correct <= total,
            'Invalid correct/total')
    require(total == expected_total, 'Correct/total differs from real loader count')
    require(isinstance(accuracy, (int, float)) and math.isfinite(accuracy) and
            math.isclose(accuracy, 100.0 * correct / total, rel_tol=0, abs_tol=5e-4),
            'Accuracy differs from correct/total')


def full_signature(template_sha):
    return {'format': 2, 'trainer': 'CoOp_TextConsistency_V2_FG', 'n_ctx': 4,
            'aggregation': 'hierarchical', 'feature_adapter': True, 'n_groups': 4,
            'n_views': 20, 'ablation_group': None, 'clip_sha256': study.CLIP_SHA256,
            'template_sha256': template_sha, 'view_order': 'contiguous', 'order_seed': 0, 'order_version': 1}


def validate_full_manifest(manifest, dataset, seed, split, kd, evidence):
    expected = loader(evidence, seed, split)
    fields = dict(dataset=NAMES[dataset], seed=seed, shots=16, subsample=split,
                  trainable_parameters=790018, code_sha256=study.MODEL_SHA256,
                  model_signature=full_signature(evidence['template']['sha256']),
                  effective_losses={'kd': float(kd), 'adaptive_ce': 1.0, 'diversity': 0.0, 'contrastive': 0.0},
                  description_counts=expected['description_counts'], view_order_manifest=expected['view_order_manifest'])
    for part in ('train', 'val', 'test'):
        fields[f'{part}_samples_sha256'] = expected[f'{part}_sha256']
    for key, value in fields.items():
        require(manifest.get(key) == value, f'Full model provenance differs: {key}')


def inspect_eval(directory, cp, manifest, dataset, seed, split, branch, kd, epoch, evidence,
                 expected_config=None, stdout=None, relocated_checkpoint=False):
    directory = Path(directory)
    metric = read_json(directory / 'metrics.json')
    evaluation_manifest = read_json(directory / 'run_manifest.json')
    validate_full_manifest(metric, dataset, seed, split, kd, evidence)
    validate_full_manifest(evaluation_manifest, dataset, seed, split, kd, evidence)
    require(metric.get('state') == 'loaded_checkpoint' and metric.get('split') == 'test' and
            metric.get('branch') == branch and metric.get('ensemble_alpha') == 0.5,
            'Metric is not fixed loaded-checkpoint branch evaluation')
    cfg = read_yaml(directory / 'effective_config.yaml')
    if expected_config is not None:
        common.validate_effective_config(cfg, expected_config, seed, directory)
    checked_metric = copy.deepcopy(metric)
    if relocated_checkpoint:
        records = checked_metric.get('checkpoints', [])
        require(len(records) == 1, 'Historical metric checkpoint record missing')
        records[0]['path'] = str(cp.resolve())
    common.checkpoint_audit(cp, checked_metric, manifest, 'hierarchical_adapter', epoch)
    import torch
    saved = torch.load(cp, map_location='cpu', weights_only=False).get('run_provenance', {})
    validate_full_manifest(saved, dataset, seed, 'base', kd, evidence)
    correct, total = common.parse_log(directory / 'log.txt')
    accuracy = metric['metrics']['accuracy']
    accuracy_counts(accuracy, correct, total, loader(evidence, seed, split)['test_count'])
    if stdout is not None:
        require(common.parse_log(Path(stdout)) == (correct, total), 'Raw stdout differs from evaluation log')
    return dict(accuracy=accuracy, correct=correct, total=total,
                test_sha256=metric['test_samples_sha256'], checkpoint_sha256=study.sha256(cp),
                metrics_sha256=study.sha256(directory / 'metrics.json'), config=cfg)


def inspect_historical(source, plan, row, evidence):
    dataset, seed, directory = row['dataset'], row['seed'], Path(row['model_dir'])
    manifest = read_json(directory / 'run_manifest.json')
    validate_full_manifest(manifest, dataset, seed, 'base', 1, evidence)
    cfg = read_yaml(directory / 'effective_config.yaml')
    commands = study.full_commands(source, plan['data_root'], directory.parent.parent, [seed], 20, 1, plan['python'], dataset)
    expected_train = study.common.effective_config_for_command(source, commands[0])
    require(cfg.get('SEED') == seed and relocated(cfg) == relocated(expected_train),
            'Historical full training config differs from actual frozen source config')
    cp = directory / 'textconsistency/model.pth.tar-20'
    require(study.sha256(cp) == row['checkpoint_sha256'], 'Historical checkpoint differs from receipt')
    scores = {}
    for branch in study.BRANCHES:
        scores[branch] = {}
        for split in ('base', 'new'):
            evaluation = directory / f'eval_{split}_{branch}'
            expected_command = next(c for c in commands if Path(study.value(c, '--output-dir')).name == evaluation.name)
            expected_eval = study.common.effective_config_for_command(source, expected_command)
            actual_eval = read_yaml(evaluation / 'effective_config.yaml')
            require(relocated(actual_eval) == relocated(expected_eval), 'Historical full eval config differs from source')
            scores[branch][split] = inspect_eval(evaluation, cp, manifest, dataset, seed, split, branch, 1, 20,
                                                evidence, relocated_checkpoint=True)
    return dict(dataset=dataset, seed=seed, model_dir=str(directory), checkpoint_sha256=row['checkpoint_sha256'],
                train_config=cfg, scores=scores)


def historical(prepared, dataset, seed):
    rows = [r for r in prepared['historical'] if r['dataset'] == dataset and r['seed'] == seed]
    require(len(rows) == 1, 'Historical reference is missing or duplicated')
    return rows[0]


def paired_row(dataset, seed, branch, scores, **extra):
    base, novel = scores['base']['accuracy'], scores['new']['accuracy']
    return dict(dataset=dataset, seed=seed, branch=branch, base=base, novel=novel,
                hm=harmonic(base, novel), counts={s: {'correct': scores[s]['correct'], 'total': scores[s]['total']}
                                               for s in ('base', 'new')}, **extra)


def harmonic(a, b):
    return 2 * a * b / (a + b) if a + b else 0.0


def validate_anchor(actual, reference):
    for field in ('correct', 'total', 'test_sha256'):
        require(actual[field] == reference[field], f'Anchor/historical exact agreement failed: {field}')


def audit_full(stage, prepared, root):
    evidence = prepared['datasets'][stage['dataset']]
    configs = prepared['expected_effective_configs'][stage['name']]
    rows, train_configs = [], {}
    for seed in stage['seeds']:
        directory = Path(stage['root']) / 'hierarchical_adapter' / f'seed{seed}'
        model_dir = Path(study.row_for(prepared['protocol']['receipt'], 'historical_kd1', stage['dataset'], seed)['model_dir']) if stage['kind'] == 'historical' else directory
        manifest = read_json(model_dir / 'run_manifest.json')
        validate_full_manifest(manifest, stage['dataset'], seed, 'base', stage['kd_weight'], evidence)
        cfg = read_yaml(model_dir / 'effective_config.yaml')
        train_configs[seed] = cfg
        if stage['kind'] != 'historical':
            index = next(i for i, c in enumerate(stage['commands']) if '--eval-only' not in c and int(study.value(c, '--seed')) == seed)
            common.validate_effective_config(cfg, configs[index], seed, model_dir)
            # Anchor every new formal config against independently source-checked historical YAML.
            old = copy.deepcopy(historical(prepared, stage['dataset'], seed)['train_config'])
            old['OUTPUT_DIR'] = cfg['OUTPUT_DIR']; old['DATASET']['ROOT'] = cfg['DATASET']['ROOT']
            old['OPTIM']['MAX_EPOCH'] = stage['epochs']
            old['TRAINER']['TEXTCONSISTENCY']['KD_WEIGHT'] = stage['kd_weight']
            require(cfg == old, 'New config differs from historical anchor outside registered changes')
        cp = model_dir / 'textconsistency' / f"model.pth.tar-{stage['epochs']}"
        for branch in study.BRANCHES:
            scores = {}
            for split in ('base', 'new'):
                evaluation = directory / f'eval_{split}_{branch}'
                index = next(i for i, c in enumerate(stage['commands']) if study.value(c, '--output-dir') == str(evaluation))
                scores[split] = inspect_eval(evaluation, cp, manifest, stage['dataset'], seed, split, branch,
                    stage['kd_weight'], stage['epochs'], evidence, configs[index], root / 'logs' / stage['name'] / f'{index:03d}.stdout.log')
                if stage['kind'] in ('anchor', 'historical'):
                    old = historical(prepared, stage['dataset'], seed)['scores'][branch][split]
                    validate_anchor(scores[split], old)
                    if stage['kind'] == 'historical':
                        require(scores[split]['checkpoint_sha256'] == old['checkpoint_sha256'], 'Historical checkpoint changed')
            rows.append(paired_row(stage['dataset'], seed, branch, scores, kd_weight=stage['kd_weight']))
    return {'rows': rows, 'train_configs': train_configs}


def validate_predictions(metric, directory, expected):
    path = Path(directory) / 'predictions.jsonl'
    require(metric['predictions']['path'] == str(path) and study.sha256(path) == metric['predictions']['sha256'],
            'Prediction file path/SHA differs')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    require(len(records) == expected['test_count'] == metric['predictions']['count'], 'Prediction count differs')
    require(len({r['relative_path'] for r in records}) == len(records), 'Duplicate prediction samples')
    classes = expected['classes']
    sample_records = []
    counts = {branch: 0 for branch in study.FUSION_BRANCHES}
    for record in records:
        label = record['label']; path = Path(record['relative_path'])
        require(not path.is_absolute() and '..' not in path.parts, 'Prediction sample path is not data-relative')
        require(type(label) is int and 0 <= label < len(classes) and record['label_classname'] == classes[label],
                'Prediction target class differs')
        sample_records.append((str(path), label, classes[label]))
        require(set(record['predictions']) == set(study.FUSION_BRANCHES), 'Missing or extra prediction branch')
        for branch, prediction in record['predictions'].items():
            predicted = prediction['label']
            require(type(predicted) is int and 0 <= predicted < len(classes) and prediction['classname'] == classes[predicted],
                    'Prediction branch class differs')
            counts[branch] += int(predicted == label)
    digest = hashlib.sha256(json.dumps(sample_records, ensure_ascii=False).encode()).hexdigest()
    require(digest == expected['test_sha256'], 'Prediction samples/order differ from real loader hash')
    require(set(metric['metrics']) == set(study.FUSION_BRANCHES), 'Missing or extra fusion metric branch')
    for branch, correct in counts.items():
        stats = metric['metrics'][branch]
        require(stats['correct'] == correct, 'Prediction correct count differs from metric')
        accuracy_counts(stats['accuracy'], stats['correct'], stats['total'], len(records))


def audit_fusion(stage, prepared):
    evidence = prepared['datasets_by_source'][stage['source_root']][stage['dataset']]
    rows = []
    for seed in stage['seeds']:
        splits = {}
        receipt = study.row_for(prepared['protocol']['receipt'], 'fusion_baselines', stage['dataset'], seed)
        for split in ('base', 'new'):
            directory = Path(stage['root']) / f'seed{seed}' / split
            metric = read_json(directory / 'metrics.json'); splits[split] = metric
            expected = loader(evidence, seed, split)
            require(metric.get('status') == 'passed' and metric.get('dataset_key') == stage['dataset'] and
                    metric.get('dataset') == NAMES[stage['dataset']] and metric.get('seed') == seed and
                    metric.get('subsample') == split and metric.get('split') == 'test', 'Fusion identity/status differs')
            index = next(i for i, c in enumerate(stage['commands']) if study.value(c, '--output-dir') == str(directory))
            common.validate_effective_config(metric['effective_config'], prepared['expected_effective_configs'][stage['name']][index], seed, directory)
            require(metric['class_order'] == expected['classes'] and metric['description']['counts'] == expected['description_counts'] and
                    metric['description']['view_order_manifest'] == expected['view_order_manifest'], 'Fusion class/description order differs')
            for part in ('train', 'val', 'test'):
                require(metric['samples'][f'{part}_sha256'] == expected[f'{part}_sha256'], 'Fusion sample hash differs')
            require(metric['assets'] == {'clip_sha256': study.CLIP_SHA256, 'template_sha256': evidence['template']['sha256']}, 'Fusion asset SHA differs')
            cp = Path(receipt['baseline_dir']) / 'textconsistency/model.pth.tar-20'
            require(metric['checkpoint'] == {'path': str(cp), 'sha256': receipt['checkpoint_sha256'],
                    'sidecar': str(cp) + '.sha256', 'epoch': 20}, 'Fusion checkpoint differs')
            require(metric['source'] == {'root': stage['source_root'], 'model_sha256': receipt['source_sha256'],
                    'expected_model_sha256': receipt['source_sha256'],
                    'evaluator_sha256': study.sha256(HERE / 'evaluate_frozen_fusion.py')}, 'Fusion source differs')
            historical_dir = Path(receipt['baseline_dir']) / f'eval_{split}_student'
            hashes = {'checkpoint': study.sha256(cp), 'sidecar': study.sha256(str(cp) + '.sha256'),
                      'historical_metrics': study.sha256(historical_dir / 'metrics.json'),
                      'historical_log': study.sha256(historical_dir / 'log.txt')}
            require(metric['input_hashes_before'] == metric['input_hashes_after'] == hashes, 'Fusion immutable inputs differ')
            protocol = metric['fixed_protocol']; scale = protocol.get('shared_logit_scale')
            require(protocol.get('epoch') == 20 and protocol.get('alpha') == .5 and isinstance(scale, (float, int)) and
                    math.isfinite(scale) and scale > 0 and math.isclose(protocol['attention_tau'], 1 / scale), 'Fusion fixed protocol differs')
            validate_predictions(metric, directory, expected)
            old_correct, old_total = common.parse_log(historical_dir / 'log.txt')
            accuracy_counts(receipt['historical'][split]['accuracy'], old_correct, old_total, expected['test_count'])
            require(metric['metrics']['student'] == metric['historical_student'] ==
                    {'correct': old_correct, 'total': old_total, 'accuracy': receipt['historical'][split]['accuracy']},
                    'Fusion student differs from historical baseline')
            fusion = study.common.load_file(HERE / 'evaluate_frozen_fusion.py', 'audit_fusion_tensors')
            fusion._validate_checkpoint_tensors(cp, read_json(Path(receipt['baseline_dir']) / 'run_manifest.json'))
        for branch in study.FUSION_BRANCHES:
            rows.append(paired_row(stage['dataset'], seed, branch,
                                  {split: splits[split]['metrics'][branch] for split in ('base', 'new')}))
    return {'rows': rows}


def audit_stage(stage, prepared, root, status):
    require(read_json(Path(stage['root']) / 'protocol.json') == stage, 'Stage protocol differs')
    require(read_json(Path(stage['root']) / 'data_preflight.json') ==
            prepared['datasets_by_source'][stage['source_root']][stage['dataset']], 'Stage data preflight differs')
    records = [r for r in status['commands'] if r['stage'] == stage['name'] and isinstance(r['index'], int)]
    require(len(records) == len(stage['commands']) and [r['index'] for r in records] == list(range(len(records))),
            'Raw command records incomplete or duplicated')
    for record, command in zip(records, stage['commands']):
        require(record['command'] == command and record.get('exit_code') == 0 and
                record.get('ended') is not None and record['ended'] >= record['started'], 'Raw command failed or unfinished')
        require(Path(record['stdout_log']).is_file() and Path(record['stderr_log']).is_file(), 'Raw logs missing')
    return audit_fusion(stage, prepared) if stage['kind'] == 'fusion' else audit_full(stage, prepared, root)


def indexed_rows(rows, keys, expected):
    indexed = {tuple(r[k] for k in keys): r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == expected, 'Incomplete or duplicate result identities')
    for row in rows:
        require(all(math.isfinite(row[m]) and 0 <= row[m] <= 100 for m in ('base', 'novel', 'hm')), 'Invalid result score')
        require(math.isclose(row['hm'], harmonic(row['base'], row['novel']), abs_tol=1e-10), 'Seed HM differs')
    return indexed


def summarize(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, values in groups.items():
        means = {m: mean(r[m] for r in values) for m in ('base', 'novel', 'hm')}
        result.append(dict(zip(keys, key)) | {'mean': means,
            'sample_sd': {m: stdev(r[m] for r in values) for m in ('base', 'novel', 'hm')},
            'hm_of_means': harmonic(means['base'], means['novel']), 'mean_seed_hm': means['hm']})
    return result


def final_analysis(full, fusion, ucf_full):
    full_index = indexed_rows(full, ('dataset', 'kd_weight', 'seed', 'branch'),
        {(d, k, s, b) for d in study.DATASETS for k in (0, 1) for s in study.SEEDS for b in study.BRANCHES})
    fusion_index = indexed_rows(fusion, ('dataset', 'seed', 'branch'),
        {(d, s, b) for d in study.ALL_DATASETS for s in study.SEEDS for b in study.FUSION_BRANCHES})
    ucf_index = indexed_rows(ucf_full, ('dataset', 'seed', 'branch'), {('ucf101', s, 'ensemble') for s in study.SEEDS})
    comparisons = []
    def difference(dataset, label, left, right, branch=None):
        rows = [{'seed': s, **{m: left(s)[m] - right(s)[m] for m in ('base', 'novel', 'hm')}} for s in study.SEEDS]
        comparisons.append({'dataset': dataset, 'comparison': label, 'branch': branch, 'per_seed': rows,
            'mean': {m: mean(r[m] for r in rows) for m in ('base', 'novel', 'hm')},
            'sample_sd': {m: stdev(r[m] for r in rows) for m in ('base', 'novel', 'hm')}})
    for dataset in study.DATASETS:
        for branch in study.BRANCHES:
            difference(dataset, 'KD1-KD0', lambda s: full_index[dataset, 1, s, branch], lambda s: full_index[dataset, 0, s, branch], branch)
    for dataset in study.ALL_DATASETS:
        for left, right in (('simple_fusion', 'student'), ('mean_probability_fusion', 'simple_fusion'),
                            ('adk_probability_fusion', 'mean_probability_fusion')):
            label = f'{left}-baseline' if right == 'student' else f'{left}-{right}'
            difference(dataset, label, lambda s: fusion_index[dataset, s, left], lambda s: fusion_index[dataset, s, right])
        for branch in ('simple_fusion', 'adk_probability_fusion'):
            difference(dataset, f'full_ensemble-{branch}',
                lambda s: ucf_index['ucf101', s, 'ensemble'] if dataset == 'ucf101' else full_index[dataset, 1, s, 'ensemble'],
                lambda s: fusion_index[dataset, s, branch])
    return {'passed': True, 'full_seed_branch_rows': full, 'fusion_seed_branch_rows': fusion,
            'ucf_historical_ensemble_rows': ucf_full, 'full_summaries': summarize(full, ('dataset', 'kd_weight', 'branch')),
            'fusion_summaries': summarize(fusion, ('dataset', 'branch')), 'comparisons': comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', required=True, type=Path)
    parser.add_argument('--preparation-dir', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage'); mode.add_argument('--final', action='store_true')
    args = parser.parse_args(argv)
    root = args.study_root.resolve()
    prepared = read_json(args.preparation_dir / 'preparation.json'); plan = prepared['protocol']
    require(str(root) == plan['output_root'] and read_json(root / 'protocol.json') == plan, 'Study protocol differs')
    require(read_json(root / 'data_preflight.json') == prepared['datasets_by_source'], 'Study preflight missing or changed')
    study.verify_frozen(prepared, check_data=True)
    status = read_json(root / 'status.json')
    require(status.get('state') in ('running', 'passed'), 'Study failed or invalid state')
    if args.stage:
        stages = [s for s in plan['stages'] if s['name'] == args.stage]
        require(len(stages) == 1, 'Unknown stage')
        index = plan['stages'].index(stages[0])
        require(status['completed'][:index] == [s['name'] for s in plan['stages'][:index]], 'Previous stages incomplete')
        result = audit_stage(stages[0], prepared, root, status)
        destination = root / 'audit' / f'{args.stage}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        study.atomic_json(destination, {'passed': True, **result}); return
    current_history = study.isolated(plan['source_root'], dict(action='historical', source=plan['source_root'],
        plan=plan, rows=plan['receipt']['historical_kd1'] + plan['receipt']['ucf101_full'],
        datasets=prepared['datasets']), plan['python'])
    require(current_history == prepared['historical'], 'Historical reference differs from independent audit')
    current_configs = {}
    for source in prepared['datasets_by_source']:
        current_configs.update(study.isolated(source, dict(action='configs', source=source, plan=plan,
            stages=[s for s in plan['stages'] if s['source_root'] == source]), plan['python']))
    require(current_configs == prepared['expected_effective_configs'], 'Prepared cfg differs from actual frozen setup_cfg')
    expected_stages = [s['name'] for s in plan['stages']]
    expected_records = {(s['name'], i) for s in plan['stages'] for i in range(len(s['commands']))}
    expected_records |= {(s['name'], 'audit') for s in plan['stages']}
    observed = [(r['stage'], r['index']) for r in status['commands'] if r['stage'] != 'final_audit']
    require(len(observed) == len(expected_records) and set(observed) == expected_records, 'Unexpected or duplicate execution records')
    require(sum(isinstance(r['index'], int) for r in status['commands']) == 124, 'Expected exactly 124 raw commands')
    final_records = [r for r in status['commands'] if r['stage'] == 'final_audit']
    require(len(final_records) <= 1, 'Duplicate final audit records')
    if final_records:
        require(final_records[0]['exit_code'] in (None, 0), 'Prior final audit failed')
    require(status['completed'] in (expected_stages, expected_stages + ['final_audit']), 'Stages incomplete or duplicated')
    results = {}
    for stage in plan['stages']:
        audit_records = [r for r in status['commands'] if r['stage'] == stage['name'] and r['index'] == 'audit']
        require(len(audit_records) == 1 and audit_records[0]['exit_code'] == 0 and audit_records[0]['ended'] is not None,
                'Stage audit failed or missing')
        result = audit_stage(stage, prepared, root, status); results[stage['name']] = result
        require(read_json(root / 'audit' / f"{stage['name']}.json") == {'passed': True, **json.loads(json.dumps(result))},
                'Stage audit report missing or changed')
    full, fusion, anchors = [], [], []
    for stage in plan['stages']:
        rows = results[stage['name']]['rows']
        if stage['kind'] in ('formal', 'historical'):
            full += rows
        elif stage['kind'] == 'fusion':
            fusion += rows
        elif stage['kind'] == 'anchor':
            anchors += rows
    for dataset in study.DATASETS:
        for seed in study.SEEDS:
            kd0 = results[f'formal_{dataset}_kd0']['train_configs'][seed]
            kd1 = copy.deepcopy(results[f'historical_{dataset}_kd1']['train_configs'][seed])
            kd1['DATASET']['ROOT'] = kd0['DATASET']['ROOT']
            compare_kd_configs(kd0, kd1)
    ucf = [paired_row('ucf101', seed, 'ensemble', historical(prepared, 'ucf101', seed)['scores']['ensemble']) for seed in study.SEEDS]
    result = final_analysis(full, fusion, ucf)
    result['anchor_rows_excluded_from_formal_pairs'] = anchors
    study.verify_frozen(prepared, check_data=True)
    study.atomic_json(root / 'analysis.json', result)


if __name__ == '__main__':
    main()
