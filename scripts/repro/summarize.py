#!/usr/bin/env python3
"""Reject missing or mislabelled runs; aggregate matched checkpoint evaluations."""
import argparse
import csv
import json
import math
import copy
import sys
from pathlib import Path
from statistics import mean, stdev
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.repro.structure_spec import EXTRA_VARIANTS, STRUCTURE_VARIANTS, STRUCTURE_PARAMETER_COUNT, order_settings

VARIANTS = {
    'baseline': ('CoOp_TextConsistency_Ablation', 'none', False, None, 'baseline'),
    'mean': ('CoOp_TextConsistency_V2_FG', 'mean', False, 4, None),
    'flat': ('CoOp_TextConsistency_V2_FG', 'flat', False, 4, None),
    'flat_adapter': ('CoOp_TextConsistency_V2_FG', 'flat', True, 4, None),
    'hierarchical': ('CoOp_TextConsistency_V2_FG', 'hierarchical', False, 4, None),
    'hierarchical_adapter': ('CoOp_TextConsistency_V2_FG', 'hierarchical', True, 4, None),
}
VARIANTS.update({name: ('CoOp_TextConsistency_V2_FG', mode, adapter, 4, None)
                 for name, (mode, adapter) in EXTRA_VARIANTS.items()})


def _expected_loss_identity(mode, temperature, effective_weight):
    weighting = {
        'full_kl': 'none_full_distribution_kl',
        'weighted_nckd': 'teacher_probability_mass_outside_ground_truth',
    }
    if mode not in weighting:
        raise ValueError(f'Unknown protocol KD loss mode: {mode}')
    return {
        'format': 1,
        'mode': mode,
        'temperature': float(temperature),
        'effective_weight': float(effective_weight),
        'active': effective_weight > 0,
        'non_target_weighting': weighting[mode],
    }


def _validate_loss_identity(data, expected, path):
    identities = (data.get('loss_identity'), data.get('model_signature', {}).get('loss_identity'))
    if identities == (None, None) and expected['mode'] == 'full_kl':
        return
    if any(identity != expected for identity in identities):
        raise ValueError(f'KD loss identity differs from protocol: {path}')


def _read_config(path):
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f'Invalid effective configuration: {path}')
    return data


def _comparable_config(data):
    data = copy.deepcopy(data)
    for field in ('OUTPUT_DIR', 'MODEL_DIR', 'RESUME', 'EVAL_ONLY', 'LOAD_EPOCH'):
        data.pop(field, None)
    data.get('DATASET', {}).pop('SUBSAMPLE_CLASSES', None)
    data.get('TRAINER', {}).get('TEXTCONSISTENCY', {}).pop('INFERENCE_BRANCH', None)
    return data


def _validate_config(data, variant, kd, kd_mode, fixed, dataset, shots, epochs, path, order=None):
    trainer, aggregation, adapter, _, _ = VARIANTS[variant]
    tc = data.get('TRAINER', {}).get('TEXTCONSISTENCY', {})
    expected = {'N_CTX': fixed['n_ctx'], 'N_VIEWS': fixed['n_views'], 'N_GROUPS': fixed['n_groups'],
                'KD_WEIGHT': kd, 'KD_TEMP': fixed['kd_temp'],
                'ADAPTIVE_CE_WEIGHT': fixed['adaptive_ce_weight'],
                'CONTRASTIVE_WEIGHT': fixed['contrastive_weight'],
                'DIVERSITY_WEIGHT': fixed['diversity_weight'], 'ENSEMBLE_ALPHA': fixed['ensemble_alpha'],
                'AGGREGATION': 'mean' if variant == 'baseline' else aggregation,
                'USE_FEATURE_ADAPTER': adapter,
                'ABLATION_GROUP': 'baseline' if variant == 'baseline' else 'full'}
    if order is not None:
        expected.update(VIEW_ORDER=order['policy'], ORDER_SEED=order['seed'])
    if data.get('TRAINER', {}).get('NAME') != trainer:
        raise ValueError(f'Trainer differs from variant protocol: {path}')
    if data.get('DATASET', {}).get('NAME') != dataset or data.get('DATASET', {}).get('NUM_SHOTS') != shots:
        raise ValueError(f'Dataset configuration differs from protocol: {path}')
    if data.get('OPTIM', {}).get('MAX_EPOCH') != epochs:
        raise ValueError(f'Training epoch configuration differs from protocol: {path}')
    configured_mode = tc.get('KD_LOSS_MODE')
    if configured_mode is None:
        if kd_mode != 'full_kl':
            raise ValueError(f'Missing KD loss mode in effective configuration: {path}')
    elif configured_mode != kd_mode:
        raise ValueError(f'Effective configuration KD loss mode differs from protocol: {path}')
    for field, value in expected.items():
        if tc.get(field) != value:
            raise ValueError(f'Effective configuration differs from protocol: {field} in {path}')


def validate_order_evidence(data, order, groups, views, matched_capacity=False):
    from trainers.textconsistency_common import view_order_manifest
    record = data.get('view_order_manifest')
    counts = data.get('description_counts')
    # Baseline has no description encoder; all description-based variants have
    # explicit, class-specific records for the actual post-tokenization views.
    if data.get('model_signature', {}).get('aggregation') == 'none':
        if record is not None or counts != []:
            raise ValueError('Baseline unexpectedly reports description ordering')
        return
    if not isinstance(record, dict) or not isinstance(counts, list) or not counts:
        raise ValueError('Missing effective description order evidence')
    if any(not isinstance(n, int) or not 1 <= n <= views for n in counts):
        raise ValueError('Invalid effective description count')
    classes = record.get('classes', [])
    names = [row['classname'] for row in classes]
    expected = view_order_manifest(names, counts, order['policy'], groups, order['seed'])
    if record != expected:
        raise ValueError('Actual description order/hash differs from protocol')
    if matched_capacity and data.get('trainable_parameters') != STRUCTURE_PARAMETER_COUNT:
        raise ValueError('Structure control trainable parameter count is not matched')


def read_metric(path, seed, split, branch, epochs, dataset='UCF101', kd_weight=None,
                code_sha256=None, shots=16, fixed_settings=None, kd_loss_mode='full_kl'):
    data = json.loads(path.read_text())
    if (data['seed'], data['subsample'], data['branch'], data['dataset']) != (seed, split, branch, dataset):
        raise ValueError(f'Actual evaluation protocol differs from its directory label: {path}')
    if data.get('shots') != shots:
        raise ValueError(f'Expected the fixed 16-shot protocol: {path}')
    if data['state'] != 'loaded_checkpoint' or len(data['checkpoints']) != 1:
        raise ValueError(f'Evaluation did not load exactly one complete checkpoint: {path}')
    if data['checkpoints'][0]['epoch'] != epochs:
        raise ValueError(f'Wrong checkpoint epoch: {path}')
    if kd_weight is not None and data.get('effective_losses', {}).get('kd') != kd_weight:
        raise ValueError(f'Effective KD weight differs from protocol: {path}')
    if kd_weight is not None and fixed_settings is not None:
        expected_identity = _expected_loss_identity(kd_loss_mode, fixed_settings['kd_temp'], kd_weight)
        _validate_loss_identity(data, expected_identity, path)
    if code_sha256 is not None and data.get('code_sha256') != code_sha256:
        raise ValueError(f'Source hash differs from protocol: {path}')
    if fixed_settings:
        expected = {'adaptive_ce': fixed_settings['adaptive_ce_weight'],
                    'contrastive': fixed_settings['contrastive_weight'],
                    'diversity': fixed_settings['diversity_weight']}
        for name, value in expected.items():
            if data.get('effective_losses', {}).get(name) != value:
                raise ValueError(f'Effective {name} differs from protocol: {path}')
        if data.get('ensemble_alpha') != fixed_settings['ensemble_alpha']:
            raise ValueError(f'Ensemble coefficient differs from protocol: {path}')
    accuracy = float(data['metrics']['accuracy'])
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError(f'Invalid accuracy in {path}')
    return data


def collect(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    dataset = protocol.get('dataset', 'UCF101')
    shots = protocol.get('shots', 16)
    expected_code = protocol.get('code_sha256')
    fixed_settings = protocol.get('fixed_settings')
    kd_loss_mode = protocol.get('kd_loss_mode', 'full_kl')
    orders = protocol.get('view_orders')
    structure = protocol.get('structure_study', False)
    if structure and (protocol['variants'] != list(STRUCTURE_VARIANTS) or protocol['seeds'] != [1, 2, 3]
                      or protocol['epochs'] != 20 or dataset != 'UCF101' or protocol.get('kd_weight') != 1.0
                      or shots != 16 or not orders or not expected_code):
        raise ValueError('Structure study does not match the preregistered protocol')
    if orders is not None and orders != {v: order_settings(v) for v in protocol['variants']}:
        raise ValueError('Description order protocol differs from named variants')
    if any(v in EXTRA_VARIANTS for v in protocol['variants']) and orders is None:
        raise ValueError('New structure controls require explicit view order provenance')
    observed_code = None
    seed_samples = {}
    seed_checkpoints = {}
    control_reference = None
    description_reference = {}
    test_sample_reference = {}
    rows = []
    for variant in protocol['variants']:
        branches = ['student'] if variant == 'baseline' else ['student', 'teacher', 'ensemble']
        for branch in branches:
            bases, novels, hms, train_hashes = [], [], [], []
            for seed in protocol['seeds']:
                directory = root / variant / f'seed{seed}'
                kd = ((0.0 if variant == 'baseline' else protocol['kd_weight'])
                      if 'kd_weight' in protocol else None)
                base = read_metric(directory / f'eval_base_{branch}/metrics.json', seed, 'base', branch,
                                   protocol['epochs'], dataset, kd, expected_code, shots, fixed_settings,
                                   kd_loss_mode)
                novel = read_metric(directory / f'eval_new_{branch}/metrics.json', seed, 'new', branch,
                                    protocol['epochs'], dataset, kd, expected_code, shots, fixed_settings,
                                    kd_loss_mode)
                if base['checkpoints'] != novel['checkpoints']:
                    raise ValueError('Base and novel results came from different checkpoints')
                checkpoint_key = (variant, seed)
                if checkpoint_key in seed_checkpoints and seed_checkpoints[checkpoint_key] != base['checkpoints']:
                    raise ValueError('Evaluation branches came from different checkpoints')
                seed_checkpoints[checkpoint_key] = base['checkpoints']
                if base['model_signature'] != novel['model_signature'] or base['code_sha256'] != novel['code_sha256']:
                    raise ValueError('Base/novel model configuration or source mismatch')
                if observed_code is None:
                    observed_code = base['code_sha256']
                elif base['code_sha256'] != observed_code:
                    raise ValueError('Runs from different source hashes were mixed')
                manifest_path = directory / 'run_manifest.json'
                if 'kd_weight' in protocol:
                    manifest = json.loads(manifest_path.read_text())
                    identity = (manifest.get('dataset'), manifest.get('seed'), manifest.get('shots'),
                                manifest.get('code_sha256'), manifest.get('train_samples_sha256'))
                    if identity != (dataset, seed, shots, base['code_sha256'], base['train_samples_sha256']):
                        raise ValueError(f'Training manifest identity mismatch: {manifest_path}')
                    if manifest.get('effective_losses', {}).get('kd') != kd:
                        raise ValueError(f'Training manifest KD mismatch: {manifest_path}')
                    _validate_loss_identity(
                        manifest, _expected_loss_identity(kd_loss_mode, fixed_settings['kd_temp'], kd),
                        manifest_path)
                    if manifest.get('model_signature') != base.get('model_signature'):
                        raise ValueError(f'Training/evaluation model signature mismatch: {manifest_path}')
                    signature = manifest['model_signature']
                    trainer, aggregation, adapter, n_groups, ablation = VARIANTS[variant]
                    expected_signature = {'trainer': trainer, 'aggregation': aggregation,
                                          'feature_adapter': adapter, 'n_groups': n_groups,
                                          'ablation_group': ablation, 'n_ctx': fixed_settings['n_ctx'],
                                          'n_views': fixed_settings['n_views']}
                    if orders is not None:
                        expected_signature.update(view_order=orders[variant]['policy'],
                                                  order_seed=orders[variant]['seed'], order_version=1)
                    for field, value in expected_signature.items():
                        if signature.get(field) != value:
                            raise ValueError(f'Model signature differs from variant: {field}')
                    if orders is not None:
                        for item in (manifest, base, novel):
                            validate_order_evidence(item, orders[variant], fixed_settings['n_groups'],
                                                    fixed_settings['n_views'], structure)
                        if manifest.get('view_order_manifest') != base.get('view_order_manifest'):
                            raise ValueError('Training/Base description order differs')
                        if any(item.get('trainable_parameters') != manifest.get('trainable_parameters')
                               for item in (base, novel)):
                            raise ValueError('Training/evaluation parameter counts differ')
                    sample = manifest['train_samples_sha256']
                    if seed in seed_samples and seed_samples[seed] != sample:
                        raise ValueError('The same seed used different training samples across variants')
                    seed_samples[seed] = sample
                    train_config = _read_config(directory / 'effective_config.yaml')
                    _validate_config(train_config, variant, kd, kd_loss_mode, fixed_settings, dataset, shots,
                                     protocol['epochs'], directory / 'effective_config.yaml',
                                     orders[variant] if orders is not None else None)
                    if structure:
                        comparison = _comparable_config(train_config)
                        comparison.pop('SEED', None)
                        for field in ('AGGREGATION', 'VIEW_ORDER', 'ORDER_SEED'):
                            comparison['TRAINER']['TEXTCONSISTENCY'].pop(field, None)
                        signature_common = {k: v for k, v in signature.items()
                                            if k not in ('aggregation', 'view_order', 'order_seed')}
                        invariant = (comparison, signature_common)
                        if control_reference is None:
                            control_reference = invariant
                        elif control_reference != invariant:
                            raise ValueError('Structure controls changed settings outside aggregation/order')
                        from trainers.textconsistency_common import sha256_file
                        checkpoint = directory / 'textconsistency' / f'model.pth.tar-{protocol["epochs"]}'
                        if sha256_file(checkpoint) != base['checkpoints'][0]['sha256']:
                            raise ValueError('Actual checkpoint bytes differ from evaluation hash')
                        for metric, label in ((base, 'base'), (novel, 'new')):
                            test_hash = metric.get('test_samples_sha256')
                            if (metric.get('split') != 'test' or not isinstance(test_hash, str)
                                    or len(test_hash) != 64 or any(c not in '0123456789abcdef' for c in test_hash)):
                                raise ValueError('Structure evaluation requires test split and complete test sample hash')
                            if label == 'base' and manifest.get('test_samples_sha256') != test_hash:
                                raise ValueError('Training/Base test sample hashes differ')
                            if label in test_sample_reference and test_sample_reference[label] != test_hash:
                                raise ValueError('Test images differ across structure seeds/variants/branches')
                            test_sample_reference[label] = test_hash
                            membership = [(row['classname'], row['count'])
                                          for row in metric['view_order_manifest']['classes']]
                            if len(membership) != (51 if label == 'base' else 50):
                                raise ValueError('Structure study has incorrect UCF101 class coverage')
                            if label in description_reference and description_reference[label] != membership:
                                raise ValueError('Description membership differs across structure controls')
                            description_reference[label] = membership
                    for metric, split_name, branch_name in ((base, 'base', branch), (novel, 'new', branch)):
                        eval_path = directory / f'eval_{split_name}_{branch_name}/effective_config.yaml'
                        eval_config = _read_config(eval_path)
                        _validate_config(eval_config, variant, kd, kd_loss_mode, fixed_settings, dataset, shots,
                                         protocol['epochs'], eval_path, orders[variant] if orders is not None else None)
                        if _comparable_config(eval_config) != _comparable_config(train_config):
                            raise ValueError(f'Train/evaluation non-runtime settings differ: {eval_path}')
                b, n = float(base['metrics']['accuracy']), float(novel['metrics']['accuracy'])
                bases.append(b); novels.append(n); hms.append(2*b*n/(b+n) if b+n else 0.)
                train_hashes.append(base['train_samples_sha256'])
            if len(set(train_hashes)) != len(train_hashes):
                raise ValueError('Different seed labels reused the same few-shot training split')
            b, n = mean(bases), mean(novels)
            rows.append({'variant': variant, 'branch': branch, 'seeds': len(bases),
                         'base_mean': b, 'novel_mean': n, 'hm_of_means': 2*b*n/(b+n) if b+n else 0.,
                         'base_std': stdev(bases) if len(bases)>1 else None,
                         'novel_std': stdev(novels) if len(novels)>1 else None,
                         'mean_seed_hm': mean(hms), 'seed_hm_std': stdev(hms) if len(hms)>1 else None})
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); rows = collect(args.root)
    with (args.root / 'results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.root / 'results.json').write_text(json.dumps(rows, indent=2))
    print(f'Validated {len(rows)} variant/branch rows. Single-seed rows are pilot results only.')

if __name__ == '__main__': main()
