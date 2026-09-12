import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    path = ROOT / 'scripts/repro' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def loss_identity(mode, temperature, effective_weight):
    return {
        'format': 1,
        'mode': mode,
        'temperature': temperature,
        'effective_weight': effective_weight,
        'active': effective_weight > 0,
        'non_target_weighting': ('teacher_probability_mass_outside_ground_truth'
                                 if mode == 'weighted_nckd' else 'none_full_distribution_kl'),
    }


def write_new_run(root, variants=('baseline',), sample_by_variant=None, kd_loss_mode=None):
    fixed = {'n_ctx': 4, 'n_views': 20, 'n_groups': 4, 'kd_temp': 2.0,
             'adaptive_ce_weight': 1.0, 'contrastive_weight': 0.0,
             'diversity_weight': 0.0, 'ensemble_alpha': 0.5}
    protocol = {'dataset': 'UCF101', 'seeds': [1], 'epochs': 20, 'variants': list(variants),
                'shots': 16, 'kd_weight': 1.0, 'code_sha256': 'code', 'fixed_settings': fixed}
    if kd_loss_mode is not None:
        protocol['kd_loss_mode'] = kd_loss_mode
    (root / 'protocol.json').write_text(json.dumps(protocol))
    for variant in variants:
        aggregation = {'baseline': 'none', 'mean': 'mean', 'flat': 'flat',
                       'flat_adapter': 'flat', 'hierarchical': 'hierarchical',
                       'hierarchical_adapter': 'hierarchical'}[variant]
        adapter = variant.endswith('_adapter')
        trainer = 'CoOp_TextConsistency_Ablation' if variant == 'baseline' else 'CoOp_TextConsistency_V2_FG'
        signature = {'trainer': trainer, 'n_ctx': 4, 'n_views': 20,
                     'n_groups': None if variant == 'baseline' else 4,
                     'aggregation': aggregation, 'feature_adapter': adapter,
                     'ablation_group': 'baseline' if variant == 'baseline' else None}
        kd = 0.0 if variant == 'baseline' else 1.0
        sample = (sample_by_variant or {}).get(variant, 'samples1')
        directory = root / variant / 'seed1'; directory.mkdir(parents=True)
        manifest = {'dataset': 'UCF101', 'seed': 1, 'shots': 16, 'code_sha256': 'code',
                    'train_samples_sha256': sample, 'effective_losses': {'kd': kd},
                    'model_signature': signature}
        if kd_loss_mode is not None:
            manifest['loss_identity'] = loss_identity(kd_loss_mode, 2.0, kd)
            signature['loss_identity'] = manifest['loss_identity']
        (directory / 'run_manifest.json').write_text(json.dumps(manifest))
        config = {'DATASET': {'NAME': 'UCF101', 'NUM_SHOTS': 16, 'SUBSAMPLE_CLASSES': 'base'},
                  'OPTIM': {'MAX_EPOCH': 20},
                  'TRAINER': {'NAME': trainer, 'TEXTCONSISTENCY': {
                      'N_CTX': 4, 'N_VIEWS': 20, 'N_GROUPS': 4, 'KD_WEIGHT': kd, 'KD_TEMP': 2.0,
                      'ADAPTIVE_CE_WEIGHT': 1.0, 'CONTRASTIVE_WEIGHT': 0.0, 'DIVERSITY_WEIGHT': 0.0,
                      'ENSEMBLE_ALPHA': 0.5, 'AGGREGATION': 'mean' if variant == 'baseline' else aggregation,
                      'USE_FEATURE_ADAPTER': adapter, 'ABLATION_GROUP': 'baseline' if variant == 'baseline' else 'full',
                      'INFERENCE_BRANCH': 'student' if variant == 'baseline' else 'ensemble'}}}
        if kd_loss_mode is not None:
            config['TRAINER']['TEXTCONSISTENCY']['KD_LOSS_MODE'] = kd_loss_mode
        import yaml
        (directory / 'effective_config.yaml').write_text(yaml.safe_dump(config))
        branches = ('student',) if variant == 'baseline' else ('student', 'teacher', 'ensemble')
        for split in ('base', 'new'):
            for branch in branches:
                target = directory / f'eval_{split}_{branch}'; target.mkdir()
                eval_config = json.loads(json.dumps(config)); eval_config['DATASET']['SUBSAMPLE_CLASSES'] = split
                eval_config['TRAINER']['TEXTCONSISTENCY']['INFERENCE_BRANCH'] = branch
                (target / 'effective_config.yaml').write_text(yaml.safe_dump(eval_config))
                payload = {**manifest, 'subsample': split, 'branch': branch, 'ensemble_alpha': 0.5,
                           'state': 'loaded_checkpoint', 'checkpoints': [{'epoch': 20, 'sha256': 'checkpoint'}],
                           'effective_losses': {'kd': kd, 'adaptive_ce': 1.0, 'contrastive': 0.0, 'diversity': 0.0},
                           'metrics': {'accuracy': 70.0}}
                (target / 'metrics.json').write_text(json.dumps(payload))
    return root


class RunnerBehavior(unittest.TestCase):
    def test_new_protocol_realistic_baseline_collects(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = module('summarize').collect(write_new_run(Path(temp)))
            self.assertEqual(rows[0]['variant'], 'baseline')

    def test_weighted_protocol_with_complete_identity_collects(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = module('summarize').collect(
                write_new_run(Path(temp), ('flat',), kd_loss_mode='weighted_nckd'))
            self.assertEqual(rows[0]['variant'], 'flat')

    def test_default_full_kl_audit_rejects_weighted_identity_in_each_record(self):
        mutations = ('config', 'manifest', 'metric')
        for location in mutations:
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temp:
                root = write_new_run(Path(temp), ('flat',))
                directory = root / 'flat/seed1'
                if location == 'config':
                    import yaml
                    path = directory / 'effective_config.yaml'
                    data = yaml.safe_load(path.read_text())
                    data['TRAINER']['TEXTCONSISTENCY']['KD_LOSS_MODE'] = 'weighted_nckd'
                    path.write_text(yaml.safe_dump(data))
                else:
                    path = (directory / 'run_manifest.json' if location == 'manifest'
                            else directory / 'eval_base_student/metrics.json')
                    data = json.loads(path.read_text())
                    data['loss_identity'] = loss_identity('weighted_nckd', 2.0, 1.0)
                    path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'loss identity|KD loss mode'):
                    module('summarize').collect(root)

    def test_weighted_audit_rejects_missing_identity_field_in_each_record(self):
        for location in ('config', 'manifest', 'metric'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temp:
                root = write_new_run(Path(temp), ('flat',), kd_loss_mode='weighted_nckd')
                directory = root / 'flat/seed1'
                if location == 'config':
                    import yaml
                    path = directory / 'effective_config.yaml'
                    data = yaml.safe_load(path.read_text())
                    data['TRAINER']['TEXTCONSISTENCY'].pop('KD_LOSS_MODE')
                    path.write_text(yaml.safe_dump(data))
                else:
                    path = (directory / 'run_manifest.json' if location == 'manifest'
                            else directory / 'eval_base_student/metrics.json')
                    data = json.loads(path.read_text())
                    data['loss_identity'].pop('non_target_weighting')
                    path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'loss identity|KD loss mode'):
                    module('summarize').collect(root)

    def test_new_protocol_rejects_variant_kd_temp_and_eval_config_drift(self):
        import yaml
        mutations = [('flat', 'signature', 'aggregation', 'hierarchical'),
                     ('hierarchical', 'signature', 'aggregation', 'flat'),
                     ('flat_adapter', 'signature', 'feature_adapter', False),
                     ('flat', 'config', 'KD_TEMP', 3.0),
                     ('flat', 'eval_config', 'N_VIEWS', 19)]
        for variant, location, field, value in mutations:
            with self.subTest(variant=variant, location=location, field=field), tempfile.TemporaryDirectory() as temp:
                root = write_new_run(Path(temp), (variant,))
                if location == 'signature':
                    path = root / variant / 'seed1/run_manifest.json'; data = json.loads(path.read_text())
                    data['model_signature'][field] = value; path.write_text(json.dumps(data))
                else:
                    path = (root / variant / 'seed1/effective_config.yaml' if location == 'config'
                            else root / variant / 'seed1/eval_base_student/effective_config.yaml')
                    data = yaml.safe_load(path.read_text()); data['TRAINER']['TEXTCONSISTENCY'][field] = value
                    path.write_text(yaml.safe_dump(data))
                with self.assertRaises(ValueError): module('summarize').collect(root)

    def test_new_protocol_requires_same_seed_samples_across_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_new_run(Path(temp), ('baseline', 'flat'), {'baseline': 'a', 'flat': 'b'})
            with self.assertRaises(ValueError): module('summarize').collect(root)

    def test_new_protocol_rejects_mixed_branch_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = write_new_run(Path(temp), ('flat',))
            for split in ('base', 'new'):
                path = root / 'flat' / 'seed1' / f'eval_{split}_teacher' / 'metrics.json'
                data = json.loads(path.read_text())
                data['checkpoints'] = [{'epoch': 20, 'sha256': 'different-checkpoint'}]
                path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'branches came from different checkpoints'):
                module('summarize').collect(root)

    def test_dataset_commands_and_kd_are_explicit(self):
        runner = module('run_pilot')
        for key, display in [('ucf101', 'UCF101'), ('dtd', 'DescribableTextures'),
                             ('fgvc_aircraft', 'FGVCAircraft')]:
            commands = runner.build_commands('/data', '/results', [1], 3,
                                             ['baseline', 'flat'], dataset=key, kd_weight=0.25)
            self.assertTrue(all(str(ROOT / f'configs/datasets/{key}.yaml') in c for c in commands))
            self.assertTrue(all(str(ROOT / f'configs/repro/{key}.yaml') in c for c in commands))
            baseline = [c for c in commands if '/baseline/' in c[c.index('--output-dir') + 1]]
            flat = [c for c in commands if '/flat/' in c[c.index('--output-dir') + 1]]
            self.assertTrue(all(c[c.index('TRAINER.TEXTCONSISTENCY.KD_WEIGHT') + 1] == '0.0' for c in baseline))
            self.assertTrue(all(c[c.index('TRAINER.TEXTCONSISTENCY.KD_WEIGHT') + 1] == '0.25' for c in flat))
            self.assertEqual(runner.DATASETS[key]['name'], display)

    def test_kd_weight_rejects_negative_and_nonfinite_values(self):
        runner = module('run_pilot')
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                runner.build_commands('/data', '/results', [1], 3, ['flat'], kd_weight=value)

    def test_commands_pass_actual_distinct_seeds_and_exact_epoch(self):
        runner = module('run_pilot')
        commands = runner.build_commands('/data', '/results', [1, 2, 3], 7, ['flat'])
        training = [c for c in commands if '--eval-only' not in c]
        self.assertEqual([c[c.index('--seed')+1] for c in training], ['1','2','3'])
        self.assertEqual(len(commands), 21)
        for command in commands:
            self.assertEqual(command[command.index('TRAINER.TEXTCONSISTENCY.USE_FEATURE_ADAPTER')+1], 'False')
            if '--eval-only' in command:
                self.assertEqual(command[command.index('--load-epoch')+1], '7')

    def test_summary_rejects_fake_seed_label(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'metrics.json'
            file.write_text(json.dumps({'seed':1,'subsample':'base','branch':'ensemble','dataset':'UCF101'}))
            with self.assertRaises(ValueError): module('summarize').read_metric(file, 2, 'base', 'ensemble', 20)

    def test_missing_seed_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'protocol.json').write_text(json.dumps({'seeds':[1,2,3], 'epochs':20, 'variants':['baseline']}))
            with self.assertRaises(FileNotFoundError): module('summarize').collect(root)

    def test_matched_results_compute_hm_from_base_and_novel_means(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'protocol.json').write_text(json.dumps({'seeds':[1,2], 'epochs':20, 'variants':['baseline']}))
            for seed, scores in [(1, (60.,90.)), (2,(80.,70.))]:
                for split, score in zip(('base','new'), scores):
                    directory = root/'baseline'/f'seed{seed}'/f'eval_{split}_student'; directory.mkdir(parents=True)
                    payload = {'seed':seed,'subsample':split,'branch':'student','dataset':'UCF101','shots':16,
                               'state':'loaded_checkpoint','checkpoints':[{'epoch':20,'sha256':f'checkpoint{seed}'}],
                               'model_signature':{},'code_sha256':'code','train_samples_sha256':f'samples{seed}',
                               'metrics':{'accuracy':score}}
                    (directory/'metrics.json').write_text(json.dumps(payload))
            rows = module('summarize').collect(root)
            self.assertEqual(rows[0]['base_mean'], 70.)
            self.assertEqual(rows[0]['novel_mean'], 80.)
            self.assertAlmostEqual(rows[0]['hm_of_means'], 74.6666666667)

    def test_summary_rejects_dataset_kd_and_source_mismatch(self):
        summary = module('summarize')
        payload = {'seed': 1, 'subsample': 'base', 'branch': 'student', 'dataset': 'FGVCAircraft',
                   'shots': 16, 'state': 'loaded_checkpoint', 'checkpoints': [{'epoch': 20}],
                   'effective_losses': {'kd': 1.0}, 'code_sha256': 'actual'}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metrics.json'; path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                summary.read_metric(path, 1, 'base', 'student', 20, dataset='UCF101')
            with self.assertRaises(ValueError):
                summary.read_metric(path, 1, 'base', 'student', 20, dataset='FGVCAircraft',
                                    kd_weight=0.0)
            with self.assertRaises(ValueError):
                summary.read_metric(path, 1, 'base', 'student', 20, dataset='FGVCAircraft',
                                    kd_weight=1.0, code_sha256='expected')

if __name__ == '__main__': unittest.main()
