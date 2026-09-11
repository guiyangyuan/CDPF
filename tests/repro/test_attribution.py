import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/repro'))
import attribution_study as study
import attribution_audit as audit


def receipt():
    result = {'fusion_baselines': [], 'historical_kd1': [], 'ucf101_full': []}
    for group, datasets in (('fusion_baselines', study.ALL_DATASETS), ('historical_kd1', study.DATASETS), ('ucf101_full', ('ucf101',))):
        for dataset in datasets:
            for seed in study.SEEDS:
                row = dict(dataset=dataset, seed=seed, checkpoint_sha256='c' * 64, source_sha256=study.MODEL_SHA256)
                row['baseline_dir' if group == 'fusion_baselines' else 'model_dir'] = f'/inputs/{group}/{dataset}/seed{seed}'
                if group == 'fusion_baselines':
                    row['source_root'] = '/old-source' if dataset == 'ucf101' else str(ROOT)
                    if dataset == 'ucf101':
                        row['source_sha256'] = study.UCF_SOURCE_SHA256
                    row['historical'] = {s: dict(metrics_sha256='m' * 64, test_sha256='t' * 64, accuracy=50.) for s in ('base', 'new')}
                result[group].append(row)
    return result


def plan(temp):
    path = Path(temp) / 'receipt.json'; path.write_text(json.dumps(receipt()))
    return study.build_plan(ROOT, '/data', Path(temp) / 'study', path)


def sample_evidence():
    sample = [('images/a.jpg', 0, 'a'), ('images/b.jpg', 1, 'b')]
    sha = hashlib.sha256(json.dumps(sample, ensure_ascii=False).encode()).hexdigest()
    order = {'version': 1, 'policy': 'contiguous', 'seed': 0, 'n_groups': 4,
             'classes': [{'classname': n, 'count': 1, 'order': [0]} for n in ('a', 'b')]}
    order['sha256'] = hashlib.sha256(json.dumps(order, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    split = dict(classes=['a', 'b'], train_sha256='train', val_sha256='val', test_sha256=sha,
                 test_count=2, description_counts=[1, 1], view_order_manifest=order)
    return {'template': {'sha256': 't' * 64}, 'loader_evidence': [{'seed': 1, 'splits': {s: split for s in ('base', 'new')}}]}


def fixture_model(root):
    import torch
    import yaml
    evidence = sample_evidence(); expected = audit.loader(evidence, 1, 'base')
    manifest = dict(dataset='DescribableTextures', seed=1, shots=16, subsample='base',
        trainable_parameters=790018, code_sha256=study.MODEL_SHA256,
        model_signature=audit.full_signature('t' * 64),
        effective_losses={'kd': 1., 'adaptive_ce': 1., 'diversity': 0., 'contrastive': 0.},
        description_counts=expected['description_counts'], view_order_manifest=expected['view_order_manifest'],
        **{f'{s}_samples_sha256': expected[f'{s}_sha256'] for s in ('train', 'val', 'test')})
    directory = Path(root).resolve(); cp = directory / 'textconsistency/model.pth.tar-20'; cp.parent.mkdir(parents=True)
    torch.save(dict(epoch=20, repro_signature=manifest['model_signature'], run_provenance=manifest,
                    state_dict={'parameters': torch.ones(790018)}), cp)
    cp.with_name(cp.name + '.sha256').write_text(study.sha256(cp))
    evaluation = directory / 'eval_base_ensemble'; evaluation.mkdir()
    metric = dict(manifest, state='loaded_checkpoint', split='test', branch='ensemble', ensemble_alpha=.5,
                  checkpoints=[{'path': str(cp), 'sha256': study.sha256(cp), 'epoch': 20}], metrics={'accuracy': 50.})
    (evaluation / 'metrics.json').write_text(json.dumps(metric))
    (evaluation / 'run_manifest.json').write_text(json.dumps(manifest))
    (evaluation / 'log.txt').write_text('* total: 2\n* correct: 1\n')
    cfg = {'SEED': 1, 'DATASET': {'ROOT': '/data'}, 'OUTPUT_DIR': str(evaluation)}
    (evaluation / 'effective_config.yaml').write_text(yaml.safe_dump(cfg))
    return evaluation, cp, manifest, evidence, cfg


class PlanTests(unittest.TestCase):
    def test_exact_queue_and_history_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp)
            self.assertEqual([s['name'] for s in p['stages']], [
                'smoke_dtd_kd0', 'smoke_fgvc_aircraft_kd0', 'anchor_dtd_kd1', 'anchor_fgvc_aircraft_kd1',
                'fusion_ucf101', 'fusion_dtd', 'fusion_fgvc_aircraft', 'formal_dtd_kd0', 'formal_fgvc_aircraft_kd0',
                'historical_dtd_kd1', 'historical_fgvc_aircraft_kd1'])
            self.assertEqual([len(s['commands']) for s in p['stages']], [7, 7, 7, 7, 6, 6, 6, 21, 21, 18, 18])
            self.assertEqual(p['command_count'], 124)
            ordinary = [c for s in p['stages'] if s['kind'] != 'fusion' for c in s['commands']]
            self.assertEqual(sum('--eval-only' not in c for c in ordinary), 10)
            self.assertEqual(sum('--eval-only' in c for c in ordinary), 96)
            for stage in p['stages']:
                if stage['kind'] == 'historical':
                    for command in stage['commands']:
                        self.assertIn('--eval-only', command)
                        self.assertTrue(study.value(command, '--model-dir').startswith('/inputs/historical_kd1'))
                        self.assertTrue(study.value(command, '--output-dir').startswith(p['output_root']))
            self.assertFalse(Path(p['output_root']).exists())

    def test_operation_checkout_paths_remain_caller_owned_with_distinct_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp).resolve() / 'immutable-source'
            data = ROOT / 'path-regression-data'
            output = ROOT / 'path-regression-study'
            python = str(ROOT / 'path-regression-venv/bin/python')
            r = receipt()
            for row in r['fusion_baselines']:
                if row['dataset'] in study.DATASETS:
                    row['source_root'] = str(source)
            for row in r['historical_kd1']:
                row['model_dir'] = str(ROOT / 'path-regression-history' / row['dataset'] / f"seed{row['seed']}")
            receipt_path = Path(temp) / 'receipt.json'
            receipt_path.write_text(json.dumps(r))
            p = study.build_plan(source, data, output, receipt_path, python)
            for stage in p['stages']:
                for command in stage['commands']:
                    self.assertEqual(command[0], python)
                    destination = Path(study.value(command, '--output-dir'))
                    self.assertTrue(destination.is_relative_to(output))
                    self.assertTrue(destination.is_relative_to(Path(stage['root'])))
                    if stage['kind'] == 'fusion':
                        self.assertEqual(study.value(command, '--data-root'), str(data))
                        continue
                    self.assertEqual(command[1], str(source / 'train.py'))
                    self.assertEqual(study.value(command, '--root'), str(data))
                    self.assertEqual(study.value(command, '--config-file'), str(source / f"configs/repro/{stage['dataset']}.yaml"))
                    self.assertEqual(study.value(command, '--dataset-config-file'), str(source / f"configs/datasets/{stage['dataset']}.yaml"))
                    if '--model-dir' in command:
                        seed = int(study.value(command, '--seed'))
                        if stage['kind'] == 'historical':
                            expected = study.row_for(r, 'historical_kd1', stage['dataset'], seed)['model_dir']
                        else:
                            expected = str(Path(stage['root']) / 'hierarchical_adapter' / f'seed{seed}')
                        self.assertEqual(study.value(command, '--model-dir'), expected)
            self.assertFalse(output.exists())
            self.assertFalse(source.exists())

    def test_cli_plan_needs_neither_site_packages_nor_data_source_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp)
            result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/repro/attribution_study.py'),
                '--source-root', str(ROOT), '--data-root', '/does-not-exist', '--output-root', p['output_root'],
                '--input-receipt', p['input_receipt'], '--plan-only'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['command_count'], 124)
            self.assertFalse(Path(p['output_root']).exists())

    def test_wrong_seed_duplicate_path_and_source_rejected(self):
        for mutation in ('seed', 'duplicate', 'path', 'source'):
            r = receipt()
            if mutation == 'seed': r['historical_kd1'][0]['seed'] = 4
            if mutation == 'duplicate': r['historical_kd1'][1] = r['historical_kd1'][0]
            if mutation == 'path': r['historical_kd1'][0]['model_dir'] = r['fusion_baselines'][0]['baseline_dir']
            if mutation == 'source': r['fusion_baselines'][0]['source_sha256'] = study.MODEL_SHA256
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                study.validate_receipt(r, ROOT)

    def test_receipt_rejects_source_dot_and_symlink_aliases_without_mutation(self):
        canonical = receipt()
        study.validate_receipt(canonical, ROOT)
        dotted = copy.deepcopy(canonical)
        dotted['fusion_baselines'][0]['source_root'] = '/old-source/./'
        original = copy.deepcopy(dotted)
        with self.assertRaisesRegex(ValueError, 'canonical absolute spelling'):
            study.validate_receipt(dotted, ROOT)
        self.assertEqual(dotted, original)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp).resolve() / 'source'; source.mkdir()
            alias = source.with_name('source-alias'); alias.symlink_to(source, target_is_directory=True)
            r = receipt()
            for row in r['fusion_baselines']:
                if row['dataset'] in study.DATASETS:
                    row['source_root'] = str(source)
            study.validate_receipt(r, source)
            next(row for row in r['fusion_baselines'] if row['dataset'] == 'dtd')['source_root'] = str(alias)
            original = copy.deepcopy(r)
            with self.assertRaisesRegex(ValueError, 'canonical absolute spelling'):
                study.validate_receipt(r, source)
            self.assertEqual(r, original)

    def test_execute_no_gpu_and_existing_output_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp)
            with patch('torch.cuda.is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                study.execute(p, Path(temp) / 'absent-preparation')
            self.assertFalse(Path(p['output_root']).exists())
            Path(p['output_root']).mkdir()
            with self.assertRaises(FileExistsError): study.execute(p, '/missing')

    def test_prepare_existing_and_overlap_rejected_before_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp)
            with self.assertRaises(FileExistsError): study.prepare(p, temp)
            with self.assertRaisesRegex(ValueError, 'isolated'): study.prepare(p, Path(p['output_root']) / 'prep')

    def test_child_environment_is_isolated_and_threads_fixed(self):
        with patch.dict('os.environ', {'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1', 'PYTHONPATH': '/wrong/source'}):
            for threads in (1, 4):
                env = study.child_env('/right/source', threads)
                self.assertEqual(env['OMP_NUM_THREADS'], str(threads))
                self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                self.assertEqual(env['PYTHONPATH'], '/right/source:/right/source/Dassl.pytorch')
                self.assertNotIn('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', env)

    def test_worker_runs_each_source_in_separate_process(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return type('Result', (), {'stdout': '{}'})()
        with patch.object(study.subprocess, 'run', side_effect=fake_run):
            study.isolated('/source-old', {'action': 'configs'}, '/python')
            study.isolated('/source-new', {'action': 'configs'}, '/python')
        self.assertEqual([c[1]['cwd'] for c in calls], ['/source-old', '/source-new'])
        self.assertTrue(all(c[0][-1] == '--metadata-worker' for c in calls))

    def test_real_processes_do_not_share_source_module_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            observed = []
            for name in ('old', 'new'):
                source = Path(temp) / name; source.mkdir()
                (source / 'origin.py').write_text(f"NAME = {name!r}\n")
                (source / 'train.py').write_text(
                    'import origin\nimport json\nfrom types import SimpleNamespace\n'
                    'def setup_cfg(args):\n'
                    '    return SimpleNamespace(dump=lambda: json.dumps({"origin": origin.NAME, "SEED": args.seed}))\n')
                stage = {'name': 'fixture', 'kind': 'formal', 'commands': [
                    sys.executable, 'unused']}
                stage['commands'] = [[sys.executable, str(source / 'train.py'), '--seed', '1']]
                result = study.isolated(source, {'action': 'configs', 'source': str(source), 'plan': {}, 'stages': [stage]})
                observed.append(result['fixture'][0]['origin'])
            self.assertEqual(observed, ['old', 'new'])

    def test_protocol_preflight_written_before_train_and_fail_stops_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp); prep = Path(temp) / 'prep'; prep.mkdir()
            prepared = {'protocol': p, 'datasets_by_source': {str(ROOT): {'dtd': {'real_loader': True}}}}
            study.atomic_json(prep / 'preparation.json', prepared)
            calls = []
            def fail_first(command, stage, index, source, output, status):
                calls.append(command)
                self.assertTrue((output / 'protocol.json').is_file())
                self.assertTrue((output / 'data_preflight.json').is_file())
                self.assertTrue((Path(p['stages'][0]['root']) / 'protocol.json').is_file())
                self.assertTrue((Path(p['stages'][0]['root']) / 'data_preflight.json').is_file())
                raise RuntimeError('fixture failure')
            with patch.object(study.common, 'assert_gpus_idle'), patch.object(study, 'verify_frozen'), \
                    patch.object(study, 'run_record', side_effect=fail_first), self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                study.execute(p, prep)
            self.assertEqual(len(calls), 1)
            status = study.read_json(Path(p['output_root']) / 'status.json')
            self.assertEqual(status['state'], 'failed')
            self.assertEqual(status['completed'], [])

    def test_cpu_description_metadata_matches_historical_aircraft_token_dedup(self):
        from types import SimpleNamespace
        from clip import clip
        import trainers.textconsistency_common as tc
        base_classes = ['737-200', '737-300', '737-400']
        new_classes = ['DHC-8-100', 'Il-76', 'Saab 340']
        evidence = {'loader_evidence': [{'seed': 1, 'splits': {
            'base': {'classes': base_classes}, 'new': {'classes': new_classes}}}]}
        preflight = SimpleNamespace(DATASETS={'fgvc_aircraft': {'name': 'FGVCAircraft'}})
        with patch.object(study.common, 'dataset_evidence', return_value=evidence), \
                patch.object(study.common, 'source_module', return_value=preflight), \
                patch.object(clip, 'load', side_effect=AssertionError('CLIP model must not load')), \
                patch.object(tc, 'load_clip_checkpoint', side_effect=AssertionError('CLIP checkpoint must not load')):
            result = study.dataset_metadata(ROOT, '/unused', 'fgvc_aircraft')
        splits = result['loader_evidence'][0]['splits']
        # These exact effective counts are in the immutable seed-1 historical metrics.
        for split, raw_counts, expected in (('base', [19, 19, 16], [19, 19, 16]),
                                            ('new', [20, 19, 11], [19, 18, 10])):
            actual = splits[split]
            self.assertEqual([len(row) for row in actual['description_rows']], raw_counts)
            self.assertEqual(actual['description_counts'], expected)
            self.assertEqual([row['count'] for row in actual['view_order_manifest']['classes']], expected)
            for raw, kept, effective, order, count in zip(actual['description_rows'], actual['effective_description_indices'],
                    actual['effective_description_rows'], actual['view_order_manifest']['classes'], expected):
                self.assertEqual(effective, [raw[i] for i in kept])
                self.assertEqual(order['order'], list(range(count)))
                self.assertEqual(kept, sorted(kept))
        vocab = ROOT / 'clip/bpe_simple_vocab_16e6.txt.gz'
        self.assertEqual(result['tokenizer']['sha256'], study.sha256(vocab))
        self.assertEqual(study.source_hashes(ROOT)[str(vocab)], study.sha256(vocab))

    def test_token_dedup_preserves_first_occurrence_and_class_boundaries_after_truncation(self):
        from clip import clip
        prefix = 'aircraft ' * 100
        rows = [['HELLO', 'hello', prefix + 'one', prefix + 'two', 'different'], ['hello', 'Hello']]
        self.assertEqual(study.token_unique_indices(rows, clip.tokenize), [[0, 2, 4], [0]])

    def test_dependency_and_template_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp)
            template = Path(temp) / 'template.json'; template.write_text('{}')
            prepared = {'protocol': p, 'files_sha256': {str(template): study.sha256(template)}}
            template.write_text('{"modified": true}')
            with self.assertRaisesRegex(ValueError, 'Frozen input'):
                study.verify_frozen(prepared)

    def test_actual_source_cfg_rejects_seed_and_shared_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp)
            for index in (0, 2, 7):
                command = p['stages'][index]['commands'][0]
                cfg = study.common.effective_config_for_command(ROOT, command)
                audit.common.validate_effective_config(cfg, copy.deepcopy(cfg), 1, 'fixture')
                changed = copy.deepcopy(cfg); changed['OPTIM']['LR'] = .02
                with self.assertRaisesRegex(ValueError, 'source-derived'):
                    audit.common.validate_effective_config(changed, cfg, 1, 'fixture')
                with self.assertRaisesRegex(ValueError, 'seed differs'):
                    audit.common.validate_effective_config(cfg, cfg, 2, 'fixture')


class AuditTests(unittest.TestCase):
    def test_real_small_checkpoint_metric_and_cfg(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluation, cp, manifest, evidence, cfg = fixture_model(temp)
            result = audit.inspect_eval(evaluation, cp, manifest, 'dtd', 1, 'base', 'ensemble', 1, 20, evidence, cfg)
            self.assertEqual((result['correct'], result['total']), (1, 2))
            changed = copy.deepcopy(cfg); changed['SEED'] = 2
            with self.assertRaisesRegex(ValueError, 'source-derived'):
                audit.inspect_eval(evaluation, cp, manifest, 'dtd', 1, 'base', 'ensemble', 1, 20, evidence, changed)
            cp.write_bytes(cp.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'hash/path/epoch'):
                audit.inspect_eval(evaluation, cp, manifest, 'dtd', 1, 'base', 'ensemble', 1, 20, evidence, cfg)

    def test_full_manifest_rejects_kd_class_order_template_and_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, manifest, evidence, _ = fixture_model(temp)
            for key, value in (('seed', 2), ('description_counts', [2, 1]), ('effective_losses', {'kd': 0}),
                               ('model_signature', audit.full_signature('changed'))):
                changed = copy.deepcopy(manifest); changed[key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'provenance'):
                    audit.validate_full_manifest(changed, 'dtd', 1, 'base', 1, evidence)

    def test_invalid_correct_total_and_accuracy(self):
        audit.accuracy_counts(50, 1, 2, 2)
        for values in ((50, 1, 3, 2), (50, -1, 2, 2), (150, 3, 2, 2), (50, 1, 0, 0), (float('nan'), 1, 2, 2), (51, 1, 2, 2)):
            with self.subTest(values=values), self.assertRaises(ValueError): audit.accuracy_counts(*values)

    def test_fusion_accepts_rounded_history_but_requires_exact_counts_and_receipt(self):
        student = {'correct': 2, 'total': 3, 'accuracy': 200.0 / 3}
        historical = {'correct': 2, 'total': 3, 'accuracy': 66.667}
        audit.validate_fusion_student(student, historical, 66.667, 2, 3, 3)
        for changed in ({'correct': 1, 'total': 3, 'accuracy': 100.0 / 3},
                        {'correct': 4, 'total': 6, 'accuracy': 200.0 / 3}):
            with self.assertRaisesRegex(ValueError, 'correct/total differs'):
                audit.validate_fusion_student(changed, historical, 66.667, 2, 3, 3)
        changed = dict(historical, accuracy=66.6668)
        with self.assertRaisesRegex(ValueError, 'receipt/log identity'):
            audit.validate_fusion_student(student, changed, 66.667, 2, 3, 3)
        for changed in (dict(student, accuracy=66.668), dict(student, accuracy=float('nan'))):
            with self.assertRaisesRegex(ValueError, 'Accuracy differs'):
                audit.validate_fusion_student(changed, historical, 66.667, 2, 3, 3)
        with self.assertRaisesRegex(ValueError, 'Accuracy differs'):
            audit.validate_fusion_student(student, dict(historical, accuracy=66.668), 66.668, 2, 3, 3)

    def test_anchor_requires_exact_counts_and_test_hash(self):
        expected = {'correct': 4, 'total': 10, 'test_sha256': 'fixed'}
        audit.validate_anchor(expected, copy.deepcopy(expected))
        for key, value in (('correct', 5), ('total', 11), ('test_sha256', 'other')):
            changed = dict(expected); changed[key] = value
            with self.assertRaisesRegex(ValueError, 'exact agreement'): audit.validate_anchor(changed, expected)

    def test_kd_direction_and_other_drift(self):
        zero = {'SEED': 1, 'OUTPUT_DIR': '/new', 'OPTIM': {'LR': .002},
                'TRAINER': {'TEXTCONSISTENCY': {'KD_WEIGHT': 0.}}}
        one = copy.deepcopy(zero); one['OUTPUT_DIR'] = '/old'; one['TRAINER']['TEXTCONSISTENCY']['KD_WEIGHT'] = 1.
        audit.compare_kd_configs(zero, one)
        with self.assertRaisesRegex(ValueError, 'direction'): audit.compare_kd_configs(one, zero)
        one['OPTIM']['LR'] = .02
        with self.assertRaisesRegex(ValueError, 'outside'): audit.compare_kd_configs(zero, one)

    def test_prediction_file_hash_count_labels_and_duplicate_samples(self):
        expected = audit.loader(sample_evidence(), 1, 'base')
        rows = [{'relative_path': f'images/{name}.jpg', 'label': label, 'label_classname': name,
                 'predictions': {b: {'label': label, 'classname': name} for b in study.FUSION_BRANCHES}}
                for label, name in enumerate(('a', 'b'))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.jsonl'
            def write(records):
                path.write_text(''.join(json.dumps(r) + '\n' for r in records))
                return {'predictions': {'path': str(path), 'sha256': study.sha256(path), 'count': len(records)},
                        'metrics': {b: {'correct': 2, 'total': 2, 'accuracy': 100.} for b in study.FUSION_BRANCHES}}
            metric = write(rows); audit.validate_predictions(metric, temp, expected)
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'SHA'): audit.validate_predictions(metric, temp, expected)
            metric = write(rows[:1])
            with self.assertRaisesRegex(ValueError, 'count'): audit.validate_predictions(metric, temp, expected)
            metric = write([rows[0], rows[0]])
            with self.assertRaisesRegex(ValueError, 'Duplicate'): audit.validate_predictions(metric, temp, expected)
            metric = write(rows); metric['metrics']['student']['correct'] = 1
            with self.assertRaisesRegex(ValueError, 'correct count'): audit.validate_predictions(metric, temp, expected)
            changed = copy.deepcopy(rows); changed[0]['label_classname'] = 'b'; metric = write(changed)
            with self.assertRaisesRegex(ValueError, 'target class'): audit.validate_predictions(metric, temp, expected)

    def test_final_all_identities_differences_and_hm_conventions(self):
        def row(dataset, seed, branch, kd=None):
            base = 60 + seed + (kd or 0); novel = 40 + 2 * seed + (kd or 0)
            result = dict(dataset=dataset, seed=seed, branch=branch, base=base, novel=novel, hm=audit.harmonic(base, novel))
            if kd is not None: result['kd_weight'] = kd
            return result
        full = [row(d, s, b, k) for d in study.DATASETS for k in (0, 1) for s in study.SEEDS for b in study.BRANCHES]
        fusion = [row(d, s, b) for d in study.ALL_DATASETS for s in study.SEEDS for b in study.FUSION_BRANCHES]
        ucf = [row('ucf101', s, 'ensemble') for s in study.SEEDS]
        result = audit.final_analysis(full, fusion, ucf)
        self.assertEqual(len(result['full_seed_branch_rows']), 36)
        self.assertEqual(len(result['fusion_seed_branch_rows']), 54)
        self.assertEqual(len(result['comparisons']), 21)
        self.assertEqual(result['comparisons'][0]['mean']['base'], 1.)
        self.assertNotEqual(result['full_summaries'][0]['hm_of_means'], result['full_summaries'][0]['mean_seed_hm'])
        for f, g in ((full[:-1], fusion), (full[:-1] + [full[0]], fusion), (full, fusion[:-1])):
            with self.assertRaisesRegex(ValueError, 'Incomplete or duplicate'): audit.final_analysis(f, g, ucf)
        changed = copy.deepcopy(full); changed[0]['hm'] = 90
        with self.assertRaisesRegex(ValueError, 'HM differs'): audit.final_analysis(changed, fusion, ucf)

    def test_incomplete_or_duplicate_command_evidence_refuses_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            p = plan(temp); stage = p['stages'][0]; root = Path(stage['root']); root.mkdir(parents=True)
            study.atomic_json(root / 'protocol.json', stage)
            study.atomic_json(root / 'data_preflight.json', {})
            prepared = {'datasets_by_source': {stage['source_root']: {stage['dataset']: {}}}}
            with self.assertRaisesRegex(ValueError, 'incomplete or duplicated'):
                audit.audit_stage(stage, prepared, Path(p['output_root']), {'commands': []})

    def test_no_retries_and_failed_exit_code_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp); status = {'commands': []}
            with self.assertRaises(subprocess.CalledProcessError):
                study.run_record([sys.executable, '-c', 'import sys;sys.exit(7)'], 'smoke', 0, ROOT, output, status)
            stored = study.read_json(output / 'status.json')['commands']
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]['exit_code'], 7)
            self.assertIsNotNone(stored[0]['pid'])
            self.assertIsNotNone(stored[0]['ended'])
            self.assertTrue(Path(stored[0]['stderr_log']).is_file())


if __name__ == '__main__':
    unittest.main()
