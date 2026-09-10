"""Controls must match capacity, preserve views and survive checkpoint reload."""
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
import json
import subprocess
import sys
import copy
import hashlib
import shutil
import yaml

import torch
from torch.nn import functional as F
from trainers import textconsistency_common as common
from .test_trainers import make_trainer, CoOp_TextConsistency_V2_FG
from .test_runner import module, write_new_run, ROOT


def structure_fixture(root):
    """Synthetic scores/checkpoint bytes, solely for exercising report auditing."""
    write_new_run(root, ('hierarchical_adapter',))
    source = root / 'hierarchical_adapter/seed1'
    manifest = json.loads((source / 'run_manifest.json').read_text())
    metric = json.loads((source / 'eval_base_student/metrics.json').read_text())
    config = yaml.safe_load((source / 'effective_config.yaml').read_text())
    shutil.rmtree(root / 'hierarchical_adapter')
    variants = [('hierarchical_adapter', 'hierarchical', 'contiguous', 0),
                ('flat_dual_adapter', 'flat_dual', 'contiguous', 0),
                ('hierarchical_interleaved_adapter', 'hierarchical', 'interleaved', 0),
                ('hierarchical_shuffle17_adapter', 'hierarchical', 'shuffle', 17),
                ('hierarchical_shuffle29_adapter', 'hierarchical', 'shuffle', 29)]
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol.update(seeds=[1, 2, 3], structure_study=True, variants=[v[0] for v in variants],
                    view_orders={v[0]: {'policy': v[2], 'seed': v[3]} for v in variants})
    (root / 'protocol.json').write_text(json.dumps(protocol))
    for variant, aggregation, order, order_seed in variants:
        for seed in (1, 2, 3):
            directory = root / variant / f'seed{seed}'; directory.mkdir(parents=True)
            cp = directory / 'textconsistency/model.pth.tar-20'; cp.parent.mkdir()
            cp.write_bytes(f'synthetic-not-a-model-{variant}-{seed}'.encode())
            data = copy.deepcopy(manifest)
            data.update(seed=seed, train_samples_sha256=f'sample{seed}', trainable_parameters=790018,
                        test_samples_sha256=hashlib.sha256(b'base-test').hexdigest(),
                        description_counts=[20]*51,
                        view_order_manifest=common.view_order_manifest([f'base{i}' for i in range(51)], [20]*51, order, 4, order_seed))
            data['model_signature'].update(aggregation=aggregation, view_order=order, order_seed=order_seed, order_version=1)
            (directory / 'run_manifest.json').write_text(json.dumps(data))
            cfg = copy.deepcopy(config); cfg['SEED'] = seed
            cfg['TRAINER']['TEXTCONSISTENCY'].update(AGGREGATION=aggregation, VIEW_ORDER=order, ORDER_SEED=order_seed)
            (directory / 'effective_config.yaml').write_text(yaml.safe_dump(cfg))
            for split in ('base', 'new'):
                for branch in ('student', 'teacher', 'ensemble'):
                    folder = directory / f'eval_{split}_{branch}'; folder.mkdir()
                    payload = {**copy.deepcopy(metric), **copy.deepcopy(data), 'subsample': split, 'branch': branch,
                               'split': 'test', 'test_samples_sha256': hashlib.sha256(f'{split}-test'.encode()).hexdigest(),
                               'effective_losses': {'kd': 1., 'adaptive_ce': 1., 'contrastive': 0., 'diversity': 0.},
                               'checkpoints': [{'epoch': 20, 'sha256': hashlib.sha256(cp.read_bytes()).hexdigest()}]}
                    if split == 'new':
                        payload['description_counts'] = [20]*50
                        payload['view_order_manifest'] = common.view_order_manifest([f'novel{i}' for i in range(50)], [20]*50, order, 4, order_seed)
                    (folder / 'metrics.json').write_text(json.dumps(payload))
                    ecfg = copy.deepcopy(cfg); ecfg['DATASET']['SUBSAMPLE_CLASSES'] = split
                    ecfg['TRAINER']['TEXTCONSISTENCY']['INFERENCE_BRANCH'] = branch
                    (folder / 'effective_config.yaml').write_text(yaml.safe_dump(ecfg))
    return root


class StructureBehavior(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_dual_flat_matches_hierarchy_capacity_and_both_heads_learn(self):
        for adapter in (False, True):
            hierarchy = common.ViewAggregator(32, 7, 'hierarchical', 3, adapter)
            try:
                flat = common.ViewAggregator(32, 7, 'flat_dual', 3, adapter)
            except ValueError as exc:
                self.fail(f'Parameter-matched flat control is unavailable: {exc}')
            self.assertEqual(sum(p.numel() for p in flat.parameters()),
                             sum(p.numel() for p in hierarchy.parameters()))
            image, views = F.normalize(torch.randn(5, 32), dim=-1), F.normalize(torch.randn(4, 7, 32), dim=-1)
            out, _ = flat(image, views)
            F.cross_entropy(10 * torch.einsum('bd,bcd->bc', image, out), torch.tensor([0, 1, 2, 3, 1])).backward()
            for name in ('intra_group_query', 'intra_group_key', 'inter_group_query', 'inter_group_key'):
                grad = getattr(flat, name).weight.grad
                self.assertIsNotNone(grad)
                self.assertGreater(grad.abs().sum().item(), 0)
            for name in ('intra_group_temp', 'inter_group_temp'):
                self.assertGreater(getattr(flat, name).grad.abs().sum().item(), 0)

    def test_dual_flat_is_global_order_invariant_and_ignores_padding(self):
        try:
            model = common.ViewAggregator(16, 7, 'flat_dual', 3, True)
        except ValueError as exc:
            self.fail(f'Global flat control is unavailable: {exc}')
        image, views = torch.randn(2, 16), torch.randn(3, 7, 16)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1]*7, [1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        original, _ = model(image, views, mask)
        changed = views.clone(); changed[~mask] = 12345
        permutation = torch.tensor([6, 0, 3, 2, 5, 1, 4])
        actual, _ = model(image, changed[:, permutation], mask[:, permutation])
        torch.testing.assert_close(original, actual, atol=2e-5, rtol=2e-5)

    def test_order_controls_preserve_views_and_do_not_consume_global_rng(self):
        self.assertTrue(hasattr(common, 'description_permutation'), 'Missing controlled description ordering')
        permutation = common.description_permutation
        torch_before, python_before = torch.get_rng_state().clone(), random.getstate()
        self.assertEqual(permutation(7, 'contiguous', 3, 0), list(range(7)))
        self.assertEqual(permutation(7, 'interleaved', 3, 0), [0, 3, 6, 1, 4, 2, 5])
        for count in (1, 3, 7, 20):
            for seed in (17, 29):
                order = permutation(count, 'shuffle', 4, seed)
                self.assertEqual(sorted(order), list(range(count)))
                self.assertEqual(order, permutation(count, 'shuffle', 4, seed))
        self.assertNotEqual(permutation(20, 'shuffle', 4, 17), permutation(20, 'shuffle', 4, 29))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        for args in [(0, 'contiguous', 4, 0), (5, 'bad', 4, 0), (5, 'shuffle', 4, -1),
                     (5, 'interleaved', 0, 0), (5, 'contiguous', 4, 17)]:
            with self.assertRaises(ValueError): permutation(*args)

    def test_encoder_applies_order_and_checkpoint_rejects_other_partition(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            original = make_trainer(CoOp_TextConsistency_V2_FG, root / 'original')
            shuffled = make_trainer(CoOp_TextConsistency_V2_FG, root / 'shuffled', view_order='shuffle', order_seed=17)
            original_encoder, encoder = original.model.multiview_encoder, shuffled.model.multiview_encoder
            self.assertTrue(hasattr(encoder, 'view_order_manifest'), 'Encoder must record actual ordering')
            for i, row in enumerate(encoder.view_order_manifest['classes']):
                count, order = row['count'], row['order']
                self.assertEqual(sorted(order), list(range(count)))
                torch.testing.assert_close(encoder.multiview_features[i, :count],
                                           original_encoder.multiview_features[i, order], rtol=0, atol=0)
            self.assertTrue(torch.equal(original_encoder.view_mask, encoder.view_mask))
            self.assertNotEqual(original.repro_signature, shuffled.repro_signature)
            shuffled.save_model(0, root / 'shuffled')
            with self.assertRaisesRegex(ValueError, 'signature mismatch'):
                original.load_model(root / 'shuffled', 1)
            reloaded = make_trainer(CoOp_TextConsistency_V2_FG, root / 'reloaded', view_order='shuffle', order_seed=17)
            reloaded.load_model(root / 'shuffled', 1)
            images = torch.randn(3, 3, 32, 32)
            shuffled.model.eval(); reloaded.model.eval()
            for branch in ('student', 'teacher', 'ensemble'):
                shuffled.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
                reloaded.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
                torch.testing.assert_close(shuffled.model_inference(images), reloaded.model_inference(images), rtol=0, atol=0)

    def test_structure_cli_fixes_all_comparisons_and_never_starts_in_plan_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'not-created'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/repro/run_pilot.py'),
                                     '--data-root', '/not-needed-for-plan', '--output-root', str(target),
                                     '--structure-study', '--seeds', '1', '2', '3', '--plan-only'],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(len(plan['commands']), 105)
            self.assertEqual(plan['seeds'], [1, 2, 3])
            self.assertEqual(plan['variants'], ['hierarchical_adapter', 'flat_dual_adapter',
                                              'hierarchical_interleaved_adapter',
                                              'hierarchical_shuffle17_adapter', 'hierarchical_shuffle29_adapter'])
            self.assertFalse(target.exists())
            for command in plan['commands']:
                variant = Path(command[command.index('--output-dir')+1]).relative_to(target.resolve()).parts[0]
                order, seed = ('shuffle', '17') if 'shuffle17' in variant else (
                    ('shuffle', '29') if 'shuffle29' in variant else (
                    ('interleaved', '0') if 'interleaved' in variant else ('contiguous', '0')))
                self.assertEqual(command[command.index('TRAINER.TEXTCONSISTENCY.VIEW_ORDER')+1], order)
                self.assertEqual(command[command.index('TRAINER.TEXTCONSISTENCY.ORDER_SEED')+1], seed)

    def test_summary_rejects_mislabelled_order_even_when_every_config_agrees(self):
        import yaml
        with tempfile.TemporaryDirectory() as temp:
            root = write_new_run(Path(temp), ('hierarchical_adapter',))
            protocol = json.loads((root / 'protocol.json').read_text())
            protocol['view_orders'] = {'hierarchical_adapter': {'policy': 'contiguous', 'seed': 0}}
            (root / 'protocol.json').write_text(json.dumps(protocol))
            directory = root / 'hierarchical_adapter/seed1'
            for path in directory.rglob('effective_config.yaml'):
                data = yaml.safe_load(path.read_text())
                data['TRAINER']['TEXTCONSISTENCY'].update(VIEW_ORDER='shuffle', ORDER_SEED=17)
                path.write_text(yaml.safe_dump(data))
            for path in [directory / 'run_manifest.json', *directory.rglob('metrics.json')]:
                data = json.loads(path.read_text())
                data['model_signature'].update(view_order='shuffle', order_seed=17, order_version=1)
                path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'order|VIEW_ORDER'):
                module('summarize').collect(root)

    def test_order_audit_rejects_missing_views_wrong_hash_and_parameter_count(self):
        audit = module('summarize')
        self.assertTrue(hasattr(audit, 'validate_order_evidence'), 'Missing actual view/parameter audit')
        record = common.view_order_manifest(['one', 'two'], [7, 3], 'shuffle', 4, 17)
        payload = {'description_counts': [7, 3], 'view_order_manifest': record, 'trainable_parameters': 790018}
        audit.validate_order_evidence(payload, {'policy': 'shuffle', 'seed': 17}, 4, 20, True)
        for field in ('hash', 'view', 'parameters', 'count'):
            wrong = json.loads(json.dumps(payload))
            if field == 'hash': wrong['view_order_manifest']['sha256'] = 'wrong'
            if field == 'view': wrong['view_order_manifest']['classes'][0]['order'] = [0]*7
            if field == 'parameters': wrong['trainable_parameters'] = 656641
            if field == 'count': wrong['description_counts'] = [6, 3]
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_order_evidence(wrong, {'policy': 'shuffle', 'seed': 17}, 4, 20, True)

    def test_new_dual_flat_checkpoint_trains_and_reloads_all_branches(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            first = make_trainer(CoOp_TextConsistency_V2_FG, root / 'one', mode='flat_dual')
            image = torch.randn(3, 3, 32, 32)
            before = first.model.view_aggregator.inter_group_query.weight.detach().clone()
            first.forward_backward({'img': image, 'label': torch.tensor([0, 1, 2])})
            self.assertFalse(torch.equal(before, first.model.view_aggregator.inter_group_query.weight))
            first.save_model(0, root / 'one')
            second = make_trainer(CoOp_TextConsistency_V2_FG, root / 'two', mode='flat_dual')
            second.load_model(root / 'one', 1)
            first.model.eval(); second.model.eval()
            for branch in ('student', 'teacher', 'ensemble'):
                first.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
                second.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
                torch.testing.assert_close(first.model_inference(image), second.model_inference(image), rtol=0, atol=0)

    def test_complete_structure_report_checks_bytes_and_cross_variant_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = structure_fixture(Path(temp))
            rows = module('summarize').collect(root)
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(row['seeds'] == 3 and row['hm_of_means'] == 70. for row in rows))
            checkpoint = root / 'flat_dual_adapter/seed1/textconsistency/model.pth.tar-20'
            saved = checkpoint.read_bytes(); checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint bytes'):
                module('summarize').collect(root)
            checkpoint.write_bytes(saved)
            for path in (root / 'flat_dual_adapter/seed1').rglob('effective_config.yaml'):
                data = yaml.safe_load(path.read_text()); data['OPTIM']['LR'] = 0.1
                path.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, 'outside aggregation/order'):
                module('summarize').collect(root)

    def test_structure_report_rejects_changed_test_images_or_validation_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = structure_fixture(Path(temp))
            path = root / 'flat_dual_adapter/seed1/eval_new_teacher/metrics.json'
            original = json.loads(path.read_text())
            for field, value in [('test_samples_sha256', hashlib.sha256(b'different-images').hexdigest()),
                                 ('test_samples_sha256', None), ('split', 'val')]:
                data = copy.deepcopy(original); data[field] = value
                path.write_text(json.dumps(data))
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'test|Test'):
                    module('summarize').collect(root)


if __name__ == '__main__':
    unittest.main()
