"""CPU evidence for the fixed visual-adapter pilot; never runs training."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    study = importlib.import_module('scripts.repro.visual_adapter_study')
except ModuleNotFoundError as exc:
    if exc.name not in {'scripts.repro.visual_adapter_study',
                        'scripts.repro.visual_adapter_validation'}:
        raise
    study = None

VARIANTS = ('simple_fusion', 'visual_only', 'visual_description')


def prepared_fixture(root):
    directory = root / 'prepared'; directory.mkdir()
    data = root / 'data'; data.mkdir()
    clip = root / 'clip.pt'; clip.write_bytes(b'controlled CPU fixture')
    rows = [{'sample_id': f'val/{i}.jpg', 'label': i % 50,
             'classname': f'class-{i % 50}'} for i in range(200)]
    configs = {}
    inputs = {str(clip): hashlib.sha256(clip.read_bytes()).hexdigest()}
    for variant in VARIANTS:
        config = directory / f'{variant}.yaml'; config.write_text(f'variant: {variant}\n')
        config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
        inputs[str(config)] = config_sha
        configs[variant] = {'config_path': str(config), 'config_sha256': config_sha,
                            'config': {'variant': variant}, 'model_signature': {}}
    value = {'source_root': str(study.ROOT), 'data_root': str(data), 'clip_path': str(clip),
             'source_sha256': study.validation.SOURCE_SHA256,
             'protocol': copy.deepcopy(study.validation.PROTOCOL),
             'prepared_identity_sha256': 'a' * 64, 'class_order': [f'class-{i}' for i in range(50)],
             'samples': {'train': [], 'val': rows, 'test': []}, 'input_files': inputs,
             'cases': configs}
    path = directory / 'prepared.json'; path.write_text(json.dumps(value))
    return path, value


def export_fixtures(prepared, correct=(100, 101, 103)):
    labels = np.arange(200, dtype=np.int64) % 50
    values = {}
    for variant, count in zip(VARIANTS, correct):
        logits = np.zeros((200, 50), dtype=np.float32)
        predicted = labels.copy(); predicted[count:] = (predicted[count:] + 1) % 50
        logits[np.arange(200), predicted] = 4.0
        receipt = {'variant': variant, 'protocol': prepared['protocol'],
                   'source_sha256': prepared['source_sha256'],
                   'prepared_identity_sha256': prepared['prepared_identity_sha256'],
                   'class_order': prepared['class_order'], 'rows': prepared['samples']['val'],
                   'checkpoint': {'epoch': 20, 'path': f'/fixture/{variant}/epoch20',
                                  'sha256': 'b' * 64}, 'logits_sha256': 'c' * 64}
        values[variant] = (copy.deepcopy(receipt),
                           {'labels': labels.copy(), 'S': logits.copy(),
                            'D': logits.copy(), 'F': logits.copy()})
    return values


class StudyFixture(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(study, 'The fixed pilot study module must be implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.prepared_path, self.prepared = prepared_fixture(self.root)
        patch = mock.patch.object(study.validation, 'load_prepared', return_value=self.prepared)
        patch.start(); self.addCleanup(patch.stop)

    def plan(self):
        return study.create_plan(self.prepared_path, self.root / 'output', self.root / 'plan')


class StudyTests(StudyFixture):
    def test_python_invocation_retains_virtualenv_symlink_path(self):
        invocation = self.root / 'python'
        invocation.symlink_to(sys.executable)
        plan = study.create_plan(self.prepared_path, self.root / 'output', self.root / 'plan',
                                 python=str(invocation))
        self.assertEqual(plan['python'], str(invocation))
        self.assertTrue(all(s['argv'][0] == str(invocation) for s in plan['stages']))

    def test_plan_has_only_three_train_export_pairs_and_does_not_create_output(self):
        plan = self.plan()
        self.assertEqual([(s['variant'], s['kind']) for s in plan['stages']], [
            ('simple_fusion', 'train'), ('simple_fusion', 'export'),
            ('visual_only', 'train'), ('visual_only', 'export'),
            ('visual_description', 'train'), ('visual_description', 'export')])
        self.assertEqual(plan['wallclock_seconds'], 3600)
        self.assertEqual(plan['training_budget'], {'planned': 3, 'maximum': 3, 'retries': 0})
        self.assertFalse((self.root / 'output').exists())
        self.assertEqual(study.validate_plan(self.root / 'plan/plan.json'), plan)
        self.assertEqual(plan['stages'][1]['argv'][-4:],
                         ['--model-dir', str(self.root / 'output/simple_fusion/model'),
                          '--output-dir', str(self.root / 'output/simple_fusion/export')])

    def test_recomputed_hash_cannot_authorize_an_extra_train_or_changed_command(self):
        original = self.plan()
        for mutation in ('extra_train', 'command', 'cap', 'threshold'):
            plan = copy.deepcopy(original)
            if mutation == 'extra_train': plan['stages'].append(plan['stages'][0])
            elif mutation == 'command': plan['stages'][0]['argv'].append('TRAINER.KD_WEIGHT=99')
            elif mutation == 'cap': plan['wallclock_seconds'] = 7200
            else: plan['continuation_rule']['minimum_gain_pp'] = 0.5
            plan['plan_identity_sha256'] = study.plan_digest(plan)
            (self.root / 'plan/plan.json').write_text(json.dumps(plan))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                study.validate_plan(self.root / 'plan/plan.json')

    def test_changed_prepared_bytes_and_operation_hash_are_rejected(self):
        plan = self.plan()
        self.prepared_path.write_text(self.prepared_path.read_text() + '\n')
        with self.assertRaises(ValueError):
            study.validate_plan(self.root / 'plan/plan.json')
        self.prepared_path.write_text(json.dumps(self.prepared))
        plan['operations']['study']['sha256'] = '0' * 64
        plan['plan_identity_sha256'] = study.plan_digest(plan)
        (self.root / 'plan/plan.json').write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            study.validate_plan(self.root / 'plan/plan.json')

    def test_existing_or_overlapping_outputs_are_refused(self):
        (self.root / 'existing').mkdir()
        with self.assertRaises(FileExistsError):
            study.create_plan(self.prepared_path, self.root / 'existing', self.root / 'plan')
        for output in (self.prepared_path.parent / 'nested', study.ROOT / 'nested-pilot',
                       Path(self.prepared['data_root']) / 'nested'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                study.create_plan(self.prepared_path, output, self.root / 'plan')
        self.plan()
        (self.root / 'output').mkdir()
        with self.assertRaises(FileExistsError):
            study.run_plan(self.root / 'plan/plan.json')

    def test_no_cuda_creates_no_output_and_launches_no_child(self):
        self.plan()
        marker = self.root / 'spawned'
        with mock.patch.object(study, '_cuda_available', return_value=False), \
             mock.patch.object(study.subprocess, 'Popen', side_effect=lambda *a, **kw: marker.touch()):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                study.run_plan(self.root / 'plan/plan.json')
        self.assertFalse(marker.exists())
        self.assertFalse((self.root / 'output').exists())

    def analyze(self, values):
        with mock.patch.object(study.validation, 'load_export',
                               side_effect=lambda p, **kw: values[str(p)]):
            return study.analyze_exports(list(values), self.prepared_path)

    def test_exact_one_percentage_point_boundary_and_all_branches_are_retained(self):
        result = self.analyze(export_fixtures(self.prepared, (100, 101, 103)))
        self.assertTrue(result['continuation']['passed'])
        self.assertFalse(result['continuation']['automatic_expansion'])
        self.assertEqual(result['comparisons']['B-A']['F']['correct_difference'], 1)
        self.assertEqual(result['comparisons']['C-A']['F']['accuracy_difference_pp'], 1.5)
        self.assertEqual(result['comparisons']['C-B']['F']['accuracy_difference_pp'], 1.0)
        self.assertEqual(result['comparisons']['C-B']['F']['corrections'], 2)
        self.assertEqual(result['comparisons']['C-B']['F']['harms'], 0)
        for variant, count in zip(VARIANTS, (100, 101, 103)):
            self.assertEqual(set(result['variants'][variant]), {'S', 'D', 'F'})
            for metric in result['variants'][variant].values():
                self.assertEqual(metric['correct'], count)
                self.assertEqual(metric['total'], 200)
        below = self.analyze(export_fixtures(self.prepared, (100, 101, 102)))
        self.assertFalse(below['continuation']['passed'])
        negative = self.analyze(export_fixtures(self.prepared, (103, 101, 100)))
        self.assertFalse(negative['continuation']['passed'])
        self.assertEqual(negative['comparisons']['C-A']['F']['harms'], 3)
        self.assertEqual(negative['comparisons']['C-A']['F']['accuracy_difference_pp'], -1.5)

    def test_branches_use_their_own_predictions_and_report_paired_harm(self):
        values = export_fixtures(self.prepared)
        _, arrays = values['visual_description']
        arrays['S'][0, :] = 0; arrays['S'][0, 1] = 10
        arrays['F'] = 0.5 * arrays['S'] + 0.5 * arrays['D']
        result = self.analyze(values)
        self.assertEqual(result['variants']['visual_description']['S']['correct'], 102)
        self.assertEqual(result['variants']['visual_description']['D']['correct'], 103)
        self.assertEqual(result['variants']['visual_description']['F']['correct'], 102)
        self.assertEqual(result['comparisons']['C-B']['F']['corrections'], 2)
        self.assertEqual(result['comparisons']['C-B']['F']['harms'], 1)
        self.assertFalse(result['continuation']['passed'])

    def test_analysis_rejects_mismatched_rows_classes_labels_fusion_and_extra_variants(self):
        for mutation in ('rows', 'classes', 'labels', 'fusion', 'duplicate', 'missing',
                         'nonfinite', 'epoch', 'protocol'):
            values = export_fixtures(self.prepared)
            receipt, arrays = values['visual_description']
            if mutation == 'rows': receipt['rows'].reverse()
            elif mutation == 'classes': receipt['class_order'].reverse()
            elif mutation == 'labels': arrays['labels'][0] = 2
            elif mutation == 'fusion': arrays['F'][0, 0] += 0.001
            elif mutation == 'duplicate': receipt['variant'] = 'visual_only'
            elif mutation == 'missing': values.pop('visual_only')
            elif mutation == 'nonfinite': arrays['S'][0, 0] = np.nan
            elif mutation == 'epoch': receipt['checkpoint']['epoch'] = 19
            else: receipt['protocol']['epochs'] = 21
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.analyze(values)


class ExecutorTests(StudyFixture):
    """Patch only hardware/artifact boundaries; execute actual short CPU subprocesses."""
    def execution_fixture(self, failing_stage=None, sleeping=False):
        plan = self.plan()
        marker = self.root / 'order.txt'
        for index, stage in enumerate(plan['stages']):
            code = ("from pathlib import Path; import os, subprocess, sys, time; "
                    f"Path({str(marker)!r}).open('a').write({str(index)!r}+'\\n'); "
                    f"Path({stage['output_dir']!r}).mkdir(parents=True); ")
            if sleeping and index == 0:
                grandchild = self.root / 'grandchild-alive'
                child_code = f"import time; from pathlib import Path; time.sleep(1.5); Path({str(grandchild)!r}).touch(); time.sleep(20)"
                code += (f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                         "time.sleep(20); ")
            code += f"raise SystemExit({7 if index == failing_stage else 0})"
            stage['argv'] = [sys.executable, '-c', code]
        # Real command canonicalization has its own tests above; this isolates queue mechanics.
        plan['plan_identity_sha256'] = study.plan_digest(plan)
        (self.root / 'plan/plan.json').write_text(json.dumps(plan))
        patches = [mock.patch.object(study, 'validate_plan', return_value=plan),
                   mock.patch.object(study, '_cuda_available', return_value=True),
                   mock.patch.object(study, '_validate_stage', return_value={'validated': True}),
                   mock.patch.object(study, 'analyze_exports', return_value={'continuation': {'passed': False}})]
        for patch in patches: patch.start(); self.addCleanup(patch.stop)
        return plan, marker

    def test_success_runs_exactly_six_children_and_writes_completed_status(self):
        self.execution_fixture()
        status = study.run_plan(self.root / 'plan/plan.json')
        self.assertEqual((self.root / 'order.txt').read_text().splitlines(), ['0', '1', '2', '3', '4', '5'])
        self.assertEqual(status['state'], 'completed')
        self.assertEqual(len(status['stages']), 6)
        self.assertTrue(all(item['state'] == 'completed' for item in status['stages']))
        self.assertTrue((self.root / 'output/analysis.json').is_file())
        with self.assertRaises(FileExistsError): study.run_plan(self.root / 'plan/plan.json')

    def test_child_failure_stops_before_next_variant_and_does_not_retry(self):
        self.execution_fixture(failing_stage=1)
        with self.assertRaises(RuntimeError): study.run_plan(self.root / 'plan/plan.json')
        self.assertEqual((self.root / 'order.txt').read_text().splitlines(), ['0', '1'])
        status = json.loads((self.root / 'output/status.json').read_text())
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['stages'][-1]['returncode'], 7)
        self.assertFalse((self.root / 'output/analysis.json').exists())

    def test_changed_input_after_first_child_stops_before_export(self):
        plan, _ = self.execution_fixture()
        original = plan['stages'][0]['argv'][2]
        plan['stages'][0]['argv'][2] = original.replace('raise SystemExit(0)',
            f"Path({str(self.prepared_path)!r}).write_text('changed'); raise SystemExit(0)")
        plan['plan_identity_sha256'] = study.plan_digest(plan)
        (self.root / 'plan/plan.json').write_text(json.dumps(plan))
        with self.assertRaises(ValueError): study.run_plan(self.root / 'plan/plan.json')
        self.assertEqual((self.root / 'order.txt').read_text().splitlines(), ['0'])
        self.assertEqual(json.loads((self.root / 'output/status.json').read_text())['state'], 'failed')

    def test_invalid_artifact_halts_before_export_and_never_claims_success(self):
        self.execution_fixture()
        with mock.patch.object(study, '_validate_stage', side_effect=ValueError('epoch20 missing')):
            with self.assertRaisesRegex(ValueError, 'epoch20'):
                study.run_plan(self.root / 'plan/plan.json')
        self.assertEqual((self.root / 'order.txt').read_text().splitlines(), ['0'])
        status = json.loads((self.root / 'output/status.json').read_text())
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['stages'][0]['state'], 'failed')
        self.assertFalse((self.root / 'output/analysis.json').exists())

    def test_one_deadline_covers_the_entire_queue_not_each_child(self):
        self.execution_fixture()
        # Child startup speed is not part of the deadline contract. Advance only
        # the runner's clock after each real child has completed and been reaped.
        # The separate alarm and child/grandchild tests exercise actual wall time.
        clock = SimpleNamespace(value=100.0)
        def validated(*args):
            clock.value += 2000.0
            return {'validated': True}
        with mock.patch.object(study, 'time', SimpleNamespace(monotonic=lambda: clock.value)), \
             mock.patch.object(study, '_validate_stage', side_effect=validated):
            with self.assertRaises(TimeoutError): study.run_plan(self.root / 'plan/plan.json')
        launched = (self.root / 'order.txt').read_text().splitlines()
        self.assertEqual(launched, ['0', '1'])
        status = json.loads((self.root / 'output/status.json').read_text())
        self.assertEqual(status['state'], 'timed_out')
        self.assertEqual(status['elapsed_seconds'], 4000.0)
        self.assertFalse((self.root / 'output/analysis.json').exists())

    def test_deadline_also_interrupts_cpu_artifact_validation(self):
        self.execution_fixture()
        start = time.monotonic()
        with mock.patch.object(study, 'WALLCLOCK_SECONDS', 0.3), \
             mock.patch.object(study, '_validate_stage', side_effect=lambda *args: time.sleep(20)):
            with self.assertRaises(TimeoutError): study.run_plan(self.root / 'plan/plan.json')
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual((self.root / 'order.txt').read_text().splitlines(), ['0'])
        self.assertEqual(json.loads((self.root / 'output/status.json').read_text())['state'], 'timed_out')

    def test_termination_signal_records_stopped_and_does_not_start_next_stage(self):
        plan, _ = self.execution_fixture()
        plan['stages'][0]['argv'][2] = plan['stages'][0]['argv'][2].replace(
            'raise SystemExit(0)', 'import signal; os.kill(os.getppid(), signal.SIGTERM); time.sleep(20)')
        (self.root / 'plan/plan.json').write_text(json.dumps(plan))
        with self.assertRaises(RuntimeError): study.run_plan(self.root / 'plan/plan.json')
        self.assertEqual((self.root / 'order.txt').read_text().splitlines(), ['0'])
        status = json.loads((self.root / 'output/status.json').read_text())
        self.assertEqual(status['state'], 'stopped')
        self.assertEqual(status['stages'][0]['state'], 'stopped')

    def test_signal_during_process_creation_cannot_leave_an_untracked_child(self):
        plan, _ = self.execution_fixture()
        leaked = self.root / 'untracked-child'
        plan['stages'][0]['argv'][2] = ("from pathlib import Path; import time; time.sleep(0.5); "
                                      f"Path({str(leaked)!r}).touch()")
        (self.root / 'plan/plan.json').write_text(json.dumps(plan))
        real_popen = study.subprocess.Popen
        def interrupted_launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGTERM)
            return process
        with mock.patch.object(study.subprocess, 'Popen', side_effect=interrupted_launch):
            with self.assertRaises(RuntimeError): study.run_plan(self.root / 'plan/plan.json')
        time.sleep(0.7)
        self.assertFalse(leaked.exists())
        self.assertEqual(json.loads((self.root / 'output/status.json').read_text())['state'], 'stopped')

    def test_deadline_signal_cannot_leak_the_subprocess_wait_lock(self):
        self.execution_fixture(sleeping=True)
        real_popen, real_cleanup = study.subprocess.Popen, study._kill_group
        locks = []
        class InterruptingLock:
            def __init__(self, lock): self.lock, self.armed = lock, True
            def acquire(self, *args, **kwargs):
                acquired = self.lock.acquire(*args, **kwargs)
                if acquired and self.armed:
                    self.armed = False
                    # CPython can receive SIGALRM after acquire succeeds but before
                    # entering its try/finally release block. Reproduce that window.
                    os.kill(os.getpid(), signal.SIGALRM)
                return acquired
            def release(self): return self.lock.release()
            def __enter__(self): self.acquire(); return self
            def __exit__(self, *args): self.release()
        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            proxy = InterruptingLock(process._waitpid_lock)
            process._waitpid_lock = proxy; locks.append(proxy)
            return process
        def cleanup(process, deadline):
            leaked = process._waitpid_lock.lock.locked()
            # On an unfixed runner fail deterministically instead of hanging this test.
            if leaked: process._waitpid_lock.release()
            real_cleanup(process, deadline)
            self.assertFalse(leaked, 'An asynchronous signal leaked Popen._waitpid_lock')
        with mock.patch.object(study.subprocess, 'Popen', side_effect=launch), \
             mock.patch.object(study, '_kill_group', side_effect=cleanup):
            with self.assertRaises(TimeoutError): study.run_plan(self.root / 'plan/plan.json')
        self.assertEqual(len(locks), 1)
        self.assertFalse(locks[0].lock.locked())
        self.assertEqual(json.loads((self.root / 'output/status.json').read_text())['state'], 'timed_out')

    @unittest.skipUnless(os.name == 'posix', 'POSIX process-group timeout')
    def test_total_deadline_kills_sleeping_child_and_grandchild(self):
        self.execution_fixture(sleeping=True)
        start = time.monotonic()
        with mock.patch.object(study, 'WALLCLOCK_SECONDS', 0.3):
            with self.assertRaises(TimeoutError): study.run_plan(self.root / 'plan/plan.json')
        self.assertLess(time.monotonic() - start, 1.2)
        status = json.loads((self.root / 'output/status.json').read_text())
        self.assertEqual(status['state'], 'timed_out')
        self.assertEqual((self.root / 'order.txt').read_text().splitlines(), ['0'])
        time.sleep(1.6)
        self.assertFalse((self.root / 'grandchild-alive').exists())


if __name__ == '__main__':
    unittest.main()
