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


class RunnerBehavior(unittest.TestCase):
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

if __name__ == '__main__': unittest.main()
