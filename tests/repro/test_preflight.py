import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load_preflight():
    path = ROOT / 'scripts/repro/preflight.py'
    spec = importlib.util.spec_from_file_location('preflight_tested', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


class PreflightBehavior(unittest.TestCase):
    def test_missing_fixed_zhou_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for dataset in ('ucf101', 'dtd'):
                with self.subTest(dataset=dataset), self.assertRaises(FileNotFoundError):
                    load_preflight().check(temp, dataset=dataset)

    def test_missing_aircraft_official_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'fgvc_aircraft'; root.mkdir()
            with self.assertRaises(FileNotFoundError):
                load_preflight().check(temp, dataset='fgvc_aircraft')

    def test_unknown_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            load_preflight().check(temp, dataset='unknown')


if __name__ == '__main__': unittest.main()
