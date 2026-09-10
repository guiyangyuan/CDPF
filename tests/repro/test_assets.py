import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def assets():
    path = ROOT / 'scripts/repro/prepare_assets.py'
    spec = importlib.util.spec_from_file_location('assets_tested', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


class AssetPreparation(unittest.TestCase):
    def archive(self, root, dataset, files):
        path = root / f'{dataset}.zip'
        with zipfile.ZipFile(path, 'w') as z:
            for name, content in files.items(): z.writestr(f'{dataset}/{name}', content)
        return path

    def test_dtd_and_aircraft_verified_archives_extract(self):
        module = assets()
        cases = {
            'dtd': {'split_zhou_DescribableTextures.json': '{}'},
            'fgvc_aircraft': {'variants.txt': 'a\n', 'images_variant_train.txt': '1 a\n',
                              'images_variant_val.txt': '2 a\n', 'images_variant_test.txt': '3 a\n',
                              'images/.keep': ''},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for key, files in cases.items():
                archive = self.archive(root, key, files)
                with mock.patch.object(module, 'digest', return_value=module.DATASETS[key]['sha256']):
                    destination = module.extract_dataset(archive, root / 'data', key)
                self.assertTrue(destination.is_dir())
                self.assertEqual(json.loads((destination / 'source_archive.json').read_text())['sha256'],
                                 module.DATASETS[key]['sha256'])

    def test_refuses_wrong_hash_existing_destination_and_symlink(self):
        module = assets()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); archive = self.archive(root, 'dtd', {'split_zhou_DescribableTextures.json': '{}'})
            with self.assertRaises(ValueError): module.extract_dataset(archive, root / 'data', 'dtd')
            destination = root / 'data' / 'dtd'; destination.mkdir(parents=True)
            with mock.patch.object(module, 'digest', return_value=module.DATASETS['dtd']['sha256']), self.assertRaises(FileExistsError):
                module.extract_dataset(archive, root / 'data', 'dtd')
            destination.rmdir()
            unsafe = root / 'unsafe.zip'
            with zipfile.ZipFile(unsafe, 'w') as z:
                info = zipfile.ZipInfo('dtd/link'); info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, 'target')
            with mock.patch.object(module, 'digest', return_value=module.DATASETS['dtd']['sha256']), self.assertRaises(ValueError):
                module.extract_dataset(unsafe, root / 'data', 'dtd')


if __name__ == '__main__': unittest.main()
