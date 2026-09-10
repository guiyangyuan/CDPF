"""Behavior regressions identified in the September paper/code audit."""
import ast
import os.path as osp
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]


def source_symbol(filename, class_name, method=None):
    """Load real source without constructing CLIP or importing dataset plugins."""
    tree = ast.parse((ROOT / 'trainers' / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method) if method else cls
    namespace = {'osp': osp}
    try:
        import torch
        namespace.update(torch=torch, nn=torch.nn, F=torch.nn.functional)
    except ImportError:
        pass
    if (ROOT / 'trainers/textconsistency_common.py').exists():
        import trainers.textconsistency_common as common
        namespace.update({name: getattr(common, name) for name in dir(common) if not name.startswith('_')})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(ROOT / 'trainers' / filename), 'exec'), namespace)
    return namespace[method or class_name]


TRAINERS = [('coop_textconsistency_v2_fg.py', 'CoOp_TextConsistency_V2_FG'),
            ('cocoop_textconsistency_v2_fg.py', 'CoCoOp_TextConsistency_V2_FG'),
            ('coop_textconsistency_ablation.py', 'CoOp_TextConsistency_Ablation')]


class ReproductionRegressions(unittest.TestCase):
    def test_missing_checkpoint_must_stop_evaluation(self):
        for filename, cls in TRAINERS:
            with self.subTest(trainer=cls), tempfile.TemporaryDirectory() as directory:
                loader = source_symbol(filename, cls, 'load_model')
                trainer = types.SimpleNamespace(get_model_names=lambda: ['textconsistency'])
                with self.assertRaises(FileNotFoundError):
                    loader(trainer, directory, epoch=42)

    def test_prediction_rule_does_not_depend_on_base_or_novel_label(self):
        for filename, cls in TRAINERS[:2]:
            method = source_symbol(filename, cls, 'model_inference')
            values = []
            for split in ('base', 'new'):
                tc = types.SimpleNamespace(INFERENCE_BRANCH='ensemble', ENSEMBLE_ALPHA=0.5)
                cfg = types.SimpleNamespace(DATASET=types.SimpleNamespace(SUBSAMPLE_CLASSES=split),
                                            TRAINER=types.SimpleNamespace(TEXTCONSISTENCY=tc))
                trainer = types.SimpleNamespace(cfg=cfg, model=lambda *a, **kw: (2., 10., None, None))
                values.append(method(trainer, None))
            self.assertEqual(values, [6., 6.])

    def test_last_view_is_not_discarded_when_group_count_does_not_divide_views(self):
        import torch
        cls = source_symbol('coop_textconsistency_v2_fg.py', 'HierarchicalViewAggregator')
        model = cls(dim=8, n_views=5, n_groups=2)
        # Uniform attention isolates view coverage from learned weighting.
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        query = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])
        views = torch.randn(2, 5, 8, requires_grad=True)
        features, _ = model(query, views)
        features[..., 0].sum().backward()
        self.assertGreater(views.grad[:, -1, :].abs().sum().item(), 0.)

if __name__ == '__main__':
    unittest.main()
