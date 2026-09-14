"""Verify copied bytes and recompute metrics; no training or new inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--backup-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(); root = args.backup_root
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((root / 'backup-manifest.json').read_text())
for entry in manifest['files']:
    p = root / entry['path']
    assert p.is_relative_to(root) and p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], entry['path']
status = json.loads((root / 'run/status.json').read_text())
assert status['state'] == 'completed'
assert len(status['stages']) == 4 and all(x['status'] == 'passed' for x in status['stages'])
assert status['source_sha256'] == '9a50256a2067f586c21b46f6de86514daa28b3a916197df379c59e89a28a9433'
new = ('attentive', 'centered'); old = ('classname', 'description', 'contrast')
arrays, records, checkpoints, correct = {}, {}, {}, {}
for name in (*old, *new):
    folder = root / ('run/' + name + '/export' if name in new else 'context/references/' + name)
    record = json.loads((folder / 'export.json').read_text())
    assert sha(folder / 'logits.npz') == record['logits_sha256']
    with np.load(folder / 'logits.npz', allow_pickle=False) as z:
        x = {k: z[k].copy() for k in z.files}
    assert set(x) == {'labels', 'S', 'M', 'R', 'Z', 'F'}
    assert x['labels'].shape == (200,) and x['labels'].dtype == np.int64
    for key in ('S', 'M', 'R', 'Z', 'F'):
        assert x[key].shape == (200, 50) and np.isfinite(x[key]).all()
    np.testing.assert_array_equal(x['labels'], [row['label'] for row in record['rows']])
    np.testing.assert_array_equal(x['Z'], x['S'] + x['R'])
    np.testing.assert_array_equal(x['F'], .5 * x['S'] + .5 * x['M'])
    counts = {k: int((x[k].argmax(1) == x['labels']).sum()) for k in ('S', 'M', 'Z', 'F')}
    assert counts == record['correct']
    expected_source = status['source_sha256'] if name in new else '143031dbd46e1c024bf62259d64906dc07ab209bea21e2faccb82dc8cef7d6e6'
    assert record['provenance']['code_sha256'] == expected_source
    assert record['provenance']['trainable_parameters'] == 67584
    if name in new:
        path = root / 'run' / name / 'model/textconsistency/model.pth.tar-20'
        assert sha(path) == record['checkpoint'][0]['sha256']
        cp = torch.load(path, map_location='cpu', weights_only=False)
        assert cp['epoch'] == 20 and cp['scheduler']
        assert cp['repro_signature'] == record['provenance']['model_signature']
        assert cp['repro_signature']['format'] == 5
        spec = cp['repro_signature']['correction']
        assert spec['kind'] == 'classname_anchored_visual_text_detail' and spec['detail_pooling'] == name
        assert cp['optimizer']['format'] == 'student_sgd_residual_adamw_v1'
        assert cp['optimizer']['student']['state'] and cp['optimizer']['residual']['state']
        assert all(float(s['step']) == 1000 for s in cp['optimizer']['residual']['state'].values())
        assert cp['optimizer']['residual']['param_groups'][0]['lr'] == .001
        assert cp['state_dict']['text_residual.up.weight'].shape == (512, 64)
        assert cp['state_dict']['text_residual.up.weight'].count_nonzero() > 0
        lines = (root / 'run' / (name + '-train.log')).read_text().splitlines()
        assert any(l.startswith('epoch [20/20] batch [50/50]') for l in lines)
        checkpoints[name] = cp
    arrays[name], records[name], correct[name] = x, record, counts
for name in arrays:
    assert records[name]['rows'] == records['classname']['rows']
    assert records[name]['class_order'] == records['classname']['class_order']
    for key in ('S', 'M', 'F'):
        np.testing.assert_array_equal(arrays[name][key], arrays['classname'][key])
torch.testing.assert_close(checkpoints['attentive']['state_dict']['prompt_learner.ctx'],
                           checkpoints['centered']['state_dict']['prompt_learner.ctx'], rtol=0, atol=0)
analysis = json.loads((root / 'run/analysis.json').read_text())
assert correct == analysis['correct']
pred = lambda name, branch: arrays[name][branch].argmax(1) == arrays[name]['labels']
def paired(candidate, controls):
    result = {}; p = pred(candidate, 'Z')
    for name, branch in controls:
        r = pred(name, branch); delta = int(p.sum()) - int(r.sum())
        result[f'{candidate}.Z - {name}.{branch}'] = {'gain_pp': delta / 2,
            'corrected': int((p & ~r).sum()), 'harmed': int((~p & r).sum()), 'passes_1pp': delta >= 2}
    return result
controls = [('classname', 'S'), ('classname', 'F'), ('classname', 'Z'), ('description', 'Z'), ('contrast', 'Z')]
primary = paired('centered', controls + [('attentive', 'Z')]); secondary = paired('attentive', controls)
assert primary == analysis['comparisons'] and secondary == analysis['attentive_comparisons']
assert analysis['continuation_signal'] == all(x['passes_1pp'] for x in primary.values())
assert analysis['attentive_signal'] == all(x['passes_1pp'] for x in secondary.values())
result = {'status': 'verified', 'verified_file_count': len(manifest['files']) + 1,
    'new_model_checkpoints': 2, 'new_training_or_inference': 0, 'correct': correct,
    'accuracy_pct': {n: {k: v / 2 for k, v in row.items()} for n, row in correct.items()},
    'comparisons': primary, 'attentive_comparisons': secondary,
    'continuation_signal': analysis['continuation_signal'], 'attentive_signal': analysis['attentive_signal'],
    'historical_S_M_F_max_abs_diff': 0, 'between_new_models_student_prompt_max_abs_diff': 0,
    'scope': 'Aircraft seed1 Base development validation, no Novel/test inference'}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
