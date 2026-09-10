"""CPU numeric regression against the preserved pre-control source snapshot."""
import hashlib
import json
from pathlib import Path
import subprocess
import types
import torch
from trainers.textconsistency_common import ViewAggregator

root = Path.cwd()
source = subprocess.check_output(['git', 'show', 'f30395c:trainers/textconsistency_common.py'], text=True)
old = types.ModuleType('old_common')
old.__file__ = str(root / 'trainers/textconsistency_common.py')
exec(compile(source, old.__file__, 'exec'), old.__dict__)
torch.set_num_threads(2)
cases = []
for seed in (1, 2, 3):
    for mode, adapter in [('mean', False), ('flat', False), ('flat', True), ('hierarchical', False), ('hierarchical', True)]:
        torch.manual_seed(seed)
        first = old.ViewAggregator(32, 7, mode, 3, adapter)
        after_first = torch.get_rng_state().clone()
        torch.manual_seed(seed)
        second = ViewAggregator(32, 7, mode, 3, adapter)
        assert torch.equal(after_first, torch.get_rng_state())
        for key, value in first.state_dict().items():
            assert torch.equal(value, second.state_dict()[key]), key
        image, views = torch.randn(4, 32), torch.randn(3, 7, 32)
        mask = torch.tensor([[1]*7, [1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        one, attn_one = first(image, views, mask)
        two, attn_two = second(image, views, mask)
        assert torch.equal(one, two)
        for key in attn_one:
            assert torch.equal(attn_one[key], attn_two[key])
        if mode != 'mean':
            (one * torch.arange(32)).sum().backward()
            (two * torch.arange(32)).sum().backward()
            for (key, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
                assert torch.equal(a.grad, b.grad), key
        cases.append({'seed': seed, 'aggregation': mode, 'adapter': adapter,
                      'initial_state_max_abs_diff': 0, 'output_max_abs_diff': 0,
                      'gradient_max_abs_diff': 0 if mode != 'mean' else None, 'rng_equal': True})
capacity = {}
for mode in ('hierarchical', 'flat_dual'):
    torch.manual_seed(1)
    model = ViewAggregator(512, 20, mode, 4, True)
    capacity[mode] = {'aggregator': sum(p.numel() for p in model.parameters()),
                      'with_coop_prompt': sum(p.numel() for p in model.parameters()) + 4*512}
    if mode == 'hierarchical':
        first = model
    else:
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in first.state_dict().items())
assert capacity['hierarchical'] == capacity['flat_dual'] == {'aggregator': 787970, 'with_coop_prompt': 790018}
print(json.dumps({'passed': True, 'scope': 'CPU synthetic aggregator regression; not benchmark/GPU evidence',
                  'reference_commit': 'f30395c8659e3371fcc97e36521cf486f2f14c55',
                  'reference_common_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'cases': cases, 'capacity': capacity, 'matched_initial_state_equal': True}, indent=2))
