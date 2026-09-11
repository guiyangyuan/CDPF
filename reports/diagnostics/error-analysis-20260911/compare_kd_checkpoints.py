"""Compare only hash-verified local experiment tensors, without a CLIP forward."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--history-receipt', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert not args.output.exists()
    receipt = json.loads(args.receipt.read_text())
    history_receipt = json.loads(args.history_receipt.read_text())
    assert receipt['state'] == 'local_complete_backup_verified'
    assert history_receipt['all_24_checkpoint_hashes_verified']
    assert sha(args.study/'protocol.json') == receipt['study_file_sha256']['protocol.json']
    protocol = json.loads((args.study/'protocol.json').read_text())
    historical = {(x['dataset'],x['seed']):x for x in protocol['receipt']['historical_kd1']}
    history_hashes = {x['path']:x['sha256'] for x in history_receipt['checkpoints']}
    pairs = []
    def load(path, expected):
        actual = sha(path)
        assert actual == expected == path.with_name(path.name+'.sha256').read_text().strip()
        # Own experiment files, authenticated against previously verified backup receipts.
        # No downloaded/unverified pickle, no model construction and no device transfer.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['epoch'] == 20
        assert all(torch.is_tensor(v) and torch.isfinite(v).all() for v in payload['state_dict'].values())
        return payload
    for dataset in ('dtd','fgvc_aircraft'):
        for seed in (1,2,3):
            new_rel = f'formal_{dataset}_kd0/hierarchical_adapter/seed{seed}/textconsistency/model.pth.tar-20'
            old_rel = f'runs/crossdataset-study-20260910-v1/formal/{dataset}/hierarchical_adapter/seed{seed}/textconsistency/model.pth.tar-20'
            p0,p1 = args.study/new_rel,args.history/old_rel
            h0,h1 = receipt['study_file_sha256'][new_rel],history_hashes[old_rel]
            assert h1 == historical[dataset,seed]['checkpoint_sha256']
            a,b = load(p0,h0),load(p1,h1)
            assert a['repro_signature'] == b['repro_signature']
            for field in ('seed','dataset','shots','code_sha256','train_samples_sha256',
                          'val_samples_sha256','test_samples_sha256','trainable_parameters'):
                assert a['run_provenance'][field] == b['run_provenance'][field]
            assert a['run_provenance']['seed'] == seed
            assert a['run_provenance']['effective_losses']['kd'] == 0
            assert b['run_provenance']['effective_losses']['kd'] == 1
            states = a['state_dict'],b['state_dict']
            assert set(states[0]) == set(states[1])
            groups = {}
            for prefix in ('prompt_learner.','view_aggregator.'):
                keys = [k for k in states[0] if k.startswith(prefix) and not k.endswith(('token_prefix','token_suffix'))]
                assert keys
                details = []
                for key in keys:
                    x,y = states[0][key],states[1][key]
                    assert x.shape == y.shape
                    details.append({'name':key,'numel':x.numel(),'exact_equal':torch.equal(x,y),
                                    'max_abs_difference':(x-y).abs().max().item(),
                                    'l2_difference':(x-y).double().square().sum().sqrt().item()})
                groups[prefix.rstrip('.')] = {'numel':sum(x['numel'] for x in details),
                    'all_tensors_exact_equal':all(x['exact_equal'] for x in details),
                    'max_abs_difference':max(x['max_abs_difference'] for x in details),'tensors':details}
            assert groups['prompt_learner']['numel']==2048
            assert groups['view_aggregator']['numel']==787970
            assert sha(p0)==h0 and sha(p1)==h1
            pairs.append({'dataset':dataset,'seed':seed,'kd0_relative_path':new_rel,'kd1_relative_path':old_rel,
                          'kd0_sha256':h0,'kd1_sha256':h1,'groups':groups})
    files = [args.repo/'train.py']
    for directory in ('trainers','datasets','clip','Dassl.pytorch/dassl'):
        files.extend(sorted((args.repo/directory).rglob('*.py')))
    source = hashlib.sha256()
    for path in files:
        source.update(str(path.relative_to(args.repo)).encode()); source.update(path.read_bytes())
    assert source.hexdigest() == '9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba'
    report = {'passed':True,'method':'CPU tensor comparison only; no CLIP construction/inference/training',
        'torch':str(torch.__version__),'model_source_sha256':source.hexdigest(),
        'receipt_sha256':sha(args.receipt),'history_receipt_sha256':sha(args.history_receipt),
        'pairs':pairs,'teacher_exact_equal_pairs':sum(p['groups']['view_aggregator']['all_tensors_exact_equal'] for p in pairs),
        'student_changed_pairs':sum(not p['groups']['prompt_learner']['all_tensors_exact_equal'] for p in pairs),
        'limitation':'Equal final teacher weights do not measure student-teacher error correlation, probability calibration or training gradients.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},indent=2))


if __name__=='__main__':
    main()
