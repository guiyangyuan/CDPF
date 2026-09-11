"""Freeze local inputs for a proposed validation-only probe; no model execution."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); root=args.workspace.resolve()
    assert not args.output.exists()
    repo=root/'work/github-publish/textconsistency_v2_tune'
    outputs=root/'outputs'
    newroot=outputs/'textconsistency-attribution-study-20260911-42527'
    study=newroot/'unpacked/attribution-study-20260911-v1'
    receipt=json.loads((newroot/'local-backup-receipt.json').read_text())
    assert sha(study/'protocol.json')==receipt['study_file_sha256']['protocol.json']
    protocol=json.loads((study/'protocol.json').read_text())
    baselines={(r['dataset'],r['seed']):r for r in protocol['receipt']['fusion_baselines']}
    old_kd_root=outputs/'textconsistency-kd-study-20260910-38402'
    old_kd_receipt=json.loads((repo/'reports/experiments/ucf101-kd-study-20260910/backup-receipt.json').read_text())
    archive=old_kd_root/'ucf101-kd-study-20260910-38402.tar.gz'
    assert sha(archive)==old_kd_receipt['sha256']
    ucf_hashes={}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if '/hierarchical_adapter/' in member.name and member.name.endswith('model.pth.tar-20'):
                assert member.isfile()
                ucf_hashes[member.name]=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
    assert len(ucf_hashes)==6
    history_root=outputs/'textconsistency-crossdataset-study-20260911-55424'
    old_hashes={(r['dataset'],r['seed']):r['checkpoint_sha256'] for r in protocol['receipt']['historical_kd1']}
    cases=[]
    for dataset in ('ucf101','dtd','fgvc_aircraft'):
        for seed in (1,2,3):
            if dataset=='ucf101':
                # Historical consolidated directories contain symlinks to the original pilot runs.
                candidates=sorted({p.resolve() for p in (outputs/'textconsistency-ucf101-s123-20260910-163256/runs').glob(f'*/baseline/seed{seed}/textconsistency/model.pth.tar-20')})
                assert len(candidates)==1
                paths={'coop':candidates[0]}
                expected={'coop':baselines[dataset,seed]['checkpoint_sha256']}
                for kd in (0,1):
                    rel=f'ucf101-kd-study-20260910-38402/kd{kd}/hierarchical_adapter/seed{seed}/textconsistency/model.pth.tar-20'
                    paths[f'kd{kd}']=old_kd_root/'unpacked'/rel
                    expected[f'kd{kd}']=ucf_hashes[rel]
            else:
                common=f'runs/crossdataset-study-20260910-v1/formal/{dataset}'
                new=f'formal_{dataset}_kd0/hierarchical_adapter/seed{seed}/textconsistency/model.pth.tar-20'
                paths={'coop':history_root/common/f'baseline/seed{seed}/textconsistency/model.pth.tar-20',
                       'kd0':study/new,'kd1':history_root/common/f'hierarchical_adapter/seed{seed}/textconsistency/model.pth.tar-20'}
                expected={'coop':baselines[dataset,seed]['checkpoint_sha256'],'kd0':receipt['study_file_sha256'][new],'kd1':old_hashes[dataset,seed]}
            refs={}; manifests={}
            for role,path in paths.items():
                digest=sha(path)
                assert digest==expected[role]==path.with_name(path.name+'.sha256').read_text().strip()
                manifest_path=path.parent.parent/'run_manifest.json'
                m=json.loads(manifest_path.read_text()); manifests[role]=m
                assert m['seed']==seed and m['shots']==16 and m['subsample']=='base'
                refs[role]={'workspace_relative_checkpoint':str(path.relative_to(root)),'sha256':digest,
                    'manifest_sha256':sha(manifest_path),'model_source_sha256':m['code_sha256'],
                    'effective_kd_weight':m['effective_losses']['kd'],'epoch':20}
            for key in ('dataset','train_samples_sha256','val_samples_sha256','test_samples_sha256','code_sha256'):
                assert len({m[key] for m in manifests.values()})==1,(dataset,seed,key)
            assert len({m['model_signature']['clip_sha256'] for m in manifests.values()})==1
            # CoOp did not use descriptive prompts during training; its null is not a mismatch.
            assert manifests['coop']['model_signature']['template_sha256'] is None
            template_hash=manifests['kd0']['model_signature']['template_sha256']
            assert template_hash==manifests['kd1']['model_signature']['template_sha256'] and template_hash
            fusion_metric=study/f'fusion_{dataset}'/f'seed{seed}'/'base/metrics.json'
            assert sha(fusion_metric)==receipt['study_file_sha256'][str(fusion_metric.relative_to(study))]
            assert json.loads(fusion_metric.read_text())['assets']['template_sha256']==template_hash
            assert [refs[role]['effective_kd_weight'] for role in ('coop','kd0','kd1')]==[0,0,1]
            cases.append({'dataset':dataset,'seed':seed,'split':'val','candidate_classes':'base',
                'train_samples_sha256':manifests['coop']['train_samples_sha256'],
                'val_samples_sha256':manifests['coop']['val_samples_sha256'],
                'clip_sha256':manifests['coop']['model_signature']['clip_sha256'],
                'template_sha256':template_hash,'coop_training_template_sha256':None,'checkpoints':refs})
    assert len(cases)==9
    payload={'state':'local_input_inventory_verified_not_gpu_ready','model_count':27,'validation_cases':9,
        'old_ucf_archive_sha256':old_kd_receipt['sha256'],'cases':cases,
        'pending':['Independent evaluation exporter implementation and tests',
                   'Real validation loader/transform hashes and no train-val image overlap check',
                   'Server deployment and CUDA preflight only after user opens server'],
        'note':'A manifest hash match is not proof of train/val disjointness or unseen-class validation; do not use novel test labels to select settings.'}
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
