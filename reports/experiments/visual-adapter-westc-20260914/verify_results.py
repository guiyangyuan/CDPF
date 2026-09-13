"""Recompute all metrics from backed-up logits; no project model/export code imported."""
from pathlib import Path
import json,hashlib,copy,re,datetime,argparse
import numpy as np
import torch
import yaml
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backup-root',type=Path,required=True,help='Extracted archive directory containing run/ and context/')
parser.add_argument('--output',type=Path,required=True,help='Fresh JSON audit receipt path')
args=parser.parse_args()
backup=args.backup_root.resolve()
run=backup/'run';context=backup/'context'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(v,key):
    v=copy.deepcopy(v);v.pop(key,None)
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
plan=read(context/'plan.json');prepared=read(context/'prepared.json');status=read(run/'status.json');analysis=read(run/'analysis.json')
assert sha(context/'plan.json')=='8f2c65b4130021d62b66e0c33920c1110b0e8af293c58d69167bdd2928eebc5c'
assert sha(context/'prepared.json')==plan['prepared_sha256']
assert plan['plan_identity_sha256']==identity(plan,'plan_identity_sha256')
assert prepared['prepared_identity_sha256']==identity(prepared,'prepared_identity_sha256')
assert status['state']=='completed' and len(status['stages'])==len(plan['stages'])==6
assert status['elapsed_seconds']<=3600
for a,b in zip(status['stages'],plan['stages']):
    assert a['stage_id']==b['stage_id'] and a['argv']==b['argv'] and a['kind']==b['kind']
    assert a['state']=='completed' and a['returncode']==0
variants=('simple_fusion','visual_only','visual_description')
labels=np.array([r['label'] for r in prepared['samples']['val']],dtype=np.int64)
assert len(labels)==200 and len(prepared['class_order'])==50
assert len(set(r['sample_id'] for r in prepared['samples']['val']))==200
metrics={};masks={};models={};states={}
for variant in variants:
    d=run/variant;export=read(d/'export/export.json');manifest=read(d/'model/run_manifest.json')
    assert export['export_identity_sha256']==identity(export,'export_identity_sha256')
    assert export['variant']==variant and export['status']=='exported' and export['device']=='cuda'
    for key,value in [('protocol',prepared['protocol']),('source_sha256',prepared['source_sha256']),('class_order',prepared['class_order']),('rows',prepared['samples']['val']),('prepared_identity_sha256',prepared['prepared_identity_sha256'])]:
        assert export[key]==value,(variant,key)
    npz=d/'export/logits.npz';assert sha(npz)==export['logits_sha256']
    with np.load(npz,allow_pickle=False) as values:
        assert set(values.files)=={'labels','S','D','F'}
        arrays={k:values[k].copy() for k in values.files}
    assert arrays['labels'].dtype==np.int64 and np.array_equal(arrays['labels'],labels)
    assert np.array_equal(arrays['F'],np.float32(.5)*arrays['S']+np.float32(.5)*arrays['D'])
    metrics[variant]={};masks[variant]={}
    for branch in ('S','D','F'):
        x=arrays[branch];assert x.dtype==np.float32 and x.shape==(200,50) and np.isfinite(x).all()
        correct=x.argmax(axis=1)==labels;count=int(correct.sum());masks[variant][branch]=correct
        m={'correct':count,'total':200,'accuracy':count/200,'accuracy_percent':count/2}
        assert analysis['variants'][variant][branch]==m
        metrics[variant][branch]=m
    model=d/'model/textconsistency/model.pth.tar-20'
    assert sha(model)==export['checkpoint']['sha256']==model.with_name(model.name+'.sha256').read_text().strip()
    assert sha(d/'model/run_manifest.json')==export['checkpoint']['manifest_sha256']
    assert sha(d/'model/effective_config.yaml')==export['checkpoint']['config_sha256']
    expected_config=copy.deepcopy(prepared['cases'][variant]['config'])
    expected_config['OUTPUT_DIR']=str(Path(plan['output_root'])/variant/'model')
    assert yaml.safe_load((d/'model/effective_config.yaml').read_text())==expected_config
    ckpt=torch.load(model,map_location='cpu',weights_only=False)
    assert ckpt['epoch']==20 and ckpt['run_provenance']==manifest
    assert ckpt['repro_signature']==prepared['cases'][variant]['model_signature']
    assert manifest['model_signature']==ckpt['repro_signature']
    assert manifest['code_sha256']==prepared['source_sha256']
    assert manifest['seed']==1 and manifest['shots']==16 and manifest['subsample']=='base'
    assert manifest['trainable_parameters']==(2048 if variant==variants[0] else 67584)
    assert manifest['effective_losses']=={'kd':0.,'adaptive_ce':1. if variant==variants[2] else 0.,'diversity':0.,'contrastive':0.}
    for split in ('train','val','test'):assert manifest[f'{split}_samples_sha256']==prepared['sample_hashes'][split]
    sd=ckpt['state_dict'];states[variant]=sd
    assert all(torch.isfinite(x).all().item() for x in sd.values())
    log=(d/'model/log.txt').read_text()
    epochs={int(x) for x in re.findall(r'epoch \[(\d+)/20\]',log)}
    assert epochs==set(range(1,21)),(variant,epochs)
    assert 'epoch [20/20] batch [50/50]' in log
    models[variant]={'checkpoint_sha256':sha(model),'epoch':20,'all_20_epochs_logged':True,'trainable_parameters':manifest['trainable_parameters']}
    if variant!=variants[0]:
        up=sd['visual_adapter.up.weight'];down=sd['visual_adapter.down.weight']
        assert up.shape==(512,64) and down.shape==(64,512) and torch.count_nonzero(up).item()>0
        models[variant]['adapter_up_l2']=torch.linalg.vector_norm(up).item()
        models[variant]['adapter_down_l2']=torch.linalg.vector_norm(down).item()
comparisons={}
for name,new,old in [('B-A',variants[1],variants[0]),('C-A',variants[2],variants[0]),('C-B',variants[2],variants[1])]:
    comparisons[name]={}
    for branch in ('S','D','F'):
        n,o=masks[new][branch],masks[old][branch]
        fixes=int((n&~o).sum());harms=int((~n&o).sum())
        values={'correct_difference':fixes-harms,'accuracy_difference_pp':(fixes-harms)/2,'corrections':fixes,'harms':harms,'both_correct':int((n&o).sum()),'both_wrong':int((~n&~o).sum())}
        assert analysis['comparisons'][name][branch]==values
        comparisons[name][branch]=values
passed=all(comparisons[name]['F']['correct_difference']>=2 for name in ('C-A','C-B'))
assert analysis['continuation']['passed']==passed
assert analysis['analysis_identity_sha256']==identity(analysis,'analysis_identity_sha256')
assert status['continuation']==analysis['continuation']
bc_difference={k:torch.linalg.vector_norm(states[variants[2]][k]-states[variants[1]][k]).item() for k in ('visual_adapter.down.weight','visual_adapter.up.weight','prompt_learner.ctx')}
assert bc_difference['visual_adapter.up.weight']>0
result={'state':'passed','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'independent NumPy logits recomputation and CPU checkpoint/provenance inspection; no model or study helper imported','source_commit':'1a3830754dbffa87f06348e842e68a07fb4054c7','training_source_sha256':prepared['source_sha256'],'plan_sha256':sha(context/'plan.json'),'stage_count':6,'new_training_runs':3,'exports':3,'validation_rows_per_variant':200,'metrics':metrics,'comparisons':comparisons,'models':models,'B_C_checkpoint_l2_differences':bc_difference,'continuation_passed':passed,'elapsed_seconds':status['elapsed_seconds'],'no_novel_evaluation':True,'no_additional_gpu_work':True}
target=args.output.resolve();assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
