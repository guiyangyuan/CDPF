"""Verify downloaded files, recompute fixed metrics, inspect final checkpoints on CPU."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

parser=argparse.ArgumentParser();parser.add_argument('--backup-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args();root=a.backup_root
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'backup-manifest.json').read_text())
for f in manifest['files']:
 p=root/f['path']
 assert p.is_relative_to(root) and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['path']
run=root/'run';status=json.loads((run/'status.json').read_text())
assert status['state']=='completed'
assert len(status['stages'])==6 and all(s['status']=='passed' for s in status['stages'])
assert status['source_sha256']=='b582a30a733c735126afccf6c77f64c0d1fdec3be9fd4b34d8a227b15e7c39bf'
variants=('classname','description','contrast');arrays={};exports={};checkpoints={};correct={}
for name in variants:
 folder=run/name;record=json.loads((folder/'export/export.json').read_text())
 assert sha(folder/'export/logits.npz')==record['logits_sha256']
 with np.load(folder/'export/logits.npz',allow_pickle=False) as z: x={key:z[key].copy() for key in z.files}
 assert x['labels'].shape==(200,) and x['labels'].dtype==np.int64
 assert set(x)=={'labels','S','M','R','Z','F'}
 for k in set(x)-{'labels'}: assert x[k].shape==(200,50) and np.isfinite(x[k]).all()
 np.testing.assert_array_equal(x['labels'],[r['label'] for r in record['rows']])
 np.testing.assert_array_equal(x['Z'],x['S']+x['R'])
 np.testing.assert_array_equal(x['F'],.5*x['S']+.5*x['M'])
 counts={k:int((x[k].argmax(1)==x['labels']).sum()) for k in ('S','M','Z','F')}
 assert counts==record['correct']
 path=folder/'model/textconsistency/model.pth.tar-20'
 assert sha(path)==record['checkpoint'][0]['sha256']
 cp=torch.load(path,map_location='cpu',weights_only=False)
 assert cp['epoch']==20 and cp['optimizer'] and cp['scheduler']
 assert cp['repro_signature']==record['provenance']['model_signature']
 assert cp['run_provenance']['code_sha256']==status['source_sha256']
 spec=cp['repro_signature']['correction']
 assert spec['enabled'] and spec['mode']==name and spec['kind']=='shared_text_residual'
 assert record['provenance']['trainable_parameters']==67584
 assert tuple(cp['state_dict']['text_residual.up.weight'].shape)==(512,64)
 assert tuple(cp['state_dict']['text_residual.down.weight'].shape)==(64,512)
 assert torch.count_nonzero(cp['state_dict']['text_residual.up.weight']).item()>0
 if name=='contrast':
  assert len(record['neighbor_indices'])==50
  assert all(len(row)==5 and len(set(row))==5 and i not in row for i,row in enumerate(record['neighbor_indices']))
 lines=(run/(name+'-train.log')).read_text().splitlines()
 assert any(line.startswith('epoch [20/20] batch [50/50]') for line in lines)
 arrays[name]=x;exports[name]=record;checkpoints[name]=cp;correct[name]=counts
for name in variants:
 assert exports[name]['rows']==exports['classname']['rows']
 assert exports[name]['class_order']==exports['classname']['class_order']
 np.testing.assert_array_equal(arrays[name]['labels'],arrays['classname']['labels'])
analysis=json.loads((run/'analysis.json').read_text())
assert analysis['correct']==correct
pred=lambda name,key:arrays[name][key].argmax(1)==arrays[name]['labels']
candidate=pred('contrast','Z');comparisons={}
for name,key in (('classname','S'),('classname','F'),('classname','Z'),('description','Z')):
 reference=pred(name,key);delta=int(candidate.sum())-int(reference.sum())
 item={'gain_pp':delta/2,'corrected':int((candidate&~reference).sum()),'harmed':int((~candidate&reference).sum()),'passes_1pp':delta>=2}
 comparisons[f'contrast.Z - {name}.{key}']=item
assert comparisons==analysis['comparisons']
assert analysis['continuation_signal']==all(c['passes_1pp'] for c in comparisons.values())
student_diffs={name:{'max_abs_logit_diff':float(np.max(np.abs(arrays[name]['S']-arrays['classname']['S']))),
 'max_abs_prompt_diff':float((checkpoints[name]['state_dict']['prompt_learner.ctx']-checkpoints['classname']['state_dict']['prompt_learner.ctx']).abs().max())} for name in variants}
assert all(v['max_abs_logit_diff']==0 and v['max_abs_prompt_diff']==0 for v in student_diffs.values())
result={'status':'verified','verified_file_count':len(manifest['files'])+1,'correct':correct,
 'accuracy_pct':{name:{k:v/2 for k,v in row.items()} for name,row in correct.items()},
 'comparisons':comparisons,'continuation_signal':analysis['continuation_signal'],'student_preservation_observed':student_diffs,
 'prediction_changes':{name:int((arrays[name]['S'].argmax(1)!=arrays[name]['Z'].argmax(1)).sum()) for name in variants}, 'checkpoint_count':3,'new_training_or_gpu_inference':0,'scope':'Aircraft seed1 Base validation only'}
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
