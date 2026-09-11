"""CPU-only real metadata checks; run each frozen source in a fresh process."""
from pathlib import Path
import importlib.util
import json
import sys
import time
root=Path('/root/autodl-tmp/textconsistency')
prep=root/'setup-logs/attribution-prep-20260911-v1'
script=prep/'fusion-evaluator-92b01a1/evaluate_frozen_fusion.py'
spec=importlib.util.spec_from_file_location('frozen_evaluator',script)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
inputs=json.loads((root/'reused-inputs/attribution-20260911/input-receipt.json').read_text())
group=sys.argv[1]
rows=[x for x in inputs['fusion_baselines'] if (x['dataset']=='ucf101')==(group=='ucf101')]
assert len(rows)==(3 if group=='ucf101' else 6)
results=[]
started=time.time()
for row in rows:
 for split in ('base','new'):
  output=root/'runs/attribution-study-20260911-v1/fusion'/row['dataset']/f"seed{row['seed']}"/split
  args=m.build_parser().parse_args(['--source-root',row['source_root'],'--baseline-dir',row['baseline_dir'],'--dataset',row['dataset'],'--seed',str(row['seed']),'--split',split,'--data-root',str(root/'data'),'--output-dir',str(output),'--expected-source-sha256',row['source_sha256'],'--validate-only'])
  result=m.evaluate(args)
  assert result['status']=='validated_metadata' and not output.exists()
  result['output_created']=False;results.append(result)
  print(json.dumps({'dataset':row['dataset'],'seed':row['seed'],'split':split,'status':result['status']}),flush=True)
payload={'state':'real_fusion_metadata_passed','group':group,'cases':results,'duration_seconds':time.time()-started,'new_gpu_evaluation_complete':False,'new_study_exists':(root/'runs/attribution-study-20260911-v1').exists()}
assert not payload['new_study_exists']
(prep/f'fusion-metadata-{group}.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({'state':payload['state'],'cases':len(results)}),flush=True)
