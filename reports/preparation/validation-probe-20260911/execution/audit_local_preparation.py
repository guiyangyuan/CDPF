"""Independent controller acceptance checks over nine real CPU receipts."""
from pathlib import Path
import argparse,csv,hashlib,json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepared-root',type=Path,required=True);parser.add_argument('--snapshot-after',type=Path,required=True);args=parser.parse_args()
 root=Path(__file__).resolve().parents[2];repo=root/'work/github-publish/textconsistency-validation'; folder=args.prepared_root.resolve()
 output=folder/'local-audit.json'
 if output.exists():raise FileExistsError(output)
 execution=json.loads((folder/'execution.json').read_text()); assert execution['state']=='nine_local_cpu_preparations_passed'
 assert len(execution['records'])==9 and all(x['returncode']==0 for x in execution['records'])
 expected={(d,s) for d in ('ucf101','dtd','fgvc_aircraft') for s in (1,2,3)}
 assert {(x['dataset'],x['seed']) for x in execution['records']}==expected
 history=json.loads((root/'work/validation-probe-20260911/historical-sampling-check.json').read_text()); historical={(x['dataset'],x['seed']):x for x in history['cases']}
 count_expected={'ucf101':(51,816,204),'dtd':(24,384,96),'fgvc_aircraft':(50,800,200)}
 rows=[];unique=set();metadata={}
 for record in execution['records']:
  d,s=record['dataset'],record['seed'];p=folder/f'{d}-seed{s}'/'prepared.json';x=json.loads(p.read_text())
  assert sha(p)==record['prepared_sha256'] and sha(Path(record['log']))==record['log_sha256']
  assert x['status']=='prepared' and (x['dataset_key'],x['seed'])==(d,s)
  proto=x['protocol'];assert (proto['split'],proto['subsample_classes'],proto['shots'],proto['val_shots'],proto['epoch'],proto['alpha'],proto['workers'],proto['sampler'])==('val','base',16,4,20,.5,0,'SequentialSampler')
  assert x['environment']['device']=='cpu'
  assert x['tool']['validation_probe_sha256']==execution['tools_sha256']['scripts/repro/validation_probe.py']
  assert (len(x['samples']['classes']),len(x['samples']['train']),len(x['samples']['val']))==count_expected[d]
  for split in ('train','val'):assert x['sample_hashes'][split]==historical[d,s][split+'_sha256']
  tr={str((Path(x['data_root'])/y['sample_id']).resolve()) for y in x['samples']['train']};val={str((Path(x['data_root'])/y['sample_id']).resolve()) for y in x['samples']['val']}
  assert not tr&val and len(tr)==len(x['samples']['train']) and len(val)==len(x['samples']['val']);unique.update(val)
  for item in x['samples']['val']:assert x['samples']['classes'][item['label']]==item['classname']
  decoded=x['transform']['decoded_val'];assert decoded['count']==len(val) and decoded['tensor_shape']==[3,224,224] and decoded['tensor_dtype']=='torch.float32'
  assert set(x['checkpoints'])=={'coop','kd0','kd1','teacher_tensor_exact_equal','teacher_tensor_keys'}
  assert x['checkpoints']['teacher_tensor_exact_equal']==x['teacher_tensor_exact_equal']
  assert x['checkpoints']['teacher_tensor_keys'] and all(k.startswith('view_aggregator.') for k in x['checkpoints']['teacher_tensor_keys'])
  case=x['inventory']['case'];assert x['source_sha256']==case['checkpoints']['coop']['model_source_sha256']
  for role in ('coop','kd0','kd1'):
   c=x['checkpoints'][role]
   assert c['checkpoint_sha256']==case['checkpoints'][role]['sha256'] and c['manifest_sha256']==case['checkpoints'][role]['manifest_sha256']
  rows.append({'dataset':d,'seed':s,'classes':len(x['samples']['classes']),'train_count':len(tr),'val_count':len(val),'decoded_count':decoded['count'],'teacher_tensor_exact_equal':x['teacher_tensor_exact_equal'],'val_samples_sha256':x['sample_hashes']['val'],'transformed_val_sha256':decoded['ordered_transform_sha256'],'prepared_sha256':sha(p)})
  metadata[str(p.relative_to(folder))]=sha(p)
 assert sum(x['val_count'] for x in rows)==1500 and len(unique)==1280
 no_cuda=json.loads((folder/'cuda-unavailable-check.json').read_text());assert no_cuda['passed'] and no_cuda['no_logits_or_success_export'] and no_cuda['returncode']!=0
 unchanged=json.loads(args.snapshot_after.read_text());assert unchanged['unchanged'] is True
 payload={'state':'cpu_checks_passed_review_pending','gpu_export_run':False,'new_training_runs':0,'revision':execution['revision'],'tools_sha256':execution['tools_sha256'],'validation_cases':9,'model_count':27,'val_records':1500,'unique_val_images':len(unique),'teacher_exact_equal_pairs':sum(x['teacher_tensor_exact_equal'] for x in rows),'prepare_total_seconds':sum(x['seconds'] for x in execution['records']),'cases':rows,'receipts_sha256':metadata,'source_data_and_checkpoint_inputs_unchanged':True,'snapshot_after_sha256':sha(args.snapshot_after),'cuda_unavailable_guard_passed':True,'limitations':['No actual CUDA model construction/forward parity or prediction run','Base validation is known-class screening, not Novel/HM confirmation','Image immutability snapshot uses path/size/mtime, not all-image content hashes']}
 output.write_text(json.dumps(payload,indent=2)+'\n')
 with (folder/'cases.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 print(json.dumps({k:v for k,v in payload.items() if k not in ('cases','receipts_sha256','tools_sha256')},indent=2))
if __name__=='__main__':main()
