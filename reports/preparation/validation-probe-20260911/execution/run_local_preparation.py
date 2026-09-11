"""Run nine independent CPU prepare processes with retained logs and fixed inputs."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
 if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):raise ValueError('Invalid label')
 root=Path(__file__).resolve().parents[2]; work=root/'work/validation-probe-20260911'; repo=root/'work/github-publish/textconsistency-validation'
 destination=work/f'prepared-{args.label}'
 if destination.exists():raise FileExistsError(destination)
 loc=json.loads((work/'local-input-locations.json').read_text())
 inventory=repo/'reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json'
 py=root/'work/tc-venv/bin/python'
 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 tools={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'scripts/repro').glob('validation_*.py'))}
 cases=[]
 for dataset in ('ucf101','dtd','fgvc_aircraft'):
  for seed in (1,2,3):
   out=destination/f'{dataset}-seed{seed}'
   command=[str(py),'-B',str(repo/'scripts/repro/validation_probe.py'),'prepare','--inventory',str(inventory),'--workspace-root',str(root),'--source-root',loc['source_roots'][dataset],'--data-root',loc['data_roots'][dataset],'--dataset',dataset,'--seed',str(seed),'--output-dir',str(out)]
   cases.append({'dataset':dataset,'seed':seed,'output_dir':str(out),'command':command})
 payload={'state':'running','revision':revision,'inventory_sha256':sha(inventory),'tools_sha256':tools,'cases':cases,'records':[],'new_training_runs':0,'gpu_export_run':False}
 destination.mkdir(parents=True)
 state=destination/'execution.json'
 def save():state.write_text(json.dumps(payload,indent=2)+'\n')
 save();env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':f'{repo}:{repo}/Dassl.pytorch','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 for case in cases:
  log=destination/f"{case['dataset']}-seed{case['seed']}.log"; start=time.monotonic()
  with log.open('w') as stream: result=subprocess.run(case['command'],cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
  record={'dataset':case['dataset'],'seed':case['seed'],'returncode':result.returncode,'seconds':time.monotonic()-start,'log':str(log),'log_sha256':sha(log)}
  prepared=Path(case['output_dir'])/'prepared.json'
  if prepared.is_file():record['prepared_sha256']=sha(prepared)
  payload['records'].append(record);save();print(json.dumps(record),flush=True)
  if result.returncode!=0 or not prepared.is_file():
   payload['state']='failed';save();raise SystemExit(1)
 if tools!={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'scripts/repro').glob('validation_*.py'))}:raise RuntimeError('Tool changed during preparation')
 payload['state']='nine_local_cpu_preparations_passed';save();print(str(state),flush=True)
if __name__=='__main__':main()
