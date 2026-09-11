"""Exercise the export CLI's real no-CUDA refusal; never start CPU inference."""
from pathlib import Path
import argparse,os,subprocess,json,hashlib,time
p=argparse.ArgumentParser();p.add_argument('--prepared-root',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];repo=root/'work/github-publish/textconsistency-validation';parent=a.prepared_root.resolve()
out=parent/'cuda-unavailable-rejection'
if out.exists():raise FileExistsError(out)
command=[str(root/'work/tc-venv/bin/python'),'-B',str(repo/'scripts/repro/validation_probe.py'),'export','--prepared',str(parent/'ucf101-seed1/prepared.json'),'--output-dir',str(out)]
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':f'{repo}:{repo}/Dassl.pytorch','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
start=time.monotonic();log=parent/'cuda-unavailable-rejection.log'
with log.open('w') as stream:r=subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
failure=json.loads((out/'failure.json').read_text())
assert r.returncode!=0 and 'CUDA' in failure['error']
assert not (out/'export.json').exists() and not (out/'logits.npz').exists()
receipt={'passed':True,'check':'CUDA unavailable refuses real model export','returncode':r.returncode,'seconds':time.monotonic()-start,'command':command,'failure':failure,'no_logits_or_success_export':True,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
(parent/'cuda-unavailable-check.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
