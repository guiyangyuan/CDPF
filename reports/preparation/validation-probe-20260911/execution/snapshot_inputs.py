"""Operational before/after snapshot; no modification of experiment inputs."""
from pathlib import Path
import argparse, hashlib, json

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--compare',type=Path);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 root=Path(__file__).resolve().parents[2]
 loc=json.loads((root/'work/validation-probe-20260911/local-input-locations.json').read_text())
 groups={f'source_{k}':Path(v) for k,v in loc['source_roots'].items() if k!='fgvc_aircraft'}
 groups.update({f'data_{k}':Path(loc['data_roots'][k])/k for k in loc['data_roots']})
 result={}
 for name,folder in groups.items():
  entries=[]
  for f in sorted(folder.rglob('*')):
   if not f.is_file() or '.git' in f.parts:continue
   st=f.stat();row={'path':str(f.relative_to(folder)),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
   if f.is_symlink():row['symlink_target']=str(f.resolve())
   if f.suffix.lower() not in ('.jpg','.jpeg','.png','.pt'):row['sha256']=sha(f)
   entries.append(row)
  result[name]={'file_count':len(entries),'entries_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest(),'entries':entries}
 inv=json.loads((root/'work/github-publish/textconsistency_v2_tune/reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json').read_text())
 cps={}
 for case in inv['cases']:
  for role,item in case['checkpoints'].items():
   path=root/item['workspace_relative_checkpoint']; paths=[path,path.with_name(path.name+'.sha256'),path.parent.parent/'run_manifest.json',path.parent.parent/'effective_config.yaml']
   for f in paths:cps[str(f.relative_to(root))]=sha(f)
 result['checkpoint_inputs']=cps
 payload={'state':'snapshot_taken','inputs':result,'images_evidence':'Path,size,mtime only for image immutability; not an image-content integrity proof. Non-image metadata and checkpoint files SHA-256.'}
 if a.compare:
  before=json.loads(a.compare.read_text());same=before['inputs']==result
  payload['unchanged']=same;payload['before_snapshot_sha256']=sha(a.compare)
  if not same:
   payload['changed_groups']=[k for k,v in result.items() if before['inputs'].get(k)!=v]
 a.output.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps({'state':payload['state'],'unchanged':payload.get('unchanged'),'groups':{k:v.get('file_count') for k,v in result.items() if 'file_count' in v},'checkpoint_related_files':len(cps),'changed_groups':payload.get('changed_groups')}))
 if payload.get('unchanged') is False:raise SystemExit(1)
if __name__=='__main__':main()
