"""Archive one completed CPU preparation run without changing its evidence."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
 if not args.label or any(c not in '0123456789abcdef' for c in args.label):raise ValueError('Expected commit label')
 root=Path(__file__).resolve().parents[2]; work=root/'work/validation-probe-20260911'; repo=root/'work/github-publish/textconsistency-validation'
 prepared=work/f'prepared-{args.label}'; evidence=repo/'reports/preparation/validation-probe-20260911/evidence'
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 audit=json.loads((prepared/'local-audit.json').read_text());execution=json.loads((prepared/'execution.json').read_text())
 assert audit['source_data_and_checkpoint_inputs_unchanged'] and audit['cuda_unavailable_guard_passed'] and audit['validation_cases']==9
 assert execution['revision'].startswith(args.label) and execution['state']=='nine_local_cpu_preparations_passed'
 archive=evidence/f'cpu-prepared-{args.label}.tar.gz'
 if archive.exists():raise FileExistsError(archive)
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(prepared.rglob('*')):
   if p.is_file():
    assert not p.is_symlink() and p.suffix in ('.json','.csv','.log'),p
    tar.add(p,arcname=str(Path(prepared.name)/p.relative_to(prepared)),recursive=False)
 for name in ('local-audit.json','cases.csv','cuda-unavailable-check.json'):
  destination=evidence/f'{args.label}-{name}'
  if destination.exists():raise FileExistsError(destination)
  shutil.copy2(prepared/name,destination)
 receipt=json.loads((evidence/'1406ca3-evidence-receipt.json').read_text())
 receipt.update(archive=archive.name,archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,after_snapshot_sha256=digest(work/f'immutable-inputs-after-{args.label}.json'))
 (evidence/f'{args.label}-evidence-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
