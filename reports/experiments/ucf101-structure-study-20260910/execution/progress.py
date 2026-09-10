"""Compact read-only snapshot of the current bounded GPU study."""
import json
from pathlib import Path
import re
import subprocess
import time

root = Path('/root/autodl-tmp/textconsistency/runs/structure-study-20260910-27946-v2')
state = json.loads((root/'status.json').read_text())
result = {'state': state['state'], 'stage': state['stage'], 'pid': state['pid'],
          'elapsed_minutes': round((time.time()-state['started_unix'])/60, 1), 'completed': state['completed'],
          'error': state.get('error'), 'phases': {}}
for phase in ('smoke', 'formal'):
    directory = root/phase
    if directory.exists():
        metrics=[]
        for path in directory.glob('*/seed*/eval_*/metrics.json'):
            try:
                data=json.loads(path.read_text())
                if data.get('state')=='loaded_checkpoint': metrics.append(str(path.relative_to(directory)))
            except (OSError,ValueError): pass
        result['phases'][phase]={'trained': len(list(directory.glob('*/seed*/textconsistency/model.pth.tar-*')))//2,
                                 'evaluations': len(metrics), 'result_present': (directory/'results.json').exists()}
        # Exclude .sha256 companions when counting actual checkpoints.
        result['phases'][phase]['trained']=len([p for p in directory.glob('*/seed*/textconsistency/model.pth.tar-*') if not p.name.endswith('.sha256')])
    log=root/f'{phase}.log'
    if log.exists():
        lines=log.read_text(errors='replace').replace('\r','\n').splitlines()
        meaningful=[s for s in lines if re.search(r'epoch \[|Running [0-9]+/|Traceback|Error:|Checkpoint saved|Loaded checkpoint:|Validated [0-9]',s)]
        result.setdefault('log_tail',{})[phase]=[line[:450] for line in meaningful[-3:]]
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(result,ensure_ascii=False,indent=2))
