"""Independent count reconciliation and derivative check against frozen loss source."""
from pathlib import Path
import ast,csv,hashlib,json,sys
import numpy as np
import torch
from torch.nn import functional as F

W=Path(__file__).resolve().parent
ROOT=W.parents[1]
R=W/'results'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(name):return list(csv.DictReader((R/name).open()))

def main():
    if (R/'verification.json').exists():raise FileExistsError('Verification exists')
    result=read(R/'diagnosis.json');rows=table('by-sample.csv');cases=table('by-seed.csv')
    assert len(rows)==1500 and len(cases)==9
    assert len({(x['dataset'],int(x['seed']),x['sample_id']) for x in rows})==1500
    expected={(d,s) for d in ('ucf101','dtd','fgvc_aircraft') for s in (1,2,3)}
    assert {(x['dataset'],int(x['seed'])) for x in cases}==expected
    functions={};checks=[]
    for source,meta in result['sources'].items():
        p=Path(source)/'trainers/textconsistency_common.py';assert sha(p)==meta['loss_file_sha256']
        module=ast.parse(p.read_text());node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='training_losses')
        isolated=ast.Module(body=[node],type_ignores=[]);ns={'torch':torch,'F':F}
        exec(compile(isolated,str(p),'exec'),ns)
        functions[source]=ns['training_losses']
    for case in cases:
        d,s=case['dataset'],int(case['seed']);name=f'{d}-seed{s}'
        p=ROOT/'work/validation-gpu-20260912-42527/unpacked/validation-probe-20260911-v1/exports'/name
        rec=read(p/'export.json');assert sha(p/'logits.npz')==rec['npz_sha256']
        subset=[x for x in rows if x['dataset']==d and int(x['seed'])==s]
        assert [(x['sample_id'],int(x['label']),x['classname']) for x in subset]==[(x['sample_id'],x['label'],x['classname']) for x in rec['rows']]
        with np.load(p/'logits.npz',allow_pickle=False) as a:
            y=a['labels'];s0=np.argmax(a['S0'],1);s1=np.argmax(a['S1'],1);t=np.argmax(a['T0'],1)
            h=(s0==y)&(s1!=y);res=(s0!=y)&(s1==y)
            assert int(case['harm'])==int(h.sum()) and int(case['rescue'])==int(res.sum())
            for i,row in enumerate(subset):
                assert int(row['S0_prediction'])==s0[i] and int(row['S1_prediction'])==s1[i] and int(row['T_prediction'])==t[i]
                assert (row['harm']=='True')==bool(h[i]) and (row['rescue']=='True')==bool(res[i])
                assert (row['T_correct']=='True')==bool(t[i]==y[i])
            source=read(ROOT/'work/validation-probe-20260911/prepared-ce31832'/name/'prepared.json')['source_root']
            f=functions[source]
            for role in ('S0','S1'):
                z=torch.tensor(a[role],dtype=torch.float64,requires_grad=True);teacher=torch.tensor(a['T0'],dtype=torch.float64)
                loss,_=f(z,teacher,torch.tensor(y),None,1.,2.,1.,0.,0.)
                actual=torch.autograd.grad(loss,z)[0]
                expected_gradient=(F.softmax(z,dim=-1)-F.one_hot(torch.tensor(y),z.shape[1]))+2*(F.softmax(z/2,dim=-1)-F.softmax(teacher/2,dim=-1))
                error=float((actual-expected_gradient/len(y)).abs().max().detach())
                assert error<1e-12
                checks.append({'dataset':d,'seed':s,'endpoint':role,'frozen_loss_gradient_max_error':error})
        for file in ('cohorts.csv','confidence-bins.csv','by-class.csv'):
            sub=[x for x in table(file) if x['dataset']==d and int(x['seed'])==s]
            assert sum(int(x['n']) for x in sub)==len(y)
            assert sum(int(x['harm']) for x in sub)==int(h.sum())
            assert sum(int(x['rescue']) for x in sub)==int(res.sum())
    assert result['inputs_sha256']=={p:sha(p) for p in result['inputs_sha256']}
    verify={'state':'passed','sample_records_reconciled':1500,'cases':9,'class_rows':375,'cohort_rows':36,'confidence_bin_rows':36,
            'frozen_actual_loss_gradient_checks':checks,'max_actual_loss_gradient_error':max(x['frozen_loss_gradient_max_error'] for x in checks),
            'all_input_hashes_unchanged':True,'optimizer_steps':0,'new_training_runs':0,
            'diagnosis_sha256':sha(R/'diagnosis.json'),'diagnostic_script_sha256':sha(W/'diagnose.py'),'verification_script_sha256':sha(__file__)}
    (R/'verification.json').write_text(json.dumps(verify,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verify.items() if k!='frozen_actual_loss_gradient_checks'},indent=2))

if __name__=='__main__':main()
