"""One-off endpoint diagnostics. No model loading, optimization, or prediction edits."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import statistics
import numpy as np
import torch
from torch.nn import functional as F

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
REPO = ROOT / 'work/github-publish/textconsistency_v2_tune'
INPUT = ROOT / 'work/validation-gpu-20260912-42527/unpacked/validation-probe-20260911-v1'
CPU = ROOT / 'work/validation-server-20260911-42527/unpacked'
LOCAL = ROOT / 'work/validation-probe-20260911/prepared-ce31832'
sys.path.insert(0, str(REPO))
from scripts.repro import validation_analysis as va
from scripts.repro import validation_probe as vp

BIN_EDGES = (0.0, 0.5, 0.8, 0.95, 1.0000000001)
EPS = 1e-12


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mean(x): return float(np.mean(x)) if len(x) else None
def pct(x): return 100.0*mean(x) if len(x) else None
def count(x): return int(np.sum(x))
def soft(x): return va.stable_softmax(x)
def logsoft(x): return va.stable_log_softmax(x)
def kl(qlog,plog): return np.sum(np.exp(qlog)*(qlog-plog),axis=1)
def csv_write(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def endpoint(student,teacher,labels,temp):
    n,c=student.shape;idx=np.arange(n);one=np.eye(c)[labels]
    p,q=soft(student/temp),soft(teacher/temp)
    ce=soft(student)-one;kd=temp*(p-q)
    dot=np.sum(ce*kd,axis=1);norm=np.linalg.norm(ce,axis=1)*np.linalg.norm(kd,axis=1)
    cosine=np.divide(dot,norm,out=np.zeros_like(dot),where=norm>EPS)
    p_log,q_log=logsoft(student/temp),logsoft(teacher/temp)
    others=np.arange(c)[None,:]!=labels[:,None]
    s_rest=student[others].reshape(n,c-1)/temp;t_rest=teacher[others].reshape(n,c-1)/temp
    pn_log,qn_log=logsoft(s_rest),logsoft(t_rest)
    logsum=lambda z:np.max(z,axis=1)+np.log(np.exp(z-np.max(z,axis=1,keepdims=True)).sum(axis=1))
    bp=np.stack([p_log[idx,labels],logsum(s_rest)-logsum(student/temp)],axis=1)
    bq=np.stack([q_log[idx,labels],logsum(t_rest)-logsum(teacher/temp)],axis=1)
    full=kl(q_log,p_log)*temp**2;target=kl(bq,bp)*temp**2;non=kl(qn_log,pn_log)*temp**2
    weighted_non=np.exp(bq[:,1])*non
    assert np.allclose(full,target+weighted_non,rtol=0,atol=1e-10)
    z=torch.tensor(student,dtype=torch.float64,requires_grad=True);t=torch.tensor(teacher,dtype=torch.float64)
    loss=F.kl_div(F.log_softmax(z/temp,dim=-1),F.softmax(t/temp,dim=-1),reduction='sum')*temp**2
    actual=torch.autograd.grad(loss,z)[0].detach().numpy()
    assert np.allclose(actual,kd,rtol=0,atol=1e-10)
    actual_ce=torch.autograd.grad(F.cross_entropy(z,torch.tensor(labels),reduction='sum'),z)[0].detach().numpy()
    assert np.allclose(actual_ce,ce,rtol=0,atol=1e-10)
    # Known non-target-only KD control: its true-class logit derivative is zero.
    nckd=np.zeros_like(student);nckd[others]=(temp*(np.exp(pn_log)-np.exp(qn_log))).ravel()
    nt_dot=np.sum(ce*nckd,axis=1)
    tensor_mask=torch.tensor(others)
    z_rest=z[tensor_mask].reshape(n,c-1)
    t_rest_tensor=t[tensor_mask].reshape(n,c-1)
    nt_loss=F.kl_div(F.log_softmax(z_rest/temp,dim=-1),F.softmax(t_rest_tensor/temp,dim=-1),reduction='sum')*temp**2
    actual_nt=torch.autograd.grad(nt_loss,z)[0].detach().numpy()
    assert np.allclose(actual_nt,nckd,rtol=0,atol=1e-10)
    return dict(kd=full,target=target,weighted_non=weighted_non,non=non,
        cosine=cosine,undefined_cos=norm<=EPS,conflict=dot < -EPS,
        target_down=kd[idx,labels]>EPS,combined_target_down=(ce+kd)[idx,labels]>EPS,
        target_probability_gap=p[idx,labels]-q[idx,labels],nt_conflict=nt_dot < -EPS,
        gradient_error=float(np.max(np.abs(actual-kd))),nckd_gradient_error=float(np.max(np.abs(actual_nt-nckd))),
        chain_error=float(np.max(np.abs(full-target-weighted_non))))


def main():
    out=WORK/'results'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    write(out/'protocol.json',dict(scope='Existing nine Base-validation endpoints only',temperature='read fixed training configs, expected 2',
        kd_weight=1,confidence_bins=list(BIN_EDGES),numeric_sign_tolerance=EPS,
        endpoints=['S0','S1'],teacher='T0, asserted exactly equal to T1',
        losses='CE and T^2 KL(q_teacher_T || p_student_T); non-target-only known control',
        do_not_claim=['parameter-gradient conflict','training trajectory causality','training outcome of selective/DKD variants','Novel/HM improvement'],
        new_training_runs=0,optimizer_steps=0))
    expected=read(ROOT/'work/validation-gpu-20260912-42527/backup-receipt.json')
    assert sha(ROOT/'work/validation-gpu-20260912-42527'/Path(expected['archive']).name)==expected['sha256']
    for item in expected['files']:
        assert sha(INPUT.parent/item['path'])==item['sha256']
    before={str(p):sha(p) for p in sorted(INPUT.rglob('*')) if p.is_file()}
    summaries=[];samples=[];cohorts=[];bins=[];branches=[];classes=[];sources={};formulas=[]
    for path in sorted((INPUT/'exports').glob('*/export.json')):
        rec,a=va.load_export(path)
        d,seed=rec['dataset_key'],rec['seed'];n=len(a['labels']);y=a['labels'];idx=np.arange(n)
        prepared_path=CPU/Path(rec['prepared']).relative_to('/root/autodl-tmp/textconsistency/setup-logs')
        assert sha(prepared_path)==rec['prepared_sha256'];pr=read(prepared_path)
        cfg=pr['effective_configs']['kd1']['TRAINER']['TEXTCONSISTENCY'];temp=cfg['KD_TEMP']
        assert temp==2 and cfg['KD_WEIGHT']==1 and pr['effective_configs']['kd0']['TRAINER']['TEXTCONSISTENCY']['KD_WEIGHT']==0
        source=Path(read(LOCAL/f'{d}-seed{seed}/prepared.json')['source_root'])
        if str(source) not in sources:
            assert vp.model_source_digest(source)==rec['provenance']['source_sha256']
            f=source/'trainers/textconsistency_common.py';code=f.read_text();node=next(x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef) and x.name=='training_losses')
            sources[str(source)]={'source_sha256':rec['provenance']['source_sha256'],'loss_file_sha256':sha(f),'loss_function':ast.get_source_segment(code,node)}
        assert np.array_equal(a['T0'],a['T1'])
        pred={b:np.argmax(a[b],axis=1) for b in vp.BRANCHES};correct={b:pred[b]==y for b in vp.BRANCHES}
        probs={b:soft(a[b]) for b in vp.BRANCHES}
        t=correct['T0'];h=correct['S0']&~correct['S1'];r=~correct['S0']&correct['S1'];samewrong=(pred['S1']==pred['T0'])&~t
        ep={b:endpoint(a[b].astype(np.float64),a['T0'].astype(np.float64),y,temp) for b in ('S0','S1')}
        margin=lambda z:z[idx,y]-np.max(np.where(np.eye(z.shape[1])[y].astype(bool),-np.inf,z),axis=1)
        gaps={b:margin(a[b]) for b in ('S0','S1','T0')};conf=probs['T0'].max(axis=1)
        good,bad=conf[t],conf[~t];auc=float(((good[:,None]>bad).sum()+.5*(good[:,None]==bad).sum())/(len(good)*len(bad))) if len(good)*len(bad) else None
        row=dict(dataset=d,seed=seed,n=n,S0_accuracy_pct=pct(correct['S0']),S1_accuracy_pct=pct(correct['S1']),
            T_accuracy_pct=pct(t),student_delta_pp=pct(correct['S1'])-pct(correct['S0']),harm=count(h),rescue=count(r),
            harm_teacher_wrong=count(h&~t),harm_teacher_correct=count(h&t),harm_same_wrong_teacher=count(h&samewrong),
            rescue_teacher_correct=count(r&t),rescue_teacher_wrong=count(r&~t),teacher_wrong=count(~t),
            teacher_confidence_correct=mean(conf[t]),teacher_confidence_wrong=mean(conf[~t]),teacher_confidence_correctness_auc=auc,
            student_teacher_agreement_S0=pct(pred['S0']==pred['T0']),student_teacher_agreement_S1=pct(pred['S1']==pred['T0']),
            mean_true_margin_delta=mean(gaps['S1']-gaps['S0']))
        assert row['rescue']-row['harm']==count(correct['S1'])-count(correct['S0'])
        for b,e in ep.items():
            row.update({f'{b}_'+k:mean(e[k]) for k in ('kd','target','weighted_non','non','target_probability_gap')})
            row.update({f'{b}_'+k+'_pct':pct(e[k]) for k in ('conflict','target_down','combined_target_down','nt_conflict')})
            formulas.append(dict(dataset=d,seed=seed,endpoint=b,gradient_max_error=e['gradient_error'],nckd_gradient_max_error=e['nckd_gradient_error'],kl_chain_max_error=e['chain_error']))
        summaries.append(row)
        for b in vp.BRANCHES:
            p=probs[b];lp=logsoft(a[b]);sort=np.sort(p,axis=1)
            branches.append(dict(dataset=d,seed=seed,branch=b,accuracy_pct=pct(correct[b]),nll=mean(-lp[idx,y]),
                true_probability=mean(p[idx,y]),max_probability=mean(p.max(axis=1)),entropy=mean(-np.sum(p*lp,axis=1)),
                probability_margin=mean(sort[:,-1]-sort[:,-2]),logit_std=mean(np.std(a[b],axis=1))))
        for student_state in (False,True):
            for teacher_state in (False,True):
                mask=(correct['S0']==student_state)&(t==teacher_state)
                co=dict(dataset=d,seed=seed,S0_correct=student_state,T_correct=teacher_state,n=count(mask),harm=count(h&mask),rescue=count(r&mask),
                    S1_correct=count(correct['S1']&mask),harm_rate_pct=pct(h[mask]),rescue_rate_pct=pct(r[mask]),teacher_confidence=mean(conf[mask]))
                for b,e in ep.items():
                    for k in ('kd','target','weighted_non','target_probability_gap'):co[b+'_'+k]=mean(e[k][mask])
                    for k in ('conflict','target_down','combined_target_down','nt_conflict'):co[b+'_'+k+'_pct']=pct(e[k][mask])
                cohorts.append(co)
        for lower,upper in zip(BIN_EDGES[:-1],BIN_EDGES[1:]):
            mask=(conf>=lower)&(conf<upper)
            bins.append(dict(dataset=d,seed=seed,lower=lower,upper=min(upper,1),n=count(mask),teacher_correct=count(t&mask),
                S0_correct=count(correct['S0']&mask),harm=count(h&mask),rescue=count(r&mask),teacher_wrong_harm=count(h&~t&mask),
                mean_teacher_confidence=mean(conf[mask]),teacher_accuracy_pct=pct(t[mask])))
        for label,name in enumerate(rec['class_order']):
            m=y==label;classes.append(dict(dataset=d,seed=seed,label=label,classname=name,n=count(m),S0_correct=count(correct['S0']&m),
                S1_correct=count(correct['S1']&m),T_correct=count(t&m),harm=count(h&m),rescue=count(r&m)))
        for i,x in enumerate(rec['rows']):
            sample=dict(dataset=d,seed=seed,**x,S0_prediction=int(pred['S0'][i]),S1_prediction=int(pred['S1'][i]),T_prediction=int(pred['T0'][i]),
                S0_correct=bool(correct['S0'][i]),S1_correct=bool(correct['S1'][i]),T_correct=bool(t[i]),harm=bool(h[i]),rescue=bool(r[i]),
                same_wrong_teacher=bool(samewrong[i]),teacher_confidence=float(conf[i]),margin_delta=float(gaps['S1'][i]-gaps['S0'][i]))
            for b,e in ep.items():
                for k in ('kd','target','weighted_non','conflict','target_down','combined_target_down','target_probability_gap','cosine','undefined_cos','nt_conflict'):
                    v=e[k][i];sample[b+'_'+k]=bool(v) if isinstance(v,np.bool_) else float(v)
            samples.append(sample)
    assert len(summaries)==9 and len(samples)==1500 and len({(s['dataset'],s['sample_id']) for s in samples})==1280
    assert len(classes)==375 and len(branches)==54 and len(cohorts)==36 and len(bins)==36
    assert before=={p:sha(p) for p in before}
    for name,rows in [('by-seed',summaries),('by-sample',samples),('by-class',classes),('cohorts',cohorts),('confidence-bins',bins),('branches',branches),('formula-checks',formulas)]:csv_write(out/(name+'.csv'),rows)
    datasets={}
    for d in ('ucf101','dtd','fgvc_aircraft'):
        group=[x for x in summaries if x['dataset']==d]
        totals={k:sum(x[k] for x in group) for k in ('n','harm','rescue','harm_teacher_wrong','harm_teacher_correct','harm_same_wrong_teacher','rescue_teacher_correct','rescue_teacher_wrong','teacher_wrong')}
        metrics={k:{'mean':statistics.mean(x[k] for x in group),'seed_values':[x[k] for x in group],'seed_sd':statistics.stdev(x[k] for x in group)} for k in group[0] if k not in ('dataset','seed',*totals) and group[0][k] is not None}
        datasets[d]={'totals':totals,'metrics':metrics}
    result=dict(state='diagnostics_complete',new_training_runs=0,optimizer_steps=0,server_connected=False,cases=9,records=1500,unique_images=1280,
        datasets=datasets,inputs_sha256=before,sources=sources,script_sha256=sha(__file__),
        formula_checks={'cases':18,'closed_form_KD_and_CE_and_NCKD_match_float64_autograd':True,'kl_chain_identity_passed':True,
        'max_gradient_error':max(r['gradient_max_error'] for r in formulas),'max_kl_chain_error':max(r['kl_chain_max_error'] for r in formulas)},
        limitations=['Endpoint association cannot identify actual training trajectory causality','Analytic logit derivatives are not prompt-parameter gradients','Repeated seeds share images and are not independent samples','Teacher correctness cohorts and fixed confidence bins are diagnostic labels, not trained gates','No new Novel/HM results or intervention training outcomes'])
    write(out/'diagnosis.json',result)
    print(json.dumps({'state':result['state'],'datasets':datasets,'formula_checks':result['formula_checks']},indent=2))


if __name__=='__main__':main()
