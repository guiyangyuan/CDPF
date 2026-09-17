#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

release=Path('/root/autodl-tmp/textconsistency/releases/投稿关键对照-20260917-04b6de0')
server=Path('/root/autodl-tmp/textconsistency')
out=server/'runs/投稿关键对照-八数据集-20260917-04b6de0'
path=release/'scripts/repro/运行投稿关键对照.py'
spec=importlib.util.spec_from_file_location('关键对照执行器',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
plan=module.生成计划(release,server,server/'data',out,server/'.venv/bin/python')
plan['命令']=[r for r in plan['命令'] if r['数据集键'] not in {'dtd','fgvc_aircraft'}]
ucf_source=server/'releases/统一推理效率-UCF101-20260916-d07157d'
for row in plan['命令']:
    if row['数据集键']=='ucf101':
        baseline=server/f"reused-inputs/统一推理效率-20260916/ucf101/baseline/seed{row['种子']}"
        row['历史源码']=str(ucf_source); row['历史检查点']=str(baseline)
        cmd=row['命令']
        cmd[cmd.index('--source-root')+1]=str(ucf_source)
        cmd[cmd.index('--baseline-dir')+1]=str(baseline)
    source=Path(row['历史源码']); baseline=Path(row['历史检查点'])
    required=[source/'train.py',source/'clip/ViT-B-16.pt',baseline/'run_manifest.json',baseline/'textconsistency/model.pth.tar-20',baseline/f"eval_{row['分支']}_student/metrics.json"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError(f"{row['数据集']} seed{row['种子']} {row['分支']} 缺少: {missing}")
for i,row in enumerate(plan['命令'],1): row['序号']=i
plan['协议']='投稿类别名称与权重稳健性-v1-八数据集阶段'
plan['状态']='待执行'; plan['模式']='GPU冻结推理-八数据集阶段'
plan['推理命令数']=len(plan['命令']); plan['数据集']=[d for d in plan['数据集'] if d not in {'dtd','fgvc_aircraft'}]
plan['阶段说明']='DTD与FGVCAircraft历史主表检查点不在当前实例；本阶段不重训，只运行其余八数据集。'
print(json.dumps(module.执行(plan,out,release,server/'.venv/bin/python'),ensure_ascii=False,indent=2))
