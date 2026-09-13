"""Independent software checks for the new feature-aware pilot protocol."""
import copy
import argparse
import ast
import json
import yaml
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn

# Imported lazily so the first red run reports the absent feature as an assertion.
class VisualValidationTests(unittest.TestCase):
    def api(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('scripts.repro.visual_adapter_validation'),
                             'The feature-aware validation protocol must exist')
        from scripts.repro import visual_adapter_validation
        return visual_adapter_validation

    def test_configs_are_same_budget_and_only_the_two_experimental_controls_differ(self):
        v=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            configs=[v.case_config(name,root/'data',root/'clip.pt',root/'model') for name in v.VARIANTS]
            for cfg in configs:
                self.assertTrue(cfg.USE_CUDA)
                self.assertTrue(cfg.TEST.NO_TEST)
                self.assertEqual(cfg.TEST.SPLIT,'val')
                self.assertEqual(cfg.DATASET.SUBSAMPLE_CLASSES,'base')
                self.assertEqual(cfg.SEED,1)
                self.assertEqual(cfg.OPTIM.MAX_EPOCH,20)
                self.assertEqual(cfg.TRAINER.TEXTCONSISTENCY.KD_WEIGHT,0.)
            a,b,c=[copy.deepcopy(cfg) for cfg in configs]
            for cfg in (a,b,c):
                cfg.defrost();cfg.TRAINER.TEXTCONSISTENCY.USE_VISUAL_ADAPTER=False
                cfg.TRAINER.TEXTCONSISTENCY.ADAPTIVE_CE_WEIGHT=0.
            self.assertEqual(a.dump(),b.dump());self.assertEqual(a.dump(),c.dump())
            with self.assertRaises(ValueError):v.case_config('unknown',root,root/'clip',root/'out')

    def test_new_export_uses_returned_adapted_features_and_keeps_all_three_branches(self):
        v=self.api()
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.scale=nn.Parameter(torch.ones(()))
                self.visual_adapter=nn.Identity()
            def forward(self,images,return_attn=False):
                # F must come from actual S,D, not from an assumed frozen feature.
                s=images[:,:2];d=images[:,2:4]
                return s,d,images[:,4:],{}
        model=Model()
        images=torch.tensor([[1.,4.,3.,0.,100.,-100.],[7.,0.,1.,6.,-100.,100.]])
        batch={'img':images,'label':torch.tensor([1,0]),'impath':['/data/a','/data/b']}
        arrays,paths=v.export_batches(model,[batch],device='cpu')
        np.testing.assert_array_equal(arrays['F'],[[2.,2.],[4.,3.]])
        np.testing.assert_array_equal(arrays['labels'],[1,0])
        self.assertEqual(set(arrays),{'labels','S','D','F'})
        self.assertEqual(paths,['/data/a','/data/b'])

    def test_export_rejects_nan_empty_or_malformed_predictions(self):
        v=self.api()
        class Model(nn.Module):
            def forward(self,x,return_attn=False):return x,x,None,{}
        model=Model()
        for x,labels,paths in [(torch.tensor([[float('nan'),1.]]),torch.tensor([0]),['a']),
                                (torch.ones(2,2),torch.tensor([0]),['a']),
                                (torch.ones(2,2),torch.tensor([0,1]),['a','a'])]:
            with self.subTest(shape=x.shape),self.assertRaises(ValueError):
                v.export_batches(model,[{'img':x,'label':labels,'impath':paths}],device='cpu')
        with self.assertRaises(ValueError):v.export_batches(model,[],device='cpu')

    def test_fresh_paths_reject_existing_and_input_overlap(self):
        v=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            with self.assertRaises(FileExistsError):v.fresh(source)
            with self.assertRaises(ValueError):v.fresh(source/'new',[source])
            self.assertEqual(v.fresh(root/'new',[source]),(root/'new').resolve())

    def test_native_training_artifacts_require_full_identity_shapes_and_sidecar(self):
        v=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();model=root/'model';model.mkdir();(model/'textconsistency').mkdir()
            cfg=v.case_config('visual_description',root/'data',root/'clip',model)
            config=yaml.safe_load(cfg.dump())
            prepared={'cases':{'visual_description':{'config':config,'model_signature':v.expected_signature('visual_description')}},
                      'description_counts':[20]*50}
            manifest={'dataset':'FGVCAircraft','seed':1,'shots':16,'subsample':'base',
                'code_sha256':v.SOURCE_SHA256,'model_signature':v.expected_signature('visual_description'),
                'loss_identity':{'format':1,'mode':'full_kl','temperature':2.,'effective_weight':0.,'active':False,
                    'non_target_weighting':'none_full_distribution_kl'},
                'effective_losses':{'kd':0.,'adaptive_ce':1.,'diversity':0.,'contrastive':0.},
                'trainable_parameters':67584,'description_counts':[20]*50,
                **{f'{name}_samples_sha256':value for name,value in v.SAMPLE_HASHES.items()}}
            state={key:torch.zeros(shape) for key,shape in {
                'prompt_learner.ctx':(4,512),'prompt_learner.token_prefix':(50,1,512),
                'prompt_learner.token_suffix':(50,72,512),
                'visual_adapter.down.weight':(64,512),'visual_adapter.up.weight':(512,64)}.items()}
            checkpoint={'epoch':20,'repro_signature':manifest['model_signature'],'run_provenance':manifest,
                        'state_dict':state,'optimizer':{'state':{}},'scheduler':{'last_epoch':20}}
            path=model/'textconsistency/model.pth.tar-20'
            def save(value):
                torch.save(value,path);path.with_name(path.name+'.sha256').write_text(v.sha256(path)+'\n')
            (model/'effective_config.yaml').write_text(yaml.safe_dump(config))
            (model/'run_manifest.json').write_text(json.dumps(manifest));save(checkpoint)
            self.assertEqual(v.validate_training_artifacts(prepared,'visual_description',model)['epoch'],20)
            for change in ('wrong_shape','missing_adapter','nan','wrong_epoch','wrong_signature','missing_optimizer'):
                current=copy.deepcopy(checkpoint)
                if change=='wrong_shape':current['state_dict']['visual_adapter.up.weight']=torch.zeros(256,128)
                if change=='missing_adapter':current['state_dict'].pop('visual_adapter.up.weight')
                if change=='nan':current['state_dict']['visual_adapter.up.weight'][0,0]=float('nan')
                if change=='wrong_epoch':current['epoch']=19
                if change=='wrong_signature':current['repro_signature']['visual_adapter']['scale']=.8
                if change=='missing_optimizer':current.pop('optimizer')
                save(current)
                with self.subTest(change=change),self.assertRaises(ValueError):
                    v.validate_training_artifacts(prepared,'visual_description',model)
            save(checkpoint);path.with_name(path.name+'.sha256').write_text('0'*64)
            with self.assertRaises(ValueError):v.validate_training_artifacts(prepared,'visual_description',model)

    def test_train_command_has_no_evaluation_or_checkpoint_warmstart(self):
        v=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();prep={'data_root':str(root/'data'),'clip_path':str(root/'clip'),
                'cases':{name:{'config_path':str(root/'prepared'/f'{name}.yaml')} for name in v.VARIANTS}}
            for variant in v.VARIANTS:
                command=v.build_train_command(prep,variant,root/'runs'/variant,python='/usr/bin/python3')
                self.assertNotIn('--eval-only',command);self.assertNotIn('--model-dir',command)
                self.assertNotIn('--resume',command)
                self.assertEqual(command[command.index('--config-file')+1],prep['cases'][variant]['config_path'])
            with self.assertRaises(ValueError):v.build_train_command(prep,'other',root/'runs')
            with self.assertRaises(ValueError):v.build_train_command(prep,'simple_fusion',root/'data'/'oops')

    def test_prepared_config_tampering_cannot_be_hidden_by_rehashing_receipt(self):
        v=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();cases={}
            for variant in v.VARIANTS:
                cfg=v.case_config(variant,root/'data',root/'clip',root/'unused-model')
                path=root/f'{variant}.yaml';path.write_text(cfg.dump())
                cases[variant]={'config_path':str(path),'config':yaml.safe_load(cfg.dump()),
                    'config_sha256':v.sha256(path),'model_signature':v.expected_signature(variant)}
            tool_paths=['scripts/repro/visual_adapter_validation.py','scripts/repro/validation_probe.py',
                'scripts/repro/h1_validation.py','configs/datasets/fgvc_aircraft.yaml']
            tool_paths += [f'configs/repro/visual_adapter/{variant}.yaml' for variant in v.VARIANTS]
            prepared={'schema_version':1,'status':'prepared','protocol':v.PROTOCOL,
                'source_root':str(v.ROOT),'source_sha256':v.SOURCE_SHA256,
                'data_root':str(root/'data'),'clip_path':str(root/'clip'),
                'sample_hashes':v.SAMPLE_HASHES,'cases':cases,
                'tool_files':{str(v.ROOT/path):v.sha256(v.ROOT/path) for path in tool_paths}}
            path=root/'prepared.json'
            def save(value):
                value['prepared_identity_sha256']=v._identity(value,'prepared_identity_sha256')
                v.atomic_json(path,value)
            save(prepared);v.load_prepared(path,verify_inputs=False)
            changed=copy.deepcopy(prepared);changed['cases']['visual_description']['config']['OPTIM']['MAX_EPOCH']=21
            config=root/'visual_description.yaml';config.write_text(yaml.safe_dump(changed['cases']['visual_description']['config']))
            changed['cases']['visual_description']['config_sha256']=v.sha256(config);save(changed)
            with self.assertRaisesRegex(ValueError,'configuration'):
                v.load_prepared(path,verify_inputs=False)

    def test_generated_train_commands_match_effective_config_through_real_cli_parser(self):
        v=self.api()
        # Execute only the actual argparse block; never call main, build a model or train.
        entry=ast.parse((v.ROOT/'train.py').read_text()).body[-1]
        self.assertIsInstance(entry,ast.If)
        self.assertIsInstance(entry.body[-1],ast.Expr)
        self.assertEqual(entry.body[-1].value.func.id,'main')
        parser_code=compile(ast.Module(body=entry.body[:-1],type_ignores=[]),'train.py:parser','exec')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();prepdir=root/'prepared';prepdir.mkdir()
            prepared={'data_root':str(root/'data'),'clip_path':str(root/'clip'),'cases':{}}
            for variant in v.VARIANTS:
                config=v.case_config(variant,root/'data',root/'clip',prepdir/'unused-model')
                path=prepdir/f'{variant}.yaml';path.write_text(config.dump())
                prepared['cases'][variant]={'config_path':str(path)}
                output=root/'run'/variant
                command=v.build_train_command(prepared,variant,output)
                namespace={'argparse':argparse}
                with mock.patch.object(sys,'argv',command[1:]):exec(parser_code,namespace)
                cfg=v.training.setup_cfg(namespace['args'])
                self.assertEqual(cfg.SEED,1)
                v.ValidatedTrainer.check_cfg(None,cfg)
                self.assertEqual(cfg.dump(),v.case_config(variant,root/'data',root/'clip',output).dump())

    def test_configuration_drift_before_preparation_is_rejected(self):
        v=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source=root/'source'
            names=['configs/datasets/fgvc_aircraft.yaml']
            names += [f'configs/repro/visual_adapter/{variant}.yaml' for variant in v.VARIANTS]
            for name in names:
                target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((v.ROOT/name).read_bytes())
            for name in names:
                path=source/name;original=path.read_text();changed=yaml.safe_load(original)
                if name.endswith('fgvc_aircraft.yaml'):changed['DATASET']['NAME']='DTD'
                else:changed['OPTIM']['LR']=.02
                path.write_text(yaml.safe_dump(changed))
                with self.subTest(config=name),mock.patch.object(v,'ROOT',source),self.assertRaises(ValueError):
                    v.case_config('visual_only',root/'data',root/'clip',root/'output')
                path.write_text(original)

    def test_export_loader_rejects_wrong_rows_arrays_and_checkpoint_even_with_recomputed_hashes(self):
        v=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();path=root/'export.json'
            rows=[{'sample_id':'a','label':0,'classname':'class0'},
                  {'sample_id':'b','label':1,'classname':'class1'}]
            prepared={'prepared_identity_sha256':'prepared','class_order':['class0','class1'],
                'samples':{'val':rows},'decoded_val':{'count':2}}
            checkpoint={'path':str(root/'model/textconsistency/model.pth.tar-20'),'sha256':'checkpoint','epoch':20}
            receipt={'schema_version':1,'status':'exported','variant':'visual_description','protocol':v.PROTOCOL,
                'source_sha256':v.SOURCE_SHA256,'prepared_path':str(root/'prepared.json'),
                'prepared_identity_sha256':'prepared','class_order':prepared['class_order'],'rows':rows,
                'checkpoint':checkpoint,'decoded_val':prepared['decoded_val'],'device':'cuda'}
            arrays={'labels':np.array([0,1],dtype=np.int64),'S':np.array([[2,0],[0,3]],dtype=np.float32),
                    'D':np.array([[0,1],[1,2]],dtype=np.float32)}
            arrays['F']=.5*arrays['S']+.5*arrays['D']
            def save(r,a):
                np.savez_compressed(root/'logits.npz',**a)
                r['logits_sha256']=v.sha256(root/'logits.npz')
                r['export_identity_sha256']=v._identity(r,'export_identity_sha256');v.atomic_json(path,r)
            # Filesystem/model identities are independently tested above; only those gates are fixtures here.
            with mock.patch.object(v,'load_prepared',return_value=prepared), \
                    mock.patch.object(v,'validate_training_artifacts',return_value=checkpoint):
                save(receipt,arrays);r,loaded=v.load_export(path)
                np.testing.assert_array_equal(loaded['F'],arrays['F'])
                for change in ('wrong_rows','wrong_labels','wrong_fusion','nan','missing_branch','wrong_dtype','wrong_checkpoint'):
                    r=copy.deepcopy(receipt);a=copy.deepcopy(arrays)
                    if change=='wrong_rows':r['rows'].reverse()
                    if change=='wrong_labels':a['labels']=a['labels'][::-1]
                    if change=='wrong_fusion':a['F'][0,0]+=1.
                    if change=='nan':a['S'][0,0]=np.nan
                    if change=='missing_branch':a.pop('D')
                    if change=='wrong_dtype':a['S']=a['S'].astype(np.float64)
                    if change=='wrong_checkpoint':r['checkpoint']['sha256']='different'
                    save(r,a)
                    with self.subTest(change=change),self.assertRaises(ValueError):v.load_export(path)

if __name__=='__main__':unittest.main()
