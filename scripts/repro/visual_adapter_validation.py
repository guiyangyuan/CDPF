#!/usr/bin/env python3
"""Prepare and export the fixed Aircraft visual-adapter pilot, without old logits."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'Dassl.pytorch')]
import numpy as np
import torch
import yaml
from dassl.config import get_cfg_default
from dassl.engine.trainer import TrainerBase
from dassl.utils import set_random_seed
import train as training
from trainers.coop_textconsistency_v2_fg import CoOp_TextConsistency_V2_FG
from trainers.textconsistency_common import description_rows, prediction_logits, sha256_file
from trainers.textconsistency_runtime import ValidatedTrainer, source_digest, _loss_identity
from trainers.visual_adapter import ResidualVisualAdapter
from scripts.repro import validation_probe as probe
from scripts.repro.h1_validation import image_content_receipt, validate_fewshot_cache

VARIANTS = ('simple_fusion', 'visual_only', 'visual_description')
SOURCE_SHA256 = '38d9afc3cdc0440695cb5eca5c59069f73ace1aba4478d5afc5b656050911840'
CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
TEMPLATE_SHA256 = 'd86833c2483c1d12907abeb6198412f1d0a020bd12999a3f170967a7f549e3b8'
CONFIG_SHA256 = {
    'configs/datasets/fgvc_aircraft.yaml': '2853829da10a89ad7d4b54b55c9bad542f0d6bdeeb273bdee0551fa486ec53a5',
    'configs/repro/visual_adapter/simple_fusion.yaml': '926e5d788274b2689c707aae5336974399dcd6ab170c0700d3a9eb7faa44b088',
    'configs/repro/visual_adapter/visual_only.yaml': 'ba1b8ad0aef6a0f2014fdebd14b8620dad9784fb320687059e62fa48a73c6792',
    'configs/repro/visual_adapter/visual_description.yaml': 'dac47a0de618c7c11ef32ad5ae2a5b74669d8592f84de24fd1e05373eea12263',
}
METADATA = {
    'variants.txt': '6e14e778b0a7c94c80619a7ba2c9f90c6784c9c997e308938ff8d3ee11662f0b',
    'images_variant_train.txt': '29ac66fb762b9e1e5d007a18cb2cd2404ee326a4bce1dc0040e267c5585c46e7',
    'images_variant_val.txt': '17afd8d6aac16d6188109465637f224a69e6c68d8b6e8422a2591f87aa3a8281',
    'images_variant_test.txt': '06b5f65d559c399d6887d660dcec09dbf661576c0f3bd85e981e976b2bed3511',
}
SAMPLE_HASHES = {
    'train': 'ae5851085ef5e74b7cb612cda7454fde300601ef14ad21b707bbbfbbd3c85e9a',
    'val': '7ac44ab40ea24086902991743d3b51720e467204034efd2b5a98352416fd02bb',
    'test': '78dd9ed86b8d2d13c40d88fac9b778ec450b322707c44347521c07e7db9b714a',
}
PROTOCOL = {'kind': 'visual_adapter_aircraft_pilot', 'version': 1, 'seed': 1,
    'shots': 16, 'epochs': 20, 'split': 'val', 'subsample_classes': 'base',
    'dataset': 'FGVCAircraft', 'backbone': 'ViT-B/16', 'precision': 'fp32',
    'alpha': 0.5, 'variants': list(VARIANTS), 'max_training_runs': 3,
    'wallclock_cap_seconds': 3600, 'minimum_C_minus_A_pp': 1.0, 'minimum_C_minus_B_pp': 1.0}


def sha256(path): return sha256_file(Path(path))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n'); temp.replace(path)

def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError('Expected a JSON object')
    return value

def _disjoint(path, protected):
    path = Path(path).resolve()
    for other in protected:
        other = Path(other).resolve()
        if path == other or path in other.parents or other in path.parents:
            raise ValueError(f'Output overlaps protected input: {other}')
    return path

def fresh(path, protected=()):
    target = Path(path).resolve()
    if target.exists(): raise FileExistsError(f'Output must be fresh: {target}')
    return _disjoint(target, protected)

def _identity(value, field):
    copy_value = dict(value); copy_value.pop(field, None)
    return digest(copy_value)


def case_config(variant, data_root, clip_path, output_dir, *, cuda=True):
    if variant not in VARIANTS: raise ValueError('Unknown visual-adapter variant')
    for name, expected in CONFIG_SHA256.items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Frozen pilot configuration changed: {name}')
    cfg = get_cfg_default(); training.extend_cfg(cfg)
    cfg.merge_from_file(str(ROOT/'configs/datasets/fgvc_aircraft.yaml'))
    cfg.merge_from_file(str(ROOT/f'configs/repro/visual_adapter/{variant}.yaml'))
    cfg.SEED = 1; cfg.DATASET.ROOT = str(Path(data_root).resolve())
    cfg.OUTPUT_DIR = str(Path(output_dir).resolve())
    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(Path(clip_path).resolve())
    cfg.USE_CUDA = cuda; cfg.TRAINER.TEXTCONSISTENCY.DEVICE = 'cuda' if cuda else 'cpu'
    cfg.TEST.NO_TEST = True
    cfg.freeze(); ValidatedTrainer.check_cfg(None, cfg)
    return cfg


def expected_signature(variant):
    signature = {'format': 2, 'trainer': 'CoOp_TextConsistency_V2_FG', 'n_ctx': 4,
        'aggregation': 'mean', 'feature_adapter': False, 'n_groups': 4, 'n_views': 20,
        'ablation_group': None, 'clip_sha256': CLIP_SHA256, 'template_sha256': TEMPLATE_SHA256,
        'view_order': 'contiguous', 'order_seed': 0, 'order_version': 1,
        'loss_identity': _loss_identity('full_kl', 2., 0.)}
    if variant != 'simple_fusion':
        signature.update(format=3, visual_adapter=ResidualVisualAdapter(512, 64, .2).identity(
            1. if variant == 'visual_description' else 0.))
    return signature


def _dataset(cfg):
    train, val, test, classes, paths = probe.rebuild_source_dataset(training, cfg, 'fgvc_aircraft')
    original = probe.validate_samples(train, val, test, Path(cfg.DATASET.ROOT), classes)
    samples = {name: [{key: row[key] for key in ('sample_id', 'label', 'classname')}
                     for row in original[name]] for name in ('train', 'val', 'test')}
    hashes = {name: probe.sample_digest(items, Path(cfg.DATASET.ROOT))
              for name, items in (('train', train), ('val', val), ('test', test))}
    if hashes != SAMPLE_HASHES or len(classes) != 50 or len(train) != 800 or len(val) != 200:
        raise ValueError('Aircraft seed1 sample order/count differs from the fixed input anchor')
    ids = [set(row['sample_id'] for row in samples[name]) for name in ('train', 'val', 'test')]
    if any(ids[i] & ids[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('Official split sample IDs overlap')
    return train, val, test, classes, paths, samples


def prepare(args):
    data = Path(args.data_root).resolve(); clip = Path(args.clip_path).resolve()
    out = fresh(args.output_dir, (ROOT, data, clip))
    if source_digest() != SOURCE_SHA256: raise ValueError('Frozen model source changed')
    files = {str(clip): CLIP_SHA256,
             str(ROOT/'templates/airplane_prompts_full.json'): TEMPLATE_SHA256}
    files.update({str(data/'fgvc_aircraft'/name): value for name,value in METADATA.items()})
    for path, expected in files.items():
        if sha256(path) != expected: raise ValueError(f'Fixed asset mismatch: {path}')
    cfg = case_config('simple_fusion', data, clip, out/'unused-model', cuda=False)
    train, val, test, classes, _, samples = _dataset(cfg)
    cache_path = data/'fgvc_aircraft/split_fewshot/shot_16-seed_1.pkl'
    cache = validate_fewshot_cache(cache_path, training, data, samples['train'], samples['val'])
    files[str(cache_path)] = cache['sha256']
    content = {split: image_content_receipt(samples[split], data) for split in ('train', 'val')}
    for value in content.values():
        files.update({str(data/row['sample_id']): row['sha256'] for row in value['files']})
    rows = description_rows(_read(ROOT/'templates/airplane_prompts_full.json'), classes, 20)
    from clip import clip as clip_module
    # Validate token truncation/dedup coverage without building the CLIP model.
    counts = [len(set(tuple(token.tolist()) for token in clip_module.tokenize(row, truncate=True))) for row in rows]
    loader, transform = probe.build_val_loader(cfg, val)
    decoded = probe._decode_evidence(loader)
    tools = {str(ROOT/path): sha256(ROOT/path) for path in (
        'scripts/repro/visual_adapter_validation.py', 'scripts/repro/validation_probe.py',
        'scripts/repro/h1_validation.py')}
    for variant in VARIANTS:
        path = ROOT/f'configs/repro/visual_adapter/{variant}.yaml'; tools[str(path)] = sha256(path)
    tools[str(ROOT/'configs/datasets/fgvc_aircraft.yaml')] = sha256(ROOT/'configs/datasets/fgvc_aircraft.yaml')
    cases = {}
    out.mkdir(parents=True)
    for variant in VARIANTS:
        future = case_config(variant, data, clip, out/'unused-model')
        path = out/f'{variant}.yaml'; path.write_text(future.dump())
        cases[variant] = {'config_path': str(path), 'config_sha256': sha256(path),
            'config': yaml.safe_load(future.dump()), 'model_signature': expected_signature(variant)}
    receipt = {'schema_version': 1, 'status': 'prepared', 'protocol': PROTOCOL,
        'source_root': str(ROOT), 'data_root': str(data), 'clip_path': str(clip),
        'source_sha256': SOURCE_SHA256, 'class_order': classes, 'samples': samples,
        'sample_hashes': SAMPLE_HASHES, 'input_files': files, 'tool_files': tools,
        'image_content': content, 'description_counts': counts,
        'decoded_val': decoded, 'cases': cases,
        'real_model_loaded': False, 'training_started': False, 'gpu_used': False}
    receipt['prepared_identity_sha256'] = _identity(receipt, 'prepared_identity_sha256')
    atomic_json(out/'prepared.json', receipt)
    load_prepared(out/'prepared.json')
    return receipt


def load_prepared(path, verify_inputs=True):
    path = Path(path).resolve(); p = _read(path)
    if p.get('prepared_identity_sha256') != _identity(p, 'prepared_identity_sha256'):
        raise ValueError('Prepared receipt identity mismatch')
    if (p.get('schema_version') != 1 or p.get('status') != 'prepared' or p.get('protocol') != PROTOCOL
            or p.get('source_root') != str(ROOT) or p.get('source_sha256') != SOURCE_SHA256
            or source_digest() != SOURCE_SHA256): raise ValueError('Prepared frozen protocol/source mismatch')
    if set(p['cases']) != set(VARIANTS) or p.get('sample_hashes') != SAMPLE_HASHES:
        raise ValueError('Prepared cases or sample anchor mismatch')
    for variant in VARIANTS:
        case = p['cases'][variant]; config_path = path.parent/f'{variant}.yaml'
        expected = yaml.safe_load(case_config(variant, p['data_root'], p['clip_path'], path.parent/'unused-model').dump())
        if (case['config_path'] != str(config_path) or case['config'] != expected
                or yaml.safe_load(config_path.read_text()) != expected or sha256(config_path) != case['config_sha256']
                or case['model_signature'] != expected_signature(variant)):
            raise ValueError('Prepared training configuration or model identity mismatch')
    expected_tools = [ROOT/name for name in ('scripts/repro/visual_adapter_validation.py',
        'scripts/repro/validation_probe.py','scripts/repro/h1_validation.py','configs/datasets/fgvc_aircraft.yaml')]
    expected_tools += [ROOT/f'configs/repro/visual_adapter/{variant}.yaml' for variant in VARIANTS]
    if p['tool_files'] != {str(file): sha256(file) for file in expected_tools}:
        raise ValueError('Prepared operation source changed')
    if verify_inputs:
        cfg = case_config('simple_fusion', p['data_root'], p['clip_path'], path.parent/'unused-model', cuda=False)
        train, val, test, classes, _, samples = _dataset(cfg)
        if samples != p['samples'] or classes != p['class_order']:
            raise ValueError('Prepared samples/classes changed')
        data = Path(p['data_root']); cache = data/'fgvc_aircraft/split_fewshot/shot_16-seed_1.pkl'
        validate_fewshot_cache(cache, training, data, samples['train'], samples['val'])
        expected_paths = {str(Path(p['clip_path'])),str(ROOT/'templates/airplane_prompts_full.json'),str(cache)}
        expected_paths.update(str(data/'fgvc_aircraft'/name) for name in METADATA)
        expected_paths.update(str(data/row['sample_id']) for split in ('train','val') for row in samples[split])
        if set(p['input_files']) != expected_paths: raise ValueError('Prepared input coverage is incomplete')
        anchored = {str(Path(p['clip_path'])):CLIP_SHA256, str(ROOT/'templates/airplane_prompts_full.json'):TEMPLATE_SHA256,
                    **{str(data/'fgvc_aircraft'/key):value for key,value in METADATA.items()}}
        for file, expected in p['input_files'].items():
            if (file in anchored and expected != anchored[file]) or sha256(file) != expected:
                raise ValueError(f'Prepared input changed: {file}')
    return p


def build_train_command(prepared, variant, model_dir, python=sys.executable):
    if variant not in VARIANTS: raise ValueError('Unknown pilot variant')
    target = _disjoint(model_dir, (ROOT, prepared['data_root'], prepared['clip_path'],
                                  Path(prepared['cases'][variant]['config_path']).parent))
    return [str(python),str(ROOT/'train.py'),'--config-file',prepared['cases'][variant]['config_path'],
            '--output-dir',str(target),'--seed','1']


def validate_training_artifacts(prepared, variant, model_dir):
    if variant not in VARIANTS: raise ValueError('Unknown pilot variant')
    directory = Path(model_dir).resolve(); case = prepared['cases'][variant]
    expected = copy.deepcopy(case['config']); expected['OUTPUT_DIR'] = str(directory)
    actual_config = yaml.safe_load((directory/'effective_config.yaml').read_text())
    if actual_config != expected: raise ValueError('Training effective config differs from fixed case')
    manifest = _read(directory/'run_manifest.json')
    count = 2048 if variant == 'simple_fusion' else 67584
    identity = {'dataset':'FGVCAircraft','seed':1,'shots':16,'subsample':'base',
        'code_sha256':SOURCE_SHA256,'model_signature':case['model_signature'],
        'loss_identity':_loss_identity('full_kl',2.,0.),
        'effective_losses':{'kd':0.,'adaptive_ce':1. if variant=='visual_description' else 0.,'diversity':0.,'contrastive':0.},
        'trainable_parameters':count, 'description_counts':prepared['description_counts'],
        **{f'{split}_samples_sha256':value for split,value in SAMPLE_HASHES.items()}}
    if any(manifest.get(key)!=value for key,value in identity.items()):
        raise ValueError('Training manifest identity mismatch')
    path = directory/'textconsistency/model.pth.tar-20'
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    if (checkpoint.get('epoch')!=20 or checkpoint.get('repro_signature')!=case['model_signature']
            or checkpoint.get('run_provenance')!=manifest): raise ValueError('Checkpoint provenance mismatch')
    state = checkpoint.get('state_dict',{})
    expected_shapes = {'prompt_learner.ctx':(4,512), 'prompt_learner.token_prefix':(50,1,512),
                       'prompt_learner.token_suffix':(50,72,512)}
    if variant != 'simple_fusion':
        expected_shapes.update({'visual_adapter.down.weight':(64,512),'visual_adapter.up.weight':(512,64)})
    if set(state)!=set(expected_shapes): raise ValueError('Incomplete or unexpected checkpoint tensors')
    for key,shape in expected_shapes.items():
        value = state[key]
        if not torch.is_tensor(value) or tuple(value.shape)!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Invalid checkpoint tensor: {key}')
    if not isinstance(checkpoint.get('optimizer'),dict) or not isinstance(checkpoint.get('scheduler'),dict):
        raise ValueError('Checkpoint optimizer/scheduler state missing')
    file_sha = sha256(path)
    if path.with_name(path.name+'.sha256').read_text().strip()!=file_sha:
        raise ValueError('Checkpoint SHA sidecar mismatch')
    return {'path':str(path),'sha256':file_sha,'epoch':20,
            'manifest_sha256':sha256(directory/'run_manifest.json'),
            'config_sha256':sha256(directory/'effective_config.yaml')}


def export_batches(model, batches, *, device):
    model.eval()
    collected = {key:[] for key in ('labels','S','D','F')}; paths=[]
    with torch.inference_mode():
        for batch in batches:
            labels = batch['label']; batch_paths = [str(path) for path in batch['impath']]
            if labels.dtype!=torch.int64 or labels.ndim!=1 or len(labels)!=len(batch_paths):
                raise ValueError('Malformed batch labels/paths')
            s,d,_,_ = model(batch['img'].to(device),return_attn=False)
            if (s.ndim!=2 or d is None or s.shape!=d.shape or s.shape[0]!=len(labels)
                    or s.shape[1]<2 or not torch.isfinite(s).all() or not torch.isfinite(d).all()
                    or (labels<0).any() or (labels>=s.shape[1]).any()):
                raise ValueError('Malformed or nonfinite S/D predictions')
            f=prediction_logits(s,d,'ensemble',.5)
            for name,value in (('labels',labels),('S',s),('D',d),('F',f)):
                collected[name].append(value.detach().cpu().numpy().astype(np.int64 if name=='labels' else np.float32))
            paths.extend(batch_paths)
    if not paths or len(paths)!=len(set(paths)): raise ValueError('Empty or duplicate export samples')
    return {key:np.concatenate(value) for key,value in collected.items()},paths


def export_case(args):
    p = load_prepared(args.prepared); variant = args.variant
    if variant not in VARIANTS: raise ValueError('Unknown pilot variant')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for actual pilot export')
    model_dir=Path(args.model_dir).resolve()
    out=fresh(args.output_dir,(ROOT,p['data_root'],p['clip_path'],Path(args.prepared).resolve().parent,model_dir))
    checkpoint=validate_training_artifacts(p,variant,model_dir)
    cfg=case_config(variant,p['data_root'],p['clip_path'],out/'runtime')
    train,val,test,classes,_,samples=_dataset(cfg)
    loader,transform=probe.build_val_loader(cfg,val)
    if probe._decode_evidence(loader)!=p['decoded_val']:raise ValueError('Validation transforms changed')
    set_random_seed(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out.mkdir(parents=True)
    trainer=CoOp_TextConsistency_V2_FG.__new__(CoOp_TextConsistency_V2_FG);TrainerBase.__init__(trainer)
    trainer.cfg=cfg;trainer.dm=SimpleNamespace(dataset=SimpleNamespace(classnames=classes,train_x=train,val=val,test=test))
    trainer.check_cfg(cfg);trainer.build_model();trainer.load_model(model_dir,20)
    if trainer.repro_signature!=p['cases'][variant]['model_signature']:raise ValueError('Constructed model signature differs')
    arrays,paths=export_batches(trainer.model,loader,device='cuda')
    rows=[{'sample_id':str(Path(path).resolve().relative_to(Path(p['data_root']))),
           'label':int(label),'classname':classes[int(label)]} for path,label in zip(paths,arrays['labels'])]
    if rows!=samples['val']:raise ValueError('Export row order differs from prepared validation split')
    np.savez_compressed(out/'logits.npz',**arrays)
    receipt={'schema_version':1,'status':'exported','variant':variant,'protocol':PROTOCOL,
        'prepared_path':str(Path(args.prepared).resolve()),'prepared_identity_sha256':p['prepared_identity_sha256'],
        'source_sha256':SOURCE_SHA256,'class_order':classes,'rows':rows,'checkpoint':checkpoint,
        'logits_sha256':sha256(out/'logits.npz'),'feature_semantics':'S,D both use actual model classification feature; F=.5S+.5D',
        'decoded_val':p['decoded_val'],'device':'cuda'}
    receipt['export_identity_sha256']=_identity(receipt,'export_identity_sha256')
    atomic_json(out/'export.json',receipt)
    load_export(out/'export.json',args.prepared)
    return receipt


def load_export(path, prepared_path=None):
    path=Path(path).resolve();r=_read(path)
    if r.get('export_identity_sha256')!=_identity(r,'export_identity_sha256'):
        raise ValueError('Export receipt identity mismatch')
    p=load_prepared(prepared_path or r['prepared_path'])
    if (r.get('schema_version')!=1 or r.get('status')!='exported' or r.get('variant') not in VARIANTS
            or r.get('protocol')!=PROTOCOL or r.get('source_sha256')!=SOURCE_SHA256
            or r.get('prepared_identity_sha256')!=p['prepared_identity_sha256']
            or r.get('class_order')!=p['class_order'] or r.get('rows')!=p['samples']['val']
            or r.get('decoded_val')!=p['decoded_val'] or r.get('device')!='cuda'):
        raise ValueError('Export protocol/input identity mismatch')
    model_dir=Path(r['checkpoint']['path']).parent.parent
    if validate_training_artifacts(p,r['variant'],model_dir)!=r['checkpoint']:
        raise ValueError('Export checkpoint changed')
    npz=path.parent/'logits.npz'
    if sha256(npz)!=r['logits_sha256']:raise ValueError('Export logits hash mismatch')
    with np.load(npz,allow_pickle=False) as values: arrays={key:values[key].copy() for key in values.files}
    if set(arrays)!={'labels','S','D','F'}:raise ValueError('Missing export branches')
    labels=arrays['labels'];n=len(p['samples']['val']);c=len(p['class_order'])
    if labels.dtype!=np.int64 or not np.array_equal(labels,np.array([row['label'] for row in p['samples']['val']],dtype=np.int64)):
        raise ValueError('Export labels mismatch')
    for key in ('S','D','F'):
        value=arrays[key]
        if value.dtype!=np.float32 or value.shape!=(n,c) or not np.isfinite(value).all():
            raise ValueError('Export arrays must be finite FP32 [N,C]')
    if not np.array_equal(arrays['F'],np.float32(.5)*arrays['S']+np.float32(.5)*arrays['D']):
        raise ValueError('Export fusion is not the fixed rule')
    return r,arrays


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--clip-path',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('export');p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('check');p.add_argument('--prepared',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':value=prepare(args)
    elif args.command=='export':value=export_case(args)
    else:value=load_prepared(args.prepared)
    print(json.dumps({'state':value['status'],'protocol':value['protocol']},indent=2))

if __name__=='__main__':main()
