"""Independent comparison of copied local/server preparation evidence; no model execution."""
import copy
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
here = Path(__file__).resolve().parent
local_file = root/'work/visual-adapter-pilot-20260913/prepared-v2/prepared.json'
server_dir = here/'server-evidence'
server_file = server_dir/'prepared/prepared.json'
local = json.loads(local_file.read_text())
remote = json.loads(server_file.read_text())
checks = {}

def check(name, actual, expected):
    assert actual == expected, name
    checks[name] = True

for field in ('protocol','source_sha256','class_order','samples','sample_hashes','image_content','description_counts'):
    check(field, remote[field], local[field])

def tools(p):
    return {str(Path(path).relative_to(p['source_root'])): value for path, value in p['tool_files'].items()}

check('operation_and_config_files', tools(remote), tools(local))
for variant in local['cases']:
    check(variant+'_model_signature', remote['cases'][variant]['model_signature'], local['cases'][variant]['model_signature'])
    a = copy.deepcopy(local['cases'][variant]['config']); b = copy.deepcopy(remote['cases'][variant]['config'])
    for config in (a,b):
        config['DATASET']['ROOT'] = '<data>'
        config['OUTPUT_DIR'] = '<output>'
        config['TRAINER']['TEXTCONSISTENCY']['CLIP_PATH'] = '<clip>'
    check(variant+'_effective_config', b, a)

def inputs(p):
    result = {}
    for path, sha in p['input_files'].items():
        file = Path(path)
        if path == p['clip_path']: key = 'CLIP'
        elif file.is_relative_to(p['data_root']): key = 'data/'+str(file.relative_to(p['data_root']))
        else: key = 'source/'+str(file.relative_to(p['source_root']))
        assert key not in result
        result[key] = sha
    return result

a,b = inputs(local),inputs(remote)
check('input_coverage', set(b), set(a))
cache = 'data/fgvc_aircraft/split_fewshot/shot_16-seed_1.pkl'
cache_identity = {'local_sha256':a.pop(cache), 'server_sha256':b.pop(cache),
                  'normalized_cached_rows_match_official_samples_on_each_machine':True}
check('all_noncache_input_bytes', b, a)
# Pickle serializes machine-dependent absolute paths. Both prepare commands independently
# verify its normalized rows against the same official samples and freeze each file's hash.
portable_local = json.loads((here/'local-portable-transform.json').read_text())
portable_remote = json.loads((server_dir/'portable-transform.json').read_text())
check('actual_cpu_validation_tensors_labels_relative_ids', portable_remote, portable_local)
for key in ('count','tensor_shape','tensor_dtype'):
    check('decoded_'+key,remote['decoded_val'][key],local['decoded_val'][key])
ready = json.loads((server_dir/'server-ready.json').read_text())
status = json.loads((server_dir/'cpu-status.json').read_text())
check('six_cpu_stages',len(status['stages']),6)
check('cpu_stages_passed', [stage['state'] for stage in status['stages']], ['passed']*6)
check('no_cuda_guard_exit',[stage['returncode'] for stage in status['stages']],[0,0,0,0,1,0])
check('tests',ready['cpu_tests'],234)
check('no_real_training',ready['gpu_training_started'],False)
check('no_run_output',ready['output_exists'],False)
receipt = {'state':'passed','checks':checks,'local_prepared_sha256':hashlib.sha256(local_file.read_bytes()).hexdigest(),
           'server_prepared_sha256':hashlib.sha256(server_file.read_bytes()).hexdigest(),
           'input_count_each':len(local['input_files']),'noncache_bytes_equal_files':len(a),
           'cache':cache_identity,'portable_transform':portable_local,
           'absolute_path_transform_hashes_are_machine_specific':True,
           'server_plan_sha256':ready['plan_sha256'],'server_plan_identity_sha256':ready['plan_identity_sha256'],
           'training_started':False,'gpu_used':False,'accuracy_results':False}
target = here/'cross-machine-verification.json'
assert not target.exists()
target.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
