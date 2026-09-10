"""Bounded execution: verified prepared source -> GPU smoke -> fixed study."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

ROOT = Path('/root/autodl-tmp/textconsistency')
SOURCE = ROOT / 'releases/structure-20260910-v1'
STUDY = ROOT / 'runs/structure-study-20260910-27946-v2'
EXPECTED = '9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba'
ENV = os.environ.copy()
ENV.update(PYTHONPATH=f'{SOURCE}:{SOURCE}/Dassl.pytorch', OMP_NUM_THREADS='4',
           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
ENV.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
os.environ.update({k: ENV[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
sys.path[:0] = [str(SOURCE), str(SOURCE / 'Dassl.pytorch')]


def main():
    STUDY.mkdir(parents=True, exist_ok=False)
    state = {'state': 'running', 'stage': 'runtime_check', 'pid': os.getpid(), 'started_unix': time.time(),
             'source_commit': '13b34978eeb8b23532b53abd1066af4719a0f909', 'source_sha256': EXPECTED,
             'completed': [], 'thread_caps': 4, 'runs': {}, 'new_outputs_only': True}
    def save():
        tmp = STUDY / 'status.json.tmp'; tmp.write_text(json.dumps(state, indent=2)); tmp.replace(STUDY / 'status.json')
    save()
    stopped = threading.Event()
    def sample_gpu():
        with (STUDY / 'gpu-samples.jsonl').open('w') as log:
            while not stopped.is_set():
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True)
                log.write(json.dumps({'unix': time.time(), 'sample': result.stdout.strip(), 'returncode': result.returncode})+'\n')
                log.flush(); stopped.wait(5)
    monitor = None
    try:
        import torch
        from trainers.textconsistency_runtime import source_digest
        from trainers.textconsistency_common import sha256_file
        from scripts.repro.summarize import collect
        from scripts.repro.structure_spec import STRUCTURE_VARIANTS
        assert torch.cuda.is_available(), 'CUDA unavailable'
        assert source_digest() == EXPECTED, 'Prepared training source changed'
        ready = json.loads((ROOT / 'setup-logs/structure-prep-20260910/ready.json').read_text())
        for name, digest in ready['files'].items():
            assert sha256_file(SOURCE / name) == digest, name
        assert sha256_file(ROOT / 'assets/ViT-B-16.pt') == ready['clip_sha256']
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        assert not active, f'GPU already has compute jobs: {active}'
        runtime = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0),
                   'memory_limit_bytes': int(Path('/sys/fs/cgroup/memory.max').read_text()),
                   'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
                   'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True), 'code_sha256': EXPECTED}
        # A real CUDA backward step verifies runtime support before full CLIP smoke.
        value = torch.randn(128, 128, device='cuda', requires_grad=True)
        loss = (value @ value.T).square().mean(); loss.backward(); torch.cuda.synchronize()
        assert torch.isfinite(value.grad).all()
        del value, loss
        torch.cuda.empty_cache()
        runtime['cuda_forward_backward_passed'] = True
        (STUDY / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        monitor = threading.Thread(target=sample_gpu, daemon=True); monitor.start()
        for stage, options, epoch, seeds in [
                ('smoke', ['--variants', *STRUCTURE_VARIANTS, '--seeds', '1', '--epochs', '1'], 1, [1]),
                ('formal', ['--structure-study', '--seeds', '1', '2', '3'], 20, [1, 2, 3])]:
            output = STUDY / stage
            assert not output.exists()
            command = [sys.executable, '-u', str(SOURCE / 'scripts/repro/run_pilot.py'),
                       '--data-root', str(ROOT / 'data'), '--output-root', str(output), *options]
            started = time.time()
            state.update(stage=stage); state['runs'][stage] = {'command': command, 'started_unix': started}; save()
            with (STUDY / f'{stage}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=SOURCE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                state['runs'][stage]['pid'] = child.pid; save()
                code = child.wait()
            state['runs'][stage].update(returncode=code, seconds=time.time()-started); save()
            if code: raise RuntimeError(f'{stage} runner exited {code}; inspect its log')
            assert len(collect(output)) == 15
            audits = []
            for variant in STRUCTURE_VARIANTS:
                for seed in seeds:
                    folder = output / variant / f'seed{seed}'
                    manifest = json.loads((folder / 'run_manifest.json').read_text())
                    assert manifest['trainable_parameters'] == 790018
                    checkpoint = folder / 'textconsistency' / f'model.pth.tar-{epoch}'
                    digest = sha256_file(checkpoint)
                    for split in ('base', 'new'):
                        for branch in ('student', 'teacher', 'ensemble'):
                            data = json.loads((folder / f'eval_{split}_{branch}/metrics.json').read_text())
                            assert data['checkpoints'][0]['sha256'] == digest and data['checkpoints'][0]['epoch'] == epoch
                    audits.append({'variant': variant, 'seed': seed, 'epoch': epoch, 'sha256': digest})
            (STUDY / f'{stage}-audit.json').write_text(json.dumps({'passed': True, 'models': audits}, indent=2))
            state['completed'].append(stage); save()
        state.update(state='completed', stage='awaiting_analysis_and_backup', finished_unix=time.time())
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        stopped.set()
        if monitor is not None: monitor.join(timeout=10)
        save()


if __name__ == '__main__':
    main()
