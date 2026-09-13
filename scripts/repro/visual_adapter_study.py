#!/usr/bin/env python3
"""One fixed Aircraft Base-validation visual-adapter pilot; no resume or expansion."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import datetime as dt
from fractions import Fraction
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.repro import visual_adapter_validation as validation

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('simple_fusion', 'visual_only', 'visual_description')
WALLCLOCK_SECONDS = 3600
CONTINUATION_RULE = {
    'metric': 'final_epoch20_base_val_F_accuracy', 'minimum_gain_pp': 1.0,
    'comparisons': ['C-A', 'C-B'], 'automatic_expansion': False,
}


def plan_digest(plan: Mapping[str, Any]) -> str:
    value = dict(plan)
    value.pop('plan_identity_sha256', None)
    return validation.digest(value)


def _disjoint(path: Path, protected: Sequence[Path]) -> Path:
    path = path.resolve()
    for other in protected:
        other = Path(other).resolve()
        if path == other or path in other.parents or other in path.parents:
            raise ValueError(f'Pilot output must be disjoint from input: {other}')
    return path


def _operations(prepared: Mapping[str, Any]) -> dict[str, Any]:
    if Path(prepared['source_root']).resolve() != ROOT:
        raise ValueError('Prepared receipt must bind the current source checkout')
    paths = {'train': ROOT / 'train.py',
             'validation': Path(validation.__file__).resolve(),
             'study': Path(__file__).resolve()}
    return {key: {'path': str(path), 'sha256': validation.sha256(path)}
            for key, path in paths.items()}


def _assemble(prepared_path: Path, output_root: Path, plan_path: Path,
              python: str, *, existing_plan: bool = False) -> dict[str, Any]:
    prepared_path, plan_path = prepared_path.resolve(), plan_path.resolve()
    prepared = validation.load_prepared(prepared_path, verify_inputs=True)
    protected = [prepared_path.parent, Path(prepared['source_root']),
                 Path(prepared['data_root']), Path(prepared['clip_path']),
                 *[Path(path) for path in prepared['input_files']]]
    output_root = _disjoint(output_root, protected)
    if output_root.exists():
        raise FileExistsError(f'Pilot output must be fresh; resume is forbidden: {output_root}')
    plan_directory = _disjoint(plan_path.parent, [*protected, output_root])
    if not existing_plan and plan_directory.exists():
        raise FileExistsError(f'Plan directory must be fresh: {plan_directory}')
    # A venv executable is often a symlink; resolving it loses the venv's site-packages.
    python_path = Path(os.path.abspath(os.path.expanduser(python)))
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError('Python must be an existing executable path')
    operations = _operations(prepared)
    stages = []
    for variant in VARIANTS:
        model = output_root / variant / 'model'
        export = output_root / variant / 'export'
        train_command = validation.build_train_command(prepared, variant, model, python=str(python_path))
        export_command = [str(python_path), operations['validation']['path'], 'export',
                          '--prepared', str(prepared_path), '--variant', variant,
                          '--model-dir', str(model), '--output-dir', str(export)]
        for kind, argv, directory, result in (
            ('train', train_command, model, model / 'textconsistency/model.pth.tar-20'),
            ('export', export_command, export, export / 'export.json'),
        ):
            stages.append({'stage_id': f'{len(stages) + 1:02d}-{variant}-{kind}',
                           'variant': variant, 'kind': kind, 'argv': list(argv),
                           'output_dir': str(directory), 'result_path': str(result)})
    plan = {
        'schema_version': 1, 'state': 'planned_not_executed',
        'study': 'Aircraft seed1 visual-adapter pilot',
        'plan_path': str(plan_path), 'output_root': str(output_root), 'python': str(python_path),
        'python_sha256': validation.sha256(python_path),
        'prepared_path': str(prepared_path), 'prepared_sha256': validation.sha256(prepared_path),
        'prepared_identity_sha256': prepared['prepared_identity_sha256'],
        'source_sha256': prepared['source_sha256'], 'protocol': copy.deepcopy(prepared['protocol']),
        'training_budget': {'planned': 3, 'maximum': 3, 'retries': 0},
        'wallclock_seconds': WALLCLOCK_SECONDS,
        'continuation_rule': copy.deepcopy(CONTINUATION_RULE),
        'operations': operations, 'stages': stages,
        'analysis_path': str(output_root / 'analysis.json'),
    }
    plan['plan_identity_sha256'] = plan_digest(plan)
    return plan


def create_plan(prepared_path: Path, output_root: Path, plan_dir: Path,
                python: str = sys.executable) -> dict[str, Any]:
    """Validate CPU inputs and save the exact queue without creating run outputs."""
    plan = _assemble(Path(prepared_path), Path(output_root), Path(plan_dir) / 'plan.json', python)
    Path(plan['plan_path']).parent.mkdir(parents=True, exist_ok=False)
    validation.atomic_json(Path(plan['plan_path']), plan)
    return plan


def validate_plan(plan_path: Path) -> dict[str, Any]:
    """Rebuild from authenticated inputs; a recomputed digest cannot change the queue."""
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict) or plan.get('plan_identity_sha256') != plan_digest(plan):
        raise ValueError('Pilot plan identity changed')
    try:
        expected = _assemble(Path(plan['prepared_path']), Path(plan['output_root']), path,
                             plan['python'], existing_plan=True)
    except KeyError as exc:
        raise ValueError('Pilot plan is incomplete') from exc
    if plan != expected:
        raise ValueError('Pilot plan commands, protocol, paths, or bound inputs changed')
    return plan


def _check_inputs(plan: Mapping[str, Any], plan_path: Path, plan_file_sha: str) -> dict[str, Any]:
    if validation.sha256(plan_path) != plan_file_sha:
        raise ValueError('Pilot plan file changed during execution')
    path = Path(plan['prepared_path'])
    if validation.sha256(path) != plan['prepared_sha256']:
        raise ValueError('Prepared input changed during execution')
    prepared = validation.load_prepared(path, verify_inputs=True)
    if (prepared['prepared_identity_sha256'] != plan['prepared_identity_sha256'] or
            _operations(prepared) != plan['operations'] or
            validation.sha256(Path(plan['python'])) != plan['python_sha256']):
        raise ValueError('Pilot operation or prepared identity changed during execution')
    return prepared


def analyze_exports(export_paths: Sequence[Path], prepared_path: Path) -> dict[str, Any]:
    """Retain every branch and compute the preregistered decision using exact counts."""
    if len(export_paths) != 3 or len({str(Path(p).resolve()) for p in export_paths}) != 3:
        raise ValueError('Analysis requires exactly three distinct variant exports')
    prepared = validation.load_prepared(Path(prepared_path), verify_inputs=True)
    expected_rows = prepared['samples']['val']
    labels = np.array([row['label'] for row in expected_rows], dtype=np.int64)
    classes = prepared['class_order']
    total = len(labels)
    if total == 0 or len(classes) == 0:
        raise ValueError('Analysis requires nonempty fixed Base-validation samples/classes')
    correct_masks = {}
    variant_metrics = {}
    exports = {}
    for path in export_paths:
        receipt, arrays = validation.load_export(Path(path), prepared_path=Path(prepared_path))
        variant = receipt.get('variant')
        if variant not in VARIANTS or variant in variant_metrics:
            raise ValueError('Analysis requires each of the three expected variants exactly once')
        expected = {'prepared_identity_sha256': prepared['prepared_identity_sha256'],
                    'source_sha256': prepared['source_sha256'], 'protocol': prepared['protocol'],
                    'class_order': classes, 'rows': expected_rows}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Export protocol, sample rows, or class alignment mismatch: {variant}')
        if receipt.get('checkpoint', {}).get('epoch') != 20:
            raise ValueError('Only the fixed final epoch20 checkpoint may be analyzed')
        if not np.array_equal(arrays.get('labels'), labels):
            raise ValueError('Export label alignment mismatch')
        for branch in ('S', 'D', 'F'):
            value = arrays.get(branch)
            if not isinstance(value, np.ndarray) or value.dtype != np.float32 or \
                    value.shape != (total, len(classes)) or not np.isfinite(value).all():
                raise ValueError(f'Invalid aligned finite FP32 branch logits: {variant}/{branch}')
        if not np.array_equal(arrays['F'], 0.5 * arrays['S'] + 0.5 * arrays['D']):
            raise ValueError('F must equal the fixed 0.5*S + 0.5*D fusion')
        correct_masks[variant] = {}
        variant_metrics[variant] = {}
        for branch in ('S', 'D', 'F'):
            mask = arrays[branch].argmax(axis=1) == labels
            count = int(mask.sum())
            correct_masks[variant][branch] = mask
            variant_metrics[variant][branch] = {
                'correct': count, 'total': total, 'accuracy': float(Fraction(count, total)),
                'accuracy_percent': float(Fraction(100 * count, total)),
            }
        exports[variant] = {'path': str(Path(path).resolve()),
                            'checkpoint': copy.deepcopy(receipt['checkpoint']),
                            'logits_sha256': receipt['logits_sha256']}
    comparisons = {}
    for name, new, old in (('B-A', VARIANTS[1], VARIANTS[0]),
                           ('C-A', VARIANTS[2], VARIANTS[0]),
                           ('C-B', VARIANTS[2], VARIANTS[1])):
        comparisons[name] = {}
        for branch in ('S', 'D', 'F'):
            newer, older = correct_masks[new][branch], correct_masks[old][branch]
            corrections = int(np.count_nonzero(newer & ~older))
            harms = int(np.count_nonzero(~newer & older))
            difference = corrections - harms
            comparisons[name][branch] = {
                'correct_difference': difference,
                'accuracy_difference_pp': float(Fraction(100 * difference, total)),
                'corrections': corrections, 'harms': harms,
                'both_correct': int(np.count_nonzero(newer & older)),
                'both_wrong': int(np.count_nonzero(~newer & ~older)),
            }
    passes = all(Fraction(100 * comparisons[name]['F']['correct_difference'], total) >= 1
                 for name in ('C-A', 'C-B'))
    result = {
        'schema_version': 1, 'state': 'analyzed', 'primary_metric': 'accuracy',
        'prepared_identity_sha256': prepared['prepared_identity_sha256'],
        'protocol': copy.deepcopy(prepared['protocol']), 'class_order': classes,
        'rows': expected_rows, 'variants': variant_metrics, 'comparisons': comparisons,
        'exports': exports,
        'continuation': {
            **copy.deepcopy(CONTINUATION_RULE), 'passed': passes,
            'decision': ('signal_for_independent_multiseed_confirmation' if passes
                         else 'stop_expansion'),
            'interpretation': ('This single-seed pilot is only a signal for independent or '
                               'multi-seed confirmation; it establishes neither method novelty '
                               'nor publication readiness.'),
        },
    }
    result['analysis_identity_sha256'] = validation.digest(result)
    return result


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


class _Stopped(RuntimeError):
    pass


def _interruption(signum: int) -> RuntimeError | TimeoutError:
    if signum == signal.SIGALRM:
        return TimeoutError('Pilot exceeded its full-queue wallclock cap')
    return _Stopped(f'Pilot stopped by signal {signum}; no further stages launched')


@contextmanager
def _queue_guard(seconds: float):
    """Bound CPU validation/analysis as well as subprocesses and handle user termination."""
    if os.name != 'posix' or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Pilot execution requires a POSIX main thread for deadline/process guards')
    previous = {}
    state = {'subprocess_critical': False, 'pending': None}
    def terminate(signum, frame):
        if state['subprocess_critical']:
            # Popen creation must return its handle; wait/poll must release CPython's
            # internal waitpid lock before an asynchronous exception can unwind.
            state['pending'] = signum
            return
        raise _interruption(signum)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] > 0:
        raise RuntimeError('An existing process timer prevents installing the pilot deadline')
    try:
        for kind in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
            previous[kind] = signal.signal(kind, terminate)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield state
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def _kill_group(process: subprocess.Popen, deadline: float) -> None:
    """Reap the direct child and terminate descendants even when their parent exited."""
    expired = time.monotonic() >= deadline
    try:
        try:
            os.killpg(process.pid, signal.SIGKILL if expired else signal.SIGTERM)
        except ProcessLookupError:
            pass
        # At the deadline use SIGKILL immediately; cleanup gets no fresh wait budget.
        grace = min(0.15, max(0.0, deadline - time.monotonic()))
        if grace and process.poll() is None:
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        if not expired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        # Also reap the direct child if an OS restriction prevented group signaling.
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _pending_interruption(guard):
    if guard['pending'] is not None:
        raise _interruption(guard['pending'])


def _wait_for_child(process, deadline, guard):
    """Defer asynchronous exceptions inside Popen.wait, checking signals every 0.2s."""
    while True:
        _pending_interruption(guard)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Pilot exceeded its full-queue wallclock cap')
        code = None
        guard['subprocess_critical'] = True
        try:
            try:
                code = process.wait(timeout=min(0.2, remaining))
            except subprocess.TimeoutExpired:
                pass
        finally:
            guard['subprocess_critical'] = False
        _pending_interruption(guard)
        if code is not None:
            return code


def _validate_stage(stage: Mapping[str, Any], prepared: Mapping[str, Any],
                    prepared_path: Path) -> dict[str, Any]:
    if stage['kind'] == 'train':
        return validation.validate_training_artifacts(
            prepared, stage['variant'], Path(stage['output_dir']))
    receipt, _ = validation.load_export(Path(stage['result_path']), prepared_path=prepared_path)
    if receipt.get('variant') != stage['variant']:
        raise ValueError('Export stage produced the wrong variant')
    return {'export': stage['result_path'], 'export_sha256': validation.sha256(Path(stage['result_path'])),
            'checkpoint': receipt['checkpoint'], 'logits_sha256': receipt['logits_sha256']}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_plan(plan_path: Path) -> dict[str, Any]:
    """Run the authenticated queue once, fail closed, and keep durable stage evidence."""
    plan_path = Path(plan_path).resolve()
    plan = validate_plan(plan_path)
    plan_file_sha = validation.sha256(plan_path)
    if not _cuda_available():
        raise RuntimeError('CUDA is required; no output directory or child process was created')
    output = Path(plan['output_root'])
    # The independent check also closes a race after plan validation.
    if output.exists():
        raise FileExistsError(f'Pilot output already exists; resume is forbidden: {output}')
    started = time.monotonic()
    deadline = started + WALLCLOCK_SECONDS
    status = {'schema_version': 1, 'state': 'running', 'started_at': _utc_now(),
              'plan_path': str(plan_path), 'plan_identity_sha256': plan['plan_identity_sha256'],
              'wallclock_seconds': WALLCLOCK_SECONDS, 'stages': []}
    active = None
    current = None
    with _queue_guard(WALLCLOCK_SECONDS) as guard:
        output.mkdir(parents=True, exist_ok=False)
        status_path = output / 'status.json'
        try:
            validation.atomic_json(status_path, status)
            logs = output / 'logs'; logs.mkdir()
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'Dassl.pytorch'),
                                               env.get('PYTHONPATH', '')])
            for stage in plan['stages']:
                prepared = _check_inputs(plan, plan_path, plan_file_sha)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Pilot full-queue wallclock cap reached before next stage')
                if Path(stage['output_dir']).exists():
                    raise FileExistsError(f'Stage output may not be reused: {stage["output_dir"]}')
                log = logs / f'{stage["stage_id"]}.log'
                current = {'stage_id': stage['stage_id'], 'variant': stage['variant'],
                           'kind': stage['kind'], 'state': 'running', 'argv': stage['argv'],
                           'started_at': _utc_now(), 'log_path': str(log)}
                status['stages'].append(current)
                validation.atomic_json(status_path, status)
                with log.open('xb') as handle:
                    guard['subprocess_critical'] = True
                    try:
                        active = subprocess.Popen(stage['argv'], cwd=ROOT, env=env, stdout=handle,
                                                  stderr=subprocess.STDOUT, start_new_session=True)
                    finally:
                        guard['subprocess_critical'] = False
                    _pending_interruption(guard)
                    current['pid'] = active.pid
                    validation.atomic_json(status_path, status)
                    code = _wait_for_child(active, deadline, guard)
                    current['returncode'] = code
                    guard['subprocess_critical'] = True
                    try:
                        _kill_group(active, deadline)
                    finally:
                        guard['subprocess_critical'] = False
                    _pending_interruption(guard)
                    active = None
                if code != 0:
                    raise RuntimeError(f'Pilot stage {stage["stage_id"]} failed with exit code {code}')
                prepared = _check_inputs(plan, plan_path, plan_file_sha)
                current['artifacts'] = _validate_stage(stage, prepared, Path(plan['prepared_path']))
                current['state'] = 'completed'
                current['finished_at'] = _utc_now()
                validation.atomic_json(status_path, status)
                current = None
            analysis = analyze_exports([Path(s['result_path']) for s in plan['stages']
                                        if s['kind'] == 'export'], Path(plan['prepared_path']))
            _check_inputs(plan, plan_path, plan_file_sha)
            validation.atomic_json(Path(plan['analysis_path']), analysis)
            status['analysis_path'] = plan['analysis_path']
            status['continuation'] = analysis['continuation']
            status['state'] = 'completed'
        except BaseException as exc:
            # Disable the alarm before cleanup so it cannot interrupt process-group termination.
            signal.setitimer(signal.ITIMER_REAL, 0)
            if active is not None:
                guard['subprocess_critical'] = True
                try:
                    _kill_group(active, deadline)
                except OSError as cleanup_error:
                    status['cleanup_error'] = str(cleanup_error)
                    status['process_group_cleanup_confirmed'] = False
                finally:
                    guard['subprocess_critical'] = False
                if current is not None:
                    current['returncode'] = active.returncode
            if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
                status['state'] = 'timed_out'
            elif isinstance(exc, (_Stopped, KeyboardInterrupt, SystemExit)):
                status['state'] = 'stopped'
            else:
                status['state'] = 'failed'
            if current is not None:
                current['state'] = status['state']
                current['finished_at'] = _utc_now()
            status['error'] = f'{type(exc).__name__}: {exc}'
            if isinstance(exc, subprocess.TimeoutExpired):
                raise TimeoutError('Pilot exceeded its full-queue wallclock cap') from exc
            raise
        finally:
            status['finished_at'] = _utc_now()
            status['elapsed_seconds'] = time.monotonic() - started
            validation.atomic_json(status_path, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--prepared', type=Path, required=True)
    plan.add_argument('--output-root', type=Path, required=True)
    plan.add_argument('--plan-dir', type=Path, required=True)
    plan.add_argument('--python', default=sys.executable)
    run = commands.add_parser('run')
    run.add_argument('--plan', type=Path, required=True)
    analyze = commands.add_parser('analyze')
    analyze.add_argument('--prepared', type=Path, required=True)
    analyze.add_argument('--exports', type=Path, nargs=3, required=True)
    analyze.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        result = create_plan(args.prepared, args.output_root, args.plan_dir, args.python)
    elif args.command == 'run':
        result = run_plan(args.plan)
    else:
        result = analyze_exports(args.exports, args.prepared)
        output = _disjoint(args.output_dir, [args.prepared.parent, *[p.parent for p in args.exports], ROOT])
        if output.exists():
            raise FileExistsError('Analysis output must be fresh')
        output.mkdir(parents=True, exist_ok=False)
        validation.atomic_json(output / 'analysis.json', result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
