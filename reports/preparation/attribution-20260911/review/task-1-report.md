# Task 1 implementation report

Status: DONE_WITH_CONCERNS

## Changed files

- `scripts/repro/evaluate_frozen_fusion.py` (owned, committed)
- `tests/repro/test_frozen_fusion.py` (owned, committed)
- `.superpowers/sdd/2026-09-11-attribution/task-1-report.md` (handoff report, intentionally not included in the owned-files-only commit)

No existing training source, configuration, metrics, checkpoint, or dataset file was modified.

## Implementation

- Added importable `frozen_fusion_logits(student_logits, image_features, description_features, view_mask, logit_scale)` with six fixed branches: `student`, `description_mean`, `simple_fusion`, `description_instance`, `mean_probability_fusion`, and `adk_probability_fusion`.
- The formulas normalize image and description features, mask invalid descriptions, share the exact positive `exp(CLIP.logit_scale)` scale, use fixed alpha 0.5, and expose attention/features for independent CPU verification.
- The ADK-named result is explicitly identified as a controlled CoOp adaptation of public formulas, not official code or published-score replication. Metrics preserve the actual numeric scale and `tau=1/scale` interpretation.
- Added a standalone CLI with the required source/baseline/dataset/seed/split/data/output/source-SHA arguments and optional `--validate-only`. It exposes no epoch, alpha, training, or search controls.
- Metadata validation checks the pinned source digest, baseline dataset/seed/shot/signature/effective losses/sample hashes, source-derived train/eval configs, historical student metrics plus log counts, checkpoint bytes plus SHA sidecar and epoch evidence. It does not load checkpoint tensors or CLIP.
- Full execution refuses missing CUDA before creating output, invokes the selected source tree's `run_pilot.build_commands`, `train.setup_cfg`, original baseline trainer, and original `load_model`, then checks live loader/model provenance and full finite 2048-parameter checkpoint tensors.
- Full output is fresh and outside source/baseline/data. It writes all six predictions for every loader test image as data-root-relative JSONL, verifies counts/accuracies, records the JSONL SHA, complete effective config, class/description order, hashes, formulas, environment, and input before/after checks. Post-output failures write `status: failed`; incomplete work is never marked passed.
- Old `d74...` UCF encoders lack `view_order_manifest`; the evaluator derives and hashes the equivalent explicit contiguous `[0..count-1]` evidence from frozen effective counts without changing old source.
- Narrow approved UCF compatibility: historical effective YAML declares `KD_WEIGHT=1.0` while the available source-derived baseline command declares `0.0`; acceptance requires UCF only, exact equality of every other config field, and manifest/checkpoint effective KD `0.0`. Output preserves historical declared, source-derived/current, and effective values. DTD/Aircraft and nonzero effective KD remain rejected.

## Exact verification

Focused tests:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_frozen_fusion -q
----------------------------------------------------------------------
Ran 21 tests in 0.103s

OK
```

All repro tests (59 pre-existing plus 21 new):

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest discover -s tests/repro -t . -q
----------------------------------------------------------------------
Ran 80 tests in 2.699s

OK
```

Syntax and whitespace:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m py_compile scripts/repro/evaluate_frozen_fusion.py tests/repro/test_frozen_fusion.py
git diff --check
(no output; exit 0)
```

Real frozen UCF metadata validation, seed 1 base:

```text
status=validated_metadata
source_sha256=d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f
checkpoint_sha256=84cc4b8201232832949cda3585a989d11f3089984639addb1409ec27f7ea89df
historical_student=1606/1934 (83.04033092037228)
output directory absent after validation
```

Real frozen UCF metadata validation, seed 1 new:

```text
status=validated_metadata
source_sha256=d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f
checkpoint_sha256=84cc4b8201232832949cda3585a989d11f3089984639addb1409ec27f7ea89df
historical_student=1189/1849 (64.30502974580854)
output directory absent after validation
```

## Key interfaces

- Python: `frozen_fusion_logits(...) -> dict[str, torch.Tensor]`
- Python metadata-only: `validate_baseline_metadata(...) -> dict`
- Python old/new order bridge: `description_order_evidence(encoder, classnames) -> dict`
- CLI: `scripts/repro/evaluate_frozen_fusion.py --source-root ... --baseline-dir ... --dataset {ucf101,dtd,fgvc_aircraft} --seed N --split {base,new} --data-root ... --output-dir ... --expected-source-sha256 ... [--validate-only]`
- Full `metrics.json`: top-level `status`, dataset identity, fixed protocol/scale, method identity, formulas, six-branch `metrics`, historical student, checkpoint/source/assets, complete effective config, declared/effective KD evidence, description/class order, sample hashes, prediction manifest, input before/after hashes, and environment.

## Concerns

- Per assignment, no GPU execution was performed locally. The full six-branch tensor path therefore still requires the parent's server run. CPU math, guards, real UCF source/config/checkpoint metadata, and both historical split records were verified.
- The ADK temperature prose is ambiguous; implementation pins the required standard CLIP interpretation `tau=1/exp(logit_scale)`, records the actual scale, and performs no search.

## Round 1 review fix: frozen seed domain

- Restricted the CLI `--seed` choices and both public evaluation/metadata entry points to exactly seeds `1`, `2`, and `3`.
- Invalid seeds `-1`, `0`, and `4` are rejected before source/checkpoint metadata reads and before output directory creation. Tests also exercise successful metadata validation for each valid seed.

Exact focused verification after the fix:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_frozen_fusion -q
----------------------------------------------------------------------
Ran 25 tests in 0.157s

OK
```
