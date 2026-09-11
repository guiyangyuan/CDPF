# Task 1 implementation report

## Scope and commit

Implementation commit: `76922fa` (`feat: add frozen validation probe tooling`), based on parent commit `158c49a`.

Owned implementation and tests:

- `scripts/repro/validation_probe.py`
- `scripts/repro/validation_analysis.py`
- `tests/repro/test_validation_probe.py`
- `tests/repro/test_validation_analysis.py`

No model, trainer, dataset, CLIP, historical checkpoint, configuration, or parent-owned preparation report was changed. Parent-owned untracked files under `reports/preparation/validation-probe-20260911/` were left unstaged.

## Implemented behavior

### CPU preparation

`validation_probe.py prepare` accepts one fixed dataset/seed case and a fresh output directory. It validates the inventory case, all three role checkpoints and their sidecars/manifests, epoch and provenance fields, role-specific schemas and losses, model signatures, checkpoint hashes, source hashes, CLIP/template hashes, and exact T0/T1 aggregator tensors.

The command imports the requested frozen source directly with bytecode writing disabled and rejects source-module contamination. It rebuilds the effective source configuration with explicit evaluation overrides. The legacy UCF CoOp declared-KD1/effective-KD0 exception is handled only for the inventory case that records it.

Dataset reconstruction avoids constructors that write caches. It uses the frozen source's fixed-split readers, seeded few-shot generation in the historical order (train16, then val4), and Base-class subsampling. Aircraft uses a read-only instance and source reader. It validates file existence, labels, class order, symlink-resolved train/val disjointness, and exact train/val/test sample digests. Test membership is metadata-only; test images are not decoded.

The source evaluation transform, `DatasetWrapper`, and sequential loader are built for validation only with zero workers. Preparation decodes every validation image and records deterministic transform evidence, source metadata hashes, complete sample inventories, effective configuration, protocol, source/assets/checkpoint provenance, environment data, and the tool hash in `prepared.json`. Preparation never constructs CLIP or runs model inference.

### CUDA-only export

`validation_probe.py export` accepts a prepared receipt and a fresh output directory. It revalidates the prepared inputs and dataset/transform identities before loading the models and after evaluation. CUDA availability is mandatory; there is no CPU inference fallback. A rejection writes `failure.json` and does not write a success receipt.

The exporter constructs the frozen source model classes directly rather than instantiating a Trainer. All source model instances are constructed before the shared CLIP module is moved to CUDA. It manually strict-loads the prompt learner and aggregator `ModuleDict`, including class-dependent token buffers, and rejects missing, unexpected, non-finite, or wrong-shape state. It freezes all parameters, sets evaluation mode, and uses inference mode. Tests make optimizer creation and backward invocation fail immediately if attempted.

Each validation image batch is transferred once to the model device and supplied to each role's direct full forward. The exporter checks shared frozen image features and CLIP logit scale. It exports finite FP32 `C`, `S0`, `S1`, `M`, `T0`, and `T1` arrays plus int64 labels in a non-pickle NPZ. `M` uses the frozen masked normalized descriptor mean and historical CLIP scale. T0 and T1 remain separate, with tensor equality and output equality recorded independently. `export.json` records row identities, class order, precision, configuration, transform, source/assets/checkpoint/tool hashes, and the NPZ hash.

### CPU analysis and aggregation

`validation_analysis.py analyze` validates the complete export receipt and NPZ status, hashes, fixed schema, sample order, class order, shapes, dtypes, and finite values. It reports accuracy, NLL, multiclass Brier score (per-sample class sum, then sample mean), and explicitly named probability top1-top2 margin for all six raw branches and all 18 fixed student/teacher half-fusions. Both logits and probability fusion are included. Probability-fusion NLL uses stable log-softmax and log-add-exp rather than clipping. The analysis also saves complete predictions, student/teacher error overlap and disagreement, and paired rescue/harm counts versus C.

`validation_analysis.py aggregate` requires exactly the nine unique dataset/seed analyses, validates Base/validation protocol, source/assets/checkpoint provenance, compatible settings, and unique role checkpoint hashes, and rejects missing, duplicate, or mixed cases. It records every dataset/seed result and reports three-seed means and sample standard deviations for every branch/combination and metric, followed by equal-dataset macro weighting. The fixed guards compare S1+M logits with C+M logits and probability, require a non-negative macro delta and each dataset mean delta of at least -1 percentage point, and require S1+M logits to exceed S0+M logits macroscopically. T0/T1 do not double-weight guards. The output makes no Novel/HM, significance, or automatic-training claim.

## CLI interfaces

All commands run from the validation checkout with `PYTHONPATH=.:Dassl.pytorch`; the project Python is `../../tc-venv/bin/python`.

Prepare one case:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python -B scripts/repro/validation_probe.py prepare \
  --inventory <proposed-probe-inputs.json> \
  --workspace-root <original-task-workspace> \
  --source-root <frozen-source-root> \
  --data-root <dataset-root> \
  --dataset {ucf101,dtd,fgvc_aircraft} \
  --seed {1,2,3} \
  --output-dir <new-directory>
```

Export one prepared case on CUDA:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python -B scripts/repro/validation_probe.py export \
  --prepared <prepared.json> \
  --output-dir <new-directory>
```

Analyze one completed export on CPU:

```sh
PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python scripts/repro/validation_analysis.py analyze \
  --export <export.json> \
  --output-dir <new-directory>
```

Aggregate the exact nine analyses on CPU:

```sh
PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python scripts/repro/validation_analysis.py aggregate \
  --analyses <nine analysis.json paths> \
  --output-dir <new-directory>
```

Every output directory must be absent before invocation; existing destinations are rejected.

## TDD evidence

The implementation followed `superpowers:test-driven-development` and its `writing-good-tests.md` guidance.

Initial CLI test command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python \
  -m unittest tests.repro.test_validation_probe tests.repro.test_validation_analysis -v
```

The first run was red with two failures because both requested scripts were absent. Minimal CLI skeletons made only the help tests green. Adding the full behavioral tests produced the next red run with ten errors from missing implementation functions before those functions were written.

Further focused red/green cycles covered issues found during review:

- Shared CLIP construction order: red because `instantiate_frozen_models` did not exist, then green after all source models were constructed before moving the shared CLIP module.
- CUDA rejection receipt: red because CUDA-unavailable export did not write `failure.json`, then green after the availability check moved into the receipt-producing failure path.
- Batch device transfer: red because `export_batches` had no device argument, then green after each image batch was moved to the requested device before direct forwards.
- Stable probability-fusion NLL: red because the stable helper did not exist, then green after probability fusion used log-softmax and log-add-exp.

The final focused command passed all 16 targeted tests. These tests include hand-derived metric and fusion fixtures, strict checkpoint loading with tiny real torch modules, non-finite and shape failures, missing validation data, class/sample permutation and hash failures, output collision, CUDA-unavailable failure receipt, no optimizer/backward invariant, shared CLIP construction order, image device transfer, safe NPZ schema, and a complete synthetic export-to-analysis-to-nine-case-aggregate path with negative aggregation cases.

## Real CPU preparation evidence

Two real source/data integration cases were run read-only with `-B` and `PYTHONDONTWRITEBYTECODE=1`. Logs and outputs are outside the checkout and outside all immutable inputs:

- `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/dtd-seed1-prepare.log`
- `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/dtd-seed1-prepared/prepared.json`
- `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/ucf101-seed1-prepare.log`
- `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/ucf101-seed1-prepared/prepared.json`

DTD seed 1 regenerated 24 Base classes, 384 training samples, 96 validation samples, and 864 metadata-only test samples. Its train/val/test digests were `53f6e066...a1ff49`, `28a600bf...54e4b`, and `cdffe9f9...8cf4a`; all 96 validation images decoded as float32 `[3,224,224]`, and all role teacher tensors matched exactly.

UCF101 seed 1 regenerated 51 Base classes, 816 training samples, 204 validation samples, and 1,934 metadata-only test samples. Its train/val/test digests were `7c98face...1cec2`, `768126b3...95a153`, and `5e65b514...149c9`; all 204 validation images decoded as float32 `[3,224,224]`, and all role teacher tensors matched exactly.

These results agree with the independently authenticated historical sampling inventory. The parent will run all nine CPU preparation cases against the reviewed commit.

## Final verification

Focused suite:

```text
Ran 16 tests
OK
```

Complete suite, using the required one-thread environment:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 126 tests in 6.267s
OK
```

`git diff --check` and the equivalent whitespace checks for the new untracked files produced no errors before commit. A prohibited-API scan found no `build_trainer`, `torch.optim`, `.backward(`, or `test_loader` use in either implementation script.

## Remaining limitations

No server or GPU was used in this task. Full CUDA export, real frozen CLIP/model forward execution, direct-forward parity on the historical models, and the resulting measured analysis and aggregate metrics remain pending for a later user-authorized server/GPU turn. This turn establishes local CPU preparation, CPU analysis/aggregation, synthetic export behavior, and explicit CUDA-unavailable rejection; it does not establish GPU readiness or any research result.

The parent next runs only all nine CPU preparation cases plus the CUDA-unavailable rejection against reviewed code. No measured performance claims should be made from the synthetic fixtures or the two preparation receipts.

## Fix round 1: decision and provenance integrity

Review base: `04a9142`. Fix implementation commit: `3888aa3` (`fix: bind validation decisions to trusted inputs`). The fixes address all four Important findings in `task-1-review.md`; the task protocol and CLI signatures did not change.

### Fixed main-candidate guard

The sole candidate is now always `S1+M_logits`. Both the macro and per-dataset probability-baseline deltas subtract `C+M_probability` accuracy from `S1+M_logits` accuracy. `S1+M_probability` remains in the complete exploratory metric table but no longer substitutes for the fixed candidate in any guard. Aggregate output records `fixed_candidate: S1+M_logits`.

The regression fixture uses three classes, with `S1=[-6,-4,-6]` and `M=[1,-2,0]`. Equal-logit fusion predicts class 0 while equal-probability fusion predicts class 1. Across the synthetic nine cases, the fixed logit candidate and `C+M_probability` both score 100%, so the required delta is 0 percentage points and passes. The previous implementation used the wrong probability-fusion candidate, scored 0%, and failed this same guard.

### Authenticated effective configuration

Preparation now requires the exact authenticated inventory file hash. Model signatures are compared as complete mappings, including format, trainer, prompt/view/group counts, aggregation, feature-adapter behavior, ablation role, CLIP/template hashes, and format-2 view-order/version fields.

Each role's source-built effective configuration is then bound to its authenticated manifest signature and the fixed protocol. The validation covers the role trainer, all behavior-bearing `TEXTCONSISTENCY` values, effective losses, legacy UCF CoOp declared/effective KD exception, seed, CUDA/device mode, validation/Base split, shots, deterministic loader and batch size, backbone, evaluation image size/interpolation/normalization, and resolved CLIP path. All three role configurations and their authenticated bindings are persisted; the config and compatibility hashes now cover the complete role set rather than only CoOp.

The same role-config validation runs again after the CUDA overrides during export. Prepared-receipt revalidation also binds the selected inventory case, source/assets, train/val hashes, and every checkpoint/manifest hash back to the authenticated inventory.

### Evaluated transform identity

Export decodes and hashes the complete validation loader before CLIP/model construction. The digest includes each contiguous pre-device transform tensor, label tensor, and ordered path. `export_batches` independently hashes the exact pre-device batches it consumes, and that digest must equal preparation evidence. After inference, export revalidates sample metadata, rebuilds the validation transform/loader, decodes the complete split again, and requires the same transform representation and ordered tensor/label/path digest before it can write success. The evaluated digest is persisted separately as `evaluated_transform`.

Regression coverage changes the image tensor, label order, and path order independently and verifies that each mutation is rejected.

### Trusted aggregate provenance

Analysis and aggregation authenticate the repository's fixed nine-case inventory by its pinned SHA-256 (`858bfbf1f64d71e93c5ca89f706bd757ca69b1f6250b2cadb55a521dcda4d185`). Every export/analysis case must match the exact inventory case digest, source, CLIP and dataset-template hashes, train/val sample hashes, and role-specific checkpoint and manifest hashes. Config and compatibility digests must also be full SHA-256 values. Aggregate still checks compatible settings within each dataset after these exact comparisons.

The negative regression changes every one of the nine cases to consistently wrong CLIP, template, inventory, checkpoint, and manifest hashes. These remain mutually consistent and unique where applicable, but aggregation now rejects them against the trusted inventory before reading metric tables.

### Fix-round TDD evidence

Focused red command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python \
  -m unittest tests.repro.test_validation_probe tests.repro.test_validation_analysis -v
```

The initial regression run was red with 17 tests: one failure and three errors. The guard fixture failed because `all_passed` was false; config validation raised `AttributeError` because `validate_effective_role_config` did not exist; the exporter omitted `decoded_val`; and the uniformly wrong provenance reached later metric-table processing instead of being rejected by provenance validation.

After implementation, the same focused command passed:

```text
Ran 17 tests in 1.626s
OK
```

### Real CPU integration after the fixes

Both historical source formats passed fresh CPU-only preparation with bytecode disabled:

- DTD seed 1 log: `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/dtd-seed1-round1-prepare.log`
- DTD seed 1 receipt: `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/dtd-seed1-round1-prepared/prepared.json`
- UCF101 seed 1 log: `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/ucf101-seed1-round1-prepare.log`
- UCF101 seed 1 receipt: `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/validation-probe-20260911/implementation-attempts/ucf101-seed1-round1-prepared/prepared.json`

DTD format-2 regenerated the same 384/96/864 train/val/test membership and the same ordered transform digest `81b0d6ad...87b3043`. UCF101 format-1 regenerated the same 816/204/1,934 membership and the same ordered transform digest `9c721f82...009805a`. Each receipt contains authenticated bindings for `coop`, `kd0`, and `kd1`.

The successful command form for each case was:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:Dassl.pytorch \
../../tc-venv/bin/python -B scripts/repro/validation_probe.py prepare \
  --inventory reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json \
  --workspace-root /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt \
  --source-root <authenticated-source-root> --data-root <real-data-root> \
  --dataset <dtd-or-ucf101> --seed 1 --output-dir <fresh-attempt-directory>
```

Both processes exited 0. The immutable source and data paths were read only.

### Fix-round final verification

The final complete suite ran once after the implementation and real-source checks:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=.:Dassl.pytorch ../../tc-venv/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 127 tests in 6.214s
OK
```

Python compilation, `git diff --check`, and the prohibited-API scan remained clean. No `build_trainer`, `torch.optim`, `.backward(`, or `test_loader` use was introduced in either implementation script.

### Remaining limitation after fix round 1

No server or GPU was used. Real CUDA model construction, historical direct-forward parity, and full CUDA export remain pending for a later user-authorized server/GPU turn. The parent will rerun all nine CPU preparation cases and CUDA-unavailable rejection against `3888aa3`; those CPU checks cannot establish GPU readiness or measured research performance.

## Final fix wave: preprocessing, row identity, and metric integrity

Review base: `8326ad3`. Final implementation commit: `62771b4` (`fix: authenticate validation analysis inputs`). This wave addresses all three Important findings in `final-review.md` together. The command-line interfaces and fixed research candidate remain unchanged.

### Exact historical input transform

`validate_effective_role_config` now requires the exact frozen `INPUT.TRANSFORMS` list:

```text
random_resized_crop, random_flip, normalize
```

Removing `normalize`, inserting `instance_norm`, reordering the transforms, or adding another transform changes the effective-config protocol and is rejected during preparation and again during export configuration reconstruction. Pixel mean/std, interpolation, size, and `NO_TRANSFORM` remain independently checked.

### Authenticated ordered rows and class order

Analysis now recomputes the historical validation digest from every ordered `(sample_id, label, classname)` export row and compares it with the selected authenticated inventory case. It also requires unique nonempty sample IDs, internal label/class consistency, and complete class-label coverage. An internally coherent class-order relabeling therefore changes the authenticated digest and fails.

When the referenced prepared receipt is locally available, analysis verifies its exact file hash and binds its dataset, seed, protocol, class order, validation rows, validation hash, and inventory case to the export. Relocated or downloaded exports remain analyzable without historical source/model/data directories: the authenticated ordered row digest supplies the self-contained identity binding, a missing original prepared path is not dereferenced, and an unavailable absolute NPZ path falls back to the same-named sibling only after the pinned NPZ hash succeeds.

Synthetic tests no longer attach real historical provenance to unrelated fixture rows. They create a temporary complete nine-case inventory, hash it at the test boundary, and pass that exact path/hash only through keyword-only Python test hooks. Production CLI calls retain the repository's pinned inventory path/hash and expose no inventory override.

### Aggregate metric recomputation

The analysis calculation is factored into the pure `compute_analysis` function. `analyze` uses it to produce all branch metrics, all 18 fusion metrics, predictions, counts, student/teacher audits, and paired C comparisons.

Before aggregation, each analysis must have schema version 1, a present referenced export, the exact recorded export JSON hash, and the current analysis-tool hash. Aggregation loads and fully validates every referenced export and NPZ, then calls the same pure computation and compares the complete derived payload with the submitted analysis. This rejects plausible finite score edits as well as impossible values, non-finite metrics, changed counts, changed predictions, missing schema, missing exports, export-hash changes, row/class changes, and malformed or hash-changed NPZ data. Range checks are not used as a substitute for recomputation.

The fixed guard remains `S1+M_logits` against both `C+M_logits` and `C+M_probability`, plus `S0+M_logits`; no search, tuning, Novel/HM, significance, or automatic-training claim was added.

### Final-wave TDD evidence

Focused red command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:Dassl.pytorch \
../../tc-venv/bin/python -B -m unittest \
  tests.repro.test_validation_probe tests.repro.test_validation_analysis -v
```

The first run was red with 18 tests, two failures, and three errors. Both altered transform lists were accepted. `load_export` lacked the fixture-inventory validation boundary, and `analyze_export`/aggregation lacked the path that recomputes reports against referenced exports.

A separate relocation regression was then added and failed because `load_export` dereferenced the unavailable original absolute NPZ path instead of the downloaded sibling. After the hash-gated relocation fallback was implemented, the complete focused suite passed:

```text
Ran 18 tests in 1.822s
OK
```

The final regressions cover normalization removal, `instance_norm` insertion, sample-ID-only edits, coherent class-order/classname edits, prepared-class mismatch even after updating its file hash, relocated NPZ resolution, plausible finite metric edits, non-finite metrics, count edits, prediction edits, missing analysis schema, missing export files, and changed export hashes. The end-to-end synthetic export-to-analysis-to-nine-case-aggregate test continues to distinguish `S1+M_logits` from `S1+M_probability` and exercises every method table.

### Final-wave complete verification

After the focused suite and static review, the complete suite was run once:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:Dassl.pytorch \
../../tc-venv/bin/python -B -m unittest discover -s tests -v
```

Result: 128 tests passed, exit code 0. The run completed in 11.9 seconds wall time. Python compilation, `git diff --check`, and the prohibited-API scan were also clean. No `build_trainer`, `torch.optim`, `.backward(`, or `test_loader` use was introduced.

### Remaining limitation after the final fix wave

No GPU, server, network, training, source edit, or full CLIP inference was used. Real CUDA model construction, historical direct-forward parity, and full CUDA logits export remain pending for a later authorized GPU turn. The parent will regenerate and archive all nine CPU preparations plus the CUDA-unavailable rejection against `62771b4`; those checks validate preparation and refusal behavior only and do not establish GPU readiness or measured research results.
