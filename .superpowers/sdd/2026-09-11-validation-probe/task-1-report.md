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
