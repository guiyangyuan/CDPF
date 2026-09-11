# Task 2 implementation report

Status: DONE_WITH_CONCERNS

Owned implementation files (only these are staged/committed):

- `scripts/repro/attribution_study.py`
- `scripts/repro/attribution_audit.py`
- `tests/repro/test_attribution.py`

This handoff report is intentionally outside the owned-files-only commit. No existing model/source/config file was edited. No server access, GPU run, push, subagent or reviewer was used.

## Implemented fixed protocol

The eleven ordered stages contain exactly `7, 7, 7, 7, 6, 6, 6, 21, 21, 18, 18` raw commands, totaling 124:

1. `smoke_dtd_kd0`, `smoke_fgvc_aircraft_kd0`
2. `anchor_dtd_kd1`, `anchor_fgvc_aircraft_kd1`
3. `fusion_ucf101`, `fusion_dtd`, `fusion_fgvc_aircraft`
4. `formal_dtd_kd0`, `formal_fgvc_aircraft_kd0`
5. `historical_dtd_kd1`, `historical_fgvc_aircraft_kd1`

Every raw command must finish successfully before its stage audit. Every stage audit must pass before the next stage. There are no retries, alternate seeds, tuning controls or branch selection. Historical reload commands point `--model-dir` to the original receipt model, with all new outputs under the new study root. KD command values use float strings (`0.0`, `1.0`), required by the original YACS config types.

## Interfaces

Study CLI:

```text
python scripts/repro/attribution_study.py \
  --source-root SOURCE --data-root DATA --output-root FRESH_STUDY \
  --input-receipt VERIFIED_RECEIPT_JSON \
  --plan-only

python scripts/repro/attribution_study.py \
  --source-root SOURCE --data-root DATA --output-root FRESH_STUDY \
  --input-receipt VERIFIED_RECEIPT_JSON --preparation-dir FRESH_PREPARATION \
  --clip-path COMPLETE_VIT_B_16_PT --prepare-only

# Same protocol/path arguments, existing preparation, fresh study:
... --execute

# Same protocol/path arguments and existing study/preparation:
... --audit-only
```

The modes are mutually exclusive. `--python` optionally selects the child interpreter; default is the invoking Python. Plan mode can run with Python `-S` and requires only the JSON receipt, not a source/data/CLIP installation. The generated source paths are absolute and the actual plan should be generated on the deployment host because operation-script paths and input-receipt paths are frozen.

Deployment must retain these seven operational files together under `scripts/repro/`:

- `attribution_study.py`
- `attribution_audit.py`
- `evaluate_frozen_fusion.py`
- `cross_dataset_study.py`
- `cross_dataset_audit.py`
- `run_pilot.py`
- `structure_spec.py`

The existing source trees must retain their own `train.py`, `scripts/repro/preflight.py`, `scripts/repro/run_pilot.py`, `structure_spec.py`, model/loader/Dassl Python, configs and templates. Use the existing reproduction environment with torch/torchvision/YAML/YACS/Dassl dependencies; no extra dependency or model download is introduced. Do not copy operation files into or alter the pinned model-source folders after preparation; all actual operation/source dependency bytes and paths are frozen.

Independent auditor:

```text
python scripts/repro/attribution_audit.py \
  --study-root STUDY --preparation-dir PREPARATION --stage STAGE_NAME
python scripts/repro/attribution_audit.py \
  --study-root STUDY --preparation-dir PREPARATION --final
```

Importable functions include `build_plan`, `prepare`, `execute`, `verify_frozen`, `child_env`, `compare_kd_configs`, `validate_anchor`, `validate_predictions`, and `final_analysis`.

## Preparation and execution evidence

- Preparation loads real dataset loaders and hashes the actual train/validation/test samples; records counts, classes, description text order/counts/manifest, template SHA, fixed split metadata and image path/size/mtime inventory. `split_fewshot` is an allowed derived cache.
- UCF old baseline source and current full-model source loaders/descriptions must agree. Each source tree is loaded in a separate fresh CPU process, with source-specific PYTHONPATH and one CPU thread; no full CLIP model is loaded. Real subprocess tests verify distinct source modules do not share import caches.
- Complete ordinary configs are resolved through the frozen source `train.setup_cfg`; fusion configs use Task 1 `build_source_configs` in the selected source process. Historical full training/eval YAML is independently checked against source-derived configs. Current full configs are anchored to those historical configs, not merely compared with each other.
- All 18 input models' checkpoint bytes/sidecars and all historical directory files are frozen. Task 1 baseline metadata is checked against each receipt's checkpoint, metrics SHA, test hash and accuracy. Small full checkpoints are CPU-loaded and checked for epoch, signature, full provenance, finite parameters and 790018 parameter count; CLIP is only streamed for its pinned SHA.
- Operational scripts plus reused orchestration/audit dependencies and source Python/config/template dependencies are SHA frozen. Preparation is only written after immutable checks succeed. Existing or overlapping preparation/study paths are rejected.
- Execution rejects missing CUDA before reading preparation or creating the study; the inherited idle-GPU guard checks compute processes and free memory. Child CPU threads are four, CUBLAS workspace is fixed, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is removed. Source bytecode writes are disabled for child processes.
- Protocol/data preflight files exist before training starts. Atomic status records include parent/child PID, stage, started/ended/exit code and completed stages. Separate raw stdout/stderr files are retained for each raw command and audit. Failure stops the queue; interrupted owned child processes are terminated.

## Independent audit

- Ordinary evaluations require exact prepared effective config, dataset/seed/split/branch identity, complete model signature/effective losses, matching real loader hashes, exact class/description order, actual checkpoint bytes/sidecar/epoch/provenance/finite 790018 parameters, and correct/total agreement between evaluation log, raw stdout, metric accuracy and loader count.
- Anchor correct/total and test hashes must exactly equal historical seed 1 for all three branches and both splits. Historical reloads additionally require the original checkpoint hash.
- Fusion audits verify all six branches, actual prepared source config, source/evaluator/CLIP/template/checkpoint hashes, both input hash snapshots, classes/descriptions, predictions JSONL SHA/count, every label and branch prediction, duplicate paths, loader sample order/hash and recomputed correct counts. Student counts/accuracy must equal the frozen historical baseline.
- Final audit independently re-derives source configs and historical full references in fresh source processes, re-audits every stage, and rejects missing/duplicate/unexpected raw commands and failed/missing audits.
- Final output has 36 full seed/branch rows, 54 fusion rows, 3 UCF historical ensemble rows, separate anchor rows, per-branch means/sample SD, separate HM-of-means and mean-seed-HM, and 21 paired comparison groups. Comparisons cover KD1 minus KD0 for all branches, simple fusion minus baseline, probability fusion increments, and full ensemble minus simple/ADK fusion. Anchors do not enter statistical pairs.

## Exact CPU verification

Focused:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_attribution -q
----------------------------------------------------------------------
Ran 20 tests in 5.434s

OK
```

All repro tests (run once, after the substantive implementation and independent final config/history checks):

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest discover -s tests/repro -t . -q
----------------------------------------------------------------------
Ran 104 tests in 4.991s

OK
```

After that all-repro run, minor changes only tightened per-dataset source-root identity, renamed the comparison label to `simple_fusion-baseline`, and exposed metadata-worker stderr on failure; focused verification/syntax checks were repeated as appropriate.

Syntax/whitespace:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m py_compile scripts/repro/attribution_study.py scripts/repro/attribution_audit.py tests/repro/test_attribution.py
# exit 0

git diff --check
# no output, exit 0
```

Tests cover exact command counts/stages/history relocation, no-side-effect plan without site packages/data/CLIP, invalid seed/duplicate/path/source identities, no-GPU/existing-path guards, subprocess environment and real source-process isolation, pre-training protocol/preflight, stop-on-failure, real source cfg seed/shared drift, real small checkpoint tampering, provenance/KD/template/class order, wrong counts, exact anchors, JSONL SHA/count/duplicates/classes/recomputed counts, incomplete/duplicate analysis identities, difference direction and both HM conventions.

## Concerns / deployment follow-up

- No real server preparation or GPU execution was attempted by this worker. Parent should deploy the three operation scripts with their reused dependencies and run a fresh `--prepare-only` on the server before any execution.
- The final audit deliberately rehashes historical inputs, re-runs real loader metadata, and independently re-derives source configs/history. This is CPU/disk work and may be noticeable on the 2 GB server, but full CLIP tensors are never loaded in preparation/audit.
- New and historical full configs permit only documented output/data relocation for historical copied evidence and registered epoch/KD differences; all other drift fails. Historical UCF full references currently require the complete existing three-branch Base/Novel evidence in their model directories, consistent with the supplied audited full-model batch.

## Fix round 1: source-only paths and token-effective description metadata

Status: DONE_WITH_CONCERNS. Scope includes the review's Important P1 and the controller's additional real CPU-preparation failure. The two Minor review findings were not changed.

### P1 source path relocation

- `full_commands` now sets `ROOT` on its freshly loaded `run_pilot` module before calling `build_commands`. The original builder uses that root only for `train.py` and the two source config paths. It no longer rewrites arbitrary command arguments.
- Added a full 124-command plan regression with a distinct immutable source and interpreter/data/study/historical-model paths under the operation checkout. Every output stays under its requested stage/study; new model reloads stay under that study; historical reloads preserve the exact receipt model directory. Interpreter and data paths remain unchanged.
- The new regression first failed against the old implementation: even the interpreter under the operation checkout was rewritten under `immutable-source`. It passes after the fix. No output/source directory was created by the regression.

### Real preparation failure: effective description counts

- The original encoder first calls source `description_rows`, then `clip.tokenize(..., truncate=True)`, then deduplicates token rows within each class while preserving the first occurrence. Preparation had stopped after string-level deduplication.
- Added the same CPU-only token-row deduplication to metadata preparation. Metadata now retains normalized source rows, effective indices, effective rows, effective counts and their exact contiguous order manifest. This never calls CLIP encoding or loads CLIP model weights.
- Recorded the source tokenizer vocabulary SHA and included `clip/bpe_simple_vocab_16e6.txt.gz` in frozen source dependencies, so tokenization inputs cannot change after preparation.
- Actual mismatch against the supplied immutable local backups is confined to Aircraft Novel: `DHC-8-100: 20 -> 19`, `Il-76: 19 -> 18`, `Saab 340: 11 -> 10`. These three pairs differ only in capitalization and become identical under source CLIP tokenization. DTD both splits and Aircraft Base already agree at string-level counts.
- Regression uses the actual shipped Aircraft template/class cases and their historical effective counts. A separate synthetic case covers uppercase/lowercase equivalence, collisions after 77-token truncation, first-occurrence preservation and per-class boundaries. CLIP model-loading functions are patched to raise in the metadata test.
- Read-only CPU verification over **all 36** local DTD/Aircraft historical metrics (3 seeds × 2 splits × 3 branches × 2 datasets) confirmed both token-effective counts and entire order manifests, including their hashes, exactly match history. Historical bytes and provenance checks were not weakened or changed.

Exact focused suite after both fixes:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_attribution -q
----------------------------------------------------------------------
Ran 23 tests in 6.451s

OK
```

Exact syntax/whitespace commands:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m py_compile scripts/repro/attribution_study.py tests/repro/test_attribution.py
git diff --check
# no output, exit 0
```

The local historical verification imported `clip.tokenize`, source `description_rows`/`view_order_manifest`, and the new `token_unique_indices`; read existing metrics under `outputs/textconsistency-crossdataset-study-20260911-55424/runs/crossdataset-study-20260910-v1/formal`; and compared the computed counts/entire manifests to every DTD/Aircraft seed/split/branch metric with model loaders patched to raise. Output:

```text
PASS: 36 immutable local historical metrics match CPU token-effective counts and full order manifests; no CLIP model loaded
```

Only the focused covering suite was run in this fix round; the broader repro suite was not repeated. Controller should deploy a new operational release and create a **fresh preparation directory**, since both the operation-script hash and the preparation evidence schema changed. This worker did not access the server or run GPU work.

## Final consolidated fix wave: both remaining Minor findings

Status: DONE. Addressed both remaining findings from `final-review.md`, with no additional scope.

- Fusion auditing now keeps exact historical `correct`/`total` and exact receipt accuracy identity, while checking both reloaded-student and historical accuracy against those counts using the existing `accuracy_counts` absolute tolerance of `5e-4`. It no longer requires the recomputed student float to have the same representation as a rounded historical float. Test: historical `66.667` and count-derived `200 / 3` pass for exactly `2 / 3`; wrong correct/total, a changed receipt value (even if within numeric tolerance), non-finite accuracy and accuracy outside tolerance still fail.
- Receipt validation now rejects a source-root string unless it exactly equals its canonical absolute resolved spelling. This handles `/./`, trailing spelling differences and symlink aliases before raw source strings can become mismatched dictionary keys. It preserves the immutable original receipt instead of rewriting it. Tests cover both `/./` and a real temporary symlink, canonical acceptance, and unchanged receipt contents after rejection.
- The supplied canonical receipt and exact historical scores are unaffected. No scientific inputs, tuning settings, stage counts or GPU protocol changed.

Exact focused verification:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_attribution -q
----------------------------------------------------------------------
Ran 25 tests in 5.683s

OK
```

Exact syntax/whitespace verification:

```text
PYTHONPATH=.:Dassl.pytorch /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m py_compile scripts/repro/attribution_study.py scripts/repro/attribution_audit.py tests/repro/test_attribution.py
git diff --check
# no output, exit 0
```

Only the focused suite was run, as requested. No full-suite, server or GPU execution was performed by this worker. The controller will perform the final full-suite, deployment and fresh preparation after scoped review; preparation hashes must be refreshed for the changed operation files.
