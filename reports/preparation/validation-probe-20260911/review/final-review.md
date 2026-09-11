# Final whole-branch review

Reviewed base `d905cc8dd0b69aa81b5bd4834cc852574832d9eb` through head `8326ad37ad24ebe2b58886deb581b2c5bbf3748f`, using the precomputed `final-review.diff`, current implementation/tests, plan/spec, archived task reviews, CPU evidence and frozen source integration points. Product files, index and HEAD were not changed. No subagents, server/network, GPU, training or full-suite reruns were used.

### Strengths

- The preparation path rebuilds historical sampling in the correct train16 → val4 → Base order, avoids cache-writing constructors, checks resolved image identities, and decodes the actual validation loader (`scripts/repro/validation_probe.py:426`, `scripts/repro/validation_probe.py:527`). The nine actual CPU preparations substantiate this path across all 27 models and 1,500 validation records.
- Export directly calls the frozen model classes, strictly verifies parameter and class-buffer loading, freezes parameters and uses inference mode. The evaluated tensor/label/path stream is checked before, during and after inference (`scripts/repro/validation_probe.py:570`, `scripts/repro/validation_probe.py:624`, `scripts/repro/validation_probe.py:755`). The original source interfaces match the direct-forward tuple contract.
- All six raw branches and 18 fixed fusions are retained, probability NLL is numerically stable, and the corrected screening guards consistently use `S1+M_logits` against both strong controls (`scripts/repro/validation_analysis.py:212`, `scripts/repro/validation_analysis.py:319`).
- Documentation and readiness receipts distinguish completed CPU preparation from pending CUDA model construction/forward verification. No synthetic metric is presented as a measured result. Historical execution helpers are explicitly archived, rather than advertised as portable deployment commands.

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

1. **Effective configuration binding still permits disabling CLIP normalization.**
   - **File:** `scripts/repro/validation_probe.py:406` (the fixed INPUT checks), with consumption at `scripts/repro/validation_probe.py:464`.
   - The validator checks pixel mean/std and `NO_TRANSFORM`, but never checks `INPUT.TRANSFORMS`. The frozen Dassl `build_transform` reads that list, and `_build_transform_test` applies normalization only if `"normalize"` is present; `"instance_norm"` also changes evaluation behavior. An effective config changed before preparation can therefore omit CLIP normalization, pass the trusted role/config checks, and establish its altered transform as the prepared reference. All later digest comparisons then agree with that altered reference.
   - **Concrete check:** loaded the actual `1406ca3` DTD seed1 prepared config into an in-memory namespace, removed only `"normalize"` from its transforms, and called `validate_effective_role_config` with the actual authenticated case and manifest. It returned successfully. Original list: `["random_resized_crop", "random_flip", "normalize"]`; accepted changed list: `["random_resized_crop", "random_flip"]`.
   - **Impact:** a seemingly successful preparation/export can evaluate a different input distribution while claiming the fixed historical preprocessing protocol. The current nine receipts use the correct list; this finding does not invalidate those actual CPU runs.
   - **Fix:** bind the evaluation-relevant transform choices to the fixed historical configuration during prepare and export. Require normalization and reject instance normalization (or compare the exact supported historical list). Add negative tests for normalization removal and instance-normalization insertion.

2. **Analysis authenticates declared sample hashes without binding them to exported rows/classes.**
   - **File:** `scripts/repro/validation_analysis.py:159`, `scripts/repro/validation_analysis.py:175`.
   - `_validate_trusted_provenance` checks the declared `val_samples_sha256` against the inventory, but `load_export` never recomputes that digest from the actual ordered `(sample_id, label, classname)` rows. It only checks internal row numbering/label/class consistency. The prepared receipt is optional; when present, only its file hash is checked, and its samples/classes are not read or compared. Thus different sample IDs or an internally consistent relabeling can retain the original declared hash and be accepted.
   - **Concrete check:** the existing three-row synthetic DTD payload, with the real DTD seed1 provenance from the test helper and no prepared receipt, was accepted by the real `load_export` validation logic. Actual row digest was `067757f5641e03a1ca570a36692d11a3ee7fa131789c2718aecec04e6bfb1ff8`; the accepted declared historical digest was `28a600bf7140b5155e369de0eeaff9bf31e809f034f671e25568bed82fe54e4b`. I/O was supplied in memory, while the fixed inventory was read and hashed normally; validation itself was not mocked.
   - **Impact:** a stale or incorrectly assembled artifact can be analyzed as the authenticated Base validation case despite containing different images/classes. The current row-permutation test catches changed row indices, but not sample-ID-only changes or coherent metadata permutations.
   - **Fix:** recompute the historical ordered sample digest from export rows and compare it with the pinned inventory; require and validate the corresponding prepared identity/class order, or retain an equivalently authenticated self-contained binding. Add negative cases that alter sample IDs without changing labels, and coherently alter class order/row classnames. Keep tiny synthetic tests by injecting a fixture inventory at the test boundary rather than attaching real provenance to unrelated rows.

3. **Aggregation trusts editable metric tables and ignores their recorded export linkage.**
   - **File:** `scripts/repro/validation_analysis.py:290`, `scripts/repro/validation_analysis.py:353`.
   - `aggregate_files` reads JSON and immediately passes it to `aggregate_reports`; neither validates `export`/`export_sha256`, analysis schema or metric/count/prediction consistency. Provenance fields can remain correct while a metric table is stale, truncated or edited. The newly computed input-file hashes only record what was accepted and do not establish that the tables came from the referenced logits.
   - **Concrete check:** supplied nine full raw/fusion metric tables with the exact fixed provenance and nonexistent referenced exports; changed only UCF seed1 `S1+M_logits.accuracy_pct` from 100 to 1234. `aggregate_reports` returned `status="passed"`, UCF candidate mean `478.0`, and `guards.all_passed=true`. This ran entirely in memory and did not mutate any report file.
   - **Impact:** invalid or stale scores can drive the fixed continue/stop decision even when all declared source/checkpoint hashes are correct. The impossible value demonstrates missing validation; a plausible but incorrect value would likewise pass.
   - **Fix:** validate each analysis against its referenced export/hash and authenticated sample identity, and verify the complete metrics/counts/predictions against the exported arrays before applying guards (factor out a pure analysis computation for reuse). Also reject missing schema, non-finite/out-of-range values and inconsistent counts. Merely bounding accuracy would not catch plausible stale metrics. Add regressions for changed finite metrics, missing/hash-mismatched exports and non-finite metrics.

#### Minor (Nice to Have)

None.

### Verification evidence

- Accepted the recorded focused17/full127 test results and all nine actual CPU preparations on `1406ca3`, including unchanged-input snapshots and explicit CUDA-unavailable refusal. These were not rerun.
- Accepted the controller's archive-publication audit: both archive hashes/sizes, all nine command/log/prepared hashes per archive, safe members, current tool hashes, README links, and unchanged model source.
- Ran only the three concrete in-memory checks above with `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=.:Dassl.pytorch`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `../../tc-venv/bin/python -B -`. Both processes exited 0, demonstrating the unexpected acceptance rather than a test harness failure.
- Frozen transform source checked directly: `outputs/textConsistency-repro/Dassl.pytorch/dassl/data/transforms/transforms.py:191` selects `cfg.INPUT.TRANSFORMS`; lines 343–353 conditionally apply normalization and instance normalization. No original source was modified.
- Actual CUDA construction, GPU forward parity and real logits export remain explicitly unexecuted; that is an accepted phase boundary, not an additional merge-blocking finding here.

The aggregate reproduction used all 24 method entries with `{accuracy_pct: 100, nll: 0, multiclass_brier: 0, probability_top1_top2_margin: 1}` and the existing test helper `_provenance(dataset, seed)` for each of the nine cases. Only the one candidate accuracy was changed to 1234 before calling `aggregate_reports`. The row-binding reproduction used `_synthetic_payload("dtd", 1)` and `_provenance("dtd", 1)`, a valid schema/protocol receipt, an in-memory NPZ context containing the fixture arrays, and ordinary fixed-inventory reads. The config reproduction used the actual receipt at `work/validation-probe-20260911/prepared-1406ca3/dtd-seed1/prepared.json` under the task workspace, converted `effective_configs.coop` recursively to `SimpleNamespace`, removed `normalize`, and invoked `validate_effective_role_config` with that receipt's manifest/case/source root.

### Recommendations

Address the three integrity boundaries in one focused fix wave, preserving the current fixed candidate and research scope. Reuse pure metric computation to verify analysis tables without introducing new inference or training. Existing real CPU evidence can remain archived as historical evidence; tie any updated readiness claims to the final tool hashes and the checks affected by the changes.

### Assessment

**Ready to merge? With fixes.**

**Reasoning:** The actual CPU preparation and direct frozen-model design are sound, and the branch respects the research/GPU boundaries. Three remaining validation gaps can accept altered preprocessing, misidentified rows or invalid metrics as authenticated inputs to the screening decision; these should be fixed before calling the complete toolchain ready.
