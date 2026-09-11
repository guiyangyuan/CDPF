- **Effective configuration permits disabling CLIP normalization — ADDRESSED.** `scripts/repro/validation_probe.py:412` now compares the exact historical `INPUT.TRANSFORMS` list; the existing prepare/export calls both enforce this validator. Regression assertions reject normalization removal and instance-normalization insertion (`tests/repro/test_validation_probe.py:182`).
- **Exported rows/classes are not bound to the authenticated validation sample hash — ADDRESSED.** `scripts/repro/validation_analysis.py:154` uses the historical ordered tuple serialization, and `scripts/repro/validation_analysis.py:227` requires its digest to equal the trusted case's validation hash, with complete label coverage. Available prepared receipts additionally bind their hash, case, rows and class order (`scripts/repro/validation_analysis.py:159`). Sample-ID-only changes, coherent class-name/order changes and prepared-class mismatches are covered (`tests/repro/test_validation_analysis.py:166`, `tests/repro/test_validation_analysis.py:171`, `tests/repro/test_validation_analysis.py:200`). Missing historical prepared paths no longer prevent local CPU analysis; the authenticated row digest remains mandatory. NPZ relocation falls back only when the original absolute file is absent and still verifies the recorded content hash (`scripts/repro/validation_analysis.py:202`).
- **Aggregation trusts editable metric tables without verifying export linkage — ADDRESSED.** `scripts/repro/validation_analysis.py:320` validates schema, referenced export existence/hash, its NPZ and sample identity, then recomputes the complete analysis through `compute_analysis` and checks every derived field plus the analysis-tool hash. Every case passes this check before guards run (`scripts/repro/validation_analysis.py:383`). Regressions cover a plausible finite score edit, NaN, count/prediction edits, missing export, wrong export hash and missing schema (`tests/repro/test_validation_analysis.py:241`).

### New Breakage in the Fix Diff

- None found. Pure analysis extraction preserves the six raw branches, 18 fusion combinations and fixed candidate/guards. Fixture inventory hooks are Python keyword arguments; the production CLI retains the pinned inventory and exposes no override (`scripts/repro/validation_analysis.py:459`).

### Out-of-Scope Observations

- None.

### Checks

- Reviewed the supplied `8326ad3..ce31832` fix package and appended implementation report; no additional Git diff or broad review was performed.
- Verified the reported regression coverage against the changed tests. Accepted recorded focused18 passing in 1.822 seconds and full128 passing with exit 0 in 11.9 seconds; suites were not rerun. No unresolved code-reading doubt required another test.
- Controller reports final `ce31832` nine-case CPU preparation passed in 41.89 seconds: 27 models, 1,500 validation records, 1,280 unique images and nine identical teacher pairs. Actual no-CUDA CLI refusal passed with no NPZ/success artifact; immutable before/after inputs match. These controller checks were not independently rerun.
- Real CUDA construction, forward parity and logits export remain explicitly pending; this verdict does not establish GPU readiness.

### Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.** Three of three original findings addressed; no open findings. Reviewed implementation `62771b4` and report head `ce31832`; only this review report was written.
