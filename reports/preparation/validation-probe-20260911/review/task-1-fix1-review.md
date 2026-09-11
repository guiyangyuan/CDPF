### Fix Round 1 Verdict

- **Finding 1 — ADDRESSED.** The fixed probability-baseline delta now uses `S1+M_logits` as the candidate at both macro and dataset levels (`scripts/repro/validation_analysis.py:321`, `scripts/repro/validation_analysis.py:325`), and the aggregate explicitly records that fixed candidate (`scripts/repro/validation_analysis.py:337`). The regression fixture proves logits and probability fusion can choose different classes before asserting the corrected delta (`tests/repro/test_validation_analysis.py:147`, `tests/repro/test_validation_analysis.py:157`).
- **Finding 2 — ADDRESSED.** Preparation now pins the inventory hash, requires the complete authenticated model signature, and validates behavior-bearing role configuration against the signature and protocol (`scripts/repro/validation_probe.py:40`, `scripts/repro/validation_probe.py:348`, `scripts/repro/validation_probe.py:508`). All three bindings are persisted and the same validation runs after CUDA overrides during export (`scripts/repro/validation_probe.py:523`, `scripts/repro/validation_probe.py:746`). Targeted negative coverage includes changed context, view, group, and feature-adapter settings (`tests/repro/test_validation_probe.py:155`).
- **Finding 3 — ADDRESSED.** Export hashes the exact pre-device tensor/label/path stream consumed by inference, compares it with preparation evidence, and repeats loader/transform validation after evaluation (`scripts/repro/validation_probe.py:481`, `scripts/repro/validation_probe.py:495`, `scripts/repro/validation_probe.py:755`, `scripts/repro/validation_probe.py:767`, `scripts/repro/validation_probe.py:777`). The evaluated digest is persisted separately (`scripts/repro/validation_probe.py:785`), with independent tensor, label-order, and path-order mutation coverage (`tests/repro/test_validation_probe.py:247`).
- **Finding 4 — ADDRESSED.** Analysis and aggregation validate every case against the pinned inventory, including exact source, CLIP, template, sample, checkpoint, and manifest hashes (`scripts/repro/validation_analysis.py:114`, `scripts/repro/validation_analysis.py:124`, `scripts/repro/validation_analysis.py:159`, `scripts/repro/validation_analysis.py:272`). The new uniformly-wrong nine-case regression is rejected before metric processing (`tests/repro/test_validation_analysis.py:184`, `tests/repro/test_validation_analysis.py:200`).

### New Breakage Introduced by Fix Diff

- None found.

### Verification Evidence

- Accepted without rerun: focused 17 tests and full 127 tests passed; fresh DTD and UCF101 CPU preparations passed. The controller's nine-case CPU preparation rerun on `1406ca3` is still in progress.
- ⚠️ Real CUDA model construction, direct-forward parity, and full export remain pending. This scoped fix review does not establish GPU readiness.

### Assessment

**Fix status:** Approved — 4/4 original Important findings addressed, 0 not addressed, 0 new regressions found.
