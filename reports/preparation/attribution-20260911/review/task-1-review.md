### Spec Compliance

- ❌ Issues found: the fixed seed protocol is not enforced. `scripts/repro/evaluate_frozen_fusion.py:258` accepts every nonnegative seed and `scripts/repro/evaluate_frozen_fusion.py:668` exposes an unrestricted integer CLI argument, although this task is bound to seeds 1, 2, and 3. A self-consistent seed-0 or seed-4 baseline can therefore pass provenance checks and be evaluated as if it belonged to the frozen comparison.
- ⚠️ Cannot verify from diff: the full CUDA path, six-branch predictions, and final artifact hashes have not run on a GPU. The controller should perform one planned frozen server evaluation before treating runtime output as validated; the reported CPU tests and metadata-only UCF checks do not exercise `scripts/repro/evaluate_frozen_fusion.py:493-659`.

### Strengths

- `scripts/repro/evaluate_frozen_fusion.py:74-141` implements the six requested branches in one pure tensor path, including per-description normalization, masked mean, shared CLIP scale, description attention, fixed logit fusion, and both probability fusions. `tests/repro/test_frozen_fusion.py:17-94` independently checks the formulas, invalid-mask isolation, finite-input guards, and zero description-term gradient to the student context proxy.
- `scripts/repro/evaluate_frozen_fusion.py:245-329` validates provenance, signatures, sample hashes, checkpoint bytes and sidecar, epoch 20, historical student metrics, and exact source-derived configs. The UCF exception remains field-specific and still requires effective KD zero through the common provenance validation at `scripts/repro/evaluate_frozen_fusion.py:196-220`.
- `scripts/repro/evaluate_frozen_fusion.py:466-493` checks a fresh disjoint output, pinned source digest, source-derived configs, historical metadata, and CUDA before creating output. `scripts/repro/evaluate_frozen_fusion.py:507-516` then uses the frozen trainer's original strict `load_model` and original `CachedMultiViewTextEncoder`.
- `scripts/repro/evaluate_frozen_fusion.py:549-659` saves all six per-sample predictions and metrics, checks the reloaded student against historical correct/total/accuracy, hashes the JSONL, checks source/data/baseline inputs before and after inference, records method identity and formulas, and writes `passed` only after those checks.
- Focused unchanged-source check for model invocation: both pinned trainer variants' `trainers/coop_textconsistency_ablation.py:71-81` return `(student, teacher, features, attention)`, matching the evaluator call, and their original loader performs signature/provenance/key/shape validation in `trainers/textconsistency_common.py:84-123`.
- Focused unchanged-source check for UCF compatibility: the pinned `scripts/repro/run_pilot.py:24-55` derives baseline KD weight 0 even when its general KD argument is 1, while `trainers/textconsistency_runtime.py:72-96` records the baseline group's effective KD as 0.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `scripts/repro/evaluate_frozen_fusion.py:258,668`: restrict `seed` to `{1, 2, 3}` in the validation boundary and CLI. Add a focused test using an otherwise self-consistent seed-4 fixture; the existing wrong-seed test at `tests/repro/test_frozen_fusion.py:177` only proves that requested provenance must match stored provenance, so it does not catch this protocol escape.

#### Minor (Nice to Have)

- None.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The evaluator is careful and closely matches the frozen formulas and evidence requirements, and the focused source checks support its two integration assumptions. The unrestricted seed admits runs outside the explicitly frozen 1/2/3 experiment and should be fixed before the downstream queue treats this interface as authoritative.
