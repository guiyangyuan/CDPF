### Re-review Verdict

- ✅ Fix round 1 is approved with one Minor evidence-wording correction. All prior Important issues 2–6 and the seed-label issue are addressed. Prior issue 1 is addressed under the binding clarification that deterministic `split_fewshot` caches are authorized derived preparation outputs.
- ✅ No new Critical or Important breakage appears in `task-1-fix1.diff`.

### Prior Findings

- **Issue 1 — ADDRESSED by ruling and implementation.** The updated constraint expressly permits loaders to generate/reuse deterministic `split_fewshot` caches. Preparation records that policy and inventories other dataset files before/after loader evidence collection (`scripts/repro/cross_dataset_study.py:103`, `scripts/repro/cross_dataset_study.py:153`). No loader rewrite or cache redirection is required.
- **Issue 2 — ADDRESSED.** `structure_spec.py` is now part of the immutable source-root dependency set, and both externally deployed operational scripts have independent byte hashes recorded at planning/preparation and rechecked before execute/audit (`scripts/repro/cross_dataset_study.py:58`, `scripts/repro/cross_dataset_study.py:73`, `scripts/repro/cross_dataset_study.py:263`, `scripts/repro/cross_dataset_audit.py:29`). This works with an old immutable `--source-root`: source files remain resolved beneath that root, while `cross_dataset_study.py` and its colocated `cross_dataset_audit.py` resolve from the external operational directory.
- **Issue 3 — ADDRESSED.** The plan freezes `order_settings()` for all variants; scoped protocols carry those records; `summarize.collect` therefore activates its exact order-policy, signature, and manifest checks; the external audit additionally compares loader class order/counts (`scripts/repro/cross_dataset_study.py:99`, `scripts/repro/cross_dataset_study.py:177`, `scripts/repro/cross_dataset_audit.py:109`, `scripts/repro/cross_dataset_audit.py:146`).
- **Issue 4 — ADDRESSED.** Execution now rejects active compute processes using `nvidia-smi`, fails closed when that query cannot be made, then retains the 90%-free-memory check (`scripts/repro/cross_dataset_study.py:184`). The focused test covers an active low-memory process and an idle GPU (`tests/repro/test_cross_dataset.py:53`).
- **Issue 5 — ADDRESSED.** Each dataset stage atomically writes its scoped protocol and prepared dataset evidence before the first raw training command; audit is read-only for those inputs and rejects either missing or changed (`scripts/repro/cross_dataset_study.py:217`, `scripts/repro/cross_dataset_audit.py:106`). The test verifies producer ordering before command launch (`tests/repro/test_cross_dataset.py:79`).
- **Issue 6 — ADDRESSED.** Stage and final audit subprocesses now use named stdout/stderr logs, record those paths and exit status, and failures flow through the structured failed-status handler (`scripts/repro/cross_dataset_study.py:231`, `scripts/repro/cross_dataset_study.py:242`, `scripts/repro/cross_dataset_study.py:278`).
- **Seed labels — ADDRESSED.** Every paired difference row now includes its explicit seed (`scripts/repro/cross_dataset_audit.py:183`).

### Strengths

- The fix closes the exact producer/consumer gap created by calling `build_commands` directly without reintroducing command duplication (`scripts/repro/cross_dataset_study.py:88`, `scripts/repro/cross_dataset_study.py:221`).
- Frozen-code verification is performed independently by standalone audit, so `--audit-only` cannot silently use changed source or operational scripts (`scripts/repro/cross_dataset_audit.py:29`, `scripts/repro/cross_dataset_audit.py:198`).
- Exception handling now preserves actionable state even for launch errors and interrupts after study creation (`scripts/repro/cross_dataset_study.py:280`).

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

None.

#### Minor (Nice to Have)

- `scripts/repro/cross_dataset_study.py:134` and `scripts/repro/cross_dataset_study.py:155`: `immutable_input_inventory.sha256` hashes JSON containing relative path, file size, and `mtime_ns`; it is an inventory digest, not a byte-content digest of dataset images. The preparation policy text says images are “hash-checked,” which overstates this evidence. Describe images as inventory-checked (path/size/mtime), while retaining byte-hash language for fixed split metadata/templates or separately citing the controller's already verified source archive ZIP digest. A multi-GB image rehash is not required for this task.

### Remaining Cannot-Verify Items

- ⚠️ The diff cannot verify actual DTD/Aircraft loader hashes, counts, exact class lists, or that legitimately different seed/split hashes were produced. The controller should retain the real local/server preparation evidence for these values.
- ⚠️ The diff cannot verify the server's complete CLIP checkpoint bytes, pinned model/source hashes, or the deployed operational-script hashes. These must match during CPU preparation; the new code fails closed on later drift.
- ⚠️ Checkpoint tensor layout/counts, real eval logs, all 84 formal evaluations, and final analysis remain unavailable until GPU training exists. GPU execution remains intentionally deferred.

### Assessment

**Task quality:** Approved

**Reasoning:** The fix resolves every prior execution and audit blocker without changing the immutable training source or duplicating runner logic. CPU-only deployment/preparation may proceed under the clarified cache policy; report the image evidence accurately as an inventory check, and continue to defer GPU execution until GPUs are enabled and the runtime gates pass.
