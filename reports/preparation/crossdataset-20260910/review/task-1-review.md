### Spec Compliance

- ❌ Issues found: preparation can write few-shot cache directories/files into the original dataset root; the frozen dependency set omits operational code that directly controls commands and auditing; the audit does not pin the required description-order policy; and the GPU-idle gate does not establish that GPUs have no active compute users (`scripts/repro/cross_dataset_study.py:58`, `scripts/repro/cross_dataset_study.py:92`, `scripts/repro/cross_dataset_study.py:166`, `scripts/repro/cross_dataset_audit.py:97`).
- ⚠️ Cannot verify from diff: actual DTD/Aircraft loader hashes/counts/classes, full CLIP bytes, source hashes, checkpoint tensor layout/counts, and real training/evaluation artifacts do not exist in this review package. The controller should compare the already collected real-data evidence and source hashes, then run CPU preparation only after the preparation-side-effect issue is resolved. GPU smoke/formal execution remains unverified and should not begin while the Important issues below remain.

### Strengths

- The plan is deterministic, dataset-separated, and reuses `run_pilot.build_commands`; the four stages and 17/17/51/51 command counts encode the required 6/28 smoke and 18/84 formal train/eval totals (`scripts/repro/cross_dataset_study.py:72`).
- Execution refuses an existing study root and unavailable CUDA before creating that root, recomputes real loader evidence, fixes child CPU-thread variables, sets deterministic CUBLAS behavior, and removes the unsafe inherited Torch override (`scripts/repro/cross_dataset_study.py:149`, `scripts/repro/cross_dataset_study.py:160`, `scripts/repro/cross_dataset_study.py:173`).
- The audit calls the existing `summarize.collect` and adds checkpoint-byte/sidecar/epoch/provenance/signature checks, explicit CPU `weights_only=False` loading, finite tensor and variant-specific parameter counts, prepared sample-hash/count checks, and log-to-metric accuracy reconciliation (`scripts/repro/cross_dataset_audit.py:56`, `scripts/repro/cross_dataset_audit.py:104`, `scripts/repro/cross_dataset_audit.py:143`).
- Final analysis requires the complete 42 dataset/seed/branch rows and 14 summaries, uses only preregistered primary branches, and computes paired Base/Novel/HM means and sample SDs (`scripts/repro/cross_dataset_audit.py:160`).
- Focused tests cover the plan shape/no study output, existing-root/no-CUDA rejection, incomplete results, checkpoint byte/count tampering, loader-derived counts, sample hashes/split, and optimizer drift (`tests/repro/test_cross_dataset.py:17`, `tests/repro/test_cross_dataset.py:53`, `tests/repro/test_cross_dataset.py:68`, `tests/repro/test_cross_dataset.py:95`, `tests/repro/test_cross_dataset.py:107`).

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

- `scripts/repro/cross_dataset_study.py:92`: `prepare()` invokes real dataset loaders, which unconditionally create `DATASET_ROOT/{dtd,fgvc_aircraft}/split_fewshot` and write `shot_16-seed_*.pkl` when absent (`datasets/dtd.py:21`, `datasets/dtd.py:40`, `datasets/fgvc_aircraft.py:19`, `datasets/fgvc_aircraft.py:43`). This violates both “prepare-only writes only the explicit evidence directory” and the global prohibition on modifying original data. Make preparation use a read-only loader path/cache redirection or preflight against an isolated cache, and add a test asserting the entire data-root tree remains unchanged.
- `scripts/repro/cross_dataset_study.py:58`: the frozen dependency manifest omits `scripts/repro/structure_spec.py`, even though both `run_pilot.py` and `summarize.py` import it to define variants, ordering, and parameter-count expectations. It also omits the externally located `cross_dataset_study.py` and `cross_dataset_audit.py` themselves, so those operational scripts can change after preparation without execution noticing. Record and recheck all operational-script hashes, including `structure_spec.py` and the two new scripts, while preserving the immutable `--source-root`/external-script layout.
- `scripts/repro/cross_dataset_audit.py:97`: the scoped protocol passed to `summarize.collect` omits `view_orders`. Consequently the existing summary skips its order-policy/signature/evidence validation, while the added audit checks only class names and counts (`scripts/repro/cross_dataset_audit.py:133`). A consistently tampered VIEW_ORDER/ORDER_SEED or order manifest can pass, contrary to the fixed original contiguous grouping and exact description-order requirements. Populate preregistered `view_orders` from `structure_spec.order_settings`, and validate each manifest exactly rather than accepting any ordering with the right classes/counts.
- `scripts/repro/cross_dataset_study.py:166`: “at least 90% memory free” is not evidence that a GPU is idle; an active compute process with a small allocation passes. This can start formal work on occupied GPUs. Query active compute processes (or an equivalent authoritative device API) and require none on selected visible GPUs in addition to the memory check; cover both low-memory-active and genuinely idle cases.
- `scripts/repro/cross_dataset_audit.py:94`: per-dataset `protocol.json` is first created by the post-stage audit (`scripts/repro/cross_dataset_audit.py:100`) rather than being fixed before the stage runs. `execute()` bypasses `run_pilot.main`, whose producer contract normally writes `protocol.json` and `data_preflight.json` before commands (`scripts/repro/run_pilot.py:115`). The audit can run because it synthesizes the missing protocol, but this makes the claimed protocol identity retrospective and leaves no stage-local pre-run data-preflight artifact. Write each scoped protocol and its prepared-data identity before the first stage command, then make audit read-only and reject missing artifacts.
- `scripts/repro/cross_dataset_study.py:202`: audit subprocess stdout/stderr are not sent to the stage log directory, unlike train/eval commands (`scripts/repro/cross_dataset_study.py:185`). This misses the requirement that every phase have separated stdout/stderr logs and makes audit failures harder to diagnose. Capture audit and final-audit streams in named stage logs and retain their paths/status atomically.

#### Minor (Nice to Have)

- `scripts/repro/cross_dataset_audit.py:170`: `comparisons[*].per_seed` stores an ordered list of metric differences without the seed identifier. The code happens to append seeds 1,2,3 in order, but the JSON does not self-describe which result belongs to which seed. Add `seed` to each row.
- `tests/repro/test_cross_dataset.py:17`: the tests do not exercise external immutable `--source-root`, preparation data-root immutability, active-GPU rejection, exact description order, or the raw-command/stage-protocol producer-consumer path. These are the integration seams where the defects above occur; add focused CPU fixtures/mocks rather than a GPU suite.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** The overall decomposition and strict result/checkpoint accounting are strong, and bypassing `run_pilot.main` does not immediately crash because the audit creates the protocol consumed by `summarize.collect`. However, preparation currently mutates source data, execution does not freeze all controlling code, and key GPU-idle and description-order guarantees are weaker than the binding protocol, so CPU preparation and later GPU execution are not yet safe to authorize.
