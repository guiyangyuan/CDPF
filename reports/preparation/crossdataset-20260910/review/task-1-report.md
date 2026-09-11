# Task 1 report: cross-dataset orchestration and audit

Status: DONE_WITH_CONCERNS

## Implementation

- Added `scripts/repro/cross_dataset_study.py` with mutually exclusive `--plan-only`, `--prepare-only`, `--execute`, and `--audit-only` modes and explicit source/data/output/preparation roots.
- The plan reuses `run_pilot.build_commands` from `--source-root`. It fixes DTD then Aircraft ordering for smoke and formal stages, producing 17/17/51/51 commands (6 smoke training + 28 smoke reloads; 18 formal training + 84 formal reloads).
- Plan mode reads no datasets, requires no GPU, and creates no study or preparation directory. Preparation writes only a fresh evidence directory after source, dependency, full CLIP, environment, fixed-split, 16-shot, description, and actual-loader checks succeed.
- Preparation records actual per-seed/per-split class lists, train/validation/test counts, and hashes from each dataset loader. Execute recomputes all loader evidence and source/CLIP hashes before creating the study root.
- Execute rejects existing study roots and non-CUDA or non-idle GPU environments. Child processes use four CPU threads, deterministic CUBLAS configuration, and remove inherited `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`.
- Atomic `status.json` includes PID, state/stage, every training/evaluation/audit command with timestamps and exit status, and completed stages. Logs are separated by stage and stdout/stderr. Each dataset stage passes an audit gate before the next stage; the final combined audit runs before state `passed`.
- Added an independent audit which first calls `summarize.collect`, then checks loader-derived split hashes/counts/classes, description order, protocol/config invariants, checkpoint bytes/sidecar/epoch/provenance/signature, finite state tensors and fixed parameter counts (baseline 2,048; both full models 790,018), exact branch checkpoint identity, test-only evaluation, and log correct/total/accuracy agreement.
- Final `analysis.json` requires all 84 formal evaluations and contains 42 dataset/seed/branch rows, 14 summary rows, and primary-branch paired hierarchical-minus-flat and hierarchical-minus-baseline Base/Novel/HM differences with means and sample SDs.

## Verification

- Focused: `/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python -m unittest tests.repro.test_cross_dataset -v` — 8 tests passed in 0.819s.
- Syntax: the same Python with `-m py_compile scripts/repro/cross_dataset_study.py scripts/repro/cross_dataset_audit.py` — passed.
- Existing repro suite: `PYTHONPATH="$PWD:$PWD/Dassl.pytorch" .../tc-venv/bin/python -m unittest tests.repro.test_assets tests.repro.test_components tests.repro.test_preflight tests.repro.test_regressions tests.repro.test_runner tests.repro.test_runtime_compatibility tests.repro.test_structure tests.repro.test_trainers -v` — 45 tests passed in 2.586s.
- A first `unittest discover -s tests/repro` attempt ran 36 tests successfully but reported three import errors because it removed package context and omitted `Dassl.pytorch` from `PYTHONPATH`; the corrected module-based command above passed all 45 existing tests.
- Plan CLI produced model digest `9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba` and stage command counts 17, 17, 51, 51 without creating its output root.

## Constraints and concerns

- No GPU or remote execution was performed. The connected server is intentionally in no-GPU mode, so only CPU preparation is appropriate there.
- The full audit is deliberately strict and has been exercised with small CPU fixtures for missing results, checkpoint/hash/parameter tampering, split/sample-hash tampering, configuration drift, and loader-derived non-UCF counts. It has not yet consumed real DTD/Aircraft training outputs because those outputs do not exist.
- `--source-root` may point to the unchanged release while the two operational scripts reside together elsewhere; source commands use the source root, while audit subprocess resolution uses the operational script's own directory.

## Review round 1 fixes

- Added `scripts/repro/structure_spec.py` to the immutable source dependency set and separately pins the externally deployed study/audit scripts. Execute and standalone audit reject either set changing after preparation.
- Added explicit contiguous `view_orders` to the global and scoped protocols. The existing summary validator now recomputes and checks exact signature/order manifests, while the external audit continues checking loader class order and effective description counts.
- Replaced the memory-only GPU test with an authoritative `nvidia-smi` active-compute-process query for the visible device selection, followed by the 90% free-memory guard.
- Each scoped `protocol.json` and `data_preflight.json` is atomically written before its first raw command. Audit is read-only for those inputs and rejects missing or changed copies.
- Stage audits and the final audit now have named stdout/stderr logs recorded in `status.json`. Launch errors, interrupts, and other exceptions after study creation atomically transition status to `failed` with structured error evidence.
- Comparison rows now include explicit seed labels.
- Preparation metadata documents that deterministic `split_fewshot` files are authorized derived loader caches. It inventories all other dataset files before/after loader preparation and rejects changes to images or fixed split metadata; description files remain content-hashed in the immutable source set.

Review-fix verification:

- Focused command: `PYTHONPATH="$PWD:$PWD/Dassl.pytorch" .../tc-venv/bin/python -m unittest tests.repro.test_cross_dataset -v` — 13 tests passed in 1.209s. Added seams cover external source and operational hashes, dependency tampering, order-protocol tampering, active/idle GPU states, pre-command scoped evidence, named audit logs, and launch-error status.
- Syntax and whitespace: `py_compile` for both operational scripts and `git diff --check` — passed.
- Existing repro command from the first report — 45 tests passed in 2.556s.
- Plan CLI still reports model digest `9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba`, exact contiguous order records for all three variants, and hashes for both external operational scripts plus the source-root structure specification.

Post-review metadata correction: clarified that images use a path/size/mtime inventory check, while fixed split metadata and description templates use SHA-256. This changes evidence wording only; runtime behavior is unchanged, so the approved test results above remain applicable.
