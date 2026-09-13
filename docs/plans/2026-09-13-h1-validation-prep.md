# H1 Base-validation queue and no-GPU deployment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute the checked tasks sequentially and preserve evidence.

**Goal:** Prepare an auditable, bounded H1 training/validation queue and deploy it without training.
**Architecture:** Add a separate H1 validation tool for current-source configuration, CPU asset preparation and strict new-checkpoint export. Add an independent study planner/executor and fixed analysis using authenticated historical seed1 references. Keep historical tools and model source unchanged.
**Tech Stack:** Existing Python, torch, numpy, yaml, unittest; no new dependencies.
**Spec:** reports/diagnostics/kd-failure-20260912/NEXT_STEP.md; reports/preparation/h1-cpu-20260913/reference-snapshot.json; docs/KD_LOSS_CONTROL.md.

## Global Constraints
- No model/trainer/dataset/CLIP/Dassl source edits and no old checkpoint provenance bypass. Training source SHA stays 387ea6e46aa0fa6cba5109e64baab94a548b35c200a1945267d392bc35a36677.
- This turn is CPU/no-GPU preparation only, zero training on real data. No paper edits. GitHub publication and independent no-GPU server deployment are authorized.
- Fixed seed1,16-shot,20epochs,ViT-B/16,fp32,T2,KD1,teacher CE1,existing online teacher and architecture, original train sampler/batch/workers/optimizer from authenticated historical kd1 config. Only loss mode changes. TEST.NO_TEST=True and TEST.FINAL_MODEL=last_step prevent automatic final/test selection; TEST.SPLIT=val, Base classes only. Export sequential val batches/workers0 separately.
- Four future trains maximum: DTD full_kl anchor once, then UCF101/DTD/fgvc_aircraft weighted_nckd once each if anchor passes. No automatic retries/resume/restarts, no other seeds or tuning.
- Preserve all failures/output dirs. Fresh disjoint outputs only. Authenticate source, config, CLIP, templates, samples/rows, source metadata, historical logits and operation scripts; check before/after execution. No downloads initiated by prepare, no training/CUDA model construction in prepare.
- Candidate SN+M_logits fixed alpha=.5. All raw S,T,M and original ensemble retained as diagnostic. Same historical seed1 row/class order. Base validation only; no Novel/HM outputs or claims.
- Anchor: same DTD rows/classes, S/T/M argmax predictions exactly equal historical S1/T1/M; logits require numpy.allclose(rtol=1e-5,atol=1e-4), report max absolute errors. Save/reload and identity must pass first. No re-tuning tolerance after seeing results.
- Candidate numeric guards: versus each C+M logits/probability macro delta>=0pp and each dataset delta>=-1pp; versus S0+M logits macro delta>0pp; equal dataset weight. Failure stops H1; passing is exploratory only, not novelty proof.
- Python /Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python. PYTHONPATH=.:Dassl.pytorch:tests OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1. Full suite: -m unittest discover -s tests -v. Logs outside repo in ../../h1-prep-20260913. Commit only owned files; no subagents for implementers.

## Task 1: Authenticated H1 configuration, CPU prepare and checkpoint export
Ownership: create scripts/repro/h1_validation.py and tests/repro/test_h1_validation.py. Historical validation_probe.py/validation_analysis.py may be imported, not modified. Reuse their source/dataset/transform/hash helpers without extending historical fixed formats.
Interfaces: CLI prepare --source-root --data-root --historical-export --historical-config --dataset --mode --output-dir creates prepared.json; CLI export --prepared --model-dir --output-dir creates export.json/logits.npz. Public build_train_command(prepared, model_dir, python=sys.executable) returns argv; load_prepared(path) revalidates assets and returns dict; load_export(path) validates new output plus prepared/checkpoint/identities and returns (receipt, arrays). CPU prepare output contains future training config and fixed source/asset/sample/historical bindings; Task2 consumes these only.
- [x] Write failing behavior tests for mode/seed/source/config/reference mismatch, duplicate or reordered samples, finite raw branches, prepare no model/optimizer calls, actual tiny-model strict reload/export, fresh-output protection and no-CUDA export rejection.
Example rejection contract:
```python
with self.assertRaises(ValueError):
    validate_historical_reference(wrong_seed_or_changed_sha)
```
Use the existing tiny CLIP fixture where practical; names of private helpers may differ.
- [x] Implement authenticated historical input loading: fixed reference-snapshot file SHA and fixed inventory; use old validation_analysis.load_export and verify exact snapshot export/NPZ hash for dataset seed1. Historical kd1 effective config must match its inventory checkpoint bundle SHA; reject arbitrary YAML. Preserve original settings, changing only paths, mode, and documented evaluation controls. Validate config with current setup_cfg/check_cfg.
- [x] Reconstruct actual train/val/test metadata with cache-free validation_probe.rebuild_source_dataset; verify train and val digests to historical trusted inventory. Read/decode only train/val images, no test inference. Hash actual sampled training and validation image contents; record all metadata/config/templates/CLIP/source dependencies. No full CLIP loading during prepare.
- [x] Store immutable prepared receipt with normalized future training config, source/tool hashes and exact mode loss_identity. build_train_command uses current source train.py directly, --seed1, new model dir and saved config; never call old pilot.
- [x] Export requires CUDA, validates prepared state and complete epoch20 checkpoint manifest/config/signature/source/loss identity before loading parameters. Construct one current CustomCLIP, strict complete trainable-module load (existing helper), use Base val loader, retain S,T,M,ensemble with finite N,C arrays. M uses normalized valid description vectors, masked mean then normalize, same historical formula. Store relative sample IDs/classes, transform verification, checkpoint hash and mode identity. Never relax runtime source checks. Pure batch-export helper may run tiny CPU tests; CLI CUDA guard remains.
- [x] load_export must recompute/validate NPZ schema, rows, labels, class order, hashes, source/mode/checkpoint/prepared bindings; do not trust summary metrics.
- [x] Run focused RED/GREEN then full suite once; self-review and commit. Report interface details for Task2.

## Task 2: Fixed study planner, gates and auditable execution
Ownership: create scripts/repro/h1_study.py, scripts/repro/h1_analysis.py, tests/repro/test_h1_study.py, tests/repro/test_h1_analysis.py, docs/H1_VALIDATION_STUDY.md. Consume Task1 public interfaces; no model/historical-tool edits.
Interfaces: h1_study.py plan --prepared full-anchor.json weighted-ucf.json weighted-dtd.json weighted-aircraft.json --output-root --output-dir produces plan.json without starting any train. h1_study.py run --plan plan.json is future GPU-only execution. h1_analysis.py accepts validated exports/reference for anchor audit or fixed three-case aggregation.
- [x] Write RED tests using actual temp artifact files and tiny fake executable commands for ordering/fail-fast/budget/duplicate refusal; pure arrays for fixed fusion/equal-dataset guards. Tests must show rejected wrong seed/mode/order, skipped anchor, changed inputs, preexisting outputs, no-CUDA, partial/mixed exports and non-finite values. No real-data optimization.
- [x] Implement plan from exactly four independently validated prepared receipts (DTD full anchor +3weighted candidates), identical fixed hyperparameters and per-dataset historical binding. Freeze argv/stages/paths/input+operation hashes; four training commands, four export commands and anchor/aggregate audits. Plan refuses arbitrary/missing/extra modes/cases or aliased/overlapping inputs/outputs. Mark planned_not_executed.
- [x] Implement anchor audit and candidate analysis by reloading authentic Task1 outputs and original authenticated historical NPZ. Match exact rows/classes. Independently calculate S,T,M,ensemble,S+M diagnostic accuracy and three comparator metrics, all guard deltas and pass/stop; no scores invented while planning. Preserve fixed tolerance and hypothesis limitations in results.
- [x] Executor rejects no-CUDA before spawning any training. Fresh study root, exclusive lock / no reuse, immutable validated command plan, fail-fast atomic status and command logs. Enforce anchor train/export/audit before any candidate. Anchor failure means no candidate command. No retries or resume. Candidate failure stops. Per-stage hashes bind model/manifests/configs/export and before/after input immutability. Final aggregate only after all3complete, not highest seed/branch. Leave state passed/stopped/failed with evidence; no automatic GPU power changes.
- [x] CPU prepare/plan commands are safe on a GPU-enabled host too: never initiate train. Document future exact CLI, boundary, fixed budget, status recovery and no automatic execution.
- [x] Run focused RED/GREEN and complete suite once, self-review and commit only owned files.

## Task 3: Controller verification, no-GPU deployment and publication
Ownership: operational scripts/logs under ../../h1-prep-20260913; report reports/preparation/h1-server-20260913; PROJECT_STATUS.md and this plan.
- [x] Scoped reviews after each task; final full-branch review, resolve concrete blockers.
- [x] Independently use actual local seed1 historical exports/config/data to prepare four receipts, check unchanged inputs/source and no GPU/model trains. Validate generated plan schedule/budget/guards, no-CUDA run rejection, final suite.
- [x] Deploy exact reviewed commit via archive to a fresh server release under /root/autodl-tmp/textconsistency/releases; verify archive/source/tool hashes. Reuse verified CLIP and actual data, never overwrite old releases/results. Current bjb1:42527 is reachable, no GPU and2GiB memory; preserve low-memory process isolation for real CPU prepares.
- [x] Server full tests plus four actual CPU prepares and future plan; no training. Capture environment, asset/sample hashes, config checks, command paths and proposed output absence. Download and independently verify evidence/receipts, compare train/val sample+image identities to local.
- [x] Save report with accurate local/server preparation status, failures and next GPU step; archive evidence excluding images/weights/credentials. Update PROJECT_STATUS, commit/ff main/push, verify remote equality and save external recovery note. No claim of new accuracy or optimized model effectiveness.


Completion evidence: code efcd3db;193 local and193 server CPU tests;4 local/4 server actual prepares;24 downloaded evidence files verified;13-stage/4-train plan ready, zero real training. See reports/preparation/h1-server-20260913. Final report/evidence publication uses the previously authorized commit/ff/push workflow; remote equality is recorded after push in the external session receipt.
