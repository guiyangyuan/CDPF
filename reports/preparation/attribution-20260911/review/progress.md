# SDD ledger — plan: docs/plans/2026-09-11-attribution.md

Spec is the plan Global Constraints and accepted chat recommendation.
Base be9de02; plan commit0908eeb. Worktree codex/attribution-20260911.
Baseline: 59 tests pass with unittest discover -s tests/repro -t .; initial invocation lacking -t caused only relative import error, corrected without edits.
Server CPU mode confirmed, cgroup2GB, no available GPU. No paid training launched.

| Pair/task | Interface / internal consistency | Finding |
|---|---|---|
| Task1 | frozen baseline models -> new fusion metrics and predictions | Consistent; source version must match historical checkpoint |
| Task2 | historical KD1 + six newKD0 + two anchors -> paired audit | Consistent; detailed CLI to be frozen after Task1 |
| Task3 | model/source mapping and prior-work -> scripts / documented deployment | Consistent; author-code availability not assumed |
| 1/2 | standalone fusion CLI consumed by queue | No shared owned files; command schema to freeze before Task2 dispatch |
| 1/3 | controller restores baselines, Task1 validates | Hashes of actual restored files required, no rewritten checkpoints |
| 2/3 | prepare/execute schemas consumed by deployment | CPU only preparation now; GPU gate later |

Task 1: in_progress (BASE0908eeb)
Task 1: scope clarified before any GPU evaluation: add ADK public-formula controlled CoOp adaptation and probability-mean control on same frozen descriptions. Six branches total, no new training/search.
Ruling: use explicitly labeled ADK formula adaptation because AAPE code is unavailable and ADK author repo is README-only — preserves a close-method mechanism comparison within authorized scope — if paper interpretation is wrong, that auxiliary comparator needs correction and re-evaluation; it is not an author-code/published-score replication.
Research subtask attribution_priorwork complete; author-code absence verified; concise receipt in reports/preparation/attribution-20260911/closest-methods.md.
Ruling: ADK tau interpreted as standard CLIP temperature 1/exp(logit_scale), shared for attention and class probabilities, with actual scale logged — formula/prose terminology is ambiguous and no author code disambiguates — if incorrect only the explicitly labeled ADK comparator requires revision, not primary simple-logit fusion or KD study.
Ruling: accept the single historical UCF baseline declared KD_WEIGHT1 vs current source-derived baselineKD0 discrepancy only when checkpoint/manifest prove effective_losses.kd0, with all other config fields exact after path relocation — original baseline group disabled KD despite declared value — if this narrow compatibility interpretation is wrong, UCF fusion comparison must be rerun with a fresh matched baseline; DTD/Aircraft remain exact. Implementer instructed to test rejection of other drift and record declared/effective/current values.
Task 1: reviewer interim Important — public metadata/evaluation seed guard allows any nonnegative seed, but frozen protocol is1/2/3. Begin fix round1 on this scoped guard while reviewer finishes immutable bcfa9db snapshot; final full findings will be included before scoped re-review.
Task 1: fix round1/5 (1 addressed,0 open; commitsbcfa9db..92b01a1); focused25passed, scopedre-reviewapproved. Prior full80passed includes59existing+21priornew; new4seedguardscoveredfocused25.
Task 1: complete (commits0908eeb..92b01a1, review clean); CUDA path remains deliberately pending user GPU availability, to be exercised by fixed queue.
Task 2: in_progress; BASE92b01a1. Parent verified all18 referenced server model hashes against actual bytes in independently retained local full backups; evidence local-backup-correspondence.json.

Controller: Task1 deployed read-only evaluator92b01a1 SHAe964e0... to independentpreparationdir; all18 actualserverbaseline metadata casespassed (UCF6,other12), fullCLIPnotloaded, formalstudyabsent. Evidence copiedtopreparationreport.

Task 2: implementation7e02cb1 complete; focused20/full104 passed locally. attribution_task_review reviewing92b01a1..7e02cb1. Independent snapshot deployed releaseattribution-ops-20260911-7e02cb1; server104tests exit0, realCPUprepare pending, GPU study absent. Reviewer identified commandpath rewriting risk when data/output underoperationroot; consolidated findings pending. Actualdeploymentpaths are disjoint from operationroot and noGPUrunoccurred.

Task 2: fix round1/5 started on Important caller-path relocation (review7e02cb1). Original implementer attribution_queue resumed; focused regression requested.
Task 2: minor (deferred): fusion audit exact accuracy-dict comparison stricter than Task1 5e-4 tolerance; actual current receipts unrounded, exact count comparison correct. Final review to triage.
Task 2: minor (deferred): canonical source-root aliases accepted then raw source strings used as metadata keys; actual receipt canonical, final review to triage.

Task 2: controller realCPUprepare7e02cb1 failed after118s on Aircraft historical description_counts; no prepared directory or formalstudy created. Source encoder token truncation/dedup occurs after text dedup; dataset_metadata counted only text rows. Actual Aircraft Basecounts include8-20 vs rawtext20. Original implementer adds sameCPUtokenizer step and tokenizer vocabulary freeze tocurrentfixround, nofullCLIPload/nohistoricaledits. Failurelog retained work/attribution-20260911/prepare-7e02cb1.log.

Task 2: fix round1/5 (2 addressed,0 Important open — source-owned-only paths and exactCPUtoken dedup; commits7e02cb1..778d9d0); focused23passed, 36actualhistoricalmetrics count/order checks pass; scoped review approved.
Task 2: complete (commits92b01a1..778d9d0, taskreview clean; 2minor deferred tofinaltriage). Realserverprepare778d9d0 running; Task3 still inprogress.

Controller: real serverprepared778d9d0 PASSED160.399s; serverfull107passed15.880s; completepreparation12366835bytes SHAaf70fc356f979d708f39323890b20d168664b834a1b1575ffea9af80db327f18 copiedlocal andgzipmetadataarchivedreports/attempt-778d9d0. Formalstudyabsent.
Finalreview1c0144a: ReadyYes, noCritical/Important, twoMinorcompatibilityedgesconfirmednonblockingactualreceipt. ONEfinalfixwave attribution_queue dispatched to alignaccuracytolerancewhilecountsexact and rejectnoncanonicalsourcealiases; no scientific/protocolchange. Willscopedrereview andfreshpreparebecauseoperationalhashchanges.

Finalfixwave43e537c: bothMinoraddressed; focused25passed, final-rereviewapproved/no newbreakage. No remainingreviewfindings. Finalrelease43e537c deployed; finalCPUpreparation running becauseoperationSHAchanged. NoGPUrun.

Task 3: complete — final release43e537c, realCPUpreparepassed; localmergedtree/serverfull109tests passed; actualnoCUDAexecutionrejectswithoutstudycreation; launcher/readiness/completecompressedmetadata saved. No newGPUexperiment. Final review and scoped fix review clean; no remaining findings. Git synchronization follows under existing user authorization.
