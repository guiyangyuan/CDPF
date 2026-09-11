# Final whole-branch review

Reviewed range: `be9de02..1c0144a`, using `final-review.diff` (including operational fix `778d9d0`). Requirements: `docs/plans/2026-09-11-attribution.md`, including its global constraints and the rulings in `progress.md`. This review covers code, tests, cross-task interfaces, and the retained preparation/deployment evidence. Supplemental corrected server evidence was read from `work/attribution-20260911/server-prep-778d9d0` after it arrived during review; subsequent controller documentation updates are outside this snapshot.

### Strengths

- `scripts/repro/evaluate_frozen_fusion.py:75` implements the six preregistered branches with individually normalized descriptions, valid-description masks, shared CLIP scale, separate logit/probability fusion, and fixed weights. Its GPU path uses the original CoOp trainer and strict loader, records all predictions, and checks the student against historical counts before marking success (`:470`). The UCF declared-KD exception remains narrow, with effective KD zero checked independently (`:193`, `:228`). The method identity explicitly distinguishes ADK formula adaptation from author-code or published-score reproduction (`:616`).
- `scripts/repro/attribution_study.py:87` restricts source relocation to the runner's source-owned paths. The 11 ordered stages contain the planned 124 commands, with historical checkpoints reused through `--model-dir`, fresh output paths, no tuning/retry controls, and stage audits before progression (`:100`, `:355`). Metadata/config workers isolate the two historical source versions in separate processes (`:170`, `:225`).
- `scripts/repro/attribution_study.py:180` reproduces the existing encoder's token truncation/deduplication and first-occurrence ordering. Preparation freezes the tokenizer vocabulary as well as the operational/source dependencies, historical artifacts, real loader evidence, descriptions, and CLIP hash (`:144`, `:150`, `:279`). The repair preserves historical evidence rather than relaxing it.
- `scripts/repro/attribution_audit.py:84`, `:201`, `:232` connect actual configs, checkpoint signatures/provenance, exact correct/total, real samples, and all six prediction branches. The final analysis requires exactly 36 full-model and 54 fusion seed/branch identities, excludes anchors from formal pairs, preserves every branch, and distinguishes HM-of-means from mean seed HM (`:316`, `:345`). The KD comparison permits only KD weight and output relocation after independently anchoring configs to historical source-derived settings.
- The regression tests cover the previously demonstrated source-path failure across all 124 commands, real Aircraft token collisions, failed-command stopping, no-GPU output guards, prediction file identities, and incomplete statistical identities. The retained first failed preparation is clearly a failed attempt, not a claimed completed study. The launcher targets the corrected independent operational release and does not alter source/model directories or GPU power.

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- None remaining in the reviewed snapshot. The seed-domain guard, caller-path relocation, and token-effective metadata findings are addressed.

#### Minor (Nice to Have)

1. **Historical accuracy representation is stricter in the combined audit than in the evaluator.** `scripts/repro/attribution_audit.py:269` requires exact equality of entire student dictionaries, including accuracy, whereas `scripts/repro/evaluate_frozen_fusion.py:578` accepts a historical accuracy within `5e-4` while requiring exact correct/total. A rounded historical metric can therefore pass Task 1 and later fail Task 2 despite identical predictions. Preserve exact counts and receipt provenance, but compare count-derived accuracy using the existing tolerance. **Triage: nonblocking for this frozen receipt.** A focused read-only check of all 18 retained baseline metadata cases confirmed that every supplied accuracy already equals `100.0 * correct / total` and its receipt value exactly; this is a fail-closed compatibility limitation, not a current result discrepancy.

2. **Accepted source aliases remain raw dictionary keys.** `scripts/repro/attribution_study.py:69` validates resolved source paths, while `:125`, `:292`, and `:302` retain/compare raw source spellings. A DTD/Aircraft source expressed with a symlink or `/./` can pass validation but receive only UCF dataset metadata, leading to a later missing-dataset failure. Normalize source paths once for command/stage/metadata identity, or reject noncanonical spellings at the receipt boundary; add an alias regression when broadening this interface. **Triage: nonblocking for the supplied canonical receipt.** The path issue fails closed and does not silently change model inputs or results.

### Verification and limits

- Read the supplied whole-branch package, binding plan, implementation reports, original reviews, scoped re-reviews, and relevant ledger rulings. Reviewed the full new evaluator, queue, auditor, and tests. Named unchanged dependency checks were limited to the baseline forward tuple, original description encoder/tokenization, runner-owned paths, checkpoint audit, loader evidence, GPU guard, and historical accuracy arithmetic.
- Existing evidence records 25 Task 1 focused tests; the original combined 104-test runs; 23 Task 2 focused tests after the fixes; and 36 real historical token-effective count/order checks. The corrected server log `tests-778d9d0.log` records **107 tests passing in 15.880 seconds**; its validation receipt records exit 0. These suites were not rerun by this reviewer.
- A focused read-only evidence probe confirmed all 18 retained baseline cases have exact count-derived receipt accuracy, matching checkpoint/source hashes, `validated_metadata` status, and no created output. A second probe checked the received corrected preparation: matching embedded/separate protocol, 11 stages, 124 complete effective configs, 9 historical full-model references, both expected source/dataset mappings, and matching hashes for the three reviewed operation scripts. Launcher interpreter/source/receipt/output/preparation paths agree with this actual preparation.
- The revised CPU preparation receipt records **exit 0 after 160.399 seconds**. The controller reports that the formal study remains absent; the first failed preparation is preserved verbatim. No new GPU evaluation, anchor reproduction, formal training, or contribution result is certified by this review. GPU runtime validation remains the explicit later gate, including exact historical student/anchor counts and all final audit checks.
- No source/test/model/config edits, Git/index/branch mutations, server calls, or subagents were used. This requested report is the only created file.

### Recommendations

- Retain both Minor compatibility findings as deferred work; neither requires changing this fixed receipt or weakening count/provenance checks.
- Publish the controller's completed CPU-readiness evidence and update the interim status text with the observed outcome. Preserve the explicit distinction between preparation, GPU execution, and validated scientific conclusions. Any later substantive code change needs a scoped review of its delta and refreshed preparation hashes.

### Assessment

**Ready to merge? Yes.**

**Reasoning:** The reviewed branch implements the frozen scientific protocol and connects its evaluation, queue, and evidence interfaces without remaining blocking findings. This is merge approval for the operational/preparation work, with two documented nonblocking compatibility limitations; it does not certify the still-pending GPU runtime or new experimental results.
