# H1 validation study

This entry point implements one fixed study: seed 1, 16 shots, 20 epochs,
ViT-B/16, fp32, Base validation only. It permits at most four new training
runs in this order:

1. DTD `full_kl` anchor
2. UCF101 `weighted_nckd` candidate
3. DTD `weighted_nckd` candidate
4. FGVC Aircraft `weighted_nckd` candidate

The anchor must pass exact S/T/M top-1 comparison and the frozen
`numpy.allclose(rtol=1e-5, atol=1e-4)` comparison against authenticated
historical seed-1 S1/T1/M. Candidate execution cannot begin before that audit
passes. There are no alternate modes, seeds, orders, command arguments,
retries, resumes, checkpoint selection, or tuning controls.

## CPU-only planning

Run planning with the required Python and the receipts in the exact order
shown. Both output paths must be new, disjoint absolute paths. `--output-root`
is reserved in the plan but is not created by this command.

```bash
PYTHONPATH=.:Dassl.pytorch:tests OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python \
  scripts/repro/h1_study.py plan \
  --prepared \
    /absolute/prepared/anchor-dtd/prepared.json \
    /absolute/prepared/candidate-ucf101/prepared.json \
    /absolute/prepared/candidate-dtd/prepared.json \
    /absolute/prepared/candidate-fgvc_aircraft/prepared.json \
  --output-root /absolute/fresh/future-study-root \
  --output-dir /absolute/fresh/plan-artifact
```

Planning calls Task 1 `load_prepared` on every receipt and
`build_train_command` on every model directory. It authenticates the source,
configs, data/sample bindings, historical exports, checkpoints referenced by
those exports, assets, transforms, and operation scripts. It writes only
`plan-artifact/plan.json`. It does not check CUDA, construct a model, start
training, or create the future study root, so the same command is safe on a
GPU host.

`plan.json` has `state: planned_not_executed`, `schema_version: 1`, the fixed
protocol and `{maximum: 4, planned: 4}` training budget. Its four `cases`
bind each prepared receipt, future config, historical JSON/NPZ, source and
loss identity by path and SHA-256. Its 13 `stages` contain the immutable argv,
argv hash, output directory, and expected result path for four trains, four
exports, one anchor audit, three candidate analyses, and one aggregate. The
`operations` object binds `train.py`, `h1_validation.py`, `h1_analysis.py`,
and `h1_study.py`. `plan_identity_sha256` seals the complete plan.

## Future GPU-only execution

Execution is a separate, explicit command. Do not run it on this CPU
preparation host.

```bash
PYTHONPATH=.:Dassl.pytorch:tests OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/Users/bytedance/Documents/Codex/2026-09-08/https-github-com-zhengli97-awesome-prompt/work/tc-venv/bin/python \
  scripts/repro/h1_study.py run \
  --plan /absolute/plan-artifact/plan.json
```

Before creating a subprocess, `run` re-authenticates every prepared receipt,
rebuilds all fixed commands, verifies the plan and operation hashes, requires
a fresh study root, acquires an exclusive sibling lock, and requires CUDA. A
no-CUDA host receives `study-root/status.json` with `state: failed`; no train
or export command is spawned.

Each stage revalidates authenticated inputs immediately before and after its
subprocess. An export snapshots the complete validated training directory; a
case analysis snapshots its complete validated export directory; aggregation
snapshots all three complete validated candidate-analysis directories. Each
snapshot hashes the actual files and must exactly match the prior stage's
recorded evidence before the next subprocess can start and after it exits.
Training outputs are checked with Task 1
`validate_new_checkpoint`, including the epoch-20 checkpoint, run manifest,
effective config, source, mode, and parameter state. Exports are reloaded by
Task 1 `load_export` and must match the planned receipt, seed, source, model
directory, and every checkpoint field recorded by the train stage. After an
analysis command exits, the executor independently reloads its authenticated
current and historical exports and recomputes the entire anchor or candidate
report. It similarly reloads all three candidates and recomputes the complete
aggregate; a partial or forged `passed` JSON cannot release the next stage. A
per-stage JSON file under `study-root/logs/` records argv, return code,
stdout/stderr, full before/after input manifests, their hashes, immutability,
and validated output evidence. `status.json` is replaced atomically after each
stage.

The final state is:

- `passed` only when the anchor, all three candidate cases, and aggregation
  pass;
- `stopped` when the anchor comparison or final numeric guards fail;
- `failed` for an execution, identity, immutability, checkpoint, export, or
  candidate-analysis error.

Any nonzero stage stops the queue. Existing stage or study directories are
never reused. The executor has no resume or retry path and does not change GPU
power, clocks, or persistence settings.

## Analysis artifacts and decision guards

The anchor and each candidate write `analysis/analysis.json`. These bind the
current and historical export JSON files and include dataset, seed, mode,
fixed tolerance, Base-only claims, and independently computed accuracy/NLL/
Brier/margin metrics. Raw S, T, M, and original ensemble logits remain in the
authenticated Task 1 export; the report binds that export rather than copying
or substituting arrays. `S+M_logits` is always `.5 * (S + M)`.

Candidate reports also include the three authenticated historical comparators:
`C+M_logits`, `C+M_probability`, and `S0+M_logits`. Current M must have the
same argmax and satisfy the frozen numeric tolerance against historical M;
otherwise the case is an integrity failure.

`aggregate/aggregate.json` accepts exactly one complete seed-1 candidate for
each of UCF101, DTD, and FGVC Aircraft. It reports per-dataset and equal-dataset
macro accuracy, all deltas, and these fixed guards. Guard decisions use exact
rational arithmetic from the integer `correct/total` counts; reported floats
are display values, so an exact tie cannot become positive through rounding.
The artifact retains the exact fractions in `exact_guard_arithmetic_pct`.

- versus `C+M_logits`: macro delta at least 0 percentage points and every
  dataset delta at least -1 point;
- versus `C+M_probability`: macro delta at least 0 points and every dataset
  delta at least -1 point;
- versus `S0+M_logits`: macro delta strictly greater than 0 points.

Passing is exploratory evidence for this fixed Base-validation hypothesis. It
does not establish novelty, Held-out/Novel performance, harmonic mean, or a
new training action.

## Failure evidence and recovery

Read `status.json` first, then the referenced file in `logs/`. Preserve the
entire study root. A `stopped` or `failed` study cannot be resumed, retried, or
pointed at a different command. Recovery means diagnosing the recorded
evidence only. Any further training attempt requires separate authorization
and a newly authenticated plan with a fresh study root; it is not performed by
these scripts.
