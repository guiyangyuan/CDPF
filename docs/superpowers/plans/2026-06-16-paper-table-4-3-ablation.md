# Paper Table 4.3 Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible five-group ablation pipeline aligned with paper Table 4.3.

**Architecture:** Add one controlled `CoOp_TextConsistency_Ablation` trainer that derives all module switches from `TRAINER.TEXTCONSISTENCY.ABLATION_GROUP`. Add shell runners for the three paper datasets and one Python summary parser for Base, Novel, and HM reporting.

**Tech Stack:** Python, PyTorch/Dassl trainer registry, Bash, standard-library CSV/regex parsing, `unittest` static contract tests.

---

### Task 1: Contract Tests

**Files:**
- Create: `tests/test_table_4_3_ablation_contract.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert exactly five groups, validate each group's module flags,
verify `train.py` imports the new trainer, verify runner dataset/seed coverage,
and verify the summary parser exposes `harmonic_mean`.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests/test_table_4_3_ablation_contract.py`

Expected: fail because the trainer, scripts, and parser do not exist.

### Task 2: Controlled Ablation Trainer

**Files:**
- Create: `trainers/coop_textconsistency_ablation.py`
- Modify: `train.py`

- [ ] **Step 1: Implement module flags**

Define `ABLATION_GROUPS` with `baseline`, `single_view_kd`, `no_kd`,
`v2_no_fg`, and `full`.

- [ ] **Step 2: Implement trainer**

Create `CoOp_TextConsistency_Ablation` with CoOp student logits, optional
single-view teacher logits, optional adaptive logits, optional hierarchical FG
logits, KD, adaptive CE, contrastive loss, and diversity loss.

- [ ] **Step 3: Register trainable modules**

Register one `textconsistency` `ModuleDict` containing every trainable module
for the selected group.

- [ ] **Step 4: Import trainer**

Add `import trainers.coop_textconsistency_ablation` in `train.py`.

### Task 3: Runners

**Files:**
- Create: `scripts/ablation/run_table_4_3.sh`
- Create: `scripts/ablation/run_all_table_4_3.sh`

- [ ] **Step 1: Add single-dataset runner**

Implement `DATA_ROOT`, `GPU`, `GROUP`, and `SEEDS` environment overrides.
Support one group or all five groups.

- [ ] **Step 2: Add all-dataset runner**

Run `caltech101`, `ucf101`, and `fgvc_aircraft`.

### Task 4: Summary Parser

**Files:**
- Create: `scripts/ablation/summarize_table_4_3.py`

- [ ] **Step 1: Parse logs**

Read Base and Novel `log.txt` files under the output layout and extract the
last `accuracy:` value from each.

- [ ] **Step 2: Write CSV and Markdown**

Write `table_4_3_results.csv` and `table_4_3_results.md`; report missing runs
as `MISSING`.

### Task 5: Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the ablation command**

Add the Table 4.3 runner and summarizer commands.

- [ ] **Step 2: Run checks**

Run:

```bash
python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/textconsistency-pycache python3 -m compileall -q train.py trainers datasets clip tests scripts/ablation
bash -n scripts/ablation/run_table_4_3.sh scripts/ablation/run_all_table_4_3.sh
git diff --check
```
