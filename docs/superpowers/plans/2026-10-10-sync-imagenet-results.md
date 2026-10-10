# ImageNet Result Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize the completed current-model ImageNet experiment into the repository as compact, verifiable evidence.

**Architecture:** Import only audit-critical JSON/YAML/checksum artifacts into a dedicated experiment directory, with a provenance README documenting the omitted large artifacts and their server backup. Add repository tests that pin the model identity, protocol, metrics, seeds, and documentation boundary.

**Tech Stack:** Python `unittest`, JSON, YAML text, Markdown, Git.

**Spec:** `docs/superpowers/specs/2026-10-10-sync-imagenet-results.md`

## Global Constraints

- Retain only the current fixed probability-fusion model evidence.
- Do not commit checkpoints, predictions, failed intermediate outputs, or credentials.
- Do not claim SOTA or conflate local and published results.
- Preserve exact server-produced evidence bytes where imported.

## Review Focus

- A stale or wrong trainer name must fail validation.
- Missing one of the three seeds must fail validation.
- A changed fixed-fusion metric or alpha must fail validation.
- Legacy V2 FG identifiers must fail validation.
- README and project status must not continue to say current ImageNet evidence is absent.

---

### Task 1: Import and validate the ImageNet evidence package

**Files:**
- Create: `tests/repro/test_imagenet_result_package.py`
- Create: `reports/experiments/ImageNet扩展验证-20260918/README.md`
- Create: `reports/experiments/ImageNet扩展验证-20260918/服务器备份.sha256`
- Create: selected server-produced JSON/YAML/checksum files under `reports/experiments/ImageNet扩展验证-20260918/`

**Interfaces:**
- Consumes: the server run `ImageNet投稿主实验-20260918-45ff8ba-重跑1`
- Produces: a compact experiment directory validated by `unittest`

- [ ] **Step 1: Write the failing package validation test**
- [ ] **Step 2: Run it and verify failure because the package is absent**
- [ ] **Step 3: Download, checksum, and import the selected evidence files**
- [ ] **Step 4: Add provenance documentation and excluded-artifact policy**
- [ ] **Step 5: Run the package test and the full test suite**
- [ ] **Step 6: Commit the synchronized evidence**

### Task 2: Correct repository status documentation

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_STATUS.md`
- Modify: `reports/README.md`
- Modify: `tests/repro/test_imagenet_result_package.py`

**Interfaces:**
- Consumes: the validated ImageNet experiment package from Task 1
- Produces: repository documentation that points to the evidence and states the correct eleven-dataset boundary

- [ ] **Step 1: Extend the test with stale-claim and evidence-link assertions**
- [ ] **Step 2: Run it and verify failure on the old ten-dataset text**
- [ ] **Step 3: Update documentation with the local ImageNet result and conservative claim boundary**
- [ ] **Step 4: Run the focused and full test suites**
- [ ] **Step 5: Commit the documentation correction**

### Task 3: Publish the branch update

**Files:**
- Modify: the existing GitHub PR branch `chore/remove-v2-fg-legacy-20261010`

**Interfaces:**
- Consumes: verified commits from Tasks 1 and 2
- Produces: updated GitHub PR #2 containing the synchronized evidence

- [ ] **Step 1: Review the complete branch diff for secrets and oversized files**
- [ ] **Step 2: Verify tests and repository size constraints once more**
- [ ] **Step 3: Push the branch and confirm PR #2 reflects the new commits**
