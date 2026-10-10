# Sync Current-Model ImageNet Results

## Objective

Import the completed 2026-09-18/19 ImageNet run from the authorized training server into the GitHub repository as auditable evidence for the retained fixed probability-fusion method.

## Required evidence

- Identify the run as `CoOp_TextConsistency_Ablation` plus frozen-description probability fusion with fixed `alpha=0.5`.
- Preserve the three-seed protocol, dataset audit, execution status, aggregate metrics, per-seed manifests/configuration, final Base/New metrics, and checkpoint hashes.
- Record the authoritative server paths and backup SHA-256.
- Exclude checkpoints, per-example predictions, transient failed outputs, and redundant full logs from ordinary Git history.
- Replace repository statements that incorrectly say the current method has no ImageNet result.
- Keep local experimental results distinct from published literature results and do not introduce an SOTA claim.

## Acceptance criteria

- The synchronized package reports Base `76.657`, Novel `71.621`, and HM `74.054` for fixed probability fusion.
- All retained manifests name `CoOp_TextConsistency_Ablation`; no synchronized evidence references `CoOp_TextConsistency_V2_FG`.
- Three checkpoint SHA-256 values and the complete server-backup SHA-256 are recorded.
- Repository tests verify provenance, metrics, completeness, and stale-claim removal.
