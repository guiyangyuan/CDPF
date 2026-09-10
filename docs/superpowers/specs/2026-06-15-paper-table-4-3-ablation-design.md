# Paper Table 4.3 Ablation Design

## Goal

Implement the five component-ablation groups defined in Table 4.3 of the
paper, using one controlled experiment entry point. The implementation must
not add the historical fixed-average group because it is not present in the
paper table.

## Experiment Matrix

| Group | Multi-view | Adaptive aggregation | KD | FG |
| --- | --- | --- | --- | --- |
| `baseline` | No | No | No | No |
| `single_view_kd` | Single view | No | Yes | No |
| `no_kd` | Yes | Yes | No | No |
| `v2_no_fg` | Yes | Yes | Yes | No |
| `full` | Yes | Yes | Yes | Yes |

The three paper datasets are:

- `caltech101`
- `ucf101`
- `fgvc_aircraft`

Each reported result is the mean over seeds 1, 2, and 3. The output reports
Base accuracy, Novel accuracy, and their harmonic mean.

## Architecture

Add a dedicated `CoOp_TextConsistency_Ablation` trainer rather than composing
several historical trainers whose behavior differs in more than one dimension.
The trainer uses configuration switches for:

- number of semantic views;
- fixed single-view teacher or instance-adaptive teacher;
- standard adaptive or hierarchical FG aggregation;
- KD loss;
- adaptive-branch classification loss;
- contrastive and diversity regularization.

The student branch is the CoOp prompt learner in every group. The teacher
branch is absent for `baseline`, fixed single-view for `single_view_kd`,
standard adaptive for `no_kd` and `v2_no_fg`, and hierarchical for `full`.

The adaptive classification term remains enabled whenever an adaptive
aggregator is trained. Without this term, the `no_kd` aggregator would receive
no useful supervision when KD is disabled.

## Configuration

Add `TRAINER.TEXTCONSISTENCY.ABLATION_GROUP` with one of the five group names.
The trainer derives module switches from this value so incompatible
combinations cannot be passed accidentally.

Dataset-specific paper settings are stored in one shell configuration table.
The component ablation keeps the same dataset-level `N_CTX`, `N_VIEWS`,
`KD_WEIGHT`, `KD_TEMP`, epoch count, shot count, and precision across all
groups except where the ablation definition explicitly overrides a component:

- `baseline` ignores all TextConsistency-specific values;
- `single_view_kd` forces one view;
- `no_kd` forces `KD_WEIGHT=0`;
- `v2_no_fg` disables hierarchical aggregation and FG losses;
- `full` enables the complete configuration.

No result values are hard-coded.

## Commands And Outputs

Create:

- `scripts/ablation/run_table_4_3.sh`: run one dataset, one group or all groups;
- `scripts/ablation/run_all_table_4_3.sh`: run all three datasets;
- `scripts/ablation/summarize_table_4_3.py`: parse evaluation logs and generate
  CSV and Markdown tables.

Output layout:

```text
output/ablation/table_4_3/<dataset>/<group>/seed<seed>/
```

Each seed has separate Base and Novel evaluation directories to avoid relying
on timestamped log filenames.

The summary files are:

```text
output/ablation/table_4_3/table_4_3_results.csv
output/ablation/table_4_3/table_4_3_results.md
```

Missing or malformed runs are reported as errors and are never treated as
zero-valued results.

## Compatibility

The new trainer registers all trainable modules as one Dassl checkpoint unit,
matching the fixed checkpoint behavior of the final trainers. Existing
trainers and scripts remain available, but Table 4.3 reproduction uses only
the new controlled entry point.

## Verification

Static contract tests verify:

- exactly five valid groups;
- each group maps to the paper's module matrix;
- `single_view_kd` has no adaptive or FG module;
- `no_kd` has adaptive supervision but no KD;
- only `full` enables hierarchical aggregation and FG losses;
- the runner covers the three paper datasets and three seeds;
- summaries calculate harmonic mean as `2 * base * novel / (base + novel)`;
- checkpoint registration includes every trainable module.

GPU validation requires PyTorch, the CLIP checkpoint, and prepared datasets.
