# TextConsistency

Official research implementation of **TextConsistency**, a multi-view textual
consistency framework for improving prompt learning with vision-language
models.

> **Status:** This repository is being prepared for public reproducibility.
> The core training implementations are available, but the full paper results
> have not yet been re-verified from a clean environment. See
> [Reproducibility Status](#reproducibility-status) before using the code.

## Overview

TextConsistency extends CoOp and CoCoOp with multiple class descriptions. The
current implementation includes:

- dataset-specific multi-view textual descriptions;
- fixed averaging and adaptive view aggregation variants;
- hierarchical view aggregation for fine-grained recognition;
- classification supervision for both prompt and adaptive branches;
- optional knowledge distillation, attention diversity, and contrastive losses;
- base-to-new class evaluation on the prompt-learning benchmark datasets.

The primary implementations are:

- `CoOp_TextConsistency_V2_FG`
- `CoCoOp_TextConsistency_V2_FG`

Earlier variants are retained for ablation studies.

## Repository Structure

```text
.
|-- clip/                 # CLIP model and tokenizer
|-- configs/              # dataset and trainer configurations
|-- Dassl.pytorch/        # vendored Dassl training framework
|-- datasets/             # dataset loaders
|-- docs/                 # dataset preparation and baseline documentation
|-- scripts/              # experiment scripts
|-- templates/            # multi-view descriptions for each dataset
|-- trainers/             # CoOp, CoCoOp, and TextConsistency trainers
|-- tests/                # lightweight implementation contract tests
`-- train.py              # main entry point
```

## Installation

The code is intended for Linux with an NVIDIA GPU. A CUDA-compatible PyTorch
installation is required.

```bash
conda create -n textconsistency python=3.10 -y
conda activate textconsistency

# Install PyTorch for your CUDA version first:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
pip install -e Dassl.pytorch
```

The dependency list is currently provisional. The experiments were developed
around the CoOp/Dassl codebase and have not yet been validated against every
recent package release.

## CLIP Checkpoint

Download the OpenAI CLIP ViT-B/16 checkpoint and place it at:

```text
clip/ViT-B-16.pt
```

The checkpoint is intentionally excluded from version control. The current
TextConsistency trainers load this exact local path.

## Dataset Preparation

Prepare datasets under a common root directory:

```text
/path/to/data/
|-- caltech-101/
|-- dtd/
|-- eurosat/
|-- fgvc_aircraft/
|-- food-101/
|-- imagenet/
|-- oxford_flowers/
|-- oxford_pets/
|-- stanford_cars/
|-- sun397/
`-- ucf101/
```

Detailed download instructions and expected directory layouts are provided in
[`docs/DATASETS.md`](docs/DATASETS.md).

The repository contains configurations for 11 prompt-learning datasets and
four ImageNet distribution-shift datasets:

- Caltech101, OxfordPets, StanfordCars, OxfordFlowers, Food101
- FGVCAircraft, SUN397, DTD, EuroSAT, UCF101, ImageNet
- ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R

## Quick Start

Run commands from the repository root. The editable Dassl installation above
is recommended; alternatively, add it to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/Dassl.pytorch:$PYTHONPATH"
```

The following example trains the CoOp-based final variant on the base classes
of Caltech101:

```bash
python train.py \
  --root /path/to/data \
  --seed 1 \
  --trainer CoOp_TextConsistency_V2_FG \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/vit_b16.yaml \
  --output-dir output/caltech101/seed1 \
  TRAINER.TEXTCONSISTENCY.N_CTX 4 \
  TRAINER.TEXTCONSISTENCY.N_VIEWS 20 \
  TRAINER.TEXTCONSISTENCY.ADAPTIVE_CE_WEIGHT 1.0 \
  TRAINER.TEXTCONSISTENCY.KD_WEIGHT 5.0 \
  TRAINER.TEXTCONSISTENCY.KD_TEMP 1.0 \
  TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT 0.0 \
  TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT 0.0 \
  TRAINER.TEXTCONSISTENCY.N_GROUPS 4 \
  TRAINER.TEXTCONSISTENCY.PREC fp32 \
  DATASET.NUM_SHOTS 16 \
  DATASET.SUBSAMPLE_CLASSES base \
  OPTIM.MAX_EPOCH 50
```

To use the CoCoOp-based implementation, select
`CoCoOp_TextConsistency_V2_FG` and a configuration under
`configs/trainers/CoCoOp/`.

The shell scripts under `scripts/` record existing experiment configurations.
Most contain machine-specific `DATA` paths and must be edited before use.

## Paper Table 4.3 Ablation

The paper ablation uses exactly five groups:

| Group | Description |
| --- | --- |
| `baseline` | CoOp-style prompt learner without TextConsistency modules |
| `single_view_kd` | Single textual view with KD, no adaptive aggregation, no FG |
| `no_kd` | Multi-view adaptive aggregation without KD or FG |
| `v2_no_fg` | Multi-view adaptive aggregation with KD, no FG |
| `full` | Full TextConsistency V2 + FG configuration |

Run one paper dataset:

```bash
DATA_ROOT=/path/to/data bash scripts/ablation/run_table_4_3.sh caltech101 all
```

Run all three Table 4.3 datasets:

```bash
DATA_ROOT=/path/to/data bash scripts/ablation/run_all_table_4_3.sh
```

Summaries are written to:

```text
output/ablation/table_4_3/table_4_3_results.csv
output/ablation/table_4_3/table_4_3_results.md
```

The summarizer reports missing runs explicitly rather than silently treating
them as zero-valued results.

## Available Trainers

| Trainer | Purpose |
| --- | --- |
| `ZeroshotCLIP` | Zero-shot CLIP baseline |
| `CoOp` | CoOp baseline |
| `CoCoOp` | CoCoOp baseline |
| `CoOp_TextConsistency` | Initial multi-view consistency variant |
| `CoOp_TextConsistency_Avg` | Fixed-average ablation |
| `CoOp_TextConsistency_V2` | Adaptive view aggregation |
| `CoOp_TextConsistency_V2_FG` | CoOp with hierarchical fine-grained aggregation |
| `CoCoOp_TextConsistency_V2_FG` | CoCoOp with hierarchical fine-grained aggregation |
| `CoOp_TextConsistency_Ablation` | Controlled five-group Table 4.3 ablation |

## Main Configuration Options

| Option | Description |
| --- | --- |
| `N_CTX` | Number of learnable context tokens |
| `N_VIEWS` | Maximum number of textual descriptions per class |
| `ADAPTIVE_CE_WEIGHT` | Classification-loss weight for the adaptive branch |
| `KD_WEIGHT` | Knowledge-distillation loss weight |
| `KD_TEMP` | Distillation temperature |
| `CONTRASTIVE_WEIGHT` | Optional supervised image-feature contrastive weight |
| `DIVERSITY_WEIGHT` | Attention-diversity regularization weight |
| `N_GROUPS` | Number of groups in hierarchical view aggregation |
| `PREC` | Numerical precision: `fp16`, `fp32`, or `amp` |

Command-line values override the defaults in `train.py` and the selected YAML
configuration.

## Reproducibility Status

The repository currently has the following limitations:

1. Published result tables are not included because they have not yet been
   reproduced from the cleaned public repository.
2. Several experiment scripts contain local filesystem paths and historical
   hyperparameter choices.
3. GPU end-to-end training has not yet been validated in a clean environment.
4. The optional fine-grained contrastive term currently operates on image
   features; its exact correspondence with the paper formulation is under
   review.

Until these items are resolved, treat the repository as a transparent research
release rather than a fully verified reproduction package.

## Verification

Lightweight static checks can be run without datasets:

```bash
python -m unittest discover -s tests
python -m compileall train.py trainers datasets clip
```

Full verification requires the CLIP checkpoint, prepared datasets, PyTorch,
and a CUDA-capable environment.

## Acknowledgements

This code builds on:

- [CLIP](https://github.com/openai/CLIP)
- [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp)
- [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)

Please also cite these projects when using this implementation.

## Citation

The paper citation will be added after the manuscript metadata is finalized.

## License

Licensing information for the cleaned top-level release is being finalized.
The vendored Dassl code and upstream components remain subject to their
respective licenses.
