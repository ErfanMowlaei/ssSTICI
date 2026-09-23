# ssSTICI

**Dataset-specific Split-Transformer with Integrated Convolutions for sparse scRNA-seq cell-variant imputation**

Code accompanying the manuscript:

> **De novo transformer modeling improves recovery of genetic cell types from sparse single-cell RNA sequencing**

## Overview

Single-cell RNA sequencing (scRNA-seq) can provide both expression measurements and genetic variant calls for individual cells, but the resulting scRNA-seq alignment are extremely sparse and contain false-positive and false-negative base calls. **ssSTICI** adapts the STICI Split-Transformer with Integrated Convolutions framework to this setting by training a separate model **de novo for each sparse scRNA-seq alignment**. The trained dataset-specific model reconstructs A/T/G/C probabilities at every site and can be used to produce a substantially denser matrix for downstream phylogenetic/genetic-type analyses.

Upstream STICI resources:

- STICI paper: https://www.nature.com/articles/s41467-025-56273-3
- STICI source code: https://github.com/shilab/STICI

This repository is intended to make the ssSTICI implementation self-contained. It should not be necessary to refer to unpublished changes to the upstream STICI code in order to understand the architecture, masking procedure, loss, or training parameters used here.

---

## What changed from STICI?

The distinction between **architecture changes** and **training/application changes** is important.

### Architecture-level changes

The overall STICI design is retained: learned categorical and positional embeddings, overlapping local chunks, multi-head attention, integrated multi-kernel 1D convolution blocks, cross-attention, residual connections, and a convolutional output head.

However, ssSTICI is **not an exact copy of the public STICI architecture**. The principal architecture is intact but input, output, and losses are adapted:

| Component | Public STICI | ssSTICI |
|---|---|---|
| Input representation | genotype/imputation representation used by STICI | 5-state nucleotide input: `A`, `T`, `G`, `C`, `?` |
| Prediction head | genotype probabilities | 4-state base probabilities: `A`, `T`, `G`, `C` |
| Loss functions | Categorical cross-entropy + KLD + Mach-Rsq | Categorical cross-entropy + KLD |

### Training/application changes

The main methodological changes for the scRNA-seq application are:

1. **One model is trained de novo per scRNA-seq alignment.** There is no external reference panel shared across datasets in the standard ssSTICI use case.
2. **Observed-base masking is dataset-specific training corruption.** For each training example, a fixed fraction `--mr` of positions that are currently observed is changed to `?` in the input. Positions already missing remain missing.
3. **Originally missing target positions do not contribute to the loss.** The target is the uncorrupted sequence, but loss is evaluated only where the original target has an observed A/T/G/C call.
4. **Loss is evaluated on all originally observed target positions, not only positions newly masked for that training pass.** Thus both masked and still-visible observed bases contribute to reconstruction training.
5. **The loss is categorical cross-entropy + KL divergence only.** ssSTICI does **not** use the MaCH/Minimac-style R-squared term present as an optional/default-on term in the public STICI HPC implementation. Categorical cross-entropy and KL divergence themselves are inherited from STICI; they are not new loss families introduced by ssSTICI.
6. **There is no train/validation split in this ssSTICI script.** Learning-rate reduction and early stopping monitor training `rec_loss`.
7. **The optimizer remains LAMB**, as in the STICI implementation used as the starting point.

### Exact reconstruction objective

For each batch, the model computes categorical cross-entropy (CCE) and KL divergence (KLD) over the flattened set of originally observed base positions:

```text
reconstruction loss = CCE(y, p) + KLD(y || p)
```
---

## Software environment

| Package | Version |
|---|---:|
| Python | 3.9.21 |
| TensorFlow | 2.14.1 |
| Keras | 2.14.0 |
| NumPy | 1.26.4 |
| tqdm | 4.67.1 |
| TensorFlow Addons | 0.22.0 |

### Create the environment

```bash
conda env create -f environment.yml
conda activate ssSTICI-tf214
```

TensorFlow 2.14 targets CUDA 11.8-era GPU libraries. On an HPC system, load the site's compatible CUDA/cuDNN module before running if those libraries are provided by the cluster. The included SLURM script uses `CUDA_MODULE=cuda` by default and allows this module name to be overridden.

---

## Input format

The cleaned implementation reads plain or gzip-compressed FASTA files. Each FASTA record corresponds to one cell, and every sequence must contain the same number of sites in the same order.

Example:

```text
>cell_001
A?TGC??A...
>cell_002
ACT?C??A...
>cell_003
??TGCCTA...
```

Accepted states are:

- `A`, `T`, `G`, `C`: observed base calls;
- `?`, `-`: missing/unobserved call.

Input is converted to uppercase. Other symbols are treated as missing (`?`). Multi-line FASTA records are supported as well.

**Site order is part of the model definition.** Training and target FASTA files must contain the same number of sites in the same order.

---

## Training

The following command makes all model/training choices explicit and corresponds to the important settings in the active TNBC5 block of the supplied historical SLURM job, while also making previously implicit defaults explicit:

```bash
python ssSTICI.py \
  --mode train \
  --ref path/to/input.fa.gz \
  --save-dir results/save-dir \
  --restart-training 1 \
  --batch-size-per-gpu 4 \
  --embed-dim 128 \
  --na-heads 16 \
  --cross-attention-heads 8 \
  --lr 5e-4 \
  --mr 0.8 \
  --epochs 1000 \
  --cs 2048 \
  --co 256 \
  --sites-per-model 16000 \
  --dropout-rate 0.25 \
  --random-seed 2026 \
  --verbose 2
```

`--restart-training 1` deletes an existing `--save-dir`; use `false` to resume completed/pending outer chunks. When resuming, ssSTICI checks the saved training-file SHA256 and the training/model arguments and refuses to mix incompatible chunks in one run directory.

### Training parameter reference

| Argument | Code default | Meaning |
|---|---:|---|
| `--embed-dim` | 128 | categorical/positional embedding width |
| `--na-heads` | 16 | self-attention heads |
| `--cross-attention-heads` | 8 | cross-attention heads; historical value was hard-coded |
| `--dropout-rate` | 0.25 | dropout after cross-attention |
| `--cs` | 2048 | internal central chunk size |
| `--co` | 256 | internal context/overlap size |
| `--sites-per-model` | 16000 | central sites handled by each independently saved outer model |
| `--mr` | 0.8 | fraction of currently observed bases masked in each training example |
| `--lr` | 0.002 | LAMB learning rate; manuscript/dataset jobs should pass the actual value explicitly |
| `--batch-size-per-gpu` | 8 | per-GPU training batch size; historical TNBC5 job used 4 |
| `--epochs` | 1000 | maximum epochs per outer model |
| `--random-seed` | 2024 | base random seed |
| `--deterministic-ops` | false | requests deterministic TensorFlow ops; reproducibility SLURM example sets true |
| `--which-chunk` | -1 | train all pending outer models; positive values select one 1-based chunk |
| `--verbose` | 2 | Keras verbosity |

Training callbacks are fixed in the source:

- `ReduceLROnPlateau`: monitor `rec_loss`, factor `0.5`, patience `15`;
- `EarlyStopping`: monitor `rec_loss`, patience `50`, restore best weights.

There is no validation dataset in this implementation, so both callbacks operate on training reconstruction loss.

---

## Imputation

```bash
python ssSTICI.py \
  --mode impute \
  --save-dir results/save-dir \
  --ref path/to/input.fa.gz \
  --target path/to/input.fa.gz \
  --batch-size-per-gpu 4 \
  --confidence-threshold 0.7 \
  --verbose 2
```

For inference, providing `--ref` is optional during imputation, but supplying it is useful because ssSTICI verifies that its site count matches the saved model.

The confidence threshold is applied to the maximum A/T/G/C probability. If the maximum is less than or equal to the threshold, the FASTA output contains `-` at that site.

**Important:** the prediction FASTA contains the model's predicted call at **every site**, including sites that were observed in the input. It is not a patching operation that automatically copies observed input bases back into the output.

---

## Outputs and run metadata

A training directory contains, for example:

```text
results/save-dir/
├── run_metadata.json
├── commandline_args.json
├── models/
│   ├── chunks_info.json
│   ├── ssSTICI_chunk_001.keras
│   ├── ssSTICI_chunk_002.keras
│   └── ...
└── out/
    ├── predictions_confidence_0.7.fasta
    ├── top_probability_per_site.tsv
    └── imputation_metadata.json
```

`top_probability_per_site.tsv` contains one row per cell and one column per site. The value is `max(P(A), P(T), P(G), P(C))`; the missing class is not part of the prediction head and is therefore inherently excluded.

---

## SLURM

`slurm_ssSTICI.job` replaces the repeated hard-coded dataset blocks in the historical job with one reusable one-dataset job. Submit one job per dataset and pass dataset-specific paths/parameters through environment variables.

Example for TNBC5:

```bash
sbatch --export=ALL,\
DATASET_NAME=input,\
REF_FASTA=/path/to/TNBC5.fa.gz,\
TARGET_FASTA=/path/to/TNBC5.fa.gz,\
LEARNING_RATE=5e-4,\
BATCH_SIZE=4,\
MASK_RATE=0.8,\
RANDOM_SEED=2024 \
slurm_ssSTICI.job
```

The job defaults to deterministic TensorFlow operations and explicitly passes the model/training hyperparameters. It leaves outputs in `SAVE_DIR/out` rather than repeatedly moving/renaming directories.

### Optional FastTree step

The historical job also ran FastTree after imputation. This is retained as an optional external step without hard-coding a lab-specific binary path:

```bash
sbatch --export=ALL,\
DATASET_NAME=TNBC5,\
REF_FASTA=/path/to/TNBC5.fa.gz,\
RUN_FASTTREE=1,\
FASTTREE_BIN=/path/to/FastTreeMP \
slurm_ssSTICI.job
```

The model environment does not install FastTree; provide the executable separately when this option is used.

---


## Contact

Mowlaei ME: mohammad[dot]erfan[dot]mowlaei[at]temple[dot]edu 

Kumar S: s[dot]kumar[dot][at]temple[dot]edu 
