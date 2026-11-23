# EukContigMiner

Identify eukaryotic and fungal contigs using convolutional sequence features, k-mer composition and nucleotide and protein language models.

## Introduction

Eukaryotic sequences often represent a small fraction of a metagenome assembly. Identifying these contigs is an initial step toward recovering eukaryotic genomes and investigating their diversity and function.

EukContigMiner (ECM) classifies DNA contigs using two independent tasks: **euk**, for eukaryotic origin, and **fun**, for fungal origin. Each task uses three successive prediction heads that combine DNA sequence and composition features with nucleotide and protein language-model representations. Contigs that receive a sufficiently decisive prediction leave the pipeline early; the remaining contigs proceed to the next head.

ECM requires only a DNA FASTA file for prediction. Once its model assets are installed, inference runs offline without reference-database searches, sequence alignment, taxonomic queries or coverage profiles. It supports a single CUDA GPU or CPU execution.

The euk task targets Metazoa, Viridiplantae, Fungi and other eukaryotes. The fun task targets Fungi against other eukaryotes and prokaryotes. **Host-derived organelle DNA is included in the corresponding positive class**: eukaryotic organelles are euk positives, and fungal organelles are fun positives. ECM does not provide separate nuclear, mitochondrial or plastid labels.

## Installation

The following instructions use release [v0.90](https://github.com/WeStrainGroup/EukContigMiner/releases/tag/v0.90). The package requires Python 3.11 or 3.12 and PyTorch 2.13.0.

Create an environment and install the release wheel:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.13.0"
python -m pip install \
  "https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.90/eukcontigminer-0.90-py3-none-any.whl"
```

For GPU execution, use a PyTorch 2.13.0 build compatible with your CUDA environment. Local build suffixes such as `2.13.0+cu126` are accepted. Other base versions and prerelease, development or postrelease builds are rejected by the runtime checks.

Download the model assets separately:

```bash
eukcontigminer download-models --output models
```

This command retrieves the ECM model archive and the pinned NTv3 and ESM-C assets, then verifies their checksums. Initial setup requires access to GitHub and Hugging Face. The ECM archive contains the trained DNA encoder and task heads; upstream language-model weights are obtained separately under their respective terms.

For subsequent offline prediction, retain the complete prepared model directory. Supply it with `--models`, or set `EUKCONTIGMINER_MODELS`.

### Model setup

These options belong to `eukcontigminer download-models`:

| Parameter | Default | Description |
| --- | --- | --- |
| `--output` | `models` | Destination model directory. |
| `--heads-archive` | — | Use a local v0.90 ECM model archive. |
| `--nt-dir` | — | Use an existing directory containing the required original NTv3 files. |
| `--esm-weights` | — | Use the original ESM-C 300M weight file. |

Local assets must match the pinned release checksums. Providing only the ECM archive does not replace the NTv3 and ESM-C assets.

## Quick start

Identify eukaryotic contigs:

```bash
eukcontigminer --input assembly.fasta.gz \
  --task euk \
  --models models \
  --device gpu \
  --output assembly_euk
```

Identify fungal contigs directly from the assembly:

```bash
eukcontigminer --input assembly.fasta.gz \
  --task fun \
  --models models \
  --device gpu \
  --output assembly_fun
```

Run both tasks:

```bash
eukcontigminer --input assembly.fasta.gz \
  --task all \
  --models models \
  --device gpu \
  --output assembly_ecm
```

The fun task does not require a preceding euk prediction. With `--task all`, both tasks classify each eligible input contig independently; euk labels do not gate fun predictions, and agreement between the two labels is not enforced.

Use `--device cpu` for CPU execution. Each command creates an output directory containing the classification table, a run summary and the positive FASTA records for each selected task. The default minimum contig length is 1,000 bp; add `--min-length 500` to include 500–999 bp contigs.

## Parameters

Defaults below refer to v0.90. **—** means no preset value.

### Prediction

| Parameter | Default | Description |
| --- | --- | --- |
| `--input` | — | Required DNA FASTA, optionally gzip-compressed with a `.gz` suffix. |
| `--task` | `euk` | `euk`, `fun` or `all`. Comma-separated values and repeated flags are also accepted; `both` remains an alias for `all`. |
| `--models` | `models` | Prepared model directory. `EUKCONTIGMINER_MODELS` changes the default; an explicit argument takes precedence. |
| `--output` | — | Required new output directory. Parent directories are created as needed. `-o` is also accepted. |
| `--device` | `gpu` | `gpu` or `cpu`. GPU execution uses the first visible CUDA device. |
| `--min-length` | `1000` | Minimum contig length to classify, in bp. Must be at least 500; shorter records retain their rows with `NA` predictions. |
| `--chunk-size` | `auto` | Input records per inference chunk. Accepts a positive integer or `auto`. |
| `--cpu-threads` | `2` | PyTorch CPU threads. Accepts a positive integer or `auto`. |
| `--help` | — | Show command-line options. |
| `--version` | — | Print the installed version. |

Input sequences must be nonempty IUPAC DNA. Lowercase bases are accepted. ECM retains input order and the full FASTA header after `>`, including duplicate headers. Use unique contig identifiers when connecting predictions to downstream coverage tables or binning workflows.

By default, contigs shorter than 1 kb are retained in the table with `NA` predictions and are excluded from exported FASTA files. Set `--min-length 500` to classify contigs down to the supported minimum. This option changes eligibility; it does not change the model or its classification threshold.

To select a particular GPU, set `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=1 eukcontigminer \
  --input assembly.fasta.gz --task all --models models \
  --device gpu --output assembly_ecm_gpu1
```

`--chunk-size auto` chooses 1024, 2048 or 4096 records from GPU memory capacity; on CPU it uses 2048. `--cpu-threads auto` uses `SLURM_CPUS_PER_TASK`, then `OMP_NUM_THREADS`, then the host CPU count, capped at eight. Batch changes can slightly alter floating-point scores. For comparisons that require the v0.89 execution settings, explicitly use `--chunk-size 2048 --cpu-threads 2` and match the length cutoff. The run summary records the effective settings.

## How classification works

Both tasks follow the same three-stage architecture with separate trained heads and decision settings.

| Head | Features | Input dimensions |
| --- | --- | ---: |
| H1 | Whole-contig DNA sequence and composition features, plus log-transformed length | 943 |
| H2 | H1 features plus NTv3 nucleotide representations | 2,479 |
| H3 | H2 features plus ESM-C representations of selected translated open reading frames | 4,399 |

The DNA encoder combines a convolutional sequence model with canonical k-mer features. H2 adds representations from the frozen NTv3 100M model; H3 adds representations from the frozen ESM-C 300M model. The language-model features use the runtime's prescribed sequence views and ORF selection.

At H1 and H2, a two-sided gate accepts predictions sufficiently far from the head's decision boundary. Contigs within the gate proceed to the next stage. H3 resolves all remaining eligible contigs. The expensive later-stage features are computed only for contigs that reach those stages. A contig's final score comes from the selected head; the three head predictions are not averaged.

Head-specific thresholds are incorporated into a unified decision score. For each selected task:

```text
score > 0.5   → eukaryota (euk) or fungi (fun)
score <= 0.5  → other
```

The output fields `p_euk` and `p_fun` lie between 0 and 1, but **they are decision scores, not calibrated posterior probabilities**. The CLI uses the fixed release decision boundary and does not expose an adjustable classification or rejection threshold.

## Outputs

For `--task all --output assembly_ecm`, a successful run writes:

```text
assembly_ecm/
├── predictions.tsv    # All input contigs and their classification results
├── euk.fa             # Contigs labelled eukaryota
├── fun.fa             # Contigs labelled fungi
└── run.json           # Run settings, selected counts and output checksums
```

Only the FASTA files for selected tasks are created. A selected task with no positive contigs produces an empty FASTA. A contig positive for both tasks appears in both FASTA files. Sequences are written in input order, with the full original header, uppercase bases and 80-base line wrapping. Duplicate headers remain separate records; extraction uses each record's own prediction.

The TSV always has eight columns:

| Column | Meaning |
| --- | --- |
| `contig_id` | Original FASTA header without the leading `>`. |
| `length_bp` | Input sequence length in bp. |
| `p_euk` | Unified euk decision score. |
| `label_euk` | `eukaryota` or `other`. |
| `head_euk` | Selected euk head: `1`, `2` or `3`. |
| `p_fun` | Unified fun decision score. |
| `label_fun` | `fungi` or `other`. |
| `head_fun` | Selected fun head: `1`, `2` or `3`. |

Unselected tasks and contigs below `--min-length` have `NA` in the relevant score, label and head fields. Their identifiers and lengths remain in the table. These rows were not classified; `NA` is neither a negative call nor a confidence-based rejection. Eligible contigs receive a binary label without an `unknown` class.

`run.json` records completion status, software version, selected tasks, model identifiers, thresholds, requested and resolved device, requested and effective chunk size, CPU thread count, minimum length, input row count, selected contig counts, elapsed time, and input/TSV/FASTA SHA-256 checksums.

The output directory must not already exist, even if empty. ECM stages outputs while processing and writes `run.json` last. If a handled failure occurs, it removes this run's staged and published files. After an interrupted run, use a new directory or inspect any incomplete output before removing it. Retain the classification table, FASTA files and run summary together.

## Downstream use

Use `euk.fa` for eukaryotic candidate sequences or `fun.fa` for fungal candidates. These files contain exactly the records with `label_euk = eukaryota` or `label_fun = fungi`, respectively; shorter, `NA` and negative records are excluded.

These sequences can be supplied to [EukBinPicker](https://github.com/WeStrainGroup/EukBinPicker) with compatible coverage profiles for genome binning. Map reads to the complete assembly before selecting coverage entries for the retained contigs.

ECM predicts sequence origin. A positive label alone does not establish a species assignment, a complete genome, contamination, or residency in the sampled environment. Because organelles are included among the positive targets, workflows requiring nuclear contigs need an additional organelle assessment.

## Evaluation

v0.90 retains the v0.89 model weights and decision thresholds. The following are the **previously reported v0.89 F1 values at 1% target prevalence**, not a new benchmark of the changed v0.90 defaults:

| Dataset | euk: overall | euk: 1–2 kb | fun: overall | fun: 1–2 kb |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.992503 | 0.980114 | 0.936182 | 0.851967 |
| Test, 1 kb and above | 0.991364 | 0.974514 | — | — |
| Test500, 500–999 bp | 0.904923 | — | — | — |

Source: [release metrics](https://github.com/WeStrainGroup/EukContigMiner/blob/v0.89/reports/metrics.json). The 1 kb+ Test comprises 489,080 fragments; Test500 comprises 127,976 fragments. **—** means not reported for that task/subset in the release metrics.

The release documentation states that Test evaluation used fixed models and thresholds selected on Validation, without Test-based tuning. Validation values reflect model selection and should be distinguished from Test results. These metrics describe the release's evaluation and target definitions; they do not establish comparative performance against other classifiers.

When comparing tools, align the target definition, length eligibility and handling of unclassified outputs. In particular, host-origin classification that includes organelles and nuclear-eukaryote screening are different endpoints; report organelle behavior separately when evaluating nuclear recovery.

## Updating from v0.89

v0.90 changes the command-line interface and output layout:

- Pass the FASTA with `--input` and give `--output` a new directory, rather than a TSV filename.
- Use `--task all` for both tasks (`both` remains accepted).
- Use `--device gpu` or `--device cpu`; choose the physical GPU through `CUDA_VISIBLE_DEVICES`.
- The default minimum length changes from 500 to 1,000 bp, and the default chunk size changes from 2048 to `auto`.
- Positive FASTA files are exported automatically alongside `predictions.tsv` and `run.json`.

The model architecture, weights, class definitions and decision thresholds are unchanged. Update scripts that consume the old TSV and summary paths before switching versions.

## Contact

Xinyu Wang: [wangxinyu30@westlake.edu.cn](mailto:wangxinyu30@westlake.edu.cn)

Report software issues through the [GitHub issue tracker](https://github.com/WeStrainGroup/EukContigMiner/issues).

## License

EukContigMiner code is distributed under [CC BY-NC-SA 4.0](https://github.com/WeStrainGroup/EukContigMiner/blob/v0.90/LICENSE). Model terms are separate: NTv3 and ESM-C retain their upstream licenses and access conditions. See [Third-party notices](https://github.com/WeStrainGroup/EukContigMiner/blob/v0.90/THIRD_PARTY_NOTICES.md).
