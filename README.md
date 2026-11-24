# EukContigMiner

Identify eukaryotic contigs using convolutional and k-mer sequence features together with nucleotide and protein language models.

## Introduction

Recovering eukaryotic genomes from metagenomes begins with identifying their contigs. Eukaryotic sequences often occur at low prevalence, and fragmented assemblies can contain many short contigs. Low prevalence makes false-positive calls particularly costly, while short sequences provide less information for classification. ECM was developed to address these challenges.

EukContigMiner (ECM) takes DNA contigs as input and returns a score and a binary label for eukaryotic origin. It also provides an independent fungal classification task. Each task combines DNA sequence and composition features with nucleotide and protein language-model representations through three successive prediction heads. Contigs with decisive predictions exit early, so later, more expensive representations are computed only when needed. This reduces the amount of language-model computation relative to running every stage on every contig.

ECM requires only a DNA FASTA file. Once its models are prepared, prediction runs offline without reference-database searches, sequence alignment, taxonomic queries or coverage profiles. It supports a single NVIDIA GPU or CPU execution.

In the [previously reported v0.89 evaluation](https://github.com/WeStrainGroup/EukContigMiner/blob/v0.89/reports/metrics.json), the euk model achieved F1 scores of 0.9914 on the 1 kb+ Test set and 0.9745 on its 1–2 kb subset at 1% eukaryotic prevalence. v0.91 retains those model weights; these values describe that evaluation rather than a new benchmark of the current defaults.

The euk task targets Metazoa, Viridiplantae, Fungi and other eukaryotes. The fun task targets Fungi against other eukaryotes and prokaryotes. **Host-derived organelle DNA is included in the corresponding positive class**: eukaryotic organelles are euk positives, and fungal organelles are fun positives. ECM does not provide separate nuclear, mitochondrial or plastid labels.

## Installation

### Software

Install v0.91 with Conda on Linux x86-64:

```bash
conda create -n eukcontigminer --override-channels \
  -c westraingroup -c conda-forge eukcontigminer=0.91
conda activate eukcontigminer
eukcontigminer --help
```

The Conda package prepares a private Python runtime and installs the pinned software dependencies. Installation requires internet access. Model weights are downloaded separately in the next step.

Alternatively, install the [release wheel](https://github.com/WeStrainGroup/EukContigMiner/releases/tag/v0.91) in a Python 3.11 or 3.12 environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.13.0"
python -m pip install \
  "https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.91/eukcontigminer-0.91-py3-none-any.whl"
```

For GPU execution, use a PyTorch 2.13.0 build compatible with your CUDA environment. Local build suffixes such as `2.13.0+cu126` are accepted. Other base versions and prerelease, development or postrelease builds are rejected by the runtime checks.

**Neither software installation method downloads model weights.** Complete model preparation once before prediction.

### Models

Download and prepare all required models:

```bash
eukcontigminer download-models --output /path/to/ecm_models
```

Replace `/path/to/ecm_models` with your model directory. This command downloads the ECM-trained DNA encoder and task heads from GitHub, and the pinned NTv3 and ESM-C assets from Hugging Face. It checks their SHA-256 values and arranges them in the required directory layout. v0.91 uses the unchanged v0.90 ECM model archive.

The language models retain their upstream licenses and access requirements. Once setup finishes, keep the entire directory and pass it with `--models`, or set `EUKCONTIGMINER_MODELS`. Prediction then runs offline.

### Manual model preparation

You can download the files on another computer and transfer the completed directory to an offline machine:

1. Download the [ECM model archive](https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.90/eukcontigminer-0.90-models.tar.gz) and extract it into your model directory. It provides the `euk/` and `fun/` files shown below.
2. From the pinned [NTv3 100M revision](https://huggingface.co/InstaDeepAI/NTv3_100M_pre/tree/5c685dca15891f5c5b80e0c930e23b87a217e441), download the nine files listed under `encoders/nt/` below. Keep their original filenames.
3. Download the pinned [ESM-C 300M weights](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12/resolve/7f10b20ae75017b2dbc884070e03434515709a8d/data/weights/esmc_300m_2024_12_v0.pth) into `encoders/esm/`.
4. Download ECM's [encoder manifest](https://raw.githubusercontent.com/WeStrainGroup/EukContigMiner/v0.91/eukcontigminer/assets/encoders/assets.json) as `encoders/assets.json`.

```text
/path/to/ecm_models/
├── euk/
│   ├── h1_dna.pt
│   ├── h1.pt
│   ├── h2.pt
│   └── h3.pt
├── fun/
│   ├── h1.pt
│   ├── h2.pt
│   └── h3.pt
└── encoders/
    ├── assets.json
    ├── nt/
    │   ├── MODEL-LICENSE.md
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   ├── vocab.json
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── configuration_ntv3_pretrained.py
    │   ├── modeling_ntv3_pretrained.py
    │   └── tokenization_ntv3.py
    └── esm/
        └── esmc_300m_2024_12_v0.pth
```

Verify the completed directory offline:

```bash
eukcontigminer verify-models --models /path/to/ecm_models
```

Verification checks every required file against the release manifest. It does not download files, load the models or run inference. A missing or changed file causes an error.

## Quick start

Identify eukaryotic contigs:

```bash
eukcontigminer \
  --input assembly.fasta.gz \
  --task euk \
  --models /path/to/ecm_models \
  --device gpu \
  --output assembly_euk
```

Identify fungal contigs directly from the assembly:

```bash
eukcontigminer --input assembly.fasta.gz \
  --task fun \
  --models /path/to/ecm_models \
  --device gpu \
  --output assembly_fun
```

Run both tasks:

```bash
eukcontigminer --input assembly.fasta.gz \
  --task all \
  --models /path/to/ecm_models \
  --device gpu \
  --output assembly_ecm
```

The fun task does not require a preceding euk prediction. With `--task all`, both tasks classify each eligible input contig independently; euk labels do not gate fun predictions, and agreement between the two labels is not enforced.

Use `--device cpu` for CPU execution. Each command creates an output directory containing the classification table, a run summary and the positive FASTA records for each selected task. The default minimum contig length is 1,000 bp; add `--min-length 500` to include 500–999 bp contigs.

## Parameters

Defaults below refer to v0.91. **—** means no preset value.

### Prediction

| Parameter | Default | Description |
| --- | --- | --- |
| `--input` | — | Required DNA FASTA, optionally gzip-compressed with a `.gz` suffix. |
| `--task` | `euk` | `euk`, `fun` or `all`. Comma-separated values and repeated flags are also accepted; `both` remains an alias for `all`. |
| `--models` | `models` | Prepared model directory. `EUKCONTIGMINER_MODELS` changes the default; an explicit argument takes precedence. |
| `--output` | — | Required new output directory. Parent directories are created as needed. |
| `--device` | `gpu` | `gpu` or `cpu`. GPU execution uses the first visible CUDA device. |
| `--min-length` | `1000` | Minimum contig length to classify, in bp. Must be at least 500; shorter records retain their rows with `NA` predictions. |
| `--chunk-size` | `auto` | Input records per inference chunk. Accepts a positive integer or `auto`. |
| `--cpu-threads` | `2` | PyTorch CPU threads. Accepts a positive integer or `auto`. |
| `--help` | — | Show command-line options. |
| `--version` | — | Print the installed version. |

Use the full option names shown above. Short flags and abbreviated long options are not accepted.

Input sequences must be nonempty IUPAC DNA. Lowercase bases are accepted. ECM retains input order and the full FASTA header after `>`, including duplicate headers. Use unique contig identifiers when connecting predictions to downstream coverage tables or binning workflows.

By default, contigs shorter than 1 kb are retained in the table with `NA` predictions and are excluded from exported FASTA files. Set `--min-length 500` to classify contigs down to the supported minimum. This option changes eligibility; it does not change the model or its classification threshold.

To select a particular GPU, set `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=1 eukcontigminer \
  --input assembly.fasta.gz --task all --models /path/to/ecm_models \
  --device gpu --output assembly_ecm_gpu1
```

`--chunk-size auto` chooses 1024, 2048 or 4096 records from GPU memory capacity; on CPU it uses 2048. `--cpu-threads auto` uses `SLURM_CPUS_PER_TASK`, then `OMP_NUM_THREADS`, then the host CPU count, capped at eight. Batch changes can slightly alter floating-point scores. For comparisons that require the v0.89 execution settings, explicitly use `--chunk-size 2048 --cpu-threads 2` and match the length cutoff. The run summary records the effective settings.

## How classification works

Both tasks follow the same three-stage architecture with separate trained heads and decision settings.

| Head | Features | Input dimensions |
| --- | --- | --- |
| H1 | Whole-contig DNA sequence and composition features, plus log-transformed length | 943 |
| H2 | H1 features plus NTv3 nucleotide representations | 2,479 |
| H3 | H2 features plus ESM-C representations of selected translated open reading frames | 4,399 |

The DNA encoder combines a convolutional sequence model with canonical k-mer features. H2 adds representations from the frozen NTv3 100M model; H3 adds representations from the frozen ESM-C 300M model. The language-model features use the runtime's prescribed sequence views and ORF selection.

At H1 and H2, a two-sided gate accepts predictions sufficiently far from the head's decision boundary. Contigs within the gate proceed to the next stage. H3 resolves all remaining eligible contigs. The expensive later-stage features are computed only for contigs that reach those stages. A contig's final score comes from the selected head; the three head predictions are not averaged.

Head-specific thresholds are incorporated into a unified decision score. For each selected task:

```txt
score > 0.5   → eukaryota (euk) or fungi (fun)
score <= 0.5  → other
```

The output fields `p_euk` and `p_fun` lie between 0 and 1, but **they are decision scores, not calibrated posterior probabilities**. The CLI uses the fixed release decision boundary and does not expose an adjustable classification or rejection threshold.

## Outputs

For `--task all --output assembly_ecm`, a successful run writes:

```txt
assembly_ecm/
├── ecm.tsv            # All input contigs and their classification results
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

The full `ecm.tsv` can also be supplied to [EukAsmChecker](https://github.com/WeStrainGroup/EukAsmChecker) (EAC), together with the original assembly and its coverage table, for interactive assembly inspection and filtering. Run ECM with `--task euk` or `--task all` for EAC's eukaryotic labels. EAC requires the same unique contig IDs in all three inputs and matching FASTA/ECM lengths. Because EAC uses the first whitespace-delimited FASTA identifier while ECM preserves the full header, prepare matching identifiers before analysis when headers contain descriptions; duplicate IDs must be resolved before EAC import.

ECM predicts sequence origin. A positive label alone does not establish a species assignment, a complete genome, contamination, or residency in the sampled environment. Because organelles are included among the positive targets, workflows requiring nuclear contigs need an additional organelle assessment.


## Contact

Xinyu Wang: [wangxinyu30@westlake.edu.cn](mailto:wangxinyu30@westlake.edu.cn)

## License

EukContigMiner code is distributed under [CC BY-NC-SA 4.0](https://github.com/WeStrainGroup/EukContigMiner/blob/v0.91/LICENSE). Model terms are separate: NTv3 and ESM-C retain their upstream licenses and access conditions. See [Third-party notices](https://github.com/WeStrainGroup/EukContigMiner/blob/v0.91/THIRD_PARTY_NOTICES.md).
