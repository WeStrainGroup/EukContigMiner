# EukContigMiner v0.40 model card

## Model and prediction contract

The released classifier combines the frozen two-head DNA ensemble with two
small supervised probes sharing one ESM-2 650M forward pass. Tiara is not a
runtime dependency. The contig is scored whole rather than chopped.

Each retained FASTA record receives one `p_euk` in `[0, 1]`. The strict rule is
`p_euk > 0.9821331568384042` for Eukaryota and Other otherwise. Equality is
Other and there is no Unknown class. Optional Eukaryota subclass prediction is
not part of v0.40.

## Data and labels

The genome/QC source was `genome_set_v1`: 10,646 eligible genomes released
strictly before 2026-01-01. Train and Validation are species-disjoint.
Training used 650,262 fragments spanning 2,127,354,814 bases, including 12,631
organelle fragments. Euk-derived nuclear and organelle fragments are positive;
non-Euk-derived fragments are negative. Organelles are not a separate class.
Fungal-host organelles are retained as positive in fungi+bacteria-only tests.

The Continuous Validation panel has 39,807 fragments at 10,780 lengths from
1,001 to 99,992 bp. The Formal Validation panel has 503,608 fragments at 21
lengths: 1-10 kb every 500 bp, 50 kb, and 100 kb. Its manifest SHA-256 is
`e1c81b0a0d96d9ae55b27eaeae0be727f79bc32a2fed4b22e155d6cf2471ba9e`.
Final Test was not used for model selection and was not read for this release.

## Frozen Validation results

One threshold was frozen on Continuous Validation and reused unchanged.

| Panel / scenario at 1% Euk | F1 |
| --- | ---: |
| Continuous Full | 0.980116 |
| Continuous 1-2 kb | 0.958203 |
| Continuous fungi+bacteria | 0.975311 |
| Formal Full | 0.973247 |
| Formal 1-2 kb | 0.873706 |
| Formal fungi+bacteria | 0.971559 |

## Runtime and compatibility

DNA early exit is enabled by default for obvious negatives; `--full-esm`
forces every retained contig through ESM. `--min-length 1000` is the default.
CPU-only inference uses FP32 and may be bounded with `--cpu-threads`. CUDA
inference uses one selected GPU and FP16 ESM autocast. CPU-only and RTX 4090
execution were physically tested for this release. NVIDIA V100, A40, and A100
use the same standard PyTorch CUDA/FP16 path and are compatibility targets, but
were not physically benchmarked on the development server. The release summary
records the actual device and dtype. CPU FP32 and GPU FP16 scores are not
expected to be bit-identical; the installed-wheel canary had zero label changes
across devices. The upstream 2.5 GB ESM-2 checkpoint is downloaded by
`fair-esm==2.0.0` and SHA-checked rather than redistributed.

## Limitations

The hardest validated region is 1-2 kb. Formal Full F1 is below the project
target of 0.99. Validation metrics may not transfer unchanged to a new ecology,
assembler, or prevalence. Retain `p_euk` and recalibrate on an independent,
representative Validation set when operating prevalence or error costs differ.
