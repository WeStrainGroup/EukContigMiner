# EukContigMiner v0.2.0 model card

## Model

Model ID:
`esm2_650m_orf2_no_tiara_asymmetric_v3_all_genomes_gbfix_pos380_neg320_v1`.
It combines a 0.8/0.2 DNA residual-head ensemble with ESM-2 650M evidence from
the two longest deterministic ORFs across complete six-frame translation.
The contig is never chopped. Positive and negative standardized protein
evidence use weights 3.8 and 3.2. Tiara is not a runtime dependency.

The model emits a binary screening probability, not taxonomic abundance or a
genome-bin assignment. Optional Eukaryota subclass prediction is not included
in v0.2.0.

## Data and labels

The genome/QC source was `genome_set_v1`: 10,646 eligible genomes released
strictly before 2026-01-01. Train and Validation are species-disjoint.
Training used 650,262 fragments spanning 2,127,354,814 bases, including 12,631
organelle fragments. Euk-derived nuclear and organelle fragments are positive;
non-Euk-derived fragments are negative. Organelles are not reported as an
independent class.

The frozen Continuous Validation panel contains 39,807 fragments at 10,780
lengths from 1,001 to 99,992 bp. The independent Formal Validation panel
contains 503,608 fragments at 21 lengths: 1-10 kb every 500 bp, 50 kb, and 100
kb. Its manifest SHA-256 is
`e1c81b0a0d96d9ae55b27eaeae0be727f79bc32a2fed4b22e155d6cf2471ba9e`.

## Branch comparison

Each branch independently froze one threshold on Continuous Full at 1% Euk
prevalence, then reused that threshold unchanged for all Continuous subsets
and the Formal panel. Thresholds are 0.9792097770 (DNA-only), 0.9787574018
(ESM-only), and 0.9626515924 (DNA+ESM).

| Validation panel / scenario | DNA-only | ESM-only | DNA+ESM |
| --- | ---: | ---: | ---: |
| Continuous Full | 0.953508 | 0.905854 | 0.975819 |
| Continuous 1-2 kb | 0.907673 | 0.841277 | 0.949244 |
| Continuous fungi+bacteria | 0.932117 | 0.891844 | 0.973985 |
| Formal Full | 0.959114 | 0.906982 | 0.970069 |
| Formal 1-2 kb | 0.839050 | 0.728895 | 0.860534 |
| Formal fungi+bacteria | 0.949030 | 0.901088 | 0.968441 |

The ESM probe was supervised directly with binary BCE, so
`sigmoid(probe_logit)` is a valid protein-only diagnostic score. It is much
weaker than DNA alone, especially at 1-2 kb, but the fusion gain shows that its
errors are complementary. ESM-only is diagnostic and is not a separately
packaged production model.

## Limitations

The most difficult validated region is 1-2 kb (Continuous F1 0.949244 at 1%
prevalence). Frozen Formal Full F1 is 0.970069, below the project target of
0.99. Validation metrics may not transfer unchanged to a new ecological
domain, assembly pipeline, or prevalence. Users should retain scores and
recalibrate a threshold on an independent representative Validation set when
their operating prevalence or error costs differ materially.

## Compute and artifacts

Inference was validated to fit one NVIDIA RTX 4090 24 GB GPU. ESM-2 feature
extraction measured about 46 contigs/s in the recorded canary; throughput
depends strongly on ORF lengths. The upstream ESM-2 650M checkpoint is not
redistributed and is SHA-bound after download. Every bundled learned asset is
also SHA-bound by the release configuration.
