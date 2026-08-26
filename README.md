# EukContigMiner

EukContigMiner scores each legal DNA contig with one `p_euk` in `[0,1]` and
assigns `Eukaryota` only when `p_euk > 0.9626515924384612`; equality is
`Other`, and there is no Unknown class. Eukaryota includes nuclear and
organelle DNA. The released v0.2.0 model is an original no-Tiara fusion of a
whole-contig DNA ensemble and frozen ESM-2 650M protein-language-model signal.

## Install and run

Python 3.10-3.12, a CUDA-capable PyTorch installation, and one 24 GB GPU are
required for the released DNA+ESM model.

```bash
python -m pip install eukcontigminer-0.2.0-py3-none-any.whl
eukcontigminer contigs.fasta.gz -o predictions.tsv --device cuda:0
```

The first run downloads the official `esm2_t33_650M_UR50D` checkpoint through
`fair-esm==2.0.0` (about 2.5 GB) into the PyTorch cache. EukContigMiner checks
its SHA-256 before use. The smaller DNA backbone, DNA residual heads, and ESM
probe are bundled and also hash-checked.

The output columns are `contig_id`, `length_bp`, `p_euk`, and `label`. Every
valid IUPAC-DNA FASTA record is scored as one whole contig, including lengths
outside the formal benchmark range. Empty records, duplicate identifiers, and
illegal symbols fail closed. Output and JSON provenance summary files are
written atomically and cannot overwrite existing files.

## Frozen Validation results

All values below use one global threshold selected once on Continuous Full at
1% Euk prevalence. Positive truth is Euk-derived sequence, including nuclear
and organelle DNA; negative truth is non-Euk-derived sequence. Fungal-host
organelles remain positive in the fungi+bacteria-only scenario.

| Validation panel / scenario | F1 at 1% Euk prevalence |
| --- | ---: |
| Continuous Full | 0.975819 |
| Continuous Full, 1-2 kb | 0.949244 |
| Continuous fungi+bacteria only | 0.973985 |
| Formal 21-point Full | 0.970069 |
| Formal fungi+bacteria only | 0.968441 |

Continuous Validation has 39,807 fragments, 10,780 distinct lengths, and a
1,001-99,992 bp range. Formal Validation has 503,608 frozen fragments at 1-10
kb every 500 bp plus 50 kb and 100 kb. Genomes are species-disjoint across
Train and Validation, all source release dates are before 2026-01-01, and the
training set uses all 10,646 eligible genomes. Final Test was not used to
select this release.

These are Validation results, not a claim that the project target of F1 >=
0.99 has been reached. See [MODEL_CARD.md](MODEL_CARD.md) for model scope,
benchmark provenance, limitations, and the DNA-only comparison.

## License

MIT. ESM-2 weights are downloaded from the upstream `fair-esm` release and
are not redistributed in this repository.
