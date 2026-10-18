# EukContigMiner

Classify DNA contigs by **eukaryotic** or **fungal** origin with two independent models. Host-derived organelle DNA is included in the corresponding positive class. No reference database is needed during prediction.

## Install

Use a fresh Python 3.11 environment:

```bash
pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.7/eukcontigminer-0.7-py3-none-any.whl
eukcontigminer download-models --output models
```

The one-time setup downloads the trained heads and original NTv3/ESM-C backbones (about 1.8 GB). [Upstream model terms](THIRD_PARTY_NOTICES.md) apply; Hugging Face authentication may be required. Existing backbones can be supplied with `--nt-dir` and `--esm-weights`.

## Run

```bash
eukcontigminer input.fasta.gz --task euk --models models -o euk.tsv --device cuda:0
eukcontigminer input.fasta.gz --task fun --models models -o fun.tsv --device cuda:0
```

`euk` is the default. Each task outputs contig ID, length, `p_euk` or `p_fungi`, and its binary label. Prediction runs offline on one GPU; `--device cpu` is also supported. Inputs must be at least **1,000 bp**. Output files are never overwritten.

## Validation

F1 at **1% target prevalence**, on the same 487,870-contig Validation set:

| Task | Overall | 1–2 kb |
|---|---:|---:|
| `euk` | 0.990373 | 0.973732 |
| `fun` | 0.911217 | 0.803682 |

Train fits the models; Validation selects models and thresholds. The two tasks use separate frozen decision rules. Scores are not calibrated posterior probabilities. [Evaluation details](reports/v07_evaluation.json) include auxiliary prevalences and verification scope. The earlier [v0.62](https://github.com/WeStrainGroup/EukContigMiner/releases/tag/v0.62) remains available as the retrained reference baseline.
