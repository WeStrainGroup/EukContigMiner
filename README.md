# EukContigMiner

Classify DNA contigs by **eukaryotic** or **fungal** origin with two independent models. Host-derived organelle DNA is included in the corresponding positive class. No reference database is needed during prediction.

## Install

Use a fresh Python 3.11 environment:

```bash
pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.72/eukcontigminer-0.72-py3-none-any.whl
eukcontigminer download-models --output models
```

The one-time setup downloads the new v0.71 heads, legacy encoder-compatible heads, and original NTv3/ESM-C backbones (about 2.0 GB). [Upstream model terms](THIRD_PARTY_NOTICES.md) apply; Hugging Face authentication may be required. Existing backbones can be supplied with `--nt-dir` and `--esm-weights`.

## Run

```bash
eukcontigminer input.fasta.gz --task euk --models models -o euk.tsv --device cuda:0
eukcontigminer input.fasta.gz --task fun --models models -o fun.tsv --device cuda:0
```

`euk` is the default. Each task outputs contig ID, length, `p_euk` or `p_fungi`, and its binary label. Prediction runs offline on one GPU; `--device cpu` is also supported. `euk` inputs must be at least **500 bp**; `fun` inputs must be at least **1,000 bp**. Output files are never overwritten.

## Validation

F1 at **1% target prevalence**, on the same 487,870-contig Validation set:

| Task | Overall | 1–2 kb |
|---|---:|---:|
| `euk` | **0.992477** | **0.980114** |
| `fun` | 0.911217 | 0.803682 |

Train fits the models; Validation selects models and thresholds. The new euk v0.71 heads use newly fitted thresholds and two-sided early-exit gates. Scores are not calibrated posterior probabilities. [Evaluation details](reports/v07_evaluation.json) include auxiliary prevalences and verification scope. The earlier [v0.62](https://github.com/WeStrainGroup/EukContigMiner/releases/tag/v0.62) remains available as the retrained reference baseline.

## Final Test (v0.71)

Frozen v0.71 models and fixed Validation thresholds; no Test tuning.

| Metric | Overall | 1–2 kb |
|---|---:|---:|
| F1 @ 1% prevalence | **0.991520** | **0.974524** |

489,080 Test contigs; 8-GPU sharded run in about 27 minutes. [Full report](reports/v071_final_test.json).
