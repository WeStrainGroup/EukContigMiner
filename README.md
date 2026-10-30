# EukContigMiner

Classify DNA contigs by **eukaryotic** or **fungal** origin with two independent models. Host-derived organelle DNA is included in the corresponding positive class. No reference database is needed during prediction.

## Install

Use a fresh Python 3.11 environment:

```bash
pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.75/eukcontigminer-0.75-py3-none-any.whl
eukcontigminer download-models --output models
```

The one-time setup downloads the new v0.71 heads, legacy encoder-compatible heads, and original NTv3/ESM-C backbones (about 2.0 GB). [Upstream model terms](THIRD_PARTY_NOTICES.md) apply; Hugging Face authentication may be required. Existing backbones can be supplied with `--nt-dir` and `--esm-weights`.

## Run

```bash
eukcontigminer input.fasta.gz --task euk --models models -o euk.tsv --device cuda:0
eukcontigminer input.fasta.gz --task fun --models models -o fun.tsv --device cuda:0
```

`euk` is the default; use `--task fun`, `--task both`, or `--task euk,fun`. The output schema is always `contig_id, length_bp, p_euk, label_euk, p_fun, label_fun`; an unspecified task is NA. Prediction runs offline on one GPU; `--device cpu` is also supported. Contigs shorter than the task minimum are still emitted in the output table with `length_bp`, `p_euk`/`p_fungi = NA`, and `label = NA`; they are not silently dropped. The minimum for prediction is **500 bp for both euk and fun**. Output files are never overwritten.

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

## Extra Test500 evaluation (v0.75)

A separate 500–999 bp Test extension was cut from the frozen Test genomes with the original 32/2 quota, seed 20261001, QC and no-overlap rules. Models, heads, thresholds and gates are unchanged from v0.71; the threshold was fixed at 0.5 and Test was not used for tuning.

| Metric | Overall |
|---|---:|
| F1 @ 1% prevalence | **0.904923** |
| Rows | 127,976 |

[Full report](reports/v072_test500.json). Test500 is a generalization stress test, not a replacement for the 1 kb+ Test.


## Fresh-environment install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-compatible torch build first if needed; local tags such as
# 2.13.0+cu126 are accepted by the runtime version check.
python -m pip install "torch>=2.13.0,<2.14"
python -m pip install eukcontigminer-0.75-py3-none-any.whl
eukcontigminer download-models --output models
```

`download-models` fetches the frozen model assets once; inference is then offline.

## Unified output

All task selections use the same six columns:

```text
contig_id  length_bp  p_euk  label_euk  p_fun  label_fun
```

- `--task euk` (default): `p_fun/label_fun = NA`
- `--task fun`: `p_euk/label_euk = NA`
- `--task both` or `--task euk,fun`: both tasks in one pass
- `label_euk` is `eukaryota` or `other`
- `label_fun` is `fungi` or `other`
- below-minimum contigs keep their row and `length_bp`, with NA prediction/label
