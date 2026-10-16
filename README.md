# EukContigMiner

Identify eukaryotic contigs from FASTA, including organelle DNA, without a reference database.

**v0.62** is the v0.61 architecture retrained on the revised dataset, released as a reproducible baseline.

## Install

Python 3.10–3.12. Install the wheel and follow the one-time [backbone setup](NTV3_INSTALL.md).

```bash
pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.62/eukcontigminer-0.62-py3-none-any.whl
```

## Run

```bash
export EUKCONTIGMINER_NTV3_DIR="$PWD/models/ntv3_100m"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
eukcontigminer input.fna.gz -o predictions.tsv --device cuda:0 --cpu-threads 4
```

One GPU is sufficient; CPU is supported with `--device cpu`. Output: contig ID, length, `p_euk`, and Eukaryota/Other. The frozen decision rule is `p_euk > 0.9996954815448658`; equality is Other. Keep the default 1024-record buffer for the evaluated batching.

## Results

F1 standardized to **1% eukaryotic prevalence**:

| Dataset | Overall | 1–2 kb |
|---|---:|---:|
| Validation (487,870) | 0.992132 | 0.978562 |
| Test (489,080) | 0.991848 | 0.975371 |

Train was used for learning, Validation for selection/calibration, and Test for fixed reporting. See [evaluation and checksums](reports/v062_evaluation.json), [source installation](SOURCE_INSTALL.md), and [model licences](THIRD_PARTY_NOTICES.md). Scores are decision scores, not calibrated posterior probabilities; hardware or batching changes can affect boundary labels.
