# EukContigMiner

Classify DNA contigs by **eukaryotic** or **fungal** origin with two independent H1/H2/H3 early-exit models. Host-derived organelle DNA is included in the corresponding positive class.

## Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.13.0"
python -m pip install eukcontigminer-0.86-py3-none-any.whl
eukcontigminer download-models --output models
```

Local PyTorch builds such as `2.13.0+cu126` are accepted. Unsupported base versions and prerelease/postrelease/dev builds are rejected.

## Run

```bash
eukcontigminer input.fasta.gz --task euk --models models -o result.tsv --device cuda:0
eukcontigminer input.fasta.gz --task fun --models models -o result.tsv --device cuda:0
eukcontigminer input.fasta.gz --task both --models models -o result.tsv --device cuda:0
```

`--task` accepts `euk` (default), `fun`, `both`, `euk,fun`, or repeated `--task` flags. `--min-length BP` raises the NA cutoff above the default 500 bp (minimum 500). Prediction is offline on one GPU; `--device cpu` is supported.

## Output

```text
contig_id  length_bp  p_euk  label_euk  head_euk  p_fun  label_fun  head_fun
```

- `head_euk`: `1/2/3` = H1/H2/H3
- `head_fun`: `1/2/3` = H1/H2/H3 of the fun model
- `label_euk`: `eukaryota` / `other`
- `label_fun`: `fungi` / `other`
- Unselected tasks and contigs shorter than 500 bp keep their row and `length_bp`, with `NA`
- Input order and duplicate headers are preserved
- Output files are never overwritten

## Early exit

Both tasks use H1 -> H2 -> H3, with one pair of two-sided cuts per head and no length groups.

```text
euk: H1 [-10, 8], H2 [-4, 1], H3 fallback
fun: H1 [ -4, 2.5], H2 [-2,  1], H3 fallback
```

Validation routing:

| task | H1 | H2 | H3 |
|---|---:|---:|---:|
| euk | 80.4% | 15.3% | 4.3% |
| fun | 94.5% | 3.5% | 2.0% |

## Metrics

F1 at 1% target prevalence:

| Dataset | euk overall | euk 1–2 kb | fun overall | fun 1–2 kb |
|---|---:|---:|---:|---:|
| Validation | **0.992503** | **0.980114** | **0.936182** | **0.851967** |
| 1 kb+ Test | **0.991364** | **0.974514** | — | — |
| Test500 (500–999 bp) | **0.904923** | — | — | — |

See [reports/metrics.json](reports/metrics.json). Test results use fixed Validation-selected models and thresholds; Test was not used for tuning. `p_euk` and `p_fun` are unified decision scores, not calibrated posteriors.

## License

**CC BY-NC-SA 4.0**. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
