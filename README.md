# EukContigMiner

Classify DNA contigs by **eukaryotic** or **fungal** origin with two independent models. Host-derived organelle DNA is included in the corresponding positive class. No reference database is needed during prediction.

## Install

Use Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch==2.13.0"
python -m pip install eukcontigminer-0.83-py3-none-any.whl
eukcontigminer download-models --output models
```

Local PyTorch builds such as `2.13.0+cu126` are accepted. Unsupported prerelease,
postrelease and different base versions are rejected before model hashes are scanned.
`download-models` fetches the frozen heads and upstream NTv3/ESM-C backbones once; inference is then offline.

## Run

```bash
eukcontigminer input.fasta.gz --task euk --models models -o euk.tsv --device cuda:0
eukcontigminer input.fasta.gz --task fun --models models -o fun.tsv --device cuda:0
eukcontigminer input.fasta.gz --task both --models models -o both.tsv --device cuda:0
```

`--task` accepts `euk` (default), `fun`, `both`, `euk,fun`, or repeated `--task euk --task fun`.
Prediction is offline on one GPU; `--device cpu` is also supported.

## Output

The output table always has the same eight columns:

```text
contig_id  length_bp  p_euk  label_euk  head_euk  p_fun  label_fun  head_fun
```

- `head_euk`: `1` = H1, `2` = H2, `3` = H3
- `head_fun`: `1` = independent fun head
- `label_euk`: `eukaryota` or `other`
- `label_fun`: `fungi` or `other`
- Unselected tasks are `NA`
- Contigs shorter than 500 bp keep their row and `length_bp`, with prediction, label and head = `NA`
- Inputs must be nonempty IUPAC DNA FASTA; original order and duplicate headers are preserved
- Output files are never overwritten

`p_euk` is the unified decision score for euk and must not be read as a calibrated posterior probability.

## Early exit

The euk path evaluates H1, then H2, then H3:

```text
H1:  score outside [-10, 8] -> exit
H2:  score outside [ -4, 1] -> exit
H3:  fallback
```

There are no length groups. The cuts were selected on Validation only. Validation routing: H1 80.4%, H2 15.3%, H3 4.3%.

## Metrics

F1 at 1% target prevalence:

| Dataset | euk overall | euk 1–2 kb | fun overall | fun 1–2 kb |
|---|---:|---:|---:|---:|
| Validation | **0.992503** | **0.980114** | 0.911217 | 0.803682 |
| 1 kb+ Test | **0.991364** | **0.974514** | — | — |
| Test500 (500–999 bp) | **0.904923** | — | — | — |

The Test numbers use fixed Validation-selected heads and thresholds; Test was not used for tuning.
Current reports: [Validation early exit](reports/early_exit_uniform.json), [1 kb+ Test](reports/test_1kb_v078.json), [Test500](reports/test500.json).
Historical reports remain in `reports/` with their version names.

## Notes

- 500 bp is the minimum prediction length for both tasks.
- The euk and fun paths are independent; `both` runs both and shares no classification decision.
- The eight-column output, NA behavior, strict `>` threshold and no-overwrite publication are part of the interface contract.


## Architecture

- `resources.py`: one manifest/hash/verify/copy entry
- `runtime/encoders.py`: one lazy `EncoderBundle` for DNA/NT/ESM, shared by euk and fun
- `runtime/euk.py`: H1 -> H2 -> H3 orchestration
- `runtime/fun.py`: independent fun path
- `runtime/routing.py`: one stage/exit decision and final routing
- `runtime/upstream/`: pinned upstream encoder code imported as a normal package, not copied into the model directory
