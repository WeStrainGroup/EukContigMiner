# EukContigMiner

**v0.54** screens metagenomic contigs for eukaryotic sequences using DNA, ESM-C 300M and an adapted Nucleotide Transformer 500M. It replaces v0.50 as the recommended version.

Every nonempty legal contig receives a score `p_euk` in `[0, 1]`: strictly `p_euk > 0.9983936852318344` means **Eukaryota**, otherwise **Other**. Organelles count as eukaryotic; there is no Unknown class. All lengths are scored by default; formal evaluation starts at 1,000 bp.

## Install and run

Use Python 3.10–3.12. Install the release wheel in your environment:

```bash
python -m pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.54/eukcontigminer-0.54-py3-none-any.whl
```

Download the two upstream backbones once, after accepting their applicable terms. The wheel includes the ECM models and adapter; the larger upstream backbones are downloaded separately.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("EvolutionaryScale/esmc-300m-2024-12")
snapshot_download(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    revision="06615c1660c892fc199840c18123f8385b3542a8",
    local_dir="models/nt500m",
    allow_patterns=["README.md", "config.json", "esm_config.py", "modeling_esm.py",
                    "model.safetensors", "special_tokens_map.json",
                    "tokenizer_config.json", "vocab.txt"],
)
PY

export EUKCONTIGMINER_NT500M_DIR="$PWD/models/nt500m"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
eukcontigminer input.fna.gz -o predictions.tsv --device cuda:0
```

Keep the same Hugging Face cache for download and inference. Model files are checked against frozen SHA256 hashes at startup. Inference runs locally without a reference database or sequence-similarity search. For CPU, use `--device cpu --cpu-threads 4`; `--device auto` selects an available GPU or CPU. Output columns are `contig_id`, `length_bp`, `p_euk`, and `label`.

## Performance

Frozen v8 confirmation: **1,090,186 fragments from 3,474 species absent from supervised training**, including fixed and continuous lengths. The same fragments and each version's previously frozen global threshold were used. F1 below is prevalence-adjusted to **1% Eukaryota**.

| Scope | v0.50 | v0.54 |
|---|---:|---:|
| All fragments | 0.96825 | **0.97747** |
| 1–2 kb | 0.93156 | **0.95357** |
| Continuous lengths | 0.95854 | **0.97153** |

The improvement combines a revised protein head and fusion tree with an NT adapter trained on broader sequence coverage. The latest adapter uses 1,633,859 windows from 12,197 registered Train/development species. Training/Validation genome release dates precede 2026-01-01. The model metadata discloses the authorized reassignment of historical holdout species to development.

**The all-length F1 ≥ 0.99 target is not yet met:** 16 of 21 fixed lengths reach it on v8. These are reference-genome fragments, not real metagenomic assemblies. Viral negatives are absent; Archaea and OtherEuk coverage is limited. Foundation-model pretraining overlap is unknown. See the [confirmation report](reports/v8_confirmation_report.md) for all 21 lengths, prevalence 0.1%/1%/10%, confidence intervals and limitations; [Continuous](reports/continuous_validation_report.md) and [Formal](reports/formal_validation_report.md) reports describe reused Validation results.

Release verification: 116 tests passed; the installed GPU CLI exactly reproduced the frozen 84-record check. CPU legal-input checks passed. No new physical V100/A40/A100 or representative throughput comparison is claimed; CPU/GPU scores need not be identical.

## License

Code is MIT. NT backbone weights and the derivative adapter are **CC BY-NC-SA 4.0**; ESM-C weights have separate upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
