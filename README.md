# EukContigMiner

**v0.6** screens metagenomic contigs for eukaryotic sequences using a simplified DNA branch, NTv3 100M and ESM-C 300M. The initial DNA branch has **69.7% fewer parameters** than v0.55; both language models remain.

Every nonempty legal contig receives a finite 0–1 `p_euk`. Strictly `p_euk > 0.9999054162961046` means **Eukaryota**; otherwise **Other**, including equality. Organelles are positive and there is no Unknown. All lengths are scored; formal evaluation starts at 1,000 bp. These are decision scores, not guaranteed posterior probabilities.

## Install and run

Use Python 3.10–3.12.

```bash
python -m pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.6/eukcontigminer-0.6-py3-none-any.whl
```

Follow the [one-time backbone setup](NTV3_INSTALL.md), including model terms and file verification, then run:

```bash
export EUKCONTIGMINER_NTV3_DIR="$PWD/models/ntv3_100m"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
eukcontigminer input.fna.gz -o predictions.tsv --device cuda:0
```

Use `--device cpu --cpu-threads 4` for CPU. Add `--min-length 1000` to skip shorter records, or `--full-esm` to disable confidence-based early exits. Inference needs no reference database, similarity search or runtime network. Output columns are `contig_id`, `length_bp`, `p_euk`, `label`. See [source installation](SOURCE_INSTALL.md) for a checkout.

## Performance

Actual installed inference uses each version's frozen global threshold. F1 is standardized to 1% Eukaryota prevalence using TPR/FPR; pooled counts determine the overall value.

| Dataset | Records | v0.55 F1@1% | v0.6 F1@1% | v0.55 1–2kb | v0.6 1–2kb |
|---|---:|---:|---:|---:|---:|
| main | 592,861 | 0.988787 | 0.989202 | 0.950310 | 0.951209 |
| supplement | 664,491 | 0.974781 | 0.976361 | 0.906498 | 0.912209 |
| long | 47,982 | 0.999108 | 0.999343 | — | — |

These are reused development panels of reference-genome fragments, not an untouched final test or real metagenomic assemblies. See complete [Main](reports/main_scientific_report.md), [supplementary](reports/supplement_scientific_report.md), [long-contig](reports/long_scientific_report.md) and [Validation](reports/validation_scientific_report.md) reports for 0.1%/1%/10% prevalence, 500 bp strata, FP/FN and uncertainty.

Across ten real SPIRE samples (84,990 contigs ≥1,000 bp), the ratio of summed paired runtimes was **1.71x faster than v0.55** on RTX4090 with four CPU threads. This includes startup/I/O and does not establish cross-GPU speed or classification accuracy on these unlabeled samples. [Per-sample timings](reports/efficiency.md).

The initial DNA branch reads the central 100 kb of longer inputs. A guarded early exit skips both language models for confident Other calls and recomputes the original batch when a score is close to the decision threshold. On the registered panels this preserved full-inference labels; scores and behavior near thresholds can differ across hardware or batching. Accuracy beyond 100 kb has not been validated.

**F1>=0.99 in every 500 bp bin remains unmet.** Lower false-positive counts can come with more false negatives. Supervised Train/Validation genomes predate 2026-01-01, and current fitting excludes known contaminated records, but ancestor-model exposure and foundation-model pretraining overlap are not fully resolved. This release does not claim completely decontaminated training or independent confirmation. See [methods and limits](reports/methods.md).

## License

Code is MIT. NTv3 and its adapted derivative use the included InstaDeep Open Model Licence 1.0 (April 2025), including noncommercial restrictions. ESM-C has separate upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
