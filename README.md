# EukContigMiner

**v0.55** screens metagenomic contigs for eukaryotic sequences with DNA features, ESM-C 300M and an adapted NTv3 100M. It replaces the NT-v2 500M branch in v0.54; the legacy DNA and protein branches remain. The 100M size describes one backbone, not the whole tool.

Every nonempty legal contig receives `p_euk` in [0,1]. Strictly `p_euk > 0.9994805844224834` means **Eukaryota**; otherwise **Other**. Equality is Other, organelles are positive, and there is no Unknown. All lengths are scored; formal evaluation starts at 1,000 bp.

## Install and run

Use Python 3.10–3.12.

```bash
python -m pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.55/eukcontigminer-0.55-py3-none-any.whl
```

Follow the [one-time backbone download instructions](NTV3_INSTALL.md), including model terms and pinned-file verification. Then run:

```bash
export EUKCONTIGMINER_NTV3_DIR="$PWD/models/ntv3_100m"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
eukcontigminer input.fna.gz -o predictions.tsv --device cuda:0
```

Use `--device cpu --cpu-threads 4` for CPU. Inference needs no reference database or sequence-similarity search. Output columns are `contig_id`, `length_bp`, `p_euk`, and `label`. For source checkout installation, see [source installation](SOURCE_INSTALL.md).

## Performance

Both installed packages were run on identical complete fragments with the same batching and each version's frozen global threshold. F1 is adjusted to 1% Eukaryota prevalence using TPR/FPR; overall values pool counts rather than average length-bin F1.

| Dataset | Records | v0.54 F1@1% | v0.55 F1@1% | v0.54 1–2 kb | v0.55 1–2 kb |
|---|---:|---:|---:|---:|---:|
| main | 592,861 | 0.987389 | 0.988787 | 0.941472 | 0.950310 |
| supplement | 664,491 | 0.971821 | 0.974781 | 0.894843 | 0.906498 |
| formal | 503,608 | 0.985811 | 0.987084 | 0.921007 | 0.926616 |

These are reused development panels of reference-genome fragments, not an untouched final test or real assembled metagenomic contigs. Complete length, 0.1%/1%/10% prevalence and count tables are in the [main](reports/main_scientific_report.md), [supplementary](reports/supplement_scientific_report.md) and [fixed-length diagnostic](reports/formal_scientific_report.md) reports. `raw100` in these source reports denotes the v0.55 checkpoint. Paired-species intervals, source hashes and the frozen execution/evaluation scripts accompany the reports.

The change combines a different DNA backbone with further supervised adaptation: 32,768 steps at encoder/head learning rates 2e-6/2e-5, class-balanced broad sampling plus replay. The comparison does not isolate model size or pretraining as the cause of improvement. The retained checkpoint is a single 100M branch, not a 100M+500M Agreement ensemble.

**The F1 ≥ 0.99 target in every 500 bp bin remains unmet.** Genomes used for supervised Train/Validation predate 2026-01-01. Historical former-Final development reuse is disclosed in model metadata. Contamination screening is incomplete and foundation-model pretraining overlap is unknown. Scores are decision scores, not guaranteed posterior probabilities at arbitrary prevalence.

The post-training source audit identified six of 1,234,292 broad-fit fragments overlapping unmasked upstream FCS adapter FIX intervals, with coordinates verified against public accession-version sequences. The completed continuation sampled five of these fragments seven times in total; packed sequence hashes and frozen schedules were verified. This count does not exhaust exposure in earlier training or shared components. The four affected species and six fragment IDs are absent from these evaluation panels; absence of homologous sequence or learned artifact effects is not established. They are listed for exclusion from future fits; these already-trained weights retain their historical exposure. This release does not claim fully decontaminated training. Source, masking and exposure checks are included in the reports.

The earlier research feature caches did not exactly reproduce installed inference, so release performance was re-evaluated with the actual installed CLIs. CPU/GPU numerical differences can change labels near the threshold; exact cross-hardware equality is not claimed.

## License

Code is MIT. NTv3 and the adapted derivative use the included InstaDeep Open Model Licence 1.0 (April 2025), including noncommercial restrictions. ESM-C has separate upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
