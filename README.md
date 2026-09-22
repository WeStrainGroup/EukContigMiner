# EukContigMiner

**v0.61** adds multilevel bidirectional early exits and permanently removes whole-buffer recomputation. It retains the v0.6 trained DNA, NTv3 100M, ESM-C 300M and fusion models. This is a runtime release; a simpler fusion architecture has not yet matched the complete model.

Every nonempty legal contig receives a finite 0–1 `p_euk`. Strictly `p_euk > 0.9999054162961046` means **Eukaryota**; otherwise **Other**, including equality. Eukaryotic nuclear and organelle sequences are positive. There is no Unknown. All lengths are scored; formal evaluation starts at 1,000 bp.

## Install and run

Use Python 3.10–3.12.

```bash
python -m pip install https://github.com/WeStrainGroup/EukContigMiner/releases/download/v0.61/eukcontigminer-0.61-py3-none-any.whl
```

Follow [one-time backbone setup](NTV3_INSTALL.md), including model terms and file verification:

```bash
export EUKCONTIGMINER_NTV3_DIR="$PWD/models/ntv3_100m"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
eukcontigminer input.fna.gz -o predictions.tsv --device cuda:0
```

Use `--device cpu --cpu-threads 4` for CPU. One GPU suffices. Add `--min-length 1000` to omit shorter records, or `--full-esm` to disable all confidence-based exits. Default buffer size is 1024. Output: `contig_id`, `length_bp`, `p_euk`, `label`. Inference is offline and reference-free: no reference database, alignment or taxonomy lookup. See [source installation](SOURCE_INSTALL.md).

## Routing and numerical behavior

DNA confidently Other → Other; DNA confidently Euk → Euk; otherwise NTv3 → a confident Other/Euk exit; remaining records → ESM/probe/fusion. Additional conditional tree bounds are supported. There is **no whole-buffer fallback**. New empirical gates are frozen from Validation within 1,000–100,000 bp, using three length groups and a fixed two-logit safety margin. They are empirical rules, not mathematical guarantees. Outside this range those new gates are disabled. The original DNA Other gate remains; the DNA representation reads the central 100 kb for longer inputs.

Empirical early exits emit decision scores 0 or 1; conditional exits emit bounds. `p_euk` is not a calibrated posterior probability and can differ substantially from full inference. Labels near the final threshold can also vary with batching or hardware. No universal score or label identity is claimed.

NT inference uses FP32. ESM inherits FP32 on CPU, FP16 on V100, and BF16 on supported A40/A100/4090 GPUs (otherwise FP16). Residue pooling is FP32; final aggregate ESM features retain the existing FP16 round trip. This release does not introduce INT8 quantization.

## Validation

The frozen global threshold is unchanged. F1 is standardized to 1% Euk prevalence using TPR/FPR.

| Development panel | Records | v0.6 F1@1% | v0.61 F1@1% |
|---|---:|---:|---:|
| Validation | 491,991 | 0.986761 | 0.986751 |
| Main | 592,861 | 0.989202 | 0.989293 |
| Supplement | 664,491 | 0.976361 | 0.976359 |
| Long | 47,982 | 0.999343 | 0.999343 |

All **1,797,325** rows were independently audited. Compared with public v0.6: **26 changed labels**, correcting 1 FP and 11 FN, introducing 14 FN and **no new FP**. Small changes are accepted for this runtime release; performance is not strictly improved on every panel. See [metrics, routes, lengths, score differences and prevalence 0.1%/1%/10%](reports/v061_validation.json) and [release notes](reports/v061_release.md). Historical v0.6 reports are under [reports/v060](reports/v060).

Paired CLI ABBA pilots versus the intermediate **no-buffer-fallback** version: **1.0283×** on 1,000 records at 1% Euk; **2.1406×** on 1,000 records at 90% Euk; **1.2306×** on 10,000 SPIRE MH0200 records. These include startup and I/O on RTX4090. They are not direct speedups over public v0.6, a full-panel timing claim or a 5× claim. Do not multiply separate optimization benchmarks.

**F1≥0.99 at every length remains unmet.** Panels are reused development genome fragments, not untouched final tests or real-assembly accuracy estimates. Main/Long share many species with Validation. Historical model ancestry and foundation pretraining exposure remain unresolved; these results do not prove unseen-species generalization. No new original Final/New934 data were read for v0.61. Accuracy above 100 kb is unvalidated. Historical data eligibility limitations remain documented in [methods](reports/v060/methods.md).

## License

Code is MIT. NTv3 and its adapted derivative use the included InstaDeep Open Model Licence 1.0 (April 2025), including noncommercial restrictions. ESM-C has separate upstream terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
