# v0.61 release notes

- Retain all v0.6 trained model weights and the global threshold.
- Add Validation-frozen positive DNA and two-sided NT exits; only unresolved records reach ESM.
- Remove whole-buffer reruns permanently. Retain conditional bounds and the existing optimized NT tokenizer/batching.
- Package gate JSON and conditional-bound NPZ assets explicitly; install from the matching wheel.
- Keep the original model licenses and offline backbone installation requirements.

Four development panels, 1,797,325 rows: vs public v0.6, 26 labels changed (1 FP and 11 FN corrected; 14 new FN; no new FP). Against the intermediate no-fallback runtime, 29 labels changed (16 FN corrected, 13 new FN, no FP changes). These are different baselines and must not be combined. Empirical decision scores can change substantially. Labels are not universally batch invariant.

The independent six-case Validation boundary replay found identical DNA/NT logits and reproduced all six selected flips by substituting ESM/probe features. This establishes a branch-level mechanism for these cases, not a specific kernel cause or a general bound.

Unified fusion, removing the probe, restoring length calibration, and structured residual heads were evaluated as complete models. None consistently matched or beat the full control across development panels; they are not part of this release. The probe-removal variant increased false negatives. Runtime simplification does not imply the core fusion architecture has been simplified.

The README and v061_validation.json contain current scores, timing scope, precision and scientific limits. Reports in v060 are historical. Original frozen experiment records were not rewritten; this release corrects their older blanket-FP32 and guarded-fallback descriptions.

Installation checks: 15 regression tests passed; 2,000 low/high-prevalence pilot scores exactly match the frozen candidate. CPU/RTX4090 legal-length and full-ESM checks passed. New A40/A100 canaries hit a 900-second timeout during shared-filesystem startup/loading (an owned process was observed in `cl_sync_io_wait`); V100 remained queued at source freeze. These HPC checks are incomplete and are not claimed as passes. See [installation evidence](v061_installation.json). Legacy `guarded_fallback_*` summary counters remain zero for output compatibility.
