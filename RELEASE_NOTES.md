# v0.40

- Emits one `p_euk` in `[0, 1]` and one binary label for every retained whole
  contig. `p_euk > threshold` is Eukaryota; equality is Other; there is no
  Unknown class.
- Eukaryota includes nuclear and organelle DNA. Fungal-host organelles remain
  positive in the fungi+bacteria-only benchmark.
- Uses the frozen no-Tiara DNA plus dual-probe ESM-2 model and its single global
  threshold. Final Test was not read or changed for this release.
- Keeps DNA early exit as the default and provides `--full-esm` for full protein
  inference.
- Adds `--device auto|cpu|cuda|cuda:N` and `--cpu-threads N`. CPU inference uses
  FP32; CUDA inference uses FP16 ESM autocast and runs on one GPU. The generic
  CUDA path covers common server NVIDIA V100, A40, A100, and RTX 4090 GPUs.
- Retains the default `--min-length 1000` filter and SHA-bound bundled model
  assets.

Frozen-threshold F1 at 1% Euk prevalence is `0.980116` on Continuous Full and
`0.973247` on Formal 21-point Full. The corresponding fungi+bacteria-only F1
values are `0.975311` and `0.971559`. These are Validation results; the project
target of F1 >= 0.99 has not yet been reached.
