# v0.2.0

This release replaces the v0.1.0 DNA-only command with the current frozen
Validation-best architecture:

- one `p_euk` and strict binary label per legal whole contig;
- no Unknown class and no Tiara runtime;
- Eukaryota truth includes nuclear and organelle DNA;
- DNA ensemble plus frozen ESM-2 650M top-two-ORF evidence;
- one 24 GB CUDA GPU deployment;
- bundled configuration and learned assets are SHA-bound;
- the 2.5 GB upstream ESM checkpoint is downloaded by `fair-esm==2.0.0` and
  SHA-checked rather than redistributed.

Frozen-threshold F1 at 1% Euk prevalence is 0.975819 on Continuous Full and
0.970069 on Formal 21-point Full. The fungi+bacteria-only values, retaining
fungal-host organelles, are 0.973985 and 0.968441. These are Validation
results; the F1 >= 0.99 project target has not yet been reached.

The release wheel was installed into an isolated target and scored the six
synthetic 1-200,001 bp canary records from the installed package on one RTX
4090. Its prediction table exactly reproduced the development candidate:
SHA-256 `aa2c239122325e3036e88b51b39949c52125e41748ce686a00c95f444efa476c`.

See `MODEL_CARD.md` for data provenance, branch comparison, and limitations.
