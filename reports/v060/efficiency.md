# Real SPIRE runtime comparison

Ten fixed real samples, at most10,000 contigs/sample and >=1000bp,84,990 total. Each pair runs sequentially on the sameRTX4090 with fourCPU threads; version order alternates across samples. Includes startup and I/O. One pair per sample; not a cross-hardware claim. Unlabeled inputs do not establish accuracy.

| Sample | Records | v0.55 seconds | v0.6 seconds | Speedup |
|---|---:|---:|---:|---:|
| SAMN09980693 | 1320 | 34.605 | 28.971 | 1.194 |
| SAMEA104565284 | 4859 | 125.470 | 92.478 | 1.357 |
| SAMEA1906428 | 8811 | 147.197 | 137.386 | 1.071 |
| MMWV60617913ST | 10000 | 195.245 | 73.102 | 2.671 |
| SAMN06328498 | 10000 | 159.456 | 80.458 | 1.982 |
| SAMN20081668 | 10000 | 156.057 | 80.815 | 1.931 |
| SAMEA2737839 | 10000 | 175.783 | 67.747 | 2.595 |
| SAMN11311699 | 10000 | 168.273 | 83.867 | 2.006 |
| MH0200 | 10000 | 173.523 | 142.084 | 1.221 |
| SAMEA14100735 | 10000 | 161.465 | 89.525 | 1.804 |

Ratio of summed per-sample elapsed times: **1.708x**. This is not the parallel four-GPU makespan.

[Structured counts and timings](efficiency.json).
