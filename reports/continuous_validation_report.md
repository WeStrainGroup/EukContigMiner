# Uniform NT window fusion: all-length Continuous Validation

Decision: **freeze_then_all21_formal**. Global threshold: 0.9983936852318344.

At most3 windows: first2000bp, last2000bp, centered2000bp (1999bp for odd length). Whole contig at length<=2000. Unique spans, symmetric under reverse complement.

Mean of window logits; each window logit averages forward and reverse-complement predictions.

| Scope | Tool | F1@0.1% | F1@1% | F1@10% |
|---|---|---:|---:|---:|
| full | candidate | 0.94717707 | 0.98847423 | 0.99280287 |
| full | v052_deployed | 0.91819864 | 0.98474663 | 0.99193585 |
| full | v052_recalibrated | 0.91819864 | 0.98474663 | 0.99193585 |
| full_1000_to_2000 | candidate | 0.85953909 | 0.97177072 | 0.98462718 |
| full_1000_to_2000 | v052_deployed | 0.82169240 | 0.96443668 | 0.98148706 |
| full_1000_to_2000 | v052_recalibrated | 0.82169240 | 0.96443668 | 0.98148706 |
| fungi_bacteria_only | candidate | 0.94163382 | 0.98455726 | 0.98906582 |
| fungi_bacteria_only | v052_deployed | 0.91355051 | 0.98263270 | 0.99011993 |
| fungi_bacteria_only | v052_recalibrated | 0.91355051 | 0.98263270 | 0.99011993 |

This is one global threshold applied to a uniform fusion formula at all lengths. Candidate NT windows were inferred from raw contigs; the shared v0.52 control uses frozen cached features. No confirmation was evaluated. Full confusion counts and precision/recall/FPR are in report.json.
