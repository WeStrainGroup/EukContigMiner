# v0.55 release selection

Selected: replacement_lr2_step32768_raw, checkpoint SHA256 `6721e6dc411b159f9cab40d1e587389c71c1055e56b62fef1031af0e61dbc71d`.

The final census covers 36 candidates, with 20 missing/unverified report combinations explicitly retained. Its research-cache scores rank this checkpoint first on overall Continuous F1@1%. Earlier Full100 smooth has stronger short-fragment and Formal point estimates; the selected checkpoint has higher supplementary overall F1. It does not dominate every scope, and superiority over that earlier candidate is not statistically established. Incomplete intermediate candidates are not treated as fully validated alternatives. Agreement retains both 100M and 500M and is outside this 100M replacement.

The census is selection history, not installed-release performance. The release uses the completed actual installed comparisons in main_report.json, supplement_report.json and formal_report.json, with separate count reconstruction and paired-species intervals. Overall and 1–2 kb F1@1% exceed v0.54 in all three panels. Main/supplement overall intervals are positive; Formal crosses zero. Reports are conditional on reused development panels and are not an independent final test.

Final installed CPU30/GPU4096 matched-setting outputs are exact matches to their corresponding verified references. The released wheel SHA256 is `de764605f7de74bcbc5a1cae2825a0721e3b4007909d15f591bfa6fb6ee353e7`. The post-training adapter observations and remaining QC limitations are disclosed in README and the source/exposure reports; this is not a contamination-free certification.
