"""EukContigMiner: low-prevalence eukaryotic contig screening."""

MODEL_ID = "esm2_650m_orf2_no_tiara_asymmetric_v3_all_genomes_gbfix_pos380_neg320_v1"
DEPLOYMENT_THRESHOLD = 0.9626515924384612

__all__ = ["DEPLOYMENT_THRESHOLD", "MODEL_ID"]
__version__ = "0.2.0"
