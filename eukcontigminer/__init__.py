"""EukContigMiner: low-prevalence eukaryotic contig screening."""

MODEL_ID = "esmc_tree_nt500m_shortcoverage_lora_qv8_epoch3_v1"
DEPLOYMENT_THRESHOLD = 0.9983936852318344

__all__ = ["DEPLOYMENT_THRESHOLD", "MODEL_ID"]
__version__ = "0.54"
