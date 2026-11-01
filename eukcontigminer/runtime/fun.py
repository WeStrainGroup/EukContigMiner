"""Fun pipeline: same H1 -> H2 -> H3 early-exit code as euk, fun heads."""
from .euk import EukPipeline


class FunPipeline(EukPipeline):
    def __init__(self, model_root, thresholds, gates, device='cuda:0', encoders=None):
        super().__init__(model_root, thresholds, gates, device, encoders=encoders,
                         head_dir='fun', token_budget=128000)
