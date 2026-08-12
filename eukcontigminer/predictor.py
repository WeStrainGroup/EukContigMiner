from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterable

import torch

from ._model.hierarchical import HierarchicalKmerTCNConfig
from ._model.motif_v21 import RCMotifWindowExpertConfig
from ._model.motif_v22 import HierarchicalKmerTCNRCMotifDirectionalExpert
from ._model.sequence import reverse_complement_batch

DEFAULT_THRESHOLD = 0.86
WEIGHTS_SHA256 = "f97dc3713803dc3568b27e716b9be19e28522e81a4b7c5d5536f2f7c54e76b1d"

BASE_CONFIG = {
    "vocabulary_size": 6, "pad_token": 5, "embedding_dimension": 24,
    "stem_branch_channels": 48, "stem_kernel_sizes": (7, 15, 31, 63),
    "stem_stride": 8, "channels": 192, "residual_kernel_size": 7,
    "residual_dilations": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    "kmer_orders": (3, 4, 5, 6), "kmer_dimensions": (24, 32, 48, 64),
    "hidden_dimension": 512, "dropout": 0.12, "layer_scale_initial": 0.001,
    "maximum_length": 100000, "binary_classes": 2, "detail_classes": 7,
}
MOTIF_CONFIG = {
    "maximum_synopsis_width": 4096, "stem_kernel_sizes": (9, 17, 33, 65),
    "stem_branch_channels": 24, "stem_stride": 4, "channels": 128,
    "residual_kernel_size": 7, "residual_dilations": (1, 2, 4, 8, 16, 32),
    "attention_heads": 4, "hidden_dimension": 256, "dropout": 0.1,
    "layer_scale_initial": 0.001, "gate_center_bp": 3000.0,
    "gate_log_temperature": 0.45, "fusion_raw_weight_initial": -3.0,
    "maximum_windows": 4,
}

_TOKEN = torch.full((256,), 4, dtype=torch.long)
for _char, _value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _TOKEN[ord(_char)] = _value
    _TOKEN[ord(_char.lower())] = _value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode(sequence: str) -> torch.Tensor:
    raw = sequence.encode("ascii", "replace")
    if not raw:
        raise ValueError("empty contig")
    values = torch.tensor(list(raw), dtype=torch.long)
    return _TOKEN[values.clamp(0, 255)]


class Predictor:
    """Load the bundled v2.2 model and predict one probability per contig."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested)
        self.model = HierarchicalKmerTCNRCMotifDirectionalExpert(
            HierarchicalKmerTCNConfig(**BASE_CONFIG),
            RCMotifWindowExpertConfig(**MOTIF_CONFIG),
        )
        resource = files("eukcontigminer.model_data").joinpath("v2_2_0_weights.pt")
        with as_file(resource) as weight_path:
            if _sha256(weight_path) != WEIGHTS_SHA256:
                raise RuntimeError("bundled model checksum differs")
            payload = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = payload.get("model_state_dict", payload.get("model", payload))
        if not isinstance(state, dict):
            raise RuntimeError("unsupported model checkpoint")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, sequences: Iterable[str], batch_size: int = 1) -> list[float]:
        encoded = [_encode(sequence) for sequence in sequences]
        if not encoded:
            return []
        if any(len(row) > BASE_CONFIG["maximum_length"] for row in encoded):
            raise ValueError("contig exceeds the 100,000 bp model limit")
        results: list[float] = []
        for start in range(0, len(encoded), batch_size):
            rows = encoded[start : start + batch_size]
            lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
            width = int(lengths.max())
            tokens = torch.full((len(rows), width), 5, dtype=torch.long)
            for index, row in enumerate(rows):
                tokens[index, : len(row)] = row
            tokens = tokens.to(self.device)
            lengths = lengths.to(self.device)
            reverse = reverse_complement_batch(tokens, lengths, pad_token=5)
            forward_logits, _ = self.model(tokens, lengths)
            reverse_logits, _ = self.model(reverse, lengths)
            probabilities = 0.5 * (
                torch.softmax(forward_logits.float(), dim=1)[:, 1]
                + torch.softmax(reverse_logits.float(), dim=1)[:, 1]
            )
            results.extend(float(value) for value in probabilities.cpu())
        return results
