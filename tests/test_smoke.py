from pathlib import Path

import torch

from eukcontigminer.predictor import DEFAULT_THRESHOLD, Predictor


def test_bundled_model_loads_and_is_rc_invariant():
    predictor = Predictor("cpu")
    sequence = "ACGT" * 250
    reverse_complement = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    forward, reverse = predictor.predict([sequence, reverse_complement])
    assert 0.0 <= forward <= 1.0
    assert abs(forward - reverse) < 1e-6
    assert DEFAULT_THRESHOLD == 0.86
