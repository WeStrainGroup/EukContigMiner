"""Inference-only adapter for the revised fixed two-sided diagnostic contract.

Accept bound H1/H2/H3 probabilities and their existing thresholds. This module
neither fits nor freezes gates. The unified score is not a calibrated probability.
No model, encoder, sequence selection, or training operations are changed.
"""
import numpy as np
from scipy.special import expit, logit


def centered_logits(probabilities, thresholds):
    if len(probabilities) != 3 or len(thresholds) != 3:
        raise ValueError('Require H1, H2 and H3 in order')
    values = []
    rows = None
    for probability, threshold in zip(probabilities, thresholds):
        p = np.asarray(probability)
        if p.ndim != 1 or p.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Require one-dimensional float32/float64 probabilities')
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError('Probabilities must be finite and within [0, 1]')
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Threshold must be finite and strictly within (0, 1)')
        if rows is None:
            rows = len(p)
        if len(p) != rows:
            raise ValueError('Head row counts differ')
        # Preserve each input dtype and the exact existing diagnostic expression.
        z = logit(np.clip(p, 1e-15, 1-1e-15)) - logit(threshold)
        if not np.array_equal(p > threshold, z > 0):
            raise ValueError('Centering cannot preserve strict threshold decisions')
        values.append(z)
    return np.column_stack(values)


def route_centered(centered, lengths, gates):
    z = np.asarray(centered)
    length = np.asarray(lengths)
    if z.ndim != 2 or z.shape[1] != 3 or z.dtype.kind != 'f' or np.isnan(z).any():
        raise ValueError('Require three floating centered logits per row; no NaN')
    if length.shape != (len(z),) or not np.isfinite(length).all() or not (length > 0).all() or not (length == np.floor(length)).all():
        raise ValueError('Require finite positive integer lengths')
    if set(gates) != {'H1', 'H2'}:
        raise ValueError('Require both fixed H1 and H2 gates')
    group = np.digitize(length, [2000, 10000])
    routes = np.full(len(z), 2, dtype=np.int8)
    for stage, name in enumerate(['H1', 'H2']):
        if set(gates[name]) != {'0', '1', '2'}:
            raise ValueError('Require all three length groups')
        for key, cut in gates[name].items():
            if set(cut) != {'other_below', 'euk_above'}:
                raise ValueError('Invalid gate keys')
            lower, upper = cut['other_below'], cut['euk_above']
            if not np.isfinite([lower, upper]).all() or not lower <= 0 <= upper:
                raise ValueError('Gate cuts must be finite and straddle zero')
            take = (routes == 2) & (group == int(key)) & ((z[:, stage] < lower) | (z[:, stage] > upper))
            routes[take] = stage
    selected = z[np.arange(len(z)), routes]
    labels = selected > 0
    score = expit(selected)
    # Preserve strict >.5 semantics at floating-point rounding boundaries.
    score = np.where(labels & (score == .5), np.nextafter(score, np.inf), score)
    if not np.array_equal(score > .5, labels):
        raise ValueError('Unified score decision differs from selected head')
    return dict(scores=score, routes=routes, labels=labels, centered_logits=selected)


def predict(probabilities, thresholds, lengths, gates):
    return route_centered(centered_logits(probabilities, thresholds), lengths, gates)
