"""Single two-sided early-exit implementation; no length groups."""
import numpy as np


def centered(probabilities, thresholds):
    p = np.asarray(probabilities, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    z = np.log(np.clip(p, 1e-15, 1 - 1e-15) / (1 - np.clip(p, 1e-15, 1 - 1e-15)))
    return z - np.log(t / (1 - t))


def route(probabilities, thresholds, gates):
    """Return final centered scores and routes; H1/H2 have one pair of cuts each."""
    z = centered(probabilities, thresholds)
    routes = np.full(len(z), 2, dtype=np.int8)
    for stage, name in enumerate(('H1', 'H2')):
        lo, hi = gates[name]
        take = (routes == 2) & ((z[:, stage] < lo) | (z[:, stage] > hi))
        routes = np.where(take, stage, routes).astype(np.int8)
    selected = z[np.arange(len(z)), routes]
    return 1.0 / (1.0 + np.exp(-selected)), routes
