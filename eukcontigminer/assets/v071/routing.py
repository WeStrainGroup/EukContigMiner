"""One center/route implementation for v0.71 early exit."""
import numpy as np


def centered(probabilities, thresholds):
    p = np.asarray(probabilities, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    z = np.log(np.clip(p, 1e-15, 1 - 1e-15) / (1 - np.clip(p, 1e-15, 1 - 1e-15)))
    return z - np.log(t / (1 - t))


def apply_stage(routes, group, z, stage, gates):
    name = 'H1' if stage == 0 else 'H2'
    for g in (0, 1, 2):
        lo, hi = gates[name][str(g)]
        take = (routes == 2) & (group == g) & ((z[:, stage] < lo) | (z[:, stage] > hi))
        routes[take] = stage
    return routes


def final_scores(z, routes):
    selected = z[np.arange(len(z)), routes]
    return 1.0 / (1.0 + np.exp(-selected))


def route(probabilities, thresholds, lengths, gates):
    z = centered(probabilities, thresholds)
    group = np.digitize(np.asarray(lengths), [2000, 10000])
    routes = np.full(len(z), 2, dtype=np.int8)
    routes = apply_stage(routes, group, z, 0, gates)
    routes = apply_stage(routes, group, z, 1, gates)
    return final_scores(z, routes), routes
