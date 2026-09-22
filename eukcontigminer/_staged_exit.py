import json
import numpy as np

class FrozenGates:
    def __init__(self, path):
        self.cuts = json.loads(path.read_text())["cuts"]

    def apply(self, z, lengths, branch, side):
        g = np.digitize(lengths, [2000, 10000])
        c = np.array([self.cuts[branch][str(k)][side] for k in range(3)])[g]
        legal = (lengths >= 1000) & (lengths <= 100000) & np.isfinite(z)
        return legal & ((z > c) if side == "euk_above" else (z < c))

    def dna_euk(self, z, lengths):
        return self.apply(z, lengths, "dna", "euk_above")

    def nt_sides(self, z, lengths):
        return (self.apply(z, lengths, "nt", "other_below"),
                self.apply(z, lengths, "nt", "euk_above"))
