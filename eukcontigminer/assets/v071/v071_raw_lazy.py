"""Inference-only FASTA runtime: lazy H1 -> NT/H2 -> ESM/H3 early exit."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'candidate/runtime'))
from v071_runtime import V071Runtime
from routing import centered, apply_stage, final_scores
from ecm_revised_native_h1_20261013 import load_h1
from ecm_revised_native_encoders_20261013 import FrozenNT, FrozenESM


class V071LazyRuntime:
    def __init__(self, model_root, device='cuda:0'):
        self.root = Path(model_root)
        self.device = torch.device(device)
        self.heads = V071Runtime(self.root / 'v071', device=device)
        self.t = self.heads.t
        self.cal = self.heads.cal
        self._h1 = self._nt_enc = self._esm_enc = None

    def _h1_model(self):
        if self._h1 is None:
            self._h1 = load_h1(self.root / 'candidate', self.device)
        return self._h1

    def _nt_model(self):
        if self._nt_enc is None:
            self._nt_enc = FrozenNT(self.root / 'encoders', self.device)
        return self._nt_enc

    def _esm_model(self):
        if self._esm_enc is None:
            self._esm_enc = FrozenESM(self.root / 'encoders', self.device)
        return self._esm_enc

    @torch.inference_mode()
    def infer(self, sequences):
        n = len(sequences)
        lengths = np.array([len(s) for s in sequences], np.int64)
        dna = np.empty((n, 942), np.float32)
        lut = np.full(256, 4, np.uint8)
        for j, c in enumerate(b'ACGT'):
            lut[c] = j
        lo = 0
        while lo < n:
            hi = lo + 1
            longest = int(lengths[lo])
            while hi < n and (hi - lo + 1) * max(longest, int(lengths[hi])) <= 512000:
                longest = max(longest, int(lengths[hi]))
                hi += 1
            token = np.full((hi - lo, longest), 5, np.uint8)
            for j, s in enumerate(sequences[lo:hi]):
                token[j, :len(s)] = lut[np.frombuffer(s, dtype=np.uint8)]
            r = self._h1_model().infer(torch.from_numpy(token).to(self.device, dtype=torch.long),
                                       torch.from_numpy(lengths[lo:hi]).to(self.device))
            dna[lo:hi] = r['dna'].cpu().numpy()
            lo = hi
        loglen = np.log10(lengths.astype(np.float64)).astype(np.float32)[:, None]
        h1 = np.concatenate([dna, loglen], 1)
        p1 = self.heads.head_prob('H1', h1)
        z = np.full((n, 3), np.nan)
        z[:, 0] = centered(p1, self.t[0])
        group = np.digitize(lengths, [2000, 10000])
        routes = np.full(n, 2, dtype=np.int8)
        routes = apply_stage(routes, group, z, 0, self.cal['gates'])
        p2 = np.full(n, np.nan)
        p3 = np.full(n, np.nan)
        nt_all = np.full((n, 1536), np.nan, np.float32)
        need = np.where(routes != 0)[0]
        if len(need):
            nt = self._nt([sequences[i] for i in need])
            nt_all[need] = nt
            h2 = np.concatenate([dna[need], nt, loglen[need]], 1)
            p2[need] = self.heads.head_prob('H2', h2)
            z[need, 1] = centered(p2[need], self.t[1])
            routes = apply_stage(routes, group, z, 1, self.cal['gates'])
        need = np.where(routes == 2)[0]
        if len(need):
            esm = self._esm([sequences[i] for i in need])
            h3 = np.concatenate([dna[need], nt_all[need], esm, loglen[need]], 1)
            p3[need] = self.heads.head_prob('H3', h3)
            z[need, 2] = centered(p3[need], self.t[2])
        scores = final_scores(z, routes)
        return {'scores': scores, 'routes': routes, 'probs': np.column_stack([p1, p2, p3]),
                'labels': scores > 0.5, 'lengths': lengths}

    infer_chunk = infer

    def _nt(self, seqs):
        model = self._nt_model()
        out = []
        for lo in range(0, len(seqs), 32):
            views = model.views(seqs[lo:lo + 32])
            ids, mask = model.tokens(views, self.device)
            out.append(model.features(ids, mask).float().cpu().numpy())
        return np.concatenate(out, 0)

    def _esm(self, seqs):
        model = self._esm_model()
        return np.concatenate([model.infer_block(seqs[i:i + 1024]).cpu().numpy()
                               for i in range(0, len(seqs), 1024)])
