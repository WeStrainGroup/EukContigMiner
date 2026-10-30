"""Euk pipeline: H1 -> H2 -> H3 two-sided early exit."""
from pathlib import Path
import numpy as np
import torch
from .heads import H1Flex, HeadFlex
from .routing import centered, apply_stage, final_scores
from .dna_encoder import load_dna_encoder
from .encoders import FrozenNT, FrozenESM


class EukRuntime:
    def __init__(self, model_root, thresholds, gates, device='cpu'):
        self.dir = Path(model_root) / 'euk'
        self.device = device
        self.t = np.asarray(thresholds, dtype=np.float64)
        self.gates = gates
        self._heads = {}

    def _head(self, name):
        if name not in self._heads:
            dim = {'H1': 943, 'H2': 2479, 'H3': 4399}[name]
            ck = torch.load(self.dir / f'{name.lower()}.pt', map_location='cpu', weights_only=False)
            model = (H1Flex(dict(ck['cfg'])) if name == 'H1' else HeadFlex(dim, dict(ck['cfg']))).to(self.device)
            model.load_state_dict(ck['state_dict']); model.eval().requires_grad_(False)
            self._heads[name] = model
        return self._heads[name]

    @torch.no_grad()
    def head_prob(self, name, mat, batch=8192):
        model = self._head(name)
        out = np.empty(len(mat), np.float64)
        for lo in range(0, len(mat), batch):
            x = torch.from_numpy(np.asarray(mat[lo:lo + batch], dtype=np.float32)).to(self.device)
            out[lo:lo + batch] = torch.sigmoid(model(x).double()).cpu().numpy()
        return out

    def predict_features(self, h1, h2, h3, lengths):
        p = np.column_stack([self.head_prob('H1', np.asarray(h1[:, :943])),
                             self.head_prob('H2', np.asarray(h2[:, :2479])),
                             self.head_prob('H3', np.asarray(h3[:, :4399]))])
        z = centered(p, self.t)
        group = np.digitize(np.asarray(lengths), [2000, 10000])
        routes = np.full(len(z), 2, dtype=np.int8)
        routes = apply_stage(routes, group, z, 0, self.gates)
        routes = apply_stage(routes, group, z, 1, self.gates)
        scores = final_scores(z, routes)
        return {'scores': scores, 'routes': routes, 'probs': p, 'labels': scores > 0.5}


class EukPipeline:
    def __init__(self, model_root, thresholds, gates, device='cuda:0'):
        self.root = Path(model_root)
        self.device = torch.device(device)
        self.heads = EukRuntime(model_root, thresholds, gates, device)
        self._dna = self._nt_enc = self._esm_enc = None

    def _dna_model(self):
        if self._dna is None:
            self._dna = load_dna_encoder(self.root / 'euk/h1_dna.pt', self.device)
        return self._dna

    def _nt_model(self):
        if self._nt_enc is None:
            self._nt_enc = FrozenNT(self.root / 'encoders', self.device)
        return self._nt_enc

    def _esm_model(self):
        if self._esm_enc is None:
            self._esm_enc = FrozenESM(self.root / 'encoders', self.device)
        return self._esm_enc

    @torch.inference_mode()
    def infer_chunk(self, sequences):
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
                longest = max(longest, int(lengths[hi])); hi += 1
            token = np.full((hi - lo, longest), 5, np.uint8)
            for j, s in enumerate(sequences[lo:hi]):
                token[j, :len(s)] = lut[np.frombuffer(s, dtype=np.uint8)]
            r = self._dna_model().infer(torch.from_numpy(token).to(self.device, dtype=torch.long),
                                        torch.from_numpy(lengths[lo:hi]).to(self.device))
            dna[lo:hi] = r['dna'].cpu().numpy(); lo = hi
        loglen = np.log10(lengths.astype(np.float64)).astype(np.float32)[:, None]
        p1 = self.heads.head_prob('H1', np.concatenate([dna, loglen], 1))
        z = np.full((n, 3), np.nan)
        z[:, 0] = centered(p1, self.heads.t[0])
        group = np.digitize(lengths, [2000, 10000])
        routes = np.full(n, 2, dtype=np.int8)
        routes = apply_stage(routes, group, z, 0, self.heads.gates)
        p2 = np.full(n, np.nan); p3 = np.full(n, np.nan)
        nt_all = np.full((n, 1536), np.nan, np.float32)
        need = np.where(routes != 0)[0]
        if len(need):
            nt = self._nt([sequences[i] for i in need]); nt_all[need] = nt
            h2 = np.concatenate([dna[need], nt, loglen[need]], 1)
            p2[need] = self.heads.head_prob('H2', h2)
            z[need, 1] = centered(p2[need], self.heads.t[1])
            routes = apply_stage(routes, group, z, 1, self.heads.gates)
        need = np.where(routes == 2)[0]
        if len(need):
            esm = self._esm([sequences[i] for i in need])
            h3 = np.concatenate([dna[need], nt_all[need], esm, loglen[need]], 1)
            p3[need] = self.heads.head_prob('H3', h3)
            z[need, 2] = centered(p3[need], self.heads.t[2])
        scores = final_scores(z, routes)
        return {'scores': scores, 'routes': routes, 'probs': np.column_stack([p1, p2, p3]),
                'labels': scores > 0.5, 'lengths': lengths}

    def _nt(self, seqs):
        model = self._nt_model(); out = []
        for lo in range(0, len(seqs), 32):
            views = model.views(seqs[lo:lo + 32]); ids, mask = model.tokens(views, self.device)
            out.append(model.features(ids, mask).float().cpu().numpy())
        return np.concatenate(out, 0)

    def _esm(self, seqs):
        model = self._esm_model()
        return np.concatenate([model.infer_block(seqs[i:i + 1024]).cpu().numpy()
                               for i in range(0, len(seqs), 1024)])
