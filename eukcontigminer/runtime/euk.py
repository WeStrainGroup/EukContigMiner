"""Euk pipeline: H1 -> H2 -> H3 two-sided early exit."""
from pathlib import Path
import numpy as np
import torch
from .heads import H1Flex, HeadFlex
from .routing import centered, route as route_scores, stage_exit
from .encoders import EncoderBundle, encode_dna, encode_nt


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
        scores, routes = route_scores(p, self.t, self.gates)
        return {'scores': scores, 'routes': routes, 'probs': p, 'labels': scores > 0.5}


class EukPipeline:
    def __init__(self, model_root, thresholds, gates, device='cuda:0', encoders=None):
        self.root = Path(model_root)
        self.device = torch.device(device)
        self.heads = EukRuntime(model_root, thresholds, gates, device)
        self.encoders = encoders or EncoderBundle(self.root, self.device)

    @torch.inference_mode()
    def infer_chunk(self, sequences):
        n = len(sequences)
        lengths = np.array([len(s) for s in sequences], np.int64)
        dna = encode_dna(self.encoders.dna(), sequences, 512000)
        loglen = np.log10(lengths.astype(np.float64)).astype(np.float32)[:, None]
        p1 = self.heads.head_prob('H1', np.concatenate([dna, loglen], 1))
        p2 = np.full(n, np.nan); p3 = np.full(n, np.nan)
        nt_all = np.full((n, 1536), np.nan, np.float32)
        z = np.full((n, 3), np.nan); z[:, 0] = centered(p1, self.heads.t[0])
        routes = np.full(n, 2, dtype=np.int8)
        routes = stage_exit(routes, z, 0, *self.heads.gates['H1'])
        need = np.where(routes != 0)[0]
        if len(need):
            nt = self._nt([sequences[i] for i in need]); nt_all[need] = nt
            h2 = np.concatenate([dna[need], nt, loglen[need]], 1)
            p2[need] = self.heads.head_prob('H2', h2)
            z[need, 1] = centered(p2[need], self.heads.t[1])
            routes = stage_exit(routes, z, 1, *self.heads.gates['H2'])
        need = np.where(routes == 2)[0]
        if len(need):
            esm = self._esm([sequences[i] for i in need])
            h3 = np.concatenate([dna[need], nt_all[need], esm, loglen[need]], 1)
            p3[need] = self.heads.head_prob('H3', h3)
            z[need, 2] = centered(p3[need], self.heads.t[2])
        scores, routes = route_scores(np.column_stack([p1, p2, p3]), self.heads.t, self.heads.gates)
        return {'scores': scores, 'routes': routes, 'probs': np.column_stack([p1, p2, p3]),
                'labels': scores > 0.5, 'lengths': lengths}

    def _nt(self, seqs):
        return encode_nt(self.encoders.nt(), seqs, 32)

    def _esm(self, seqs):
        model = self.encoders.esm()
        return np.concatenate([model.infer_block(seqs[i:i + 1024]).cpu().numpy()
                               for i in range(0, len(seqs), 1024)])
