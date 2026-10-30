"""Inference-only v0.71 heads and feature-level routing."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v071_heads import H1Flex, HeadFlex
from routing import route as route_scores


class V071Runtime:
    def __init__(self, model_dir, device='cpu'):
        self.dir = Path(model_dir)
        self.device = device
        self.cal = json.loads((self.dir / 'calibration.json').read_text())
        self._heads = {}
        self.t = np.array([self.cal['heads'][k]['threshold'] for k in ('H1', 'H2', 'H3')])

    def _head(self, name):
        if name not in self._heads:
            dim = {'H1': 943, 'H2': 2479, 'H3': 4399}[name]
            ck = torch.load(self.dir / f'{name}_selected.pt', map_location='cpu', weights_only=False)
            cfg = dict(ck['cfg'])
            model = (H1Flex(cfg) if name == 'H1' else HeadFlex(dim, cfg)).to(self.device)
            model.load_state_dict(ck['state_dict'])
            model.eval().requires_grad_(False)
            self._heads[name] = model
        return self._heads[name]

    @torch.no_grad()
    def head_prob(self, name, mat, batch=8192):
        dim = mat.shape[1]
        model = self._head(name)
        out = np.empty(len(mat), np.float64)
        for lo in range(0, len(mat), batch):
            x = torch.from_numpy(np.asarray(mat[lo:lo + batch, :dim], dtype=np.float32)).to(self.device)
            out[lo:lo + batch] = torch.sigmoid(model(x).double()).cpu().numpy()
        return out

    def probs(self, mat, dim, batch=8192):
        name = {943: 'H1', 2479: 'H2', 4399: 'H3'}[dim]
        return self.head_prob(name, np.asarray(mat[:, :dim]), batch)

    def route(self, p, lengths):
        return route_scores(p, self.t, lengths, self.cal['gates'])

    def predict_features(self, h1, h2, h3, lengths):
        p = np.column_stack([self.probs(h1, 943), self.probs(h2, 2479), self.probs(h3, 4399)])
        scores, routes = self.route(p, lengths)
        return {'scores': scores, 'routes': routes, 'probs': p, 'labels': scores > 0.5}
