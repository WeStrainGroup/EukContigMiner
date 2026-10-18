"""Standalone inference-only PCA gated H2 with256 squared-coordinate coefficients; all transforms arrive in a bound state_dict.
No training modules, datasets, transform directories or pretrained encoder access.
"""
import numpy as np
import torch
from torch import nn


def features(dna, nt, lengths, h1_probability, esm=None):
    length = np.asarray(lengths, dtype=np.float32)
    p = np.asarray(h1_probability, dtype=np.float64)
    if length.ndim != 1 or not np.isfinite(length).all() or not (length > 0).all():
        raise ValueError('Expected finite positive lengths')
    if p.shape != length.shape or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Expected matching finite H1 probabilities')
    parts = [np.asarray(dna, dtype=np.float32), np.asarray(nt, dtype=np.float32)]
    dims = [942, 1536]
    if esm is not None:
        parts.append(np.asarray(esm, dtype=np.float32)); dims.append(1920)
    if any(a.shape != (len(length), d) or not np.isfinite(a).all() for a, d in zip(parts, dims)):
        raise ValueError('Invalid feature shapes or nonfinite values')
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.concatenate([*parts, np.log10(length)[:, None],
                           (np.log(p/(1-p))/8).astype(np.float32)[:, None]], axis=1)


class FrozenPCA(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.register_buffer('shift', torch.zeros(dimension, dtype=torch.float32))
        self.register_buffer('projection', torch.zeros(dimension, 128, dtype=torch.float32))

    def forward(self, x):
        return (x-self.shift)@self.projection


class GatedProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(128,256)

    def forward(self, x):
        value,gate=self.project(x).chunk(2,dim=-1)
        return value*torch.sigmoid(gate)


class PCAHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes = [942, 1536]
        dim = sum(self.sizes) + 2
        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('std', torch.ones(dim, dtype=torch.float32))
        self.linear = nn.Linear(dim, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(FrozenPCA(d), GatedProjection()) for d in self.sizes])
        self.net = nn.Sequential(nn.Dropout(.3), nn.Linear(128*len(self.sizes)+2, 256),
                                 nn.GELU(), nn.Dropout(.3), nn.Linear(256, 64),
                                 nn.GELU(), nn.Linear(64, 1))

        self.register_buffer('energy_mean', torch.zeros(2, 128, dtype=torch.float32))
        self.register_buffer('energy_std', torch.ones(2, 128, dtype=torch.float32))
        self.energy_linear = nn.ModuleList([nn.Linear(128, 1, bias=False) for _ in range(2)])


    def forward(self, x):
        z = (x - self.mean) / self.std
        parts = torch.split(z[:, :-2], self.sizes, dim=1)
        embeddings, terms = [], []
        for i, (part, branch) in enumerate(zip(parts, self.branches)):
            projected = branch[0](part)
            embeddings.append(branch[1](projected))
            squared = (projected * projected - self.energy_mean[i]) / self.energy_std[i]
            terms.append(self.energy_linear[i](squared))
        score = self.linear(z) + self.net(torch.cat(embeddings + [z[:, -2:]], dim=1))
        return (score + (terms[0] + terms[1])).squeeze(-1)


class NativePCAGateEnergyH2(nn.Module):
    def __init__(self, state):
        super().__init__()
        if not isinstance(state, dict) or not state or any(
                not torch.is_tensor(v) or v.dtype != torch.float32 or not torch.isfinite(v).all()
                for v in state.values()):
            raise ValueError('Expected finite float32 state tensors')
        self.stage = 'H2'
        self.width = 2480
        if 'mean' not in state or 'std' not in state or state['mean'].shape != (self.width,) or state['std'].shape != (self.width,) or not (state['std'] > 0).all():
            raise ValueError('Invalid normalization')
        if any(k not in state or state[k].shape != (2, 128) for k in ('energy_mean', 'energy_std')) or not (state['energy_std'] >= 1e-6).all():
            raise ValueError('Invalid squared-feature normalization')
        self.head = PCAHead()
        self.head.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        if mode:
            raise ValueError('NativePCAGateEnergyH2 is inference-only')
        return super().train(False)

    def forward(self, x):
        if any(m.training for m in self.modules()) or x.ndim != 2 or x.dtype != torch.float32 or x.shape[1] != self.width or not torch.isfinite(x).all():
            raise ValueError('Expected frozen eval head and finite float32 features of exact width')
        return self.head(x)
