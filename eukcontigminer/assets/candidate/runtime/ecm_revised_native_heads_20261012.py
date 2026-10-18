"""Standalone inference for revised nested H2/H3 checkpoints.

No training modules, paths, data loading, or encoder changes. Inputs are frozen
DNA942, NT1536, ESM1920, log10(float32 length), in that order. H2 also accepts
its compact 2479-column input without ESM. Outputs are logits, not calibrated
probabilities. Caller must separately bind checkpoint, features and thresholds.
"""
import numpy as np
import torch
from torch import nn


def features(dna, nt, lengths, esm=None):
    parts = [np.asarray(dna, dtype=np.float32), np.asarray(nt, dtype=np.float32)]
    sizes = [942, 1536]
    if esm is not None:
        parts.append(np.asarray(esm, dtype=np.float32))
        sizes.append(1920)
    lengths = np.asarray(lengths, dtype=np.float32)
    if lengths.ndim != 1 or not np.isfinite(lengths).all() or not (lengths > 0).all():
        raise ValueError('Expected finite positive one-dimensional lengths')
    if any(a.shape != (len(lengths), dim) or not np.isfinite(a).all()
           for a, dim in zip(parts, sizes)):
        raise ValueError('Invalid feature dimensions or nonfinite features')
    return np.concatenate([*parts, np.log10(lengths)[:, None]], axis=1)


class _Base(nn.Module):
    def __init__(self, stage):
        super().__init__()
        self.sizes = [942, 1536] + ([1920] if stage == 'H3' else [])
        width = sum(self.sizes) + 1
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('std', torch.ones(width))
        self.linear = nn.Linear(width, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 128), nn.GELU()) for dim in self.sizes])
        self.net = nn.Sequential(nn.Dropout(.3), nn.Linear(128 * len(self.sizes) + 1, 256),
                                 nn.GELU(), nn.Dropout(.3), nn.Linear(256, 64),
                                 nn.GELU(), nn.Linear(64, 1))
        if stage == 'H3':
            self.length_gate = nn.Sequential(nn.Linear(1, 8), nn.GELU(),
                                             nn.Linear(8, len(self.sizes)))

    def forward(self, x):
        z = (x - self.mean) / self.std
        parts = torch.split(z[:, :-1], self.sizes, dim=1)
        if hasattr(self, 'length_gate'):
            gains = 2 * torch.sigmoid(self.length_gate(z[:, -1:]))
            emb = [branch(part) * gains[:, i:i+1]
                   for i, (branch, part) in enumerate(zip(self.branches, parts))]
        else:
            emb = [branch(part) for branch, part in zip(self.branches, parts)]
        fused = torch.cat(emb + [z[:, -1:]], dim=1)
        return (self.linear(z) + self.net(fused)).squeeze(-1)


class NativeHead(nn.Module):
    """Use from_state_dict to strictly construct an immutable inference head."""
    def __init__(self, stage):
        super().__init__()
        if stage not in ('H2', 'H3'):
            raise ValueError('stage must be H2 or H3')
        self.stage = stage
        self.base = _Base(stage)
        self.h1 = nn.Sequential(nn.Linear(943, 512), nn.GELU(), nn.Linear(512, 128),
                                nn.GELU(), nn.Linear(128, 1))
        self.upstream = NativeHead('H2') if stage == 'H3' else None
        self.project = nn.Sequential(nn.Linear(len(self.base.mean), 64), nn.GELU())
        self.correction = nn.Sequential(nn.Linear(65 + (stage == 'H3'), 32),
                                        nn.GELU(), nn.Linear(32, 1))
        self.eval()
        self.requires_grad_(False)

    def train(self, mode=True):
        if mode:
            raise ValueError('NativeHead is inference-only')
        return super().train(False)

    @classmethod
    def from_state_dict(cls, stage, state):
        if not state or any(not torch.is_tensor(v) or v.dtype != torch.float32
                            or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Expected finite float32 state tensors')
        model = cls(stage)
        model.load_state_dict(state, strict=True)
        for name, tensor in model.named_buffers():
            if name.endswith('std') and not (tensor > 0).all():
                raise ValueError('Normalization standard deviations must be positive')
        return model

    def forward(self, full):
        if full.ndim != 2 or full.dtype != torch.float32:
            raise ValueError('Expected a two-dimensional float32 tensor')
        allowed = (2479, 4399) if self.stage == 'H2' else (4399,)
        if full.shape[1] not in allowed:
            raise ValueError('Invalid feature width for head')
        x = full if self.stage == 'H3' or full.shape[1] == 2479 else torch.cat(
            (full[:, :2478], full[:, 4398:4399]), 1)
        scores = [torch.tanh(self.h1(torch.cat((full[:, :942], full[:, -1:]), 1)) / 8)]
        if self.upstream is not None:
            scores.append(torch.tanh(self.upstream(full)[:, None] / 8))
        normalized = (x - self.base.mean) / self.base.std
        return self.base(x) + self.correction(torch.cat(
            [self.project(normalized), *scores], 1)).squeeze(-1)
