"""Inference-only head definitions for v0.71; no training or search code."""
import torch
from torch import nn


class H1Flex(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.family = cfg["family"]
        self.inp = nn.BatchNorm1d(943) if cfg["input_norm"] else nn.Identity()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish}[cfg["activation"]]
        h = cfg["hidden"]; d = cfg["depth"]; dr = cfg["dropout"]
        if self.family == "linear":
            self.net = nn.Linear(943, 1)
        elif self.family == "mlp":
            layers = []
            last = 943
            for _ in range(d):
                layers += [nn.Linear(last, h), act(), nn.Dropout(dr)]
                last = h
            layers += [nn.Linear(last, 1)]
            self.net = nn.Sequential(*layers)
        elif self.family == "residual":
            self.proj = nn.Linear(943, h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, h), act(), nn.Dropout(dr), nn.Linear(h, h), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h)
            self.out = nn.Linear(h, 1)
        elif self.family == "gated":
            self.proj = nn.Linear(943, 2 * h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.GLU(), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h)
            self.out = nn.Linear(h, 1)
        elif self.family == "cnn":
            self.conv = nn.Sequential(nn.Conv1d(1, h, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(h), act(), nn.AdaptiveMaxPool1d(1))
            self.out = nn.Linear(h, 1)
        else:
            raise ValueError(self.family)
        if cfg["init"] == "orthogonal":
            for p in self.parameters():
                if p.ndim >= 2:
                    nn.init.orthogonal_(p)

    def forward(self, x):
        x = self.inp(x)
        if self.family == "linear" or self.family == "mlp":
            return self.net(x).squeeze(-1)
        if self.family == "residual":
            h = self.proj(x)
            for b in self.blocks:
                h = h + b(h)
            return self.out(self.norm(h)).squeeze(-1)
        if self.family == "gated":
            h = self.proj(x)
            h = h[:, :h.shape[1] // 2] * torch.sigmoid(h[:, h.shape[1] // 2:])
            for b in self.blocks:
                h = h + b(h)
            return self.out(self.norm(h)).squeeze(-1)
        if self.family == "cnn":
            h = self.conv(x.unsqueeze(1)).squeeze(-1)
            return self.out(h).squeeze(-1)


class HeadFlex(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.family = cfg["family"]
        self.inp = nn.BatchNorm1d(input_dim) if cfg["input_norm"] else nn.Identity()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish}[cfg["activation"]]
        h, d, dr = cfg["hidden"], cfg["depth"], cfg["dropout"]
        if self.family == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif self.family == "mlp":
            layers, last = [], input_dim
            for _ in range(d):
                layers += [nn.Linear(last, h), act(), nn.Dropout(dr)]
                last = h
            layers += [nn.Linear(last, 1)]
            self.net = nn.Sequential(*layers)
        elif self.family == "residual":
            self.proj = nn.Linear(input_dim, h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, h), act(), nn.Dropout(dr), nn.Linear(h, h), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h); self.out = nn.Linear(h, 1)
        elif self.family == "gated":
            self.proj = nn.Linear(input_dim, 2 * h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.GLU(), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h); self.out = nn.Linear(h, 1)
        else:
            raise ValueError(self.family)
        if cfg["init"] == "orthogonal":
            for p in self.parameters():
                if p.ndim >= 2:
                    nn.init.orthogonal_(p)

    def forward(self, x):
        x = self.inp(x)
        if self.family in ("linear", "mlp"):
            return self.net(x).squeeze(-1)
        if self.family == "residual":
            h = self.proj(x)
            for b in self.blocks:
                h = h + b(h)
            return self.out(self.norm(h)).squeeze(-1)
        h = self.proj(x)
        h = h[:, :h.shape[1] // 2] * torch.sigmoid(h[:, h.shape[1] // 2:])
        for b in self.blocks:
            h = h + b(h)
        return self.out(self.norm(h)).squeeze(-1)
