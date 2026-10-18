"""Standalone inference-only H3 activation widths; no training/data imports."""
import torch
from torch import nn
from ecm_revised_native_pca_head_20261014 import NativePCAHead, features


class FrozenWidthGELU(nn.Module):
    def __init__(self, width):
        super().__init__()
        if not torch.is_tensor(width) or width.shape != () or width.dtype != torch.float32 or not torch.isfinite(width):
            raise ValueError('Expected one finite float32 activation width')
        self.register_buffer('width', width.detach().clone())

    def forward(self, x):
        scale = torch.exp(2 * torch.tanh(self.width))
        return nn.functional.gelu(x * scale) / scale


class NativeActivationH3(NativePCAHead):
    def __init__(self, scope, state):
        if scope not in ('branches', 'shared') or not isinstance(state, dict):
            raise ValueError('Expected explicit branch/shared activation H3 state')
        names = ({f'branches.{i}.2.width' for i in range(3)} if scope == 'branches'
                 else {'net.2.width', 'net.5.width'})
        if {n for n in state if n.endswith('.width')} != names:
            raise ValueError('Activation width keys differ')
        super().__init__('H3', {n: v for n, v in state.items() if n not in names})
        if scope == 'branches':
            for i in range(3):
                self.head.branches[i][2] = FrozenWidthGELU(state[f'branches.{i}.2.width'])
        else:
            for i in (2, 5):
                self.head.net[i] = FrozenWidthGELU(state[f'net.{i}.width'])
        self.head.load_state_dict(state, strict=True)
        self.activation_scope = scope
        self.requires_grad_(False)
        self.eval()
