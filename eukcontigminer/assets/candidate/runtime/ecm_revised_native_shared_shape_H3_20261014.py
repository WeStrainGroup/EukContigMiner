"""Portable inference for two shape scalars inside a shared-activation H3."""
import torch
from torch import nn
from ecm_revised_native_activation_H3_20261014 import NativeActivationH3, FrozenWidthGELU, features

class FrozenShapeGELU(FrozenWidthGELU):
    def __init__(self, width, shape, mode):
        super().__init__(width)
        if mode not in ('silu', 'mish'):
            raise ValueError('Expected silu or mish mode')
        if not torch.is_tensor(shape) or shape.shape != () or shape.dtype != torch.float32 or not torch.isfinite(shape):
            raise ValueError('Expected a finite float32 shape scalar')
        self.register_buffer('shape', shape.detach().clone())
        self.mode = mode

    def forward(self, x):
        scale = torch.exp(2 * torch.tanh(self.width))
        z = x * scale
        original = nn.functional.gelu(z) / scale
        target = (nn.functional.silu(z) if self.mode == 'silu'
                  else z * torch.tanh(nn.functional.softplus(z))) / scale
        return original + torch.tanh(self.shape) * (target - original)

class NativeSharedShapeH3(NativeActivationH3):
    def __init__(self, mode, state):
        if mode not in ('silu', 'mish') or not isinstance(state, dict):
            raise ValueError('Expected explicit shape mode and state dictionary')
        names = {'net.2.shape', 'net.5.shape'}
        if {n for n in state if n.endswith('.shape')} != names:
            raise ValueError('Shared shape keys differ')
        super().__init__('shared', {n: v for n, v in state.items() if n not in names})
        for i in (2, 5):
            self.head.net[i] = FrozenShapeGELU(state[f'net.{i}.width'], state[f'net.{i}.shape'], mode)
        self.head.load_state_dict(state, strict=True)
        self.shape_mode = mode
        self.requires_grad_(False)
        self.eval()
