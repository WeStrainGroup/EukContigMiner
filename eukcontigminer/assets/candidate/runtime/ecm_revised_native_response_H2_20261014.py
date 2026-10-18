"""Inference-only response refinement of the existing compact gated-energy H2."""
import torch
from torch import nn
from ecm_revised_native_gateenergy_H2_20261014 import NativePCAGateEnergyH2, features


class FrozenResponse(nn.Module):
    def __init__(self, response, project=None):
        super().__init__()
        if not torch.is_tensor(response) or response.shape != () or response.dtype != torch.float32 or not torch.isfinite(response):
            raise ValueError('Expected one finite float32 response scalar')
        self.register_buffer('response', response.detach().clone())
        self.project = project

    def forward(self, x):
        scale = torch.exp(2 * torch.tanh(self.response))
        if self.project is None:
            return nn.functional.gelu(x * scale) / scale
        value, gate = self.project(x).chunk(2, dim=-1)
        return value * torch.sigmoid(gate * scale)


class NativeResponseH2(NativePCAGateEnergyH2):
    def __init__(self, scope, state):
        if scope not in ('gates', 'shared') or not isinstance(state, dict):
            raise ValueError('Expected explicit gates/shared response H2 state')
        names = ({f'branches.{i}.1.response' for i in range(2)} if scope == 'gates'
                 else {'net.2.response', 'net.5.response'})
        if {n for n in state if n.endswith('.response')} != names:
            raise ValueError('Response keys differ')
        super().__init__({n: v for n, v in state.items() if n not in names})
        if scope == 'gates':
            for i in range(2):
                self.head.branches[i][1] = FrozenResponse(
                    state[f'branches.{i}.1.response'], self.head.branches[i][1].project)
        else:
            for i in (2, 5):
                self.head.net[i] = FrozenResponse(state[f'net.{i}.response'])
        self.head.load_state_dict(state, strict=True)
        self.response_scope = scope
        self.requires_grad_(False)
        self.eval()
