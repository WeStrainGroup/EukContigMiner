"""Whole-contig DNA encoder shared by euk and fun; loads euk/h1_dna.pt."""
import torch
from torch import nn
from . import dna_features, sequence
from .dna_model import HierarchicalKmerTCN, HierarchicalKmerTCNConfig


class DNAEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = HierarchicalKmerTCN(HierarchicalKmerTCNConfig(**cfg))
        self.head = nn.Sequential(nn.Linear(943, 512), nn.GELU(), nn.Linear(512, 128),
                                  nn.GELU(), nn.Linear(128, 1))
        self.register_buffer('dna_mean', torch.zeros(942))
        self.register_buffer('dna_std', torch.ones(942))

    def infer(self, tokens, length):
        rc = sequence.reverse_complement_batch(tokens, length, pad_token=5)
        dna = 0.5 * (dna_features.whole_contig_base_features(self.encoder, tokens, length)
                     + dna_features.whole_contig_base_features(self.encoder, rc, length))
        return {'dna': dna}


def load_dna_encoder(path, device='cpu'):
    asset = torch.load(path, map_location='cpu', weights_only=True)
    if asset.get('schema') != 'ecm.revised.inference_asset.v1' or asset.get('stage') != 'H1':
        raise ValueError('Not a valid DNA encoder asset')
    model = DNAEncoder(asset['config'])
    model.load_state_dict(asset['state_dict'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite DNA encoder state')
    model.requires_grad_(False).eval()
    return model.to(device)
