"""Euk pipeline: H1 -> H2 -> H3 two-sided early exit."""
from pathlib import Path
import numpy as np
import torch
from .heads import H1Flex, HeadFlex
from .routing import centered, route as route_scores, stage_exit
from .encoders import EncoderBundle, encode_dna, encode_nt


class EukRuntime:
    def __init__(self, model_root, thresholds, gates, device='cpu', head_dir='euk'):
        self.dir = Path(model_root) / head_dir
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
    def __init__(self, model_root, thresholds, gates, device='cuda:0', encoders=None, head_dir='euk', token_budget=512000):
        self.root = Path(model_root)
        self.device = torch.device(device)
        self.heads = EukRuntime(model_root, thresholds, gates, device, head_dir=head_dir)
        self.encoders = encoders or EncoderBundle(self.root, self.device)
        self.token_budget = token_budget

    def infer_chunk(self, sequences):
        return infer_shared({'task': self}, sequences)['task']

    def _nt(self, seqs):
        return encode_nt(self.encoders.nt(), seqs, 32)

    def _esm(self, seqs):
        model = self.encoders.esm()
        return np.concatenate([model.infer_block(seqs[i:i + 1024]).cpu().numpy()
                               for i in range(0, len(seqs), 1024)])


@torch.inference_mode()
def infer_shared(pipelines, sequences):
    """Reuse DNA rows and language-model blocks with identical numerical context."""
    first = next(iter(pipelines.values()))
    if any(p.encoders is not first.encoders for p in pipelines.values()):
        raise ValueError('Joint inference requires shared encoder models')
    n = len(sequences)
    lengths = np.array([len(s) for s in sequences], np.int64)
    # Keep the smaller token budget when tasks differ, to bound peak memory.
    budget = min(p.token_budget for p in pipelines.values())
    dna = encode_dna(first.encoders.dna(), sequences, budget)
    loglen = np.log10(lengths.astype(np.float64)).astype(np.float32)[:, None]
    h1 = np.concatenate([dna, loglen], 1)
    states = {}
    for task, pipeline in pipelines.items():
        probs = np.full((n, 3), np.nan)
        probs[:, 0] = pipeline.heads.head_prob('H1', h1)
        z = np.full((n, 3), np.nan)
        z[:, 0] = centered(probs[:, 0], pipeline.heads.t[0])
        routes = stage_exit(np.full(n, 2, dtype=np.int8), z, 0,
                            *pipeline.heads.gates['H1'])
        states[task] = dict(probs=probs, z=z, routes=routes)

    # Keep NT's original 32-contig batches: changing the batch context can
    # shift floating-point features even when the sequences are unchanged.
    nt_cache = {}
    for task, pipeline in pipelines.items():
        s = states[task]
        need = np.flatnonzero(s['routes'] != 0)
        if not len(need):
            continue
        nt = np.full((n, 1536), np.nan, np.float32)
        for start in range(0, len(need), 32):
            key = tuple(need[start:start + 32])
            if key not in nt_cache:
                nt_cache[key] = first._nt([sequences[i] for i in key])
            nt[list(key)] = nt_cache[key]
        s['nt'] = nt
        h2 = np.concatenate([dna[need], nt[need], loglen[need]], 1)
        s['probs'][need, 1] = pipeline.heads.head_prob('H2', h2)
        s['z'][need, 1] = centered(s['probs'][need, 1], pipeline.heads.t[1])
        s['routes'] = stage_exit(s['routes'], s['z'], 1, *pipeline.heads.gates['H2'])

    # ESM-C uses BF16 and length-sorted ORF batches. Preserve each task's
    # original 1024-contig blocks; a changed context can shift its features.
    esm_cache = {}
    for task, pipeline in pipelines.items():
        s = states[task]
        need = np.flatnonzero(s['routes'] == 2)
        if not len(need):
            continue
        blocks = []
        for start in range(0, len(need), 1024):
            key = tuple(need[start:start + 1024])
            if key not in esm_cache:
                esm_cache[key] = first._esm([sequences[i] for i in key])
            blocks.append(esm_cache[key])
        esm = np.concatenate(blocks, 0)
        h3 = np.concatenate([dna[need], s['nt'][need], esm, loglen[need]], 1)
        s['probs'][need, 2] = pipeline.heads.head_prob('H3', h3)

    results = {}
    for task, pipeline in pipelines.items():
        probs = states[task]['probs']
        scores, routes = route_scores(probs, pipeline.heads.t, pipeline.heads.gates)
        results[task] = dict(scores=scores, routes=routes, probs=probs,
                             labels=scores > 0.5, lengths=lengths)
    return results
