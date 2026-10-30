"""Independent fun pipeline: DNA942 + NT1536 + ESM1920 + loglength -> one head."""
from pathlib import Path
import numpy as np
import torch
from .dna_encoder import load_dna_encoder
from .encoders import FrozenNT, FrozenESM
from .fun_head import CompactFungiHead


class FunPipeline:
    def __init__(self, model_root, head_sha256, device='cuda:0'):
        root = Path(model_root)
        self.device = torch.device(device)
        self.h1 = load_dna_encoder(root / 'euk/h1_dna.pt', self.device)
        self.nt = FrozenNT(root / 'encoders', self.device)
        self.esm = FrozenESM(root / 'encoders', self.device)
        self.head = CompactFungiHead.from_checkpoint(root / 'fun/head.pt', head_sha256, self.device)

    @torch.inference_mode()
    def infer_chunk(self, sequences):
        lengths = np.array([len(s) for s in sequences], dtype=np.int64)
        dna = np.empty((len(sequences), 942), np.float32)
        lut = np.full(256, 4, np.uint8)
        for j, c in enumerate(b'ACGT'):
            lut[c] = j
        lo = 0
        while lo < len(sequences):
            hi = lo + 1
            longest = int(lengths[lo])
            while hi < len(sequences) and (hi - lo + 1) * max(longest, int(lengths[hi])) <= 128000:
                longest = max(longest, int(lengths[hi])); hi += 1
            token = np.full((hi - lo, longest), 5, np.uint8)
            for j, s in enumerate(sequences[lo:hi]):
                token[j, :len(s)] = lut[np.frombuffer(s, dtype=np.uint8)]
            r = self.h1.infer(torch.from_numpy(token).to(self.device, dtype=torch.long),
                              torch.from_numpy(lengths[lo:hi]).to(self.device))
            dna[lo:hi] = r['dna'].cpu().numpy(); lo = hi
        nt = np.concatenate([self.nt.infer_batch(sequences[i:i + 8]).cpu().numpy()
                             for i in range(0, len(sequences), 8)])
        esm = np.concatenate([self.esm.infer_block(sequences[i:i + 1024]).cpu().numpy()
                              for i in range(0, len(sequences), 1024)])
        scores = self.head.predict(dna, nt, esm, lengths, batch_size=2048)
        return {'task': 'p_fungi', 'scores': scores, 'lengths': lengths, 'dna': dna, 'nt': nt, 'esm': esm}
