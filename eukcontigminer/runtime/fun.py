"""Independent fun pipeline: DNA942 + NT1536 + ESM1920 + loglength -> one head."""
from pathlib import Path
import numpy as np
import torch
from .encoders import EncoderBundle, encode_dna, encode_nt
from .fun_head import CompactFungiHead


class FunPipeline:
    def __init__(self, model_root, head_sha256, device='cuda:0', encoders=None):
        root = Path(model_root)
        self.device = torch.device(device)
        self.encoders = encoders or EncoderBundle(root, self.device)
        self.head_path = root / 'fun/head.pt'
        self.head_sha256 = head_sha256
        self._head = None

    def _head_model(self):
        if self._head is None:
            self._head = CompactFungiHead.from_checkpoint(self.head_path, self.head_sha256, self.device)
        return self._head

    @torch.inference_mode()
    def infer_chunk(self, sequences):
        lengths = np.array([len(s) for s in sequences], dtype=np.int64)
        dna = encode_dna(self.encoders.dna(), sequences, 128000)
        nt = encode_nt(self.encoders.nt(), sequences, 8)
        esm = np.concatenate([self.encoders.esm().infer_block(sequences[i:i + 1024]).cpu().numpy()
                              for i in range(0, len(sequences), 1024)])
        scores = self._head_model().predict(dna, nt, esm, lengths, batch_size=2048)
        return {'task': 'p_fungi', 'scores': scores, 'lengths': lengths, 'dna': dna, 'nt': nt, 'esm': esm}
