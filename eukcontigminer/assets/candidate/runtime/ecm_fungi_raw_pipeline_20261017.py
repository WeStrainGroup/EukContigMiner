"""Standalone broad-source fungi raw adapter; candidate preparation, not CLI acceptance.

Keep original whole-contig DNA, paired central NT and ESM ORF selection/batches.
A separately bound compact checkpoint supplies one p_fungi; no euk score input.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from ecm_revised_native_h1_20261013 import load_h1
from ecm_revised_native_encoders_20261013 import FrozenNT, FrozenESM
from ecm_fungi_compact_runtime_20261017 import CompactFungiHead


class FungiRawPipeline:
    def __init__(self, dna_bundle, dna_manifest_sha256, encoder_assets,
                 head_checkpoint, head_sha256, device='cpu'):
        root = Path(dna_bundle).resolve()
        if hashlib.sha256((root/'candidate.json').read_bytes()).hexdigest() != dna_manifest_sha256:
            raise ValueError('DNA bundle manifest differs')
        self.device = torch.device(device)
        # H1's unchanged infer() also computes its probability; it is never a head input.
        self.h1 = load_h1(root, self.device)
        self.nt = FrozenNT(encoder_assets, self.device)
        self.esm = FrozenESM(encoder_assets, self.device)
        self.head = CompactFungiHead.from_checkpoint(head_checkpoint, head_sha256, self.device)

    @torch.inference_mode()
    def infer_chunk(self, sequences):
        if not 1<=len(sequences)<=2048 or any(not isinstance(x,bytes)or not x for x in sequences):
            raise ValueError('Require 1..2048 original nonempty byte sequences')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('FP32 inference requires TF32 disabled')
        lengths=np.array([len(s)for s in sequences],dtype=np.int64)
        dna=np.empty((len(sequences),942),np.float32);p1=np.empty(len(sequences),np.float64)
        lut=np.full(256,4,np.uint8)
        for j,c in enumerate(b'ACGT'):lut[c]=j
        lo=0
        while lo<len(sequences):
            hi=lo+1;longest=int(lengths[lo])
            while hi<len(sequences)and (hi-lo+1)*max(longest,int(lengths[hi]))<=128000:
                longest=max(longest,int(lengths[hi]));hi+=1
            token=np.full((hi-lo,longest),5,np.uint8)
            for j,s in enumerate(sequences[lo:hi]):token[j,:len(s)]=lut[np.frombuffer(s,dtype=np.uint8)]
            r=self.h1.infer(torch.from_numpy(token).to(self.device,dtype=torch.long),torch.from_numpy(lengths[lo:hi]).to(self.device))
            dna[lo:hi]=r['dna'].cpu().numpy();p1[lo:hi]=r['probability'].cpu().numpy();lo=hi
        nt=np.concatenate([self.nt.infer_batch(sequences[i:i+8]).cpu().numpy()for i in range(0,len(sequences),8)])
        esm=np.concatenate([self.esm.infer_block(sequences[i:i+1024]).cpu().numpy()for i in range(0,len(sequences),1024)])
        scores = self.head.predict(dna, nt, esm, lengths, batch_size=2048)
        return dict(task='p_fungi', scores=scores, lengths=lengths,
                    dna=dna, nt=nt, esm=esm)
