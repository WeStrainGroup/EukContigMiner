"""Standalone frozen-feature fungi head. Requires only NumPy and PyTorch."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn


class CompactFungiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(4399,dtype=torch.float32))
        self.register_buffer('std',torch.ones(4399,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(4399,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,x):
        return self.net((x-self.mean)/self.std).squeeze(-1)

    @classmethod
    def from_checkpoint(cls,path,sha256,device='cpu'):
        path=Path(path)
        digest=hashlib.sha256()
        with path.open('rb')as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        got=digest.hexdigest()
        if got!=sha256:raise ValueError('checkpoint SHA256 mismatch')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(checkpoint,dict)or set(checkpoint)!={'state_dict','epoch','plan'}:
            raise ValueError('expected compact epoch checkpoint')
        state=checkpoint['state_dict'];model=cls();expected=model.state_dict()
        if not isinstance(state,dict)or state.keys()!=expected.keys():raise ValueError('checkpoint state keys differ')
        for key,value in state.items():
            if not torch.is_tensor(value)or value.dtype!=torch.float32 or value.shape!=expected[key].shape or not torch.isfinite(value).all():
                raise ValueError('invalid checkpoint tensor: '+key)
        if not(state['std']>0).all():raise ValueError('normalization std must be positive')
        model.load_state_dict(state,strict=True);model.requires_grad_(False);model.eval();return model.to(device)

    @staticmethod
    def features(dna,nt,esm,length):
        dna=np.asarray(dna);nt=np.asarray(nt);esm=np.asarray(esm);length=np.asarray(length)
        if length.ndim!=1 or not np.issubdtype(length.dtype,np.integer)or not(length>0).all():
            raise ValueError('length must be a positive integer vector')
        n=len(length)
        if dna.shape!=(n,942)or nt.shape!=(n,1536)or esm.shape!=(n,1920):raise ValueError('feature shapes differ from DNA942/NT1536/ESM1920')
        raw=np.concatenate([np.asarray(dna,np.float32),np.asarray(nt,np.float32),np.asarray(esm,np.float32),np.log10(length.astype(np.float32))[:,None]],axis=1)
        if not np.isfinite(raw).all():raise ValueError('nonfinite input features')
        return np.ascontiguousarray(raw)

    def predict(self,dna,nt,esm,length,batch_size=2048):
        if not isinstance(batch_size,int)or batch_size<=0:raise ValueError('batch_size must be positive')
        # Fixed stored Train normalization and FP64 sigmoid match training evaluation.
        dna=np.asarray(dna);nt=np.asarray(nt);esm=np.asarray(esm);length=np.asarray(length)
        if length.ndim!=1 or not np.issubdtype(length.dtype,np.integer)or not(length>0).all():raise ValueError('invalid length vector')
        n=len(length)
        if dna.shape!=(n,942)or nt.shape!=(n,1536)or esm.shape!=(n,1920):raise ValueError('feature shapes differ')
        self.eval();out=np.empty(n,np.float64);device=self.mean.device
        with torch.inference_mode():
            for lo in range(0,n,batch_size):
                hi=min(lo+batch_size,n);raw=self.features(dna[lo:hi],nt[lo:hi],esm[lo:hi],length[lo:hi])
                out[lo:hi]=torch.sigmoid(self(torch.from_numpy(raw).to(device)).double()).cpu().numpy()
        if not np.isfinite(out).all():raise ValueError('nonfinite output probabilities')
        return out
