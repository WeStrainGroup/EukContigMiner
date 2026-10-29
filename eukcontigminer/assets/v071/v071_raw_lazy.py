"""Lazy raw FASTA runtime for v0.71: H1 -> H2 -> H3 early exit."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'candidate/runtime'))
from v071_runtime import V071Runtime
from ecm_revised_native_pipeline_sharedshape_20261014 import RawPipeline

class V071LazyRuntime:
    def __init__(self,model_root,device='cuda:0'):
        root=Path(model_root);self.device=torch.device(device);self.cal=json.loads((root/'v071/calibration.json').read_text());self.heads=V071Runtime(root/'v071',device=device)
        cand=root/'candidate';enc=root/'encoders';cs=hashlib.sha256((cand/'candidate.json').read_bytes()).hexdigest();self.base=RawPipeline(cand,enc,cs,device=device)
        self.t=self.heads.t
    @torch.inference_mode()
    def infer(self,sequences):
        n=len(sequences);lengths=np.array([len(s) for s in sequences],np.int64);dna=np.empty((n,942),np.float32);p1=np.empty(n,np.float64)
        # H1 DNA features only
        lut=np.full(256,4,np.uint8)
        for j,c in enumerate(b'ACGT'):lut[c]=j
        lo=0
        while lo<n:
            hi=lo+1;longest=int(lengths[lo])
            while hi<n and (hi-lo+1)*max(longest,int(lengths[hi]))<=512000:
                longest=max(longest,int(lengths[hi]));hi+=1
            token=np.full((hi-lo,longest),5,np.uint8)
            for j,s in enumerate(sequences[lo:hi]):token[j,:len(s)]=lut[np.frombuffer(s,dtype=np.uint8)]
            r=self.base.h1.infer(torch.from_numpy(token).to(self.device,dtype=torch.long),torch.from_numpy(lengths[lo:hi]).to(self.device))
            dna[lo:hi]=r['dna'].cpu().numpy();lo=hi
        loglen=np.log10(lengths.astype(np.float64)).astype(np.float32)[:,None]
        h1=np.concatenate([dna,loglen],1);p1=self._head_prob('H1',h1)
        routes=np.full(n,2,np.int8);g=np.digitize(lengths,[2000,10000]);z=np.full((n,3),np.nan)
        z[:,0]=self._z(p1,self.t[0]);routes=self._gate(routes,g,z,0);p2=np.full(n,np.nan);p3=np.full(n,np.nan);nt_all=np.full((n,1536),np.nan,np.float32)
        # H2 only where not H1
        need=np.where(routes!=0)[0]
        if len(need):
            nt=self._nt([sequences[i] for i in need]);nt_all[need]=nt;h2=np.concatenate([dna[need],nt,loglen[need]],1);p2[need]=self._head_prob('H2',h2);z[need,1]=self._z(p2[need],self.t[1]);routes=self._gate(routes,g,z,1)
        # H3 only where still fallback
        need=np.where(routes==2)[0]
        if len(need):
            esm=self._esm([sequences[i] for i in need]);h3=np.concatenate([dna[need],nt_all[need],esm,loglen[need]],1);p3[need]=self._head_prob('H3',h3);z[need,2]=self._z(p3[need],self.t[2])
        sel=z[np.arange(n),routes];scores=1/(1+np.exp(-sel));return {'scores':scores,'routes':routes,'probs':np.column_stack([p1,p2,p3]),'labels':scores>.5,'lengths':lengths}
    infer_chunk=infer
    def _head_prob(self,k,x):
        m=self.heads.heads[k];out=np.empty(len(x),np.float64)
        for lo in range(0,len(x),8192):
            t=torch.from_numpy(np.asarray(x[lo:lo+8192],dtype=np.float32)).to(self.device);out[lo:lo+8192]=torch.sigmoid(m(t).double()).cpu().numpy()
        return out
    def _z(self,p,t):return np.log(np.clip(p,1e-15,1-1e-15)/(1-np.clip(p,1e-15,1-1e-15)))-np.log(t/(1-t))
    def _gate(self,routes,g,z,st):
        name='H1' if st==0 else 'H2'
        for k in (0,1,2):
            lo,hi=self.cal['gates'][name][str(k)];take=(routes==2)&(g==k)&((z[:,st]<lo)|(z[:,st]>hi));routes[take]=st
        return routes
    def _nt(self,seqs):
        # Bounded custom NT batches; FrozenNT.features groups exact padded widths.
        out=[]
        for lo in range(0,len(seqs),32):
            views=self.base.nt.views(seqs[lo:lo+32]);ids,mask=self.base.nt.tokens(views,self.device)
            out.append(self.base.nt.features(ids,mask).float().cpu().numpy())
        return np.concatenate(out,0)
    def _esm(self,seqs):return np.concatenate([self.base.esm.infer_block(seqs[i:i+1024]).cpu().numpy() for i in range(0,len(seqs),1024)])
