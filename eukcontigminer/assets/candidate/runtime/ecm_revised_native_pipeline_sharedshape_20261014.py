"""Unreleased projection-refined response-H2 and shared-shape-H3 raw pipeline with original H1 and frozen encoders.
Original raw sequence selection, NT8/ESM1024 and contiguous2048 chunking retained.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from ecm_revised_native_h1_20261013 import load_h1
from ecm_revised_native_response_H2_20261014 import NativeResponseH2,features as gated_features
from ecm_revised_native_shared_shape_H3_20261014 import NativeSharedShapeH3,features as pca_features
from ecm_revised_native_encoders_20261013 import FrozenNT,FrozenESM

class RawPipeline:
    def __init__(self,candidate_assets,encoder_assets,candidate_sha256,device='cpu'):
        root=Path(candidate_assets).resolve();path=root/'candidate.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=candidate_sha256:
            raise ValueError('Candidate manifest differs')
        manifest=json.loads(path.read_text())
        if manifest.get('required_pipeline')!=Path(__file__).name or manifest.get('blends') or manifest.get('dependencies')!={} or manifest.get('H3_architecture')!='fixed_train_pca_shared_activation_shape' or manifest.get('H3_shape_mode') not in ('silu','mish') or manifest.get('H3_activation_scope')!='shared' or manifest.get('H2_architecture')!='fixed_train_pca_gated_energy_response' or manifest.get('H2_response_scope')!='shared':
            raise ValueError('Require explicit PCA gated H2 and shared activation H3 manifest')
        self.device=torch.device(device);self.h1=load_h1(root,self.device);self.heads={}
        for stage in ['H2','H3']:
            item=manifest['assets'][stage];p=(root/item['file']).resolve()
            if not p.is_relative_to(root)or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Head asset differs')
            a=torch.load(p,map_location='cpu',weights_only=True)
            if a['schema']!='ecm.revised.inference_asset.v1'or a['stage']!=stage:
                raise ValueError('Head schema differs')
            if stage=='H2':
                if a.get('architecture')!='fixed_train_pca_gated_energy_response' or a.get('response_scope')!='shared':raise ValueError('H2 architecture differs')
                head=NativeResponseH2(a['response_scope'],a['state_dict'])
            else:
                if a.get('architecture')!='fixed_train_pca_shared_activation_shape' or a.get('activation_scope')!='shared' or a.get('shape_mode')!=manifest['H3_shape_mode']:raise ValueError('H3 architecture differs')
                head=NativeSharedShapeH3(a['shape_mode'],a['state_dict'])
            self.heads[stage]=head.to(self.device).eval().requires_grad_(False)
        self.nt=FrozenNT(encoder_assets,self.device);self.esm=FrozenESM(encoder_assets,self.device)
        self.manifest=manifest

    @torch.inference_mode()
    def infer_chunk(self,sequences):
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
        x2=gated_features(dna,nt,lengths,p1);x3=pca_features(dna,nt,lengths,p1,esm)
        p2=torch.sigmoid(self.heads['H2'](torch.from_numpy(x2).to(self.device)).double()).cpu().numpy()
        p3=torch.sigmoid(self.heads['H3'](torch.from_numpy(x3).to(self.device)).double()).cpu().numpy()
        probabilities={'H1':p1,'H2':p2,'H3':p3}
        if any(not np.isfinite(p).all()for p in probabilities.values()):raise ValueError('Nonfinite head probabilities')
        return dict(probabilities=probabilities,dna=dna,nt=nt,esm=esm,lengths=lengths)
