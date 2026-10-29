"""ECM v0.71 feature-level runtime: new H1/H2/H3 heads + two-sided early exit."""
import json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from v071_heads import H1Flex, HeadFlex

class V071Runtime:
    def __init__(self,model_dir,device='cpu'):
        self.dir=Path(model_dir);self.device=device
        self.heads={};self.cal=json.load(open(self.dir/'calibration.json'))
        for k,dim in (('H1',943),('H2',2479),('H3',4399)):
            ck=torch.load(self.dir/f'{k}_selected.pt',map_location='cpu',weights_only=False);cfg=dict(ck['cfg'])
            m=(H1Flex(cfg) if k=='H1' else HeadFlex(dim,cfg)).to(device);m.load_state_dict(ck['state_dict']);m.eval().requires_grad_(False);self.heads[k]=m
        self.t=np.array([self.cal['heads'][k]['threshold'] for k in ('H1','H2','H3')])
    @torch.no_grad()
    def probs(self,mat,dim,batch=8192):
        out=np.empty(len(mat),np.float64)
        for lo in range(0,len(mat),batch):
            x=torch.from_numpy(np.asarray(mat[lo:lo+batch,:dim])).to(self.device,non_blocking=True).float();out[lo:lo+batch]=torch.sigmoid(self.heads['H1' if dim==943 else 'H2' if dim==2479 else 'H3'](x).double()).cpu().numpy()
        return out
    def route(self,p,lengths):
        z=np.log(np.clip(p,1e-15,1-1e-15)/(1-np.clip(p,1e-15,1-1e-15)))-np.log(self.t/(1-self.t))
        g=np.digitize(np.asarray(lengths),[2000,10000]);routes=np.full(len(z),2,np.int8)
        for st,name in enumerate(('H1','H2')):
            for k in (0,1,2):
                lo,hi=self.cal['gates'][name][str(k)];take=(routes==2)&(g==k)&((z[:,st]<lo)|(z[:,st]>hi));routes[take]=st
        sel=z[np.arange(len(z)),routes];score=np.where(sel>0,np.nextafter(1/(1+np.exp(-sel)),1.0),1/(1+np.exp(-sel)))
        return score,routes
    def predict_features(self,h1,h2,h3,lengths):
        p=np.column_stack([self.probs(h1,943),self.probs(h2,2479),self.probs(h3,4399)])
        score,routes=self.route(p,lengths);return {'scores':score,'routes':routes,'probs':p,'labels':score>0.5}
