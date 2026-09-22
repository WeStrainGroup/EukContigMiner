"""ESM-free score upper bound, conditional on observed DNA and NT logits.

Unknown ESM/probe splits retain both branches; known DNA/length splits select
the exact branch. Summed independent tree extrema form conservative bounds.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compile_bounds(model_path, output):
    import xgboost as xgb
    tree=xgb.Booster(model_file=str(model_path));tree.set_param({'device':'cpu','nthread':4})
    nodes=[json.loads(s) for s in tree.get_dump(dump_format='json')]
    thresholds={1920:set(),1921:set()}
    def prepare(n):
        if 'leaf' in n:return ('leaf',float(n['leaf']),float(n['leaf']),False)
        c={s['nodeid']:prepare(s) for s in n['children']};a,b=c[n['yes']],c[n['no']]
        f=int(n['split'][1:]);v=np.float32(n['split_condition'])
        if f in thresholds:thresholds[f].add(float(v))
        return (f,min(a[1],b[1]),max(a[2],b[2]),f in thresholds or a[3] or b[3],v,a,b)
    prepared=[prepare(n) for n in nodes]
    dx=np.array(sorted(thresholds[1920]),np.float32);lx=np.array(sorted(thresholds[1921]),np.float32)
    dr=np.r_[np.float32(-np.inf),dx];lr=np.r_[np.float32(-np.inf),lx]
    dna,length=np.meshgrid(dr,lr,indexing='ij')
    def interval(n):
        if not n[3]:return n[1],n[2]
        a,b=interval(n[5]),interval(n[6])
        if n[0] in thresholds:
            take=(dna if n[0]==1920 else length)<n[4]
            return np.where(take,a[0],b[0]),np.where(take,a[1],b[1])
        return np.minimum(a[0],b[0]),np.maximum(a[1],b[1])
    cfg=json.loads(tree.save_config())['learner']['learner_model_param']
    bias=float(logit(float(str(cfg['base_score']).strip('[]'))))
    lower=np.full(dna.shape,bias);upper=lower.copy()
    for n in prepared:
        lo,hi=interval(n);lower+=lo;upper+=hi
    # Conservative IEEE float32 summation bound plus JSON decimal/bias allowance.
    eps=np.finfo(np.float32).eps;n=len(prepared)+4;gamma=n*eps/(1-n*eps)
    allowance=gamma*(abs(bias)+sum(max(abs(x[1]),abs(x[2])) for x in prepared))+1e-4
    lower-=allowance;upper+=allowance
    out=Path(output);assert not out.exists()
    np.savez(out,dna_thresholds=dx,length_thresholds=lx,lower=lower,upper=upper,
             tree_sha256=np.array(sha(model_path)),rounding_allowance=np.array(allowance))
    return {'tree_sha256':sha(model_path),'table_sha256':sha(out),'shape':list(lower.shape),
            'cells':lower.size,'rounding_allowance':allowance,'trees':len(prepared)}


class BoundRouter:
    def __init__(self,table,model_path,config):
        a=np.load(table,allow_pickle=False)
        assert str(a['tree_sha256'])==sha(model_path)
        self.dx=a['dna_thresholds'];self.lx=a['length_thresholds']
        self.lower=a['lower'];self.upper=a['upper'];self.config=config
        assert np.isfinite(self.lower).all() and np.isfinite(self.upper).all()
        assert (self.lower<=self.upper).all()

    def upper_probability(self,dna,nt,lengths):
        x=np.asarray(dna,np.float32);l=np.log10(lengths).astype(np.float32)
        i=np.searchsorted(self.dx,x,side='right');j=np.searchsorted(self.lx,l,side='right')
        margin=self.upper[i,j]
        # Match the published clipping and arithmetic, preserving monotonicity.
        q=expit(logit(np.clip(expit(margin),1e-15,1-1e-15))+.75*nt)
        c=self.config['model']['slim_length_calibration'];s=np.log2(np.clip(lengths,1000,10000)/2000.)
        q=expit(logit(np.clip(q,1e-15,1-1e-15))-c['short']*np.minimum(s,0)-c['long']*np.maximum(s,0))
        return np.minimum(1.,np.nextafter(q,np.inf)+1e-12)

    def lower_probability(self,dna,nt,lengths):
        x=np.asarray(dna,np.float32);l=np.log10(lengths).astype(np.float32)
        i=np.searchsorted(self.dx,x,side='right');j=np.searchsorted(self.lx,l,side='right')
        margin=self.lower[i,j]
        # Match the published clipping and arithmetic, preserving monotonicity.
        q=expit(logit(np.clip(expit(margin),1e-15,1-1e-15))+.75*nt)
        c=self.config['model']['slim_length_calibration'];s=np.log2(np.clip(lengths,1000,10000)/2000.)
        q=expit(logit(np.clip(q,1e-15,1-1e-15))-c['short']*np.minimum(s,0)-c['long']*np.maximum(s,0))
        return np.maximum(0.,np.nextafter(q,-np.inf)-1e-12)
