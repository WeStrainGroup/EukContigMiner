"""Relocatable revised H1 inference over unchanged whole sequences; no gate or encoders NT/ESM."""
from pathlib import Path
import hashlib,importlib.util,json,sys,types
import torch
from torch import nn

def _load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def load_h1(bundle,device='cpu'):
 root=Path(bundle).resolve();manifest=json.loads((root/'candidate.json').read_text())
 def checked(item):
  p=(root/item['file']).resolve()
  if not p.is_relative_to(root):raise ValueError('Asset escapes bundle')
  if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Asset hash mismatch')
  return p
 files={n:checked(manifest['runtime_sources'][n])for n in ['dna.py','dna_features.py','sequence.py']}
 name='_ecm_revised_dna_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
 package=types.ModuleType(name);package.__path__=[str(files['dna.py'].parent)];sys.modules[name]=package
 seq=_load(name+'.sequence',files['sequence.py']);backbone=_load(name+'.dna',files['dna.py']);features=_load(name+'.features',files['dna_features.py'])
 asset=torch.load(checked(manifest['assets']['H1']),map_location='cpu',weights_only=True)
 if asset['schema']!='ecm.revised.inference_asset.v1'or asset['stage']!='H1':raise ValueError('Not revised H1 inference asset')
 class NativeH1(nn.Module):
  def __init__(self,cfg):
   super().__init__();self.encoder=backbone.HierarchicalKmerTCN(backbone.HierarchicalKmerTCNConfig(**cfg))
   self.head=nn.Sequential(nn.Linear(943,512),nn.GELU(),nn.Linear(512,128),nn.GELU(),nn.Linear(128,1))
   self.register_buffer('dna_mean',torch.zeros(942));self.register_buffer('dna_std',torch.ones(942))
  def infer(self,tokens,length):
   if self.training:raise ValueError('Inference requires eval mode')
   rc=seq.reverse_complement_batch(tokens,length,pad_token=5)
   dna=.5*(features.whole_contig_base_features(self.encoder,tokens,length)+features.whole_contig_base_features(self.encoder,rc,length))
   assert dna.shape==(len(length),942)
   x=torch.cat(((dna-self.dna_mean)/self.dna_std,torch.log10(length.float())[:,None]),1)
   x=torch.nn.functional.dropout(self.head[1](self.head[0](x)),p=.5,training=self.training)
   x=torch.nn.functional.dropout(self.head[3](self.head[2](x)),p=.5,training=self.training)
   logit=self.head[4](x).squeeze(-1)
   return dict(dna=dna,logit=logit,probability=torch.sigmoid(logit.double()))
  def forward(self,tokens,length):return self.infer(tokens,length)['logit']
 model=NativeH1(asset['config']);model.load_state_dict(asset['state_dict'],strict=True)
 if not all(torch.isfinite(v).all()for v in model.state_dict().values()):raise ValueError('Nonfinite H1 state')
 if not torch.equal(model.dna_mean,torch.zeros_like(model.dna_mean))or not torch.equal(model.dna_std,torch.ones_like(model.dna_std)):raise ValueError('Unexpected H1 normalization')
 model.requires_grad_(False);model.eval();return model.to(device)
