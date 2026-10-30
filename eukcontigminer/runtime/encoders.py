"""Original frozen NT/ESM loaders with unchanged feature and sequence contracts.

Asset staging is independent of the candidate head bundle. Full GPU feature
parity and the installed end-to-end CLI require separate verification.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
from torch import nn
from packaging.version import Version, InvalidVersion

ASSET_SHA='977bb82da74acff0f5d150287e3aa4d596ac5cb0a016c1aef8ac257150e6dc58'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def checked_assets(root):
    root=Path(root).resolve()
    if sha(root/'assets.json')!=ASSET_SHA:raise ValueError('Frozen asset manifest differs')
    m=json.loads((root/'assets.json').read_text())
    for rel,b in m['files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root)or p.stat().st_size!=b['bytes']or sha(p)!=b['sha256']:
            raise ValueError('Frozen asset differs: '+rel)
    for name,version in m['observed_dependency_versions'].items():
        installed=importlib.metadata.version(name)
        try:
            compatible=Version(installed).base_version==Version(version).base_version and not Version(installed).is_prerelease and not Version(installed).is_devrelease
        except InvalidVersion:
            compatible=installed==version
        if not compatible:raise ValueError(f'Dependency version differs: {name} required {version}, installed {installed}')
    return root,m

def runtime_module(root,manifest,name):
    namespace='_ecm_revised_frozen_'+hashlib.sha256(str(root).encode()).hexdigest()[:20]
    if namespace not in sys.modules:
        pkg=types.ModuleType(namespace);pkg.__path__=[str(root/'runtime')];sys.modules[namespace]=pkg
    for stem in ['protein_model','esm_features','esm_inference','ntv3_runtime']:
        key=namespace+'.'+stem;path=root/'runtime'/(stem+'.py')
        if key not in sys.modules:
            spec=importlib.util.spec_from_file_location(key,path);mod=importlib.util.module_from_spec(spec)
            sys.modules[key]=mod;spec.loader.exec_module(mod)
        if sha(Path(sys.modules[key].__file__))!=manifest['files']['runtime/'+stem+'.py']['sha256']:
            raise ValueError('Loaded runtime differs')
    return sys.modules[namespace+'.'+name]

def finite_frozen(model):
    model.eval().requires_grad_(False)
    if any(not torch.isfinite(v).all()for v in model.state_dict().values()):raise ValueError('Nonfinite model')
    if any(v.requires_grad for v in model.parameters()):raise ValueError('Encoder must be frozen')
    return model

class FrozenNT(nn.Module):
    def __init__(self,assets,device='cpu'):
        super().__init__();root,self.manifest=checked_assets(assets)
        self.runtime=runtime_module(root,self.manifest,'ntv3_runtime')
        self.tokenizer,self.encoder=self.runtime.load_source(root/'nt',self.manifest['NT']['source_digests'],sha)
        self.device=torch.device(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        finite_frozen(self.encoder).to(self.device);self.eval()

    def views(self,sequences):
        result=[]
        for raw in sequences:
            if not isinstance(raw,bytes)or not raw:raise ValueError('Nonempty raw ASCII bytes required')
            n=len(raw);width=n if n<=5000 else 5000-n%2;a=(n-width)//2
            s=self.runtime.normalize(raw[a:a+width].decode())
            result.extend(sorted((s,s.translate(self.runtime.RC)[::-1])))
        return result

    @torch.inference_mode()
    def infer_batch(self,sequences):
        # Caller preserves original contiguous batches of 8, including final tail.
        if not 1<=len(sequences)<=8:raise ValueError('NT original batch is at most 8')
        if self.training or self.encoder.training:raise ValueError('Frozen inference requires eval mode')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:raise ValueError('NT requires no TF32')
        ids,mask=self.tokens(self.views(sequences),self.device)
        result=self.features(ids,mask)
        if result.shape!=(len(sequences),1536)or not torch.isfinite(result).all():raise ValueError('Invalid NT features')
        return result.float()

    def tokens(self, views, device):
        width = (max(map(len, views))+127)//128*128
        ids = torch.full((len(views), width), self.tokenizer.convert_tokens_to_ids('N'), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for j, s in enumerate(views):
            t = self.tokenizer(s, add_special_tokens=False)['input_ids']
            assert len(t) == len(s)
            ids[j, :len(t)] = torch.tensor(t); mask[j, :len(t)] = True
        return ids.to(device), mask.to(device)

    def features(self, ids, mask):
        # Group by exact padded width, matching the existing native NT adapter.
        widths = ((mask.sum(1)+127)//128)*128; indices = []; vectors = []
        for size in sorted(set(widths.detach().cpu().tolist())):
            take = torch.nonzero(widths == size).flatten(); indices.append(take)
            h = self.encoder.core(input_ids=ids[take, :size])[
                'embeddings_deconv_'+str(self.encoder.config.num_downsamples)].permute(0, 2, 1).float()
            valid = mask[take, :size].unsqueeze(-1)
            vectors.append(torch.cat(((h*valid).sum(1)/valid.sum(1),
                                      h.masked_fill(~valid, -torch.inf).amax(1)), 1))
        f = torch.cat(vectors)[torch.argsort(torch.cat(indices))]
        return f.reshape(-1, 2, 1536).mean(1)

class FrozenESM(nn.Module):
    def __init__(self,assets,device='cpu'):
        super().__init__();root,self.manifest=checked_assets(assets)
        self.runtime=runtime_module(root,self.manifest,'esm_inference')
        from esm.models.esmc import ESMC
        from esm.pretrained import get_esmc_model_tokenizers
        self.model=ESMC(d_model=960,n_heads=15,n_layers=30,tokenizer=get_esmc_model_tokenizers(),use_flash_attn=False)
        self.model.load_state_dict(torch.load(root/self.manifest['ESM']['weight'],map_location='cpu',weights_only=True),strict=True)
        self.device=torch.device(device);finite_frozen(self.model).to(self.device)
        if self.runtime.optimize_esmc_feature_inference(self.model)!=1:raise ValueError('ESM optimization differs')
        self.config=self.runtime.ESM2ORFInferenceConfig();self.eval()

    def select(self,sequences):
        if any(not isinstance(x,bytes)or not x for x in sequences):raise ValueError('Nonempty raw ASCII bytes required')
        return [self.runtime.select_orfs_from_contig(x,self.config)for x in sequences]

    @torch.inference_mode()
    def infer_block(self,sequences):
        # Caller preserves original 1024-row blocks, including final tail.
        if not 1<=len(sequences)<=1024:raise ValueError('ESM original block is at most 1024')
        if self.training or self.model.training:raise ValueError('Frozen inference requires eval mode')
        result=self.runtime.esmc_features_from_orfs(self.model,self.model._tokenize,self.select(sequences),device=self.device,config=self.config,quantize_float16=True)
        if result.shape!=(len(sequences),1920)or not torch.isfinite(result).all():raise ValueError('Invalid ESM features')
        return result.to(torch.float16)
