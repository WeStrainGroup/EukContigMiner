"""Pinned offline NTv3 100M inference for the ECM release package."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
import threading
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

WINDOW_RULE='whole_le2000_else_first_last_center_rc_symmetric_v1'
REVISION='5c685dca15891f5c5b80e0c930e23b87a217e441'
RC=str.maketrans('ACGTN','TGCAN')
SOURCE_NAMES={'MODEL-LICENSE.md','special_tokens_map.json','tokenizer_config.json','vocab.json','config.json','model.safetensors','configuration_ntv3_pretrained.py','modeling_ntv3_pretrained.py','tokenization_ntv3.py'}
COUNTS={'lora':494849,'full':106654897,'decoder2':7581953}
def validate_binding(b):
    required={'schema':'ecm.ntv3.runtime.ieee.v1','backbone_model_id':'InstaDeepAI/NTv3_100M_pre','backbone_revision':REVISION,'precision':'ieee_float32','window_rule':WINDOW_RULE,'forward_reverse_complement':'mean_logits','window_aggregation':'mean_logits'}
    if any(b.get(k)!=v for k,v in required.items()) or b.get('adaptation') not in COUNTS:raise ValueError('NTv3 runtime binding differs')
    if b.get('trainable_parameters')!=COUNTS[b['adaptation']] or set(b.get('source_files_sha256',{}))!=SOURCE_NAMES:raise ValueError('NTv3 state/source contract differs')
    if b.get('rank')!=(0 if b['adaptation']=='full' else 16) or b.get('lora_scale')!=(0.0 if b['adaptation']=='full' else 1.0):raise ValueError('NTv3 LoRA contract differs')
    if not isinstance(b.get('checkpoint_plan_sha256'),str) or len(b['checkpoint_plan_sha256'])!=64:raise ValueError('Missing checkpoint plan binding')
    validate_checkpoint_progress(b)
def validate_checkpoint_progress(binding):
    fields = [key for key in ('checkpoint_epoch', 'checkpoint_step') if key in binding]
    if len(fields) != 1:
        raise ValueError('NTv3 checkpoint requires exactly one epoch or step binding')
    key = fields[0]
    if type(binding[key]) is not int or binding[key] <= 0:
        raise ValueError('NTv3 checkpoint progress must be a positive integer')
    return key.removeprefix('checkpoint_')

def validate_checkpoint_provenance(cp, binding):
    progress = validate_checkpoint_progress(binding)
    if (cp.get('plan_sha256') != binding['checkpoint_plan_sha256']
            or type(cp.get(progress)) is not int
            or cp[progress] != binding['checkpoint_' + progress]):
        raise ValueError('NTv3 checkpoint provenance differs')

def normalize(s):
    if not isinstance(s,str) or not s:raise ValueError('NTv3 sequence must be nonempty text')
    return ''.join(c if c in 'ACGT' else 'N' for c in s.upper())
def window_spans(n):
    if n<1:raise ValueError('NTv3 sequence must be nonempty')
    if n<=2000:return [(0,n)]
    width=2000-n%2;start=(n-width)//2
    return sorted({(0,2000),(start,start+width),(n-2000,n)})
class LoRA(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.a=nn.Linear(base.in_features,16,bias=False);self.b=nn.Linear(16,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5));nn.init.zeros_(self.b.weight)
    def forward(self,x):return self.base(x)+self.b(self.a(x))
class Classifier(nn.Module):
    def __init__(self,encoder,mode):
        super().__init__();self.encoder=encoder;encoder.requires_grad_(True);encoder.core.lm_head.requires_grad_(False)
        width=2*encoder.config.conv_init_embed_dim
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
        if mode!='full':
            encoder.requires_grad_(False)
            for block in encoder.core.transformer_blocks:
                for projection in [block.sa_layer.query_head,block.sa_layer.value_head]:projection.linear=LoRA(projection.linear)
            if mode=='decoder2':
                if len(encoder.core.deconv_tower_blocks)!=7:raise ValueError('Unexpected decoder blocks')
                for i in [5,6]:encoder.core.deconv_tower_blocks[i].requires_grad_(True)
    def forward(self,ids,mask):
        padded=((mask.sum(1)+127)//128)*128;groups=[];indices=[]
        for size in sorted(set(padded.detach().cpu().tolist())):
            take=torch.nonzero(padded==size).flatten();indices.append(take)
            outputs=self.encoder.core(input_ids=ids[take,:size])
            h=outputs['embeddings_deconv_'+str(self.encoder.config.num_downsamples)].permute(0,2,1).float()
            valid=mask[take,:size].bool().unsqueeze(-1)
            groups.append(torch.cat([(h*valid).sum(1)/valid.sum(1),h.masked_fill(~valid,-torch.inf).amax(1)],1))
        return self.head(torch.cat(groups)[torch.argsort(torch.cat(indices))]).squeeze(-1).float()
def load_source(source,digests,sha):
    for name,digest in digests.items():
        if Path(name).name!=name or sha(source/name)!=digest:raise ValueError('NTv3 source artifact differs: '+name)
    # A private module namespace prevents collisions with other NTv3 snapshots.
    package='_ecm_ntv3_'+digests['modeling_ntv3_pretrained.py'][:20]
    if package not in sys.modules:
        mod=types.ModuleType(package);mod.__path__=[str(source)];sys.modules[package]=mod
    loaded={}
    for name in ['configuration_ntv3_pretrained','modeling_ntv3_pretrained','tokenization_ntv3']:
        key=package+'.'+name
        if key in sys.modules:
            mod=sys.modules[key]
            if sha(Path(mod.__file__))!=digests[name+'.py']:raise ValueError('Loaded NTv3 module differs')
        else:
            spec=importlib.util.spec_from_file_location(key,source/(name+'.py'));mod=importlib.util.module_from_spec(spec);sys.modules[key]=mod;spec.loader.exec_module(mod)
        loaded[name]=mod
    config=loaded['configuration_ntv3_pretrained'].Ntv3PreTrainedConfig(**json.loads((source/'config.json').read_text()))
    encoder=loaded['modeling_ntv3_pretrained'].NTv3PreTrained(config)
    from safetensors.torch import load_file
    encoder.load_state_dict(load_file(source/'model.safetensors'),strict=True)
    if sum(p.numel() for p in encoder.parameters())!=106463419:raise ValueError('NTv3 backbone shape differs')
    config.deconv_layers_to_save=(config.num_downsamples,)
    tokenizer=loaded['tokenization_ntv3'].NTv3Tokenizer(vocab_file=str(source/'vocab.json'))
    return tokenizer,encoder
_precision_lock=threading.RLock()
@contextmanager
def ieee_fp32(device):
    # Preserve other ECM branches' global backend settings. Single inference-thread contract.
    with _precision_lock:
        previous=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            with torch.autocast(device.type,enabled=False):yield
        finally:
            torch.backends.cuda.matmul.allow_tf32=previous[0];torch.backends.cudnn.allow_tf32=previous[1]
class NTV3Adapter:
    def __init__(self,binding,device,bound_path,sha256_file):
        validate_binding(binding);directory=os.environ.get('EUKCONTIGMINER_NTV3_DIR')
        if not directory:raise ValueError('Set EUKCONTIGMINER_NTV3_DIR to the pinned offline NTv3 directory')
        self.tokenizer,encoder=load_source(Path(directory),binding['source_files_sha256'],sha256_file)
        self.model=Classifier(encoder,binding['adaptation'])
        learned={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        cp=torch.load(bound_path(binding['checkpoint'],'NTv3 checkpoint'),weights_only=True,map_location='cpu')
        validate_checkpoint_provenance(cp, binding)
        state=cp['state_dict']
        if set(state)!=set(learned) or sum(p.numel() for p in learned.values())!=binding['trainable_parameters']:raise ValueError('NTv3 learned tensor set differs')
        with torch.no_grad():
            for name,value in state.items():
                if value.shape!=learned[name].shape or not torch.isfinite(value).all():raise ValueError('Invalid NTv3 learned tensor '+name)
                learned[name].copy_(value)
        self.model.to(device).eval();self.device=device;self.bf16=False;self.windows=0
    def encode(self,seqs):
        width=((max(map(len,seqs))+127)//128)*128
        ids=torch.full((len(seqs),width),self.tokenizer.convert_tokens_to_ids('N'),dtype=torch.long);mask=torch.zeros_like(ids)
        for i,s in enumerate(seqs):
            tokens=self.tokenizer(s,add_special_tokens=False)['input_ids']
            if len(tokens)!=len(s):raise ValueError('NTv3 tokenizer is not nucleotide-aligned')
            ids[i,:len(s)]=torch.tensor(tokens);mask[i,:len(s)]=1
        if (ids==self.tokenizer.pad_token_id).any():raise ValueError('Unexpected NTv3 padding token')
        return ids.to(self.device),mask.to(self.device)
    def predict_windows(self,sequences):
        sequences=[normalize(s) for s in sequences];buckets={};result=np.full(len(sequences),np.nan)
        for i,s in enumerate(sequences):
            if len(s)>2000:raise ValueError('NTv3 window exceeds 2000 bp')
            buckets.setdefault((len(s)+127)//128,[]).append(i)
        with torch.no_grad(),ieee_fp32(self.device):
            for _,indices in sorted(buckets.items()):
                for start in range(0,len(indices),4):
                    take=indices[start:start+4];views=[]
                    for i in take:views.extend(sorted([sequences[i],sequences[i].translate(RC)[::-1]]))
                    ids,mask=self.encode(views)
                    result[take]=self.model(ids,mask).double().reshape(-1,2).mean(1).cpu().numpy()
        if not np.isfinite(result).all():raise ValueError('NTv3 emitted nonfinite logits')
        self.windows+=len(sequences);return result
    def contig_logits(self,sequences,indices):
        ix=np.asarray(indices)
        if ix.ndim!=1 or (ix.size and (not np.issubdtype(ix.dtype,np.integer) or ix.min()<0 or ix.max()>=len(sequences))) or len(set(ix.tolist()))!=len(ix):raise ValueError('Invalid NTv3 contig indices')
        result=np.zeros(len(sequences),np.float64);counts=np.zeros(len(sequences),np.int8);windows=[];owners=[]
        for i in ix:
            s=normalize(sequences[int(i)]);spans=window_spans(len(s));counts[i]=len(spans)
            for a,b in spans:windows.append(s[a:b]);owners.append(i)
        for i,z in zip(owners,self.predict_windows(windows),strict=True):result[i]+=z/int(counts[i])
        return result
