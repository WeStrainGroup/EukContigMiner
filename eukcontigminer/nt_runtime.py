"""Offline supervised NT500M adapter inference with bounded symmetric windows.

The separately supplied NT backbone and derivative adapter use CC BY-NC-SA 4.0.
"""
from contextlib import nullcontext
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn

FORMULA = 'DNA_if_routed_else_sigmoid(logit(v052_tree_score)+0.5*mean_NT_window_logit)'
WINDOW_RULE = 'whole_le2000_else_first_last_center_rc_symmetric_v1'
RC = str.maketrans('ACGTN','TGCAN')

class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base=base
        self.a=nn.Linear(base.in_features,rank,bias=False)
        self.b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale=8/rank
    def forward(self, values):
        return self.base(values)+self.b(self.a(values))*self.scale

class Classifier(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.encoder.requires_grad_(False)
        for layer in encoder.encoder.layer:
            attention=layer.attention.self
            attention.query=LoRALinear(attention.query)
            attention.value=LoRALinear(attention.value)
        self.head=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self,ids,mask,cls_token_id):
        hidden=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state.float()
        valid=(mask.bool() & (ids!=cls_token_id)).unsqueeze(-1)
        if not torch.all(valid.sum(1)>0):raise ValueError('NT input has no sequence tokens')
        mean=(hidden*valid).sum(1)/valid.sum(1)
        maximum=hidden.masked_fill(~valid,-torch.inf).amax(1)
        return self.head(torch.cat((mean,maximum),dim=1)).squeeze(-1).float()

def window_spans(length):
    if length<1:raise ValueError('NT input must be nonempty')
    if length<=2000:return [(0,length)]
    width=2000-(length%2);start=(length-width)//2
    return sorted(set([(0,2000),(start,start+width),(length-2000,length)]))

def validate_binding(binding):
    if (binding.get('alpha')!=.5 or binding.get('window_rule')!=WINDOW_RULE
        or binding.get('backbone_revision')!='06615c1660c892fc199840c18123f8385b3542a8'
        or binding.get('rank')!=8 or binding.get('lora_alpha')!=8
        or binding.get('license')!='CC-BY-NC-SA-4.0'
        or binding.get('forward_reverse_complement')!='mean_logits'
        or binding.get('window_aggregation')!='mean_logits'):
        raise ValueError('NT adapter contract differs')

class NTAdapter:
    def __init__(self,binding,device,bound_path,sha256_file):
        validate_binding(binding)
        source_env=os.environ.get('EUKCONTIGMINER_NT500M_DIR')
        if not source_env:raise ValueError('Set EUKCONTIGMINER_NT500M_DIR to the frozen offline NT500M directory')
        source=Path(source_env)
        for name,digest in binding['source_files_sha256'].items():
            if Path(name).name!=name or sha256_file(source/name)!=digest:
                raise ValueError('NT source artifact differs: '+name)
        # Only the locally verified source code and weights may be loaded.
        from transformers import AutoTokenizer,AutoModelForMaskedLM
        self.tokenizer=AutoTokenizer.from_pretrained(source,trust_remote_code=True,local_files_only=True)
        mlm=AutoModelForMaskedLM.from_pretrained(source,trust_remote_code=True,local_files_only=True,use_safetensors=True)
        if mlm.config.hidden_size!=1024 or mlm.config.num_hidden_layers!=29:raise ValueError('NT backbone shape differs')
        self.model=Classifier(mlm.esm.eval()).to(device);del mlm
        state=torch.load(bound_path(binding['checkpoint'],'NT adapter'),weights_only=True,map_location='cpu')['state_dict']
        learned={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        if len(learned)!=122 or set(state)!=set(learned):raise ValueError('NT learned tensor names differ')
        with torch.no_grad():
            for name,value in state.items():
                if learned[name].shape!=value.shape or not torch.isfinite(value).all():raise ValueError('Invalid NT learned tensor '+name)
                learned[name].copy_(value.to(device))
        self.model.eval().requires_grad_(False)
        self.device=device
        self.bf16=device.type=='cuda' and torch.cuda.get_device_capability(device)[0]>=8
        self.windows=0
    def predict_windows(self,sequences):
        counts=[max(len(self.tokenizer(s,add_special_tokens=True,truncation=False)['input_ids']) for s in (sequence,sequence.translate(RC)[::-1])) for sequence in sequences]
        order=np.argsort(counts,kind='stable');result=np.full(len(sequences),np.nan);start=0
        context=torch.autocast('cuda',dtype=torch.bfloat16) if self.bf16 else nullcontext()
        with torch.inference_mode(),context:
            while start<len(order):
                stop=start+1
                while stop<len(order) and stop-start<16 and 2*(stop-start+1)*counts[order[stop]]**2<=4_000_000:stop+=1
                take=order[start:stop];batch=[sequences[int(i)] for i in take]
                views=[v for sequence in batch for v in (sequence,sequence.translate(RC)[::-1])]
                encoded=self.tokenizer(views,add_special_tokens=True,padding=True,truncation=False,return_attention_mask=True,return_tensors='pt')
                if encoded['input_ids'].shape[1]>2048:raise ValueError('NT window exceeds token limit')
                logits=self.model(encoded['input_ids'].to(self.device),encoded['attention_mask'].to(self.device),self.tokenizer.cls_token_id).double().cpu().numpy().reshape(len(batch),2)
                result[take]=logits.mean(1);start=stop
        if not np.isfinite(result).all():raise ValueError('NT emitted nonfinite window logits')
        self.windows+=len(sequences)
        return result
    def contig_logits(self,sequences,indices):
        windows=[];owners=[];counts=np.zeros(len(sequences),np.int8);result=np.zeros(len(sequences),np.float64)
        for i in indices:
            sequence=''.join(c if c in 'ACGT' else 'N' for c in sequences[int(i)])
            spans=window_spans(len(sequence));counts[i]=len(spans)
            for left,right in spans:windows.append(sequence[left:right]);owners.append(i)
        for owner,value in zip(owners,self.predict_windows(windows)):result[owner]+=value/int(counts[owner])
        return result
