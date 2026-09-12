"""Reconstruct native outputs and verify strict, prevalence-adjusted metrics."""
import csv,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from native_installed_pair_v1 import R,O,D,read,bind,checked,save

def load_panel(panel):
    plan=read(D/'plan.json');sources=[];parts=[]
    for part in (['supp_other','supp_three'] if panel=='supplement' else [panel]):
        src=plan['panels'][part]
        rp=O/'hpc_full100_formal_allwindows_v1/panel_receipt.json' if part=='formal' else checked(src['receipt'])
        r=read(rp);sources.append(bind(rp));ids=checked(r['ids']).read_text().splitlines();meta=dict(np.load(checked(r['metadata']),allow_pickle=False))
        assert len(ids)==r['records']==len(set(ids)) and len(ids)==len(meta['truth'])
        scores={a:np.full(len(ids),np.nan) for a in ['v054','raw100']};host=np.full(len(ids),'',dtype='<U32');org=np.zeros(len(ids),bool)
        for k in range(8):
            indices=np.arange(k,len(ids),8)
            manifest=list(csv.DictReader(checked(src['shards'][str(k)]['manifest']).open(),delimiter='\t'))
            assert [v['fragment_id'] for v in manifest]==[ids[i] for i in indices]
            for i,row in zip(indices,manifest,strict=True):
                assert int(row['length_bp'])==meta['length'][i]
                assert int(row['truth'])==meta['truth'][i] and str(row['species_id'])==str(meta['species'][i])
                org[i]=row.get('source_kind')=='organelle'
                host[i]=row.get('host_subclass','') if org[i] else row.get('source_label','')
            for arm in scores:
                out=D/part/(f'{k:02d}_{arm}');q=out/'receipt.json';v=read(q);sources.append(bind(q))
                assert v['status']=='native_installed_output_verified' and v['plan']==bind(D/'plan.json')
                assert v['input']==src['shards'][str(k)]['fasta'] and v['config']==plan['packages'][arm]['config']
                rows=list(csv.DictReader(checked(v['output']).open(),delimiter='\t'))
                assert [x['contig_id'] for x in rows]==[ids[i] for i in indices] and len(rows)==v['records']
                assert np.array_equal([int(x['length_bp']) for x in rows],meta['length'][indices])
                a=np.array([float(x['p_euk']) for x in rows]);assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
                assert [x['label'] for x in rows]==['Eukaryota' if v>plan['packages'][arm]['threshold'] else 'Other' for v in a]
                scores[arm][indices]=a
        assert np.all(meta['truth'][org]==1)
        parts.append((ids,meta,scores,host,org))
    ids=sum([v[0] for v in parts],[]);assert len(ids)==len(set(ids))
    meta={k:np.concatenate([v[1][k] for v in parts]) for k in ['truth','length','species']}
    scores={a:np.concatenate([v[2][a] for v in parts]) for a in ['v054','raw100']}
    host=np.concatenate([v[3] for v in parts]);org=np.concatenate([v[4] for v in parts]);y=meta['truth'];length=meta['length']
    assert set(np.unique(y))=={0,1} and (length>=1000).all()
    masks={'overall':np.ones(len(y),bool),'short_1_2kb':(length>=1000)&(length<2000)}
    if panel=='formal':masks.update({str(lo):length==lo for lo in list(range(1000,10001,500))+[50000,100000]})
    else:masks.update({str(lo):(length>=lo)&((length<lo+500) if lo<9500 else (length<=10000)) for lo in range(1000,10000,500)})
    for g in ['Fungi','Metazoa','Viridiplantae','Other_Eukaryota']:masks[g]=(y==0)|(host==g)
    masks['fungi_bacteria_only']=(host=='Fungi')|(host=='Bacteria');masks['organelle']=(y==0)|org
    masks={k:v for k,v in masks.items() if np.any(y[v]==1) and np.any(y[v]==0)}
    return plan,sources,ids,meta,scores,masks

def metrics(y,pred,pi):
    tp=int(np.count_nonzero((y==1)&pred));fp=int(np.count_nonzero((y==0)&pred));fn=int(np.count_nonzero((y==1)&~pred));tn=int(np.count_nonzero((y==0)&~pred))
    tpr=tp/(tp+fn);fpr=fp/(fp+tn);den=pi*tpr+(1-pi)*fpr;precision=pi*tpr/den if den else 0.
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tpr,fpr=fpr,precision=precision,f1=2*pi*tpr/(pi*(1+tpr)+(1-pi)*fpr))

def calculate(panel):
    p,src,ids,meta,scores,masks=load_panel(panel);y=meta['truth'];ts={a:p['packages'][a]['threshold'] for a in scores}
    mm={a:{k:{str(pi):metrics(y[m],(s>ts[a])[m],pi) for pi in [.001,.01,.1]} for k,m in masks.items()} for a,s in scores.items()}
    r=dict(status='native_cli_scores_reconstructed_requires_separate_verification',panel=panel,records=len(y),species=len(set(meta['species'])),
        thresholds=ts,metrics=mm,plan=bind(D/'plan.json'),sources=src,code=bind(__file__),
        role='Reused development evaluation. Both installed CLIs run on identical full fragments; frozen original calibration thresholds, no evaluation-set fitting. Not untouched confirmation. Genome contamination screening incomplete.',
        original_final_whole_genome_fasta_reads=0)
    return r,meta,scores,masks

def run(panel,verify=False):
    out=D/'evaluation'/panel;r,meta,scores,masks=calculate(panel);y=meta['truth'];ts=r['thresholds']
    if verify:
        prev=read(out/'report.json');bb=prev.pop('scores');assert prev==r
        for arm,s in scores.items():
            assert np.array_equal(s,np.load(checked(bb[arm])))
            for scope,mask in masks.items():
                # Independent integer counting by bincount, independent F1 expression.
                v=np.bincount(2*y[mask].astype(int)+(s[mask]>ts[arm]).astype(int),minlength=4)
                tn,fp,fn,tp=map(int,v)
                for pi in [.001,.01,.1]:
                    e=r['metrics'][arm][scope][str(pi)];assert [tn,fp,fn,tp]==[e['tn'],e['fp'],e['fn'],e['tp']]
                    wpos=pi/(tp+fn);wneg=(1-pi)/(tn+fp)
                    f=2*wpos*tp/(2*wpos*tp+wpos*fn+wneg*fp);assert abs(f-e['f1'])<1e-12
        species,inv=np.unique(meta['species'],return_inverse=True);ns=len(species);P=np.bincount(inv,weights=y,minlength=ns);N=np.bincount(inv,weights=1-y,minlength=ns)
        assert not np.any((P>0)&(N>0));rng=np.random.default_rng(2026091204);w=np.zeros((1000,ns))
        for ix in [np.flatnonzero(P>0),np.flatnonzero(N>0)]:w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=1000)
        ci={}
        for scope in ['overall','short_1_2kb','fungi_bacteria_only']:
            if scope not in masks:continue
            mask=masks[scope];pp=w@np.bincount(inv,weights=mask*y,minlength=ns);nn=w@np.bincount(inv,weights=mask*(1-y),minlength=ns);assert (pp>0).all() and (nn>0).all();b={}
            for a,s in scores.items():
                pred=s>ts[a];tpr=(w@np.bincount(inv,weights=mask&(y==1)&pred,minlength=ns))/pp;fpr=(w@np.bincount(inv,weights=mask&(y==0)&pred,minlength=ns))/nn
                b[a]=.02*tpr/(.01*(1+tpr)+.99*fpr)
            ci[scope]=np.quantile(b['raw100']-b['v054'],[.025,.5,.975]).tolist()
        save(out/'independent_readback.json',dict(status='native_scores_and_integer_counts_verified_in_separate_process',report=bind(out/'report.json'),paired_species_bootstrap_raw100_minus_v054=ci,replicates=1000,selection_correction=False))
        print(json.dumps({'panel':panel,'status':'verified','f1':{a:mm['overall']['0.01']['f1'] for a,mm in r['metrics'].items()},'ci':ci}),flush=True);return
    out.mkdir(parents=True,exist_ok=False);bb={}
    for a,s in scores.items():np.save(out/(a+'.npy'),s);bb[a]=bind(out/(a+'.npy'))
    save(out/'report.json',dict(r,scores=bb));table=[dict(model=a,scope=k,prevalence=pi,threshold=ts[a],**v) for a,mm in r['metrics'].items() for k,pp in mm.items() for pi,v in pp.items()]
    with (out/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=[f'# Installed ECM comparison: {panel}','',r['role'],'',f"Records: {r['records']}; species: {r['species']}. Prevalence-adjusted F1 from class-conditional rates. Overall counts are pooled, not averages of bin F1.",'','| Model | Scope | F1@0.1% | F1@1% | F1@10% | FP | FN |','|---|---|---:|---:|---:|---:|---:|']
    for a,mm in r['metrics'].items():
        for k,pp in mm.items():lines.append('| '+a+' | '+k+' | '+' | '.join(f"{pp[str(pi)]['f1']:.9f}" for pi in [.001,.01,.1])+f" | {pp['0.01']['fp']} | {pp['0.01']['fn']} |")
    (out/'scientific_report.md').write_text('\n'.join(lines)+'\n')

def watch():
    completed=[]
    while time.time()<1789182000:
        for panel,parts in [('main',['main']),('supplement',['supp_other','supp_three']),('formal',['formal'])]:
            if panel in completed:continue
            if not all((D/part/f'{k:02d}_{arm}'/'receipt.json').exists() for part in parts for k in range(8) for arm in ['v054','raw100']):continue
            for mode in ['run','verify']:
                subprocess.run([sys.executable,'-B',__file__,mode,panel],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            completed.append(panel)
        save(D/'evaluation_status.json',{'completed':completed,'pid':os.getpid(),'at':time.time(),'stage':'complete' if len(completed)==3 else 'waiting_half_hour'})
        if len(completed)==3:return
        time.sleep(1800)

if __name__=='__main__':
    if sys.argv[1]=='watch':watch()
    else:run(sys.argv[2],sys.argv[1]=='verify')
