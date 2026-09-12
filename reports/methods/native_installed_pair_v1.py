"""Run both unmodified installed packages on identical frozen full contigs."""
import csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

R=Path('/ssd/wangxinyu2/eukcontigminer')
O=R/'small_dna_backbone_selection_20260907_v1'
D=O/'native_installed_pair_20260912_v1'
PACKAGES={'v054':R/'release_candidate_v054_20260907/installed_final',
          'raw100':R/'release_candidate_v055_raw32768_20260912/installed'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(b):
    p=Path(b['path']);assert sha(p)==b['sha256'],p;return p
def save(p,v):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def environment(arm,gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(PACKAGES[arm]),
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',
        HF_HOME=str(R/'huggingface_esmc_v1'),HF_MODULES_CACHE=str(R/'nt500m_dna_20260906_v1/module_cache'),
        EUKCONTIGMINER_NT500M_DIR=str(R/'nt500m_dna_20260906_v1/source_files'),
        EUKCONTIGMINER_NTV3_DIR=str(O/'snapshots/NTv3_100M_pre'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

def prepare():
    assert not D.exists();D.mkdir()
    panels={}
    for name,source in [('main','continuous500_scoring_v1'),('supp_other','unused_species_generalization_scoring_v1'),('supp_three','unused_three_euk_scoring_v1')]:
        receipt=O/source/'receipt.json';r=read(receipt)
        panels[name]={'receipt':bind(receipt),'records':r['records'],'ids':r['ids'],'metadata':r['metadata'],
                      'shards':{k:{x:v[x] for x in ['records','fasta','manifest']} for k,v in r['shards'].items()}}
    f=O/'hpc_full100_formal_allwindows_v1/plan.json';r=read(f)
    panels['formal']={'receipt':bind(f),'records':r['records'],'shards':{k:{'fasta':v['panel']['fragments.fna'],'manifest':v['panel']['fragments.tsv']} for k,v in r['old_sources'].items()}}
    packages={}
    for arm,p in PACKAGES.items():
        cfg=p/'eukcontigminer/model_data/model.json';c=read(cfg)
        packages[arm]={'path':str(p),'config':bind(cfg),'threshold':c['prediction_rule']['threshold'],
            'code':{str(x.relative_to(p)):sha(x) for x in (p/'eukcontigminer').rglob('*.py')},
            'nt_checkpoint':bind(p/'eukcontigminer/model_data/nt_adapter.pt')}
    assert packages['raw100']['nt_checkpoint']['sha256']=='6721e6dc411b159f9cab40d1e587389c71c1055e56b62fef1031af0e61dbc71d'
    assert packages['raw100']['threshold']==.9994805844224834
    sys.path.insert(0,str(PACKAGES['raw100']))
    from eukcontigminer.fasta import fasta_records
    source=checked(panels['main']['shards']['0']['fasta'])
    inp=D/'canary4096.fna';n=0
    with inp.open('x') as f:
        for ident,seq in fasta_records(source):
            f.write('>'+ident+'\n'+seq+'\n');n+=1
            if n==4096:break
    assert n==4096
    save(D/'plan.json',dict(status='frozen_before_native_predictions',code=bind(__file__),packages=packages,panels=panels,
        canary={'fasta':bind(inp),'records':n},settings={'cpu_threads':6,'buffer_records':1024},
        method='Unmodified installed CLI, default precision and batching. Same full contigs, same settings, fresh process for each model/shard. Frozen existing calibration thresholds; no refitting on evaluation panels.',
        panel_order=['main','supp_other','supp_three','formal'],
        limitations='Development panels reused for selection; not untouched confirmation. Native results supersede cached scores for deployment claims. CPU/GPU exact equality not established.',
        original_final_whole_genome_fasta_reads=0))

def run_one(arm,gpu,panel,k,inp,records=None):
    p=read(D/'plan.json');checked(p['code']);pkg=p['packages'][arm]
    checked(pkg['config']);checked(pkg['nt_checkpoint'])
    for n,h in pkg['code'].items():assert sha(PACKAGES[arm]/n)==h
    fasta=checked(inp);out=D/panel/(f'{k:02d}_{arm}');out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    args=[sys.executable,'-B','-P','-c','from eukcontigminer.cli import main; raise SystemExit(main())',str(fasta),'-o',str(out/'predictions.tsv'),
          '--device','cuda:0','--cpu-threads','6','--buffer-records','1024']
    save(out/'started.json',dict(pid=os.getpid(),at=time.time(),gpu=gpu,arm=arm,input=inp,plan=bind(D/'plan.json')))
    with (out/'cli.log').open('x') as f:
        subprocess.run(args,cwd=R,env=environment(arm,gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
    seconds=time.monotonic()-start
    sys.path.insert(0,str(PACKAGES['raw100']))
    from eukcontigminer.fasta import fasta_records
    import math
    ids=set();n=0;bases=0
    with (out/'predictions.tsv').open() as f:
        for row,(ident,seq) in zip(csv.DictReader(f,delimiter='\t'),fasta_records(fasta),strict=True):
            assert row['contig_id']==ident and int(row['length_bp'])==len(seq)
            assert ident not in ids;ids.add(ident)
            s=float(row['p_euk']);assert math.isfinite(s) and 0<=s<=1
            assert row['label']==('Eukaryota' if s>pkg['threshold'] else 'Other')
            n+=1;bases+=len(seq)
    if records is not None:assert n==records
    save(out/'receipt.json',dict(status='native_installed_output_verified',arm=arm,records=n,bases=bases,seconds=seconds,
        input=inp,output=bind(out/'predictions.tsv'),plan=bind(D/'plan.json'),config=pkg['config'],threshold=pkg['threshold'],gpu=gpu))
    print(panel,k,arm,n,seconds,flush=True)

def worker(k):
    p=read(D/'plan.json')
    for panel in p['panel_order']:
        inp=p['panels'][panel]['shards'][str(k)]
        for arm in (['v054','raw100'] if k%2==0 else ['raw100','v054']):
            run_one(arm,k,panel,k,inp['fasta'],inp.get('records'))
    save(D/f'worker_{k:02d}_complete.json',{'status':'complete','at':time.time(),'pid':os.getpid()})

def free_gpus():
    busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines())
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    return [int(a) for a,u,m,v in ([x.strip() for x in r.split(',')] for r in rows) if u not in busy and int(m)<500 and int(v)<5]

def controller():
    assert not (D/'controller.json').exists();assert free_gpus()==list(range(8))
    save(D/'controller.json',{'pid':os.getpid(),'at':time.time(),'code':bind(__file__)})
    p=read(D/'plan.json')
    for arm in ['v054','raw100']:run_one(arm,0,'canary',0,p['canary']['fasta'],4096)
    c={a:read(D/'canary'/('00_'+a)/'receipt.json') for a in ['v054','raw100']}
    save(D/'canary_complete.json',{'status':'native_pair_passed','runs':c,'estimated_all_panels_seconds_per_gpu':sum(v['seconds'] for v in c.values())*sum(v['records'] for v in p['panels'].values())/(4096*8),'estimate_limit':'Extrapolation only; loading and length composition differ.'})
    assert free_gpus()==list(range(8))
    workers=[]
    for k in range(8):
        log=(D/f'worker_{k:02d}.log').open('x')
        proc=subprocess.Popen([sys.executable,'-B',__file__,'worker',str(k)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6'))
        workers.append((k,proc,log))
    save(D/'submission.json',{'at':time.time(),'controller_pid':os.getpid(),'workers':{str(k):v.pid for k,v,_ in workers}})
    results={}
    for k,v,f in workers:results[str(k)]=v.wait();f.close()
    save(D/'controller_result.json',{'exit_codes':results,'at':time.time(),'status':'complete' if all(v==0 for v in results.values()) else 'failed_preserved_no_retry'})

if __name__=='__main__':
    mode=sys.argv[1]
    try:
        if mode=='prepare':prepare()
        elif mode=='controller':controller()
        elif mode=='worker':worker(int(sys.argv[2]))
        else:raise ValueError(mode)
    except BaseException as e:
        if D.exists():save(D/f'{mode}_{os.getpid()}_failure.json',{'error':repr(e),'at':time.time()})
        raise
