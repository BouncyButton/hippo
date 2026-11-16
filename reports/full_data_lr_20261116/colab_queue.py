"""Run one LR worker per visible notebook and leave the active control workers alone."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
ds=sys.argv[1];assert ds in ('HFH','ADNI','MSD')
sys.path.insert(0,'/content')
from continue_colab import session,existing
guard=(here/'queue.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
r=json.loads((here/'plan.json').read_bytes())
keys=[k for k,v in r.items() if v['dataset']==ds and v['status']=='new' and v['model']!='medsam3']
keys+=[k for k,v in r.items() if v['model']=='medsam3' and ((v['dataset']=='MSD' and v['fold']=={'MSD':0,'ADNI':1,'HFH':3}[ds]) or (ds=='ADNI' and v['dataset']=='ADNI'))]
# Prioritize the MSD fold when the matched control has published and the training device is free.
keys.sort(key=lambda k:(r[k]['model']!='medsam3',r[k]['model']!='nnunet',k))
def done(k):return existing(drive,k+'_full_lr_v1_results.zip')
def room():
    return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
drive=session()
print('ASSIGNED',ds,len(keys),flush=True)
bad={}
while True:
    pending=[k for k in keys if not done(k)]
    if not pending:break
    free=room();ready=[]
    for k in pending:
        v=r[k]
        if bad.get(k,0)>time.time():continue
        if v['model']=='medsam3' and free<17000:continue
        if v['dataset']=='MSD' and v['model']=='medsam3' and not existing(drive,k+'_standard_fold_results.zip'):continue
        if v['model']!='medsam3' and free<{'nnunet':6500,'swinunetr':10000}[v['model']]:continue
        ready.append(k)
    if not ready:time.sleep(45);continue
    k=ready[0];print('START',k,free,flush=True)
    py=Path('/content/full_data_'+r[k]['dataset'])/'venv/bin/python'
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH='/content')
    rc=subprocess.run([str(py),str(here/'run.py'),k,'--cloud'],env=env).returncode
    if rc:
        print('RETRY AFTER INSPECTION',k,rc,flush=True);bad[k]=time.time()+3600
