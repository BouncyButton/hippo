"""Run all 36 paired BAND conditions, retaining improvements and regressions."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


class Drive:
    def __init__(self, folder):
        private=Path('/content/.hfh_storage_auth')
        self.command=[str(private/'rclone'),'--config',str(private/'rclone.conf'),
            '--tpslimit','2','--tpslimit-burst','2','--retries','3','--low-level-retries','10']
        self.folder='hfh_storage:'+folder
        self.call('mkdir',self.folder)
        entries=json.loads(self.call('lsjson','hfh_storage:','--dirs-only'))
        matches=[r for r in entries if r['Name']==folder and r.get('ID')]
        if len(matches)>1:raise ValueError('Ambiguous Drive campaign folder')
        if matches:
            self.command+=['--drive-root-folder-id',matches[0]['ID']]
            self.folder='hfh_storage:'
    def path(self,name):return self.folder+('' if self.folder.endswith(':') else '/')+name
    def call(self,*args):
        result=subprocess.run(self.command+list(args),capture_output=True,text=True,timeout=600)
        if result.returncode:raise RuntimeError('Drive transfer failed: '+result.stderr[-1000:])
        return result.stdout
    def upload(self,path,name=None):
        remote=self.path(name or path.name)
        self.call('copyto',str(path),remote,'--checksum')
        metadata=json.loads(self.call('lsjson',remote,'--stat','--hash'))
        if metadata['Size']!=path.stat().st_size or metadata['Hashes'].get('MD5',metadata['Hashes'].get('md5'))!=hashlib.md5(path.read_bytes()).hexdigest():
            raise ValueError('Remote checksum mismatch')


def extract_archive(path,durable):
    with zipfile.ZipFile(path) as zipped:
        manifest=json.loads(zipped.read('results/archive_manifest.json'))
        for name,digest in manifest.items():
            if Path(name).name!=name or hashlib.sha256(zipped.read('results/'+name)).hexdigest()!=digest:
                raise ValueError('BAND result archive integrity failed')
        for item in zipped.infolist():
            parts=Path(item.filename).parts
            if len(parts)==2 and parts[0]=='results' and parts[1] not in ('.','..'):
                target=durable/parts[1];target.parent.mkdir(parents=True,exist_ok=True)
                payload=zipped.read(item)
                if target.exists() and target.read_bytes()!=payload:raise ValueError('Local archive differs from Drive')
                target.write_bytes(payload)
    return read(durable/'paired_comparison.json')


def failed_fold(rows):
    """A declared full-matrix candidate cannot win after a nonpositive fold."""
    for model in {r['model'] for r in rows}:
        for fold in (0,1,2):
            group=[r for r in rows if r['model']==model and r['fold']==fold]
            if len(group)==3 and {r['seed'] for r in group}=={0,1,2}:
                if sum(r['delta_dice_pp'] for r in group)<=0 or any(r['band_dice']<.5 for r in group):
                    return dict(model=model,fold=fold,mean_delta_pp=sum(r['delta_dice_pp'] for r in group)/3)
    return None


def run(args):
    root=args.root;bundle=root/'bundle';payload_path=root/args.payload;payload=read(payload_path)
    precision=read(bundle/'gpu_overlay.json')['precision']
    if precision!=payload['precision']:
        raise ValueError('BAND requires controls at the same precision. Complete matching FP32 controls before comparing against FP32 BAND.')
    work=root/args.work;work.mkdir(exist_ok=True)
    drive=Drive(args.folder)
    downloads=work/'remote_archives';downloads.mkdir(exist_ok=True)
    drive.call('copy',drive.folder,str(downloads),'--include','*.zip','--include','campaign_status.json',
               '--transfers','2','--checkers','2','--checksum')
    for path in downloads.glob('*.zip'):
        if path.stem not in payload['references']:raise ValueError('Unexpected BAND archive identity')
        extract_archive(path,work/'archives'/path.stem)
        shutil.copyfile(path,work/path.name)
        write(work/(path.stem+'.uploaded.json'),dict(sha256=sha(path),verified_at=time.time()))
    prior=downloads/'campaign_status.json'
    if prior.exists():
        state=read(prior);current=state.get('current')
        if current and state['status'] in ('running','runtime_error') and not (work/'archives'/current/'cleanup.json').exists() and not (work/'runs'/current/'budget_ledger.json').exists():
            raise RuntimeError('Unfinished charged BAND attempt on previous VM requires recovery audit: '+current)
    protocols=work/'protocols'
    for model,files in payload['protocols'].items():
        folder=protocols/model;folder.mkdir(parents=True,exist_ok=True)
        for name,value in files.items():
            text=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
            path=folder/name
            if path.exists() and path.read_text()!=text:raise ValueError('Frozen BAND protocol changed')
            path.write_text(text)
        write(folder/'baseline_references.json',{k:v for k,v in payload['references'].items() if k.startswith(model+'_f')})
    sys.path.insert(0,str(bundle/'source'))
    from scripts.colab_hfh.bands_archive import archive
    from scripts.colab_hfh.runtime import verify_archive
    env=dict(os.environ,PYTHONPATH=':'.join(map(str,[bundle/'compatibility',bundle/'source',bundle/'source/scripts',bundle/'source/baselines/nnUNet'])),
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',HIPPO_HFH_GPU_CACHE='1')
    pairs=payload.get('pairs')
    state=dict(status='running',arm='dice_bands',seeds=payload['seeds'],folds=payload['folds'],expected_runs=len(pairs) if pairs is not None else len(payload['rates'])*len(payload['seeds'])*len(payload['folds']),results=[],
        interpretation='All paired outcomes retained; no validation-based seed rejection')
    status_path=work/'campaign_status.json';deadline=time.time()+args.hours*3600
    def save():write(status_path,state)
    try:
        for model in payload['rates']:
            for fold in payload['folds']:
                for seed in payload['seeds']:
                    name=f'{model}_f{fold}_s{seed}';protocol=protocols/model
                    if pairs is not None and name not in pairs:continue
                    output=work/'runs'/name;durable=work/'archives'/name
                    output.mkdir(parents=True,exist_ok=True);durable.mkdir(parents=True,exist_ok=True)
                    reference=payload['references'][name]
                    if (durable/'cleanup.json').exists():
                        verify_archive(durable);result=read(durable/'paired_comparison.json')
                        if result['baseline_archive_sha256']!=reference['archive_sha256'] or read(durable/'runtime_recipe.json')['protocol_sha256']!=sha(protocol/'protocol.json'):
                            raise ValueError('Cached BAND result belongs to another comparison')
                    else:
                        if time.time()>=deadline or (work/'STOP').exists():
                            state['status']='paused';save();drive.upload(status_path);return
                        state['current']=name;save();drive.upload(status_path)
                        print('BAND START',name,flush=True)
                        write(output/'execution_environment.json',dict(precision=precision,gpu=subprocess.check_output([
                            'nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()))
                        if not (output/'completion.json').exists():
                            command=[sys.executable,'-m','scripts.colab_hfh.bands_worker','--protocol-dir',str(protocol),
                                '--output',str(output),'--durable',str(durable),'--pkl',str(bundle/'data_index.json'),
                                '--fold',str(fold),'--seed',str(seed)]
                            with (output/'worker.log').open('a') as log:
                                subprocess.run(command,cwd=bundle/'source',env={**env,'HIPPO_DEADLINE':str(min(deadline,time.time()+3600))},stdout=log,stderr=subprocess.STDOUT,check=True)
                        result=archive(output,durable,model,fold,seed,protocol,reference)
                    zip_path=work/(name+'.zip');receipt=work/(name+'.uploaded.json')
                    if not (receipt.exists() and zip_path.exists() and read(receipt)['sha256']==sha(zip_path)):
                        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zipped:
                            for path in sorted(durable.iterdir()):
                                if path.is_file():zipped.write(path,'results/'+path.name)
                            for path in sorted(protocol.iterdir()):zipped.write(path,'protocol/'+path.name)
                            for filename in ('bands_worker.py','bands_archive.py','baseline.py','runtime.py'):
                                zipped.write(bundle/'source/scripts/colab_hfh'/filename,'source/'+filename)
                            if payload.get('strict_search'):
                                zipped.write(bundle/'source/scripts/colab_hfh/bands_variants.py','source/bands_variants.py')
                            zipped.write(__file__,'source/bands_campaign.py')
                            zipped.write(payload_path,'source/bands_v1_payload.json')
                        drive.upload(zip_path);write(receipt,dict(sha256=sha(zip_path),verified_at=time.time()))
                    print('BAND VERIFIED',name,f"baseline={100*result['baseline_dice']:.2f}% band={100*result['band_dice']:.2f}% delta={result['delta_dice_pp']:+.2f}pp",flush=True)
                    state['results'].append(result);save()
                    if payload.get('min_pair_delta_pp') is not None and (result['delta_dice_pp']<payload['min_pair_delta_pp'] or result['band_dice']<.5):
                        state.update(status='rejected_pair',rejection=dict(model=model,fold=fold,seed=seed,delta_pp=result['delta_dice_pp'],required_delta_pp=payload['min_pair_delta_pp']))
                        state.pop('current',None);save();drive.upload(status_path)
                        print('REJECTED individual pair',state['rejection'],flush=True);return
                    rejected=failed_fold(state['results']) if payload.get('reject_nonpositive_fold') else None
                    if rejected:
                        state.update(status='rejected_completed_fold',rejection=rejected)
                        state.pop('current',None);save();drive.upload(status_path)
                        print('REJECTED completed fold',rejected,flush=True);return
        state['status']='complete_all_36' if state['expected_runs']==36 else 'complete_requested_matrix';state.pop('current',None);save();drive.upload(status_path)
        print('COMPLETE:',state['expected_runs'],'paired BAND comparisons',flush=True)
    except Exception as error:
        state.update(status='runtime_error',error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--hours',type=float,default=6)
    parser.add_argument('--payload',default='bands_v1_payload.json')
    parser.add_argument('--work',default='bands_v1')
    parser.add_argument('--folder',default='results_bands_v1')
    args=parser.parse_args()
    with (args.root/'bands_campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args)


if __name__=='__main__':main()
