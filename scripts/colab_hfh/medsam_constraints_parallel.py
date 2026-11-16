"""Adaptive concurrency over real pending runs, with unchanged paired recipes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.colab_hfh import medsam_archive as archives
from scripts.colab_hfh import medsam_constraints_campaign as serial
from scripts.colab_hfh.medsam_constraints_capacity import Capacity
from scripts.colab_hfh.medsam_constraints_policy import PAIRS, STRICT_CONTROL, STRICT_RECIPES, ARMS, make_protocol, qualifies
from scripts.colab_hfh.medsam_recovery import copy_verified, rclone, remote_path
from scripts.colab_hfh.runtime import read, write, digest


def gpu_sample():
    import psutil
    command = ['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    name,total,used,util = subprocess.check_output(command,text=True).strip().split(',')
    if name.strip() != 'NVIDIA A100-SXM4-40GB':
        raise RuntimeError('Paired campaign requires the original A100-SXM4-40GB')
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'],text=True)
    processes = {}
    for line in text.strip().splitlines():
        pid, memory = line.split(',')
        processes[int(pid)] = float(memory)/1024
    return dict(total_gib=float(total)/1024, used_gib=float(used)/1024, utilization=float(util),
                processes=processes, host_free_gib=psutil.virtual_memory().available/1024**3,
                cpu_slots=max(1,len(os.sched_getaffinity(0))//4))


class Campaign:
    def __init__(self, root, hours, maximum):
        self.root, self.bundle = root, root/'bundle'
        self.work = root/'medsam_constraints_strict_v1'
        self.durable = Path('/content/hfh_storage/results_medsam3_constraints_strict_v1')
        self.work.mkdir(exist_ok=True)
        self.handles = []
        for path in (self.work/'controller.lock',root/'work_medsam3/controller.lock'):
            path.parent.mkdir(parents=True,exist_ok=True)
            handle = path.open('a')
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.handles.append(handle)
        # An orphaned worker must be recovered/attached deliberately, not duplicated.
        import psutil
        for process in psutil.process_iter(['cmdline']):
            if 'scripts.colab_hfh.medsam_constraints_worker' in (process.info['cmdline'] or []):
                raise RuntimeError('Existing MedSAM worker: wait for clean controller handoff')
        quota = json.loads(rclone('about','hfh_storage:','--json'))
        if min(quota.get('free',0),shutil.disk_usage(root).free) < 8*1024**3:
            raise RuntimeError('Need 8 GiB headroom locally and on second-account Drive')
        inventory = {r['Name']:r for r in json.loads(rclone('lsjson',
            'hfh_storage:results_medsam3_v1/compact_archives','--files-only','--hash'))}
        self.controls = {(seed,fold):serial.read_control(root,seed,fold,inventory) for seed,fold in PAIRS}
        archives.configure(self.bundle,self.work,self.durable)
        self.deadline = time.time()+hours*3600
        self.maximum = maximum
        self.stopping = False
        self.active = {}
        self.tuner = Capacity(maximum=maximum)
        self.phase_failure = None
        self.last_publish = 0
        self.state = dict(status='running_controls',gpu='NVIDIA A100-SXM4-40GB',controls=[],results=[],rejected=[],
            winner=None,recipes=list(STRICT_RECIPES),matrix_arms=list(ARMS),matrix_additional_runs=108,
            gate='All nine paired validation gains strictly positive; every Dice >=50%',
            warning='Validation-selected exploratory study. Preserve all attempts; no test-set claim.',
            scheduler='Adaptive process concurrency; scientific workers, updates, optimizer and sampling unchanged')
        signal.signal(signal.SIGTERM,self.stop)
        signal.signal(signal.SIGINT,self.stop)

    def stop(self,*unused):
        self.stopping = True

    def prepare(self, recipe, arm, seed, fold):
        name = f"{recipe['name']}__{arm}__f{fold}_s{seed}"
        output = self.work/'runs'/name
        output.mkdir(parents=True,exist_ok=True)
        control = self.controls[seed,fold]
        protocol = make_protocol(control['protocol'],control['slab_schedule'],control['training_history'],
                                 recipe,arm,control['archive_sha256'])
        path = self.work/'protocols'/(name+'.json')
        if path.exists() and read(path) != protocol:
            raise ValueError('Existing protocol differs: '+name)
        write(path,protocol)
        return dict(name=name,output=output,remote=self.durable/'runs'/name,control=control,
                    protocol=protocol,path=path,recipe=recipe,arm=arm,seed=seed,fold=fold)

    def existing(self, job):
        name = job['name']+'.zip'
        row = archives.CONTEXT['inventory'].get(name)
        if row is None:
            return None
        local = archives.CONTEXT['cache']/name
        expected = {k.lower():v for k,v in row['Hashes'].items()}['md5']
        if not local.exists() or hashlib.md5(local.read_bytes()).hexdigest() != expected:
            rclone('copyto',remote_path(archives.CONTEXT['remote']/name),str(local),'--checksum')
        if hashlib.md5(local.read_bytes()).hexdigest() != expected:
            raise ValueError('Completed archive differs: '+name)
        archives.verify_zip(local)
        if serial.zip_read(local,'protocol.json') != {**job['protocol'],'arm':job['arm']}:
            raise ValueError('Completed protocol differs: '+name)
        return serial.zip_read(local,'paired_result.json')

    def commit(self, job):
        row = serial.paired_result(job['output'],job['protocol'],job['control'],job['arm'],job['recipe'])
        write(job['output']/'paired_result.json',row)
        archives.archive(job['output'],job['remote'],'medsam3','medsam3',job['fold'],job['seed'],job['path'])
        print('PAIRED_FINAL',job['name'],json.dumps(row),flush=True)
        return row

    def launch(self, job):
        provenance = {str(p.relative_to(self.bundle/'source')):p.read_text()
                      for p in sorted((self.bundle/'source/scripts/colab_hfh').glob('medsam_constraints_*.py'))}
        for relative in ('scripts/cross_dataset_campaign/foundation.py',
                         'scripts/cross_dataset_campaign/adni_foundation_full_runtime.py'):
            provenance[relative]=(self.bundle/'source'/relative).read_text()
        write(job['output']/'constraint_sources.json',provenance)
        env = dict(os.environ,HIPPO_DEADLINE=str(self.deadline),HFH_CONTROLLER_ARCHIVES='1',
                   MNI_MANIFEST_SHA=digest(self.bundle/'manifest.json'))
        command = [sys.executable,'-m','scripts.colab_hfh.medsam_constraints_worker','--protocol',str(job['path']),
                   '--output',str(job['output']),'--durable',str(job['remote']),
                   '--base-weights',str(self.root/'sam3.pt'),'--arm',job['arm']]
        (job['output']/'supervisor_ready.json').unlink(missing_ok=True)
        with (job['output']/'worker.log').open('a') as log:
            process = subprocess.Popen(command,cwd=self.bundle/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        job.update(process=process,pid=process.pid,peak_gib=0,sent_stop=False,started=time.time())
        self.active[job['name']]=job
        print('PARALLEL_START',job['name'],'pid',process.pid,'workers',len(self.active),flush=True)

    def workers(self, gpu):
        rows = []
        for job in self.active.values():
            output=job['output']
            ledger=read(output/'checkpoint_ledger.json') if (output/'checkpoint_ledger.json').exists() else {}
            step=ledger.get('step',0)
            phase='evaluation' if (output/'training_summary.json').exists() else 'training' if step else 'starting'
            job['peak_gib']=max(job['peak_gib'],gpu['processes'].get(job['pid'],0))
            rows.append(dict(name=job['name'],pid=job['pid'],step=step,phase=phase,peak_gib=job['peak_gib'],
                             recipe=job['recipe']['name'],arm=job['arm'],seed=job['seed'],fold=job['fold']))
        return rows

    def publish(self, gpu, workers, force=False):
        self.state.update(observed_at=time.time(),workers=workers,concurrency=self.tuner.state(),
                          gpu_telemetry={k:v for k,v in gpu.items() if k!='processes'})
        self.state.pop('active',None)
        if workers:
            active=dict(workers[0]);output=self.active[active['name']]['output']
            for name in ('checkpoint_ledger','completion'):
                if (output/(name+'.json')).exists():active[name]=read(output/(name+'.json'))
            active['log_tail']=(output/'worker.log').read_text(errors='replace')[-3000:]
            self.state['active']=active
        write(self.work/'campaign_status.json',self.state)
        write(self.work/'parallel_benchmark.json',self.tuner.state())
        if force or time.monotonic()-self.last_publish>=90:
            write(self.work/'live_status.json',self.state)
            try:
                copy_verified(self.work/'live_status.json',self.durable/'monitor/live_status.json')
                copy_verified(self.work/'parallel_benchmark.json',self.durable/'monitor/parallel_benchmark.json')
                self.last_publish=time.monotonic()
            except Exception as error:
                print('STATUS_UPLOAD_RETRY',str(error)[-300:],flush=True)

    def phase(self, recipe, arm, pairs, bucket, fail_fast=False):
        self.tuner=Capacity(maximum=self.maximum)
        self.phase_failure=None
        jobs=[self.prepare(recipe,arm,*pair) for pair in pairs]
        pending=[];rows=[]
        for job in jobs:
            row=self.existing(job)
            if row is None and (job['output']/'completion.json').exists():row=self.commit(job)
            if row is None:pending.append(job)
            else:rows.append(row)
        def failed(row):
            return row['validation_dice']<.5 or (arm!='dice' and row['delta_dice_pp']<=0)
        if fail_fast and any(failed(row) for row in rows):
            self.state[bucket].extend(rows)
            return rows
        self.state[bucket].extend(rows)
        while pending or self.active:
            if time.time()>=self.deadline or (self.work/'STOP').exists():self.stopping=True
            for name,job in list(self.active.items()):
                code=job['process'].poll()
                if code is None:continue
                del self.active[name]
                if code==75:
                    if not self.stopping and not self.phase_failure:raise RuntimeError('Unexpected worker pause: '+name)
                    continue
                if code:raise RuntimeError(name+' failed: '+(job['output']/'worker.log').read_text()[-3000:])
                row=self.commit(job);rows.append(row);self.state[bucket].append(row)
                if fail_fast and failed(row):self.phase_failure=row
            gpu=gpu_sample();workers=self.workers(gpu)
            self.tuner.observe(time.monotonic(),gpu,workers)
            if gpu['total_gib']-gpu['used_gib']<2 or gpu['host_free_gib']<4:
                self.stopping=True
                self.state['resource_stop']='Low memory headroom; preserve all active checkpoints'
            if self.stopping or self.phase_failure:
                pending=[]
                for job in self.active.values():
                    if not job['sent_stop'] and (job['output']/'supervisor_ready.json').exists():
                        job['process'].send_signal(signal.SIGUSR1);job['sent_stop']=True
            elif pending and self.tuner.admission(gpu,workers):
                self.launch(pending.pop(0))
            self.publish(gpu,self.workers(gpu))
            if self.active or pending:time.sleep(5)
        self.publish(gpu_sample(),[],force=True)
        tag=recipe['name']+'__'+arm
        write(self.work/('benchmark_'+tag+'.json'),self.tuner.state())
        copy_verified(self.work/('benchmark_'+tag+'.json'),self.durable/'monitor'/('benchmark_'+tag+'.json'))
        if self.stopping:raise SystemExit(75)
        return rows

    def run(self):
        try:
            rows=self.phase(STRICT_CONTROL,'dice',PAIRS,'controls',fail_fast=True)
            if len(rows)!=9 or any(r['validation_dice']<.5 for r in rows):
                self.state['status']='control_recipe_failed';return
            for seed,fold in PAIRS:
                name=f"{STRICT_CONTROL['name']}__dice__f{fold}_s{seed}.zip"
                self.controls[seed,fold]=serial.archived_control(archives.CONTEXT['cache']/name)
            self.state['status']='running_bands'
            winner=None
            for recipe in STRICT_RECIPES:
                rows=self.phase(recipe,'dice_bands',PAIRS,'results',fail_fast=True)
                if qualifies(rows):winner=recipe;self.state['winner']=recipe;break
                self.state['rejected'].append(dict(recipe=recipe,completed_pairs=len(rows),results=rows,
                    reason='At least one pair failed; any in-flight runs were checkpointed without launching replacements'))
            if winner is None:
                self.state['status']='needs_next_hypothesis';return
            self.state['status']='running_full_matrix'
            for arm in ARMS:
                if arm not in ('dice','dice_bands'):
                    self.phase(winner,arm,PAIRS,'results')
            self.state['status']='complete_full_matrix'
        except SystemExit as error:
            if error.code!=75:raise
            self.state['status']='paused'
        except Exception as error:
            self.state.update(status='error',error=str(error)[-3500:])
            raise
        finally:
            for job in self.active.values():
                if job['process'].poll() is None:
                    while not (job['output']/'supervisor_ready.json').exists() and job['process'].poll() is None:
                        time.sleep(2)
                    if job['process'].poll() is None:job['process'].send_signal(signal.SIGUSR1)
            for job in self.active.values():job['process'].wait()
            self.publish(gpu_sample(),[],force=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/content/hippo_hfh'))
    parser.add_argument('--hours',type=float,default=18)
    parser.add_argument('--max-workers',type=int,default=0)
    args=parser.parse_args()
    if not 0<args.hours<=20 or args.max_workers<0:parser.error('Invalid session limit')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    Campaign(args.root,args.hours,args.max_workers).run()


if __name__=='__main__':main()
