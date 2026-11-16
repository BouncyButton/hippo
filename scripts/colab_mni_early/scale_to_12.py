"""Controller-only overlay: drain active workers, reuse results, dispatch up to 12."""
import json
import os
from pathlib import Path
import psutil
import signal
import subprocess
import time

ROOT=Path('/content/mni_nnunet_early_f0s0_20261113')
state=json.loads((ROOT/'status.json').read_text())
old=psutil.Process(int((ROOT/'controller.pid').read_text()))
assert str(ROOT/'runner.py') in old.cmdline() and 'controller' in old.cmdline()
old.send_signal(signal.SIGSTOP)
print('Dispatch paused; waiting for the three active runs to finish without interrupting them.',flush=True)
while True:
    remaining=[]
    for task in state['active']:
        try:
            process=psutil.Process(task['pid'])
            if process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
                remaining.append(task['id'])
                continue
        except psutil.NoSuchProcess:
            pass
        folder=ROOT/'runs'/task['id']
        assert (folder/'completion.json').exists() and (folder/'resource_usage.json').exists(),task['id']
    if not remaining:break
    time.sleep(5)
# Terminate only the stopped scheduler after all of its training workers finish.
old.send_signal(signal.SIGTERM);old.send_signal(signal.SIGCONT)
time.sleep(1)
original=(ROOT/'runner.py').read_text()
before="plan=read(ROOT/'plan.json');pending=list(plan['tasks']);active={};results=[];failed=[]"
after="""plan=read(ROOT/'plan.json');active={};failed=[]
    results=read(ROOT/'status.json')['results']
    completed={row['id'] for row in results}
    for task in plan['tasks']:
        if task['id'] not in completed and (ROOT/'runs'/task['id']/'completion.json').exists():
            results.append(archive(task,drive));completed.add(task['id'])
    pending=[task for task in plan['tasks'] if task['id'] not in completed]"""
assert original.count(before)==1
patched=original.replace(before,after)
patched=patched.replace("len(os.sched_getaffinity(0))//4", "len(os.sched_getaffinity(0))")
patched=patched.replace("max(2*1024**3,math.ceil(peak*1.4))", "max(int(1.2*1024**3),math.ceil(peak*1.5+0.4*1024**3))")
patched=patched.replace("(psutil.virtual_memory().available-8*1024**3)//(5*1024**3)", "(psutil.virtual_memory().available-4*1024**3)//(3.6*1024**3)")
compile(patched,'runner_12.py','exec')
overlay=ROOT/'runner_12.py';overlay.write_text(patched)
from storage_sdk import Drive
drive=Drive()
drive.upload(overlay,'mni_storage:source/runner_12.py')
record=dict(change='Scheduler capacity and completed-run reuse only. Training code and four CPU threads unchanged.',
    reason='Measured full-combination peak GPU reservation 570425344 bytes; each active process RSS about 3.52 GiB; 12 CPU slots.',
    gpu_headroom_gib=3,ram_headroom_gib=4,ram_per_worker_gib=3.6,max_workers=12)
(ROOT/'worker_scaling.json').write_text(json.dumps(record,indent=2))
drive.upload(ROOT/'worker_scaling.json','mni_storage:source/worker_scaling.json')
launch=json.loads((ROOT/'launch.json').read_text())
env=dict(os.environ,**launch['env'],PYTHONUNBUFFERED='1')
log=(ROOT/'controller_12.log').open('a')
proc=subprocess.Popen([launch['python'],'-u',str(overlay),'controller'],cwd=ROOT/'source',env=env,
    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'controller.pid').write_text(str(proc.pid))
print('Replacement scheduler launched:',proc.pid,'; completed runs will be reused.',flush=True)
