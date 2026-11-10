from pathlib import Path
import subprocess,sys,time,os,json
from scripts.cross_dataset_campaign.common import write
from scripts.cross_dataset_campaign import swin_unet_seed5_constraints_job as job
root=job.ROOT;dest=root/'reports'/job.NAME;stage=root/'experiments'/job.NAME
record=dest/'collector_process.json'
if record.exists():
 value=json.loads(record.read_text())
 try: os.kill(value['pid'],0)
 except ProcessLookupError: pass
 else: raise RuntimeError('Collector already running')
args=[sys.executable,'-m','scripts.cross_dataset_campaign.collect','--stage','/mnt/beegfsstudents/home/3160552/'+job.NAME,'--local-stage',str(stage),'--destination',str(dest)]
with (dest/'collector.log').open('ab') as log:
 process=subprocess.Popen(args,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
awake=subprocess.Popen(['caffeinate','-i','-w',str(process.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
write(record,dict(pid=process.pid,caffeinate_pid=awake.pid,started_at=time.time(),python=sys.executable,command=args,log=str(dest/'collector.log')))
print(record.read_text())
