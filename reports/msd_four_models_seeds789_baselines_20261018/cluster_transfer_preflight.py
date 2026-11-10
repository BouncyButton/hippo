from pathlib import Path
import io,json,shlex,subprocess,tarfile
from scripts.cross_dataset_campaign import seeds789_msd_baselines_job as job
from scripts.cross_dataset_campaign.common import REMOTE,write
stage=job.prepare()
remote=REMOTE/job.NAME
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w:gz') as tar:
 tar.add(stage,arcname=job.NAME,filter=lambda info: None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
check='from pathlib import Path; import sys; p=Path(sys.argv[1]); assert not p.exists(), str(p)+" already exists"'
subprocess.run(['ssh','bocconi-cluster',shlex.join(['python3','-c',check,str(remote)])],check=True)
subprocess.run(['ssh','bocconi-cluster',shlex.join(['tar','-xzf','-','-C',str(REMOTE)])],input=archive.getvalue(),check=True)
print('Uploaded frozen36-run batch',flush=True)
source=remote/'source'
env=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+':'.join(str(source/v) for v in ('','scripts','baselines/nnUNet'))]
args=[*env,'/home/3160552/.conda/envs/hippocampus/bin/python','-m',job.MODULE,'preflight','--stage',str(remote)]
subprocess.run(['ssh','bocconi-cluster',shlex.join(args)],check=True)
value=json.loads(subprocess.check_output(['ssh','bocconi-cluster',shlex.join(['cat',str(remote/'preflight.json')])]))
write(stage/'preflight.json',value)
write(job.ROOT/'reports'/job.NAME/'preflight.json',value)
print(json.dumps({'status':value['status'],'cli_commands_checked':value['cli_commands_checked'],'models':len(value['models']),'quota':value['quota']}),flush=True)
