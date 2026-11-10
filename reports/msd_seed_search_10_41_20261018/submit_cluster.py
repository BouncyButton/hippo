import json
import shlex
import subprocess

from scripts.cross_dataset_campaign import msd_seed_search as job
from scripts.cross_dataset_campaign.common import REMOTE, write

stage = REMOTE/job.NAME
source = stage/'source'
env = ['env', 'PYTHONDONTWRITEBYTECODE=1',
       'PYTHONPATH='+':'.join(str(source/v) for v in ('', 'scripts', 'baselines/nnUNet'))]
args = [*env, '/home/3160552/.conda/envs/hippocampus/bin/python', '-m', job.MODULE,
        'submit', '--stage', str(stage)]
subprocess.run(['ssh', 'bocconi-cluster', shlex.join(args)], check=True)
receipt = json.loads(subprocess.check_output([
    'ssh', 'bocconi-cluster', shlex.join(['cat', str(stage/'submission.json')])]))
write(job.ROOT/'reports'/job.NAME/'submission.json', receipt)
print(json.dumps(receipt))
