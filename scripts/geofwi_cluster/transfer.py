"""Stage this immutable campaign and its hashed one-GiB data subset over SSH."""
import shlex
import subprocess
import tarfile
from pathlib import Path
from .prepare import ROOT,NAME,REMOTE
from .campaign import verify,read,write,sha

stage=ROOT/'experiments'/NAME
verify(stage)
archive=ROOT/'reports'/NAME/'payload.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for path in sorted(stage.rglob('*')):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
            tar.add(path,arcname=str(Path(NAME)/path.relative_to(stage)),recursive=False)
    for path in sorted((stage/'data').resolve().rglob('*')):
        if path.is_file():tar.add(path,arcname=str(Path(NAME)/'data'/path.relative_to((stage/'data').resolve())),recursive=False)
def ssh(args,**kw):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','bocconi-cluster',shlex.join([str(x) for x in args])],check=True,**kw)
ssh(['python3','-c','from pathlib import Path;import sys;assert not Path(sys.argv[1]).exists(), "Existing stage: verify it before reusing"',REMOTE])
with archive.open('rb') as stream:ssh(['tar','-xzf','-','-C',REMOTE.parent],stdin=stream)
ssh(['python3',REMOTE/'campaign.py','verify','--stage',REMOTE])
ssh(['sbatch','--test-only',REMOTE/'run.sbatch',REMOTE])
write(ROOT/'reports'/NAME/'transfer.json',dict(status='verified',remote=str(REMOTE),manifest_sha256=sha(stage/'manifest.json'),payload_sha256=sha(archive)))
print('Transfer and scheduler syntax verified')
