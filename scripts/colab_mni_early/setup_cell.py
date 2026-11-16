# Executed in the browser notebook after the package is uploaded.
import hashlib, json, os, pathlib, subprocess, sys, zipfile
NAME='mni_nnunet_early_f0s0_20261113'
ROOT=pathlib.Path('/content')/NAME
PACKAGE=pathlib.Path('/content')/(NAME+'.zip')
assert PACKAGE.exists(), 'Upload the frozen pilot zip first.'
ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(PACKAGE) as z:
    assert all(not pathlib.PurePosixPath(n).is_absolute() and '..' not in pathlib.PurePosixPath(n).parts for n in z.namelist())
    z.extractall(ROOT)
for name,sha in json.loads((ROOT/'manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
os.environ['HFH_DRIVE_CREDENTIALS']='gcloud'
sys.path.insert(0,str(ROOT))
# The SDK helper requires a verified root; the temporary value is never used for writes.
os.environ['MNI_DRIVE_ROOT']='pending'
from storage_sdk import credentials
from googleapiclient.discovery import build
service=build('drive','v3',credentials=credentials(),cache_discovery=False)
about=service.about().get(fields='user(emailAddress),storageQuota').execute(num_retries=5)
assert about['user']['emailAddress']==os.environ.get('HIPPO_STORAGE_EMAIL', ''),about['user']
q=about['storageQuota']
if 'limit' in q: assert int(q['limit'])-int(q['usage'])>8*1024**3
folders=service.files().list(q="'root' in parents and trashed=false and name='"+NAME+"'",fields='files(id,name,mimeType,owners(emailAddress))').execute(num_retries=5)['files']
assert len(folders)<=1,'Ambiguous folder; inspect before continuing.'
if not folders:
    file_id=service.files().generateIds(count=1,space='drive').execute(num_retries=5)['ids'][0]
    folders=[service.files().create(body=dict(id=file_id,name=NAME,mimeType='application/vnd.google-apps.folder'),fields='id,name,mimeType,owners(emailAddress)').execute()]
assert os.environ.get('HIPPO_STORAGE_EMAIL', '') in [x['emailAddress'] for x in folders[0]['owners']]
os.environ['MNI_DRIVE_ROOT']=folders[0]['id']
import storage_sdk
storage_sdk.ROOT_ID=folders[0]['id']
drive=storage_sdk.Drive()
drive.upload(PACKAGE,'mni_storage:source/'+PACKAGE.name)
print('Verified durable package on storage account:',about['user']['emailAddress'])
print('Results folder: https://drive.google.com/drive/folders/'+folders[0]['id'])
VENV=ROOT/'venv';PYTHON=VENV/'bin/python'
if not PYTHON.exists(): subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(VENV)],check=True)
with (ROOT/'setup.log').open('w') as log:
    result=subprocess.run([str(PYTHON),'-m','pip','install','--disable-pip-version-check','-r',str(ROOT/'requirements-colab.txt')],stdout=log,stderr=subprocess.STDOUT)
print('\n'.join((ROOT/'setup.log').read_text().splitlines()[-20:]))
assert result.returncode==0,'Dependency setup failed; see setup.log.'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,[ROOT,ROOT/'source',ROOT/'source/scripts',ROOT/'source/baselines/nnUNet'])),
    PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',MPLCONFIGDIR=str(ROOT/'mpl'))
subprocess.run([str(PYTHON),'-c','from scripts.mni_constraints import campaign; import torch,monai; print(torch.__version__,monai.__version__,torch.cuda.get_device_name())'],env=env,cwd=ROOT/'source',check=True)
launch=dict(python=str(PYTHON),env={k:env[k] for k in ('PYTHONPATH','HFH_DRIVE_CREDENTIALS','MNI_DRIVE_ROOT','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','WANDB_MODE','MPLCONFIGDIR')})
(ROOT/'launch.json').write_text(json.dumps(launch,indent=2))
print('SETUP VERIFIED. Ready to launch 17 seed0/fold0 runs at the fixed 53.305T optimization cap.')
