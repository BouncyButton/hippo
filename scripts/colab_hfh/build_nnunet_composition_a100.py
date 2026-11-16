"""Build the A100 notebook with the MedSAM notebook's verified Drive SDK."""
import os
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'
WORK = REPORT/'nnunet_composition_a100_v1'
NAME = 'nnunet_composition_a100_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


AUTH = '''# Compute stays on account 7; authenticate only account 2 for SDK storage.
import os, subprocess, getpass
if 'login' in globals() and login.poll() is None:
    login.terminate();login.wait()
command = ['gcloud','auth','login',os.environ.get('HIPPO_STORAGE_EMAIL', ''),
           '--enable-gdrive-access','--no-launch-browser','--quiet']
login = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,text=True,bufsize=1)
buffer = ''
while True:
    char = login.stdout.read(1)
    if not char:break
    buffer += char
    if char == '\\n':
        print(buffer,end='',flush=True);buffer=''
    if 'enter' in buffer.lower() and 'code' in buffer.lower() and ':' in buffer:
        authorization = getpass.getpass('Authorization code (hidden; not saved): ')
        login.stdin.write(authorization+'\\n');login.stdin.flush()
        del authorization
        buffer=''
if buffer:print(buffer)
if login.wait():raise RuntimeError('Second-account SDK login did not complete')
print('SDK login complete. The next cell verifies the storage identity and owner.')
'''


def build():
    WORK.mkdir(exist_ok=True)
    package = WORK/'package'
    shutil.copytree(REPORT/'nnunet_composition_v1/package',package,dirs_exist_ok=True)
    source = package/'source/scripts/colab_hfh'
    for p in (ROOT/'scripts/colab_hfh').glob('*.py'):
        shutil.copyfile(p,source/p.name)
    for name in ('native','band_edge','band_edge_depth'):
        p = read(package/'controls'/name/'protocol/protocol.json')
        for key, values in p.items():
            if key.endswith('source_sha256') and isinstance(values,dict):
                for file,expected in values.items():
                    if sha(source/file) != expected:
                        raise ValueError('Historical scientific source changed: '+file)
    design = dict(name=NAME,fold=0,seed=0,gpu='A100 >=40 GB',max_workers=3,
        controls=['native','band_edge','band_edge_depth'],
        tests=['attenuated_band_edge','group_band_edge_depth'],
        native_updates=180,constraint_updates=179,optimization_cap=63952901334180,
        controls_first=True,calibration='Same preregistered rules on fresh native A100 update-60 gradients',
        primary_reference='Same hardware native final budget',
        gate='User authorized independent A100 execution; MedSAM status recorded only',compute_account=os.environ.get('HIPPO_COMPUTE_EMAIL', ''),
        storage_account=os.environ.get('HIPPO_STORAGE_EMAIL', ''))
    write(package/'a100_design.json',design)
    manifest = {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
                if p.is_file() and p != package/'manifest.json'}
    write(package/'manifest.json',manifest)
    archive = WORK/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(package.rglob('*')):
            if p.is_file():zipped.write(p,str(p.relative_to(package)))
    package_sha = sha(archive)
    # Reuse the exact SDK implementation from the requested MedSAM notebook.
    reference = read(WORK/'medsam_reference.ipynb')
    tree = ast.parse(''.join(reference['cells'][1]['source']))
    sdk = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='storage_source' for t in n.targets))
    restore = (REPORT/'unetr_completion_20261113/restore.py').read_text()
    tree = ast.parse(restore)
    node = next(n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='storage_source' for t in n.targets))
    lines = restore.splitlines(keepends=True)
    restore = ''.join(lines[:node.lineno-1])+ 'storage_source = '+repr(sdk)+'\n'+''.join(lines[node.end_lineno:])
    restore = restore.replace('results_unetr_completion_20261113','results_'+NAME)
    restore = restore.replace('verified UNETR storage preflight','verified A100 composition storage preflight')
    restore = restore.replace("print('UNETR_RESTORE_READY',hardware['gpu'],flush=True)",
        "if 'A100' not in hardware['gpu'] or hardware['precision'] != 'bf16':raise RuntimeError('A100 BF16 required')\nprint('A100_SETUP_READY',hardware['gpu'],flush=True)")
    # This only stages local files and checks storage; no optimization occurs.
    split = restore.index("VENV=ROOT/'venv'")
    staging, setup = restore[:split],restore[split:]
    deploy = f'''# Frozen package and isolated source; no active runtime is modified.
from pathlib import Path
import json,hashlib,shutil,zipfile,subprocess,os
ROOT=Path('/content/hippo_hfh_baselines_v2')
work=Path('/content/hippo_nnunet_composition_a100_v1');work.mkdir(exist_ok=True)
archive=work/'{archive.name}'
storage.download('hfh_storage:{archive.name}',archive)
if sha(archive) != '{package_sha}':raise ValueError('Experiment package differs')
package=work/'package'
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        p=Path(member.filename)
        if p.is_absolute() or '..' in p.parts or (member.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe package member')
    z.extractall(package)
for name,expected in json.loads((package/'manifest.json').read_text()).items():
    if sha(package/name)!=expected:raise ValueError('Package member differs: '+name)
if (work/'status.json').exists():raise RuntimeError('Existing attempt: inspect status and charged ledger before relaunch')
if not (work/'bundle').exists():
    shutil.copytree(ROOT/'bundle',work/'bundle',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(package/'source',work/'bundle/source',dirs_exist_ok=True)
shutil.copyfile(ROOT/'storage_sdk.py',work/'storage_sdk.py')
source=work/'bundle/source'
manifest={{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
(work/'execution_source_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(work/'execution_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in manifest:z.write(source/name,'source/'+name)
    z.write(work/'storage_sdk.py','storage_sdk.py')
    z.write(work/'execution_source_manifest.json','execution_source_manifest.json')
    z.write(ROOT/'bundle/requirements-colab.txt','requirements-colab.txt')
    z.write(ROOT/'bundle/manifest.json','original_bundle_manifest.json')
provenance=dict(hardware=hardware,package_sha256='{package_sha}',
    sdk_source_sha256=sha(work/'storage_sdk.py'),source_manifest_sha256=sha(work/'execution_source_manifest.json'),
    execution_source_zip_sha256=sha(work/'execution_source.zip'),
    compute_account=os.environ.get('HIPPO_COMPUTE_EMAIL', ''),storage_account=os.environ.get('HIPPO_STORAGE_EMAIL', ''),max_gpu_workers=3)
(work/'launch_provenance.json').write_text(json.dumps(provenance,indent=2))
run_env=dict(env,PYTHONPATH=':'.join([str(work)]+[str(work/'bundle'/p) for p in
    ('compatibility','source','source/scripts','source/baselines/nnUNet')]))
log=(work/'controller.log').open('a')
controller=subprocess.Popen([str(PYTHON),'-m','scripts.colab_hfh.nnunet_composition_a100_campaign',
    '--root',str(work),'--package',str(package)],cwd=source,env=run_env,
    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(work/'controller_pid.txt').write_text(str(controller.pid))
print('Controller PID:',controller.pid,'; independent A100 run, maximum three GPU workers.')
'''
    monitor = '''from pathlib import Path
import json,subprocess
work=Path('/content/hippo_nnunet_composition_a100_v1')
status=work/'status.json'
if status.exists():
    value=json.loads(status.read_text())
    print('Status:',value['status'],'| Phase:',value.get('phase'),'| Workers:',value.get('workers'))
    for row in value.get('results',[]):
        print(row['variant'], 'validation Dice',round(100*row['validation_dice'],5),
              'native gain pp',row.get('native_gain_pp'))
else:print('No training status yet.')
if (work/'controller.log').exists():print((work/'controller.log').read_text()[-5000:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True))
'''
    markdown = '''# HFH nnU-Net — long-budget composition tests on A100

Compute: the configured compute account. Storage: the configured storage account, using the same verified Google Drive SDK as the HFH MedSAM3 notebook. No Drive mount or rclone is used for new transfers.

Three fresh matched controls (native, BAND+EDGE, BAND+EDGE+DEPTH) run concurrently, followed by the two prepared tests: attenuated BAND+EDGE and group-calibrated BAND+EDGE+DEPTH. Maximum three GPU workers. This separate notebook does not interrupt the baseline or MedSAM sessions.

All runs use fold 0 / seed 0, unchanged native Dice+CE/deep supervision and long LR schedule, no augmentation, BF16 and the original 63,952,901,334,180 optimization FLOP cap. Native gets 180 updates; each constrained run gets 179. Every constrained run must exactly reproduce the **fresh A100 native update-60 model** before constraint updates. The two tests apply the previously frozen calibration rules to the same ten A100 training crops, without validation selection.

The user authorized this separate A100 runtime to start independently of the remaining MedSAM job. Archives and source are verified on account-2 Drive before completed checkpoints are retired. Failed runs retain recovery and are never silently restarted with a reset budget.

Run authentication, input staging, environment setup, launch, then monitor. Google SDK login details: https://docs.cloud.google.com/sdk/gcloud/reference/auth/login
'''
    def cell(kind,text,index):
        result=dict(cell_type=kind,id='composition_a100_'+str(index),metadata={},source=text.splitlines(keepends=True))
        if kind=='code':result.update(execution_count=None,outputs=[])
        return result
    texts=[('markdown',markdown),('code',AUTH),('code',staging),('code',setup),('code',deploy),('code',monitor)]
    for kind,text in texts:
        if kind=='code':ast.parse(text)
    notebook=dict(nbformat=4,nbformat_minor=5,cells=[cell(k,t,i) for i,(k,t) in enumerate(texts)],
        metadata=dict(accelerator='GPU',colab=dict(name='HFH_nnUNet_composition_A100.ipynb',gpuType='A100',machine_shape='hm'),
                      kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python')))
    write(WORK/'HFH_nnUNet_composition_A100.ipynb',notebook)
    write(WORK/'build_receipt.json',dict(package_sha256=package_sha,bytes=archive.stat().st_size,
        files=len(manifest),sdk_source_sha256=hashlib.sha256(sdk.encode()).hexdigest(),status='prepared_not_started'))
    print(json.dumps(read(WORK/'build_receipt.json'),indent=2))


if __name__=='__main__':build()
