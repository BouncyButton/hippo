"""Prepare a self-contained search package for the existing authenticated A100."""
import os
import ast
import json
from pathlib import Path
import shutil
import zipfile

from scripts.colab_hfh.nnunet_search13_policy import NAME,CAP,ARMS,ACTIVITY,candidates,make_protocol
from scripts.colab_hfh.runtime import read,write,digest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'
WORK = REPORT/NAME
REMOTE = Path('/content/hippo_'+NAME)


def build():
    WORK.mkdir(parents=True,exist_ok=True)
    package = WORK/'package'
    package.mkdir(exist_ok=True)
    source = package/'source/scripts/colab_hfh'
    source.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'scripts/colab_hfh').glob('*.py'):shutil.copyfile(p,source/p.name)
    reference = REPORT/'nnunet_composition_a100_v1/results/native.zip'
    with zipfile.ZipFile(reference) as z:
        for item in z.namelist():
            if item.startswith(('results/','protocol/')):
                target = package/'reference_a100_native'/item
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(item))
    native = package/'protocols/native'
    shutil.copytree(package/'reference_a100_native/protocol',native,dirs_exist_ok=True)
    p = read(native/'protocol.json')
    p.update(a100_native_anchor=str(REMOTE/'runs/native/run/native_update60.json'),hardware_campaign=NAME)
    write(native/'protocol.json',p)
    for recipe in candidates():
        arm = recipe['arm']
        directory = next((REPORT/d/arm/'protocol' for d in
                          ('nnunet_long_matrix_v1','nnunet_long_constraints_v1')
                          if (REPORT/d/arm/'protocol/protocol.json').exists()),None)
        if directory is None:raise ValueError('Missing frozen canonical protocol: '+arm)
        target = package/'protocols'/recipe['id']
        shutil.copytree(directory,target,dirs_exist_ok=True)
        p = make_protocol(read(directory/'protocol.json'),recipe,REMOTE/'runs/native/run/native_update60.json')
        write(target/'protocol.json',p)
        for key,values in p.items():
            if key.endswith('source_sha256') and isinstance(values,dict):
                for name,sha in values.items():
                    if digest(source/name) != sha:raise ValueError('Historical source changed: '+name)
    design = dict(name=NAME,fold=0,seed=0,arms=list(ARMS),candidates=candidates(),
        candidates_count=len(candidates()),total_runs=1+len(candidates()),optimization_cap=CAP,
        max_workers=12,initial_verification_workers=3,threads_per_worker=1,
        native='Identical original long recipe: native Dice+CE/deep supervision; SGD momentum .95; no augmentation; BF16; native180 schedule',
        activity_policy=ACTIVITY,selection='Final Dice among activity-verified candidates; also report leaders improving every included raw penalty; exploratory f0/s0 selection',
        diagnostic_accounting='Activity probe charged inside optimization cap. Post-training adherence and gradient audits separate and recorded.',
        activation='All components start together after exact matched native update60; shared9 or30 update ramp',
        storage_account=os.environ.get('HIPPO_STORAGE_EMAIL', ''),compute_account=os.environ.get('HIPPO_COMPUTE_EMAIL', ''),
        prior_native_zip_sha256=digest(reference),no_other_fold_or_seed=True)
    write(package/'search_design.json',design)
    manifest = {str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*'))
                if p.is_file() and p.name != 'manifest.json'}
    write(package/'manifest.json',manifest)
    archive = WORK/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'manifest.json']:z.write(package/name,name)
    sha = digest(archive)
    launch = f'''# Search all 13 combinations on fold 0 / seed 0; use up to 12 workers.
from pathlib import Path
import os,json,hashlib,shutil,subprocess,zipfile
prior=Path('/content/hippo_nnunet_composition_a100_v1')
search=Path('{REMOTE}')
if search.exists() and (search/'status.json').exists():raise RuntimeError('Existing charged search: inspect status before relaunch')
assert json.loads((prior/'status.json').read_text())['status']=='complete_five_arms'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
search.mkdir(exist_ok=True)
archive=search/'{archive.name}'
storage.download('hfh_storage:{archive.name}',archive)
def search_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert search_sha(archive)=='{sha}', 'Search package differs'
package=search/'package'
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        path=Path(item.filename)
        if path.is_absolute() or '..' in path.parts or (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe package')
    z.extractall(package)
for name,expected in json.loads((package/'manifest.json').read_text()).items():
    assert search_sha(package/name)==expected,name
if not (search/'bundle').exists():
    shutil.copytree(prior/'bundle',search/'bundle',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(package/'source',search/'bundle/source',dirs_exist_ok=True)
shutil.copyfile(prior/'storage_sdk.py',search/'storage_sdk.py')
source=search/'bundle/source'
manifest={{str(p.relative_to(source)):search_sha(p) for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}}
(search/'execution_source_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(search/'execution_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in manifest:z.write(source/name,'source/'+name)
    for name in ('storage_sdk.py','execution_source_manifest.json'):z.write(search/name,name)
    z.write(package/'search_design.json','search_design.json')
provenance=json.loads((prior/'launch_provenance.json').read_text())
provenance.update(package_sha256='{sha}',max_gpu_workers=12,threads_per_worker=1,
    source_manifest_sha256=search_sha(search/'execution_source_manifest.json'),
    execution_source_zip_sha256=search_sha(search/'execution_source.zip'),campaign='{NAME}')
(search/'launch_provenance.json').write_text(json.dumps(provenance,indent=2))
search_env=dict(os.environ,PYTHONPATH=':'.join([str(search)]+[str(search/'bundle'/p) for p in ('compatibility','source','source/scripts','source/baselines/nnUNet')]),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',HFH_DRIVE_CREDENTIALS='gcloud')
python=Path('/content/hippo_hfh_baselines_v2/venv/bin/python')
# Verify generated trainer hooks and numerical gradient probes on the real environment before training.
preflight=subprocess.run([str(python),'-m','scripts.colab_hfh.test_nnunet_search13','--runtime'],cwd=source,env=search_env,capture_output=True,text=True)
(search/'preflight.log').write_text(preflight.stdout+preflight.stderr)
print(preflight.stdout+preflight.stderr)
if preflight.returncode:raise RuntimeError('Search preflight failed; no training launched')
log=(search/'controller.log').open('a')
search_controller=subprocess.Popen([str(python),'-m','scripts.colab_hfh.nnunet_search13_campaign','--root',str(search),'--package',str(package),'--max-workers','12'],cwd=source,env=search_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(search/'controller_pid.txt').write_text(str(search_controller.pid))
print('Search launched:',search_controller.pid,'| 13 combinations | 78 candidates + native | fold0 seed0 | max12 workers')
'''
    ast.parse(launch)
    (WORK/'launch_cell.py').write_text(launch)
    write(WORK/'build_receipt.json',dict(package_sha256=sha,bytes=archive.stat().st_size,
        files=len(manifest),candidates=len(candidates()),status='prepared_not_started'))
    (WORK/'README.md').write_text('''# nnU-Net all-13 A100 search, fold 0 / seed 0

78 candidates plus one fresh native comparator. No other fold or seed is authorized in this stage.
Native recipe and optimization cap remain unchanged: 63,952,901,334,180 FLOPs.
Every candidate reproduces native update60 before activating all included canonical losses.
Search: strengths .05/.10/.20, ramp9; strength .10/ramp30; strength .10/ramp9 with each included component emphasized 2x in turn (multi-component arms only).

Every active update measures detached FP32 full-resolution native/auxiliary gradients on its training crop. Frozen anchor coefficients are retained when feasible; otherwise component RMS contributions are projected to a bounded simplex. At full ramp every component contributes at least 1% of native logit-gradient RMS and 10% of the sum of component norms. Their sum is at most 50% of native, bounding the joint norm even with cancellation. During the common ramp those floors scale with the ramp. A 64x anchor coefficient ceiling fails closed rather than dropping a component.

The 8192-operations-per-logit-per-term probe allowance is charged at every active optimization update; fewer updates than the uninstrumented candidates may fit the same cap. This conservative accounting is explicit, not a hidden extension of the budget. The original 180-update LR horizon is preserved. Final raw canonical penalties are evaluated on train and validation for all four components, including the native control. These are surrogate adherence measures, not hard anatomical guarantees. Post-training evaluation/probe allowances are recorded separately.

All candidates run to their capped endpoint, without intermediate validation pruning. Best Dice recipes and recipes improving every included penalty are both reported; this is exploratory validation selection, not independent confirmation. Up to12 workers (one CPU thread each), following three successful initial verification runs. Isolated source, existing account7 A100, account2 SDK storage, no Colab MCP. Verify each remote result archive before immediately retiring completed checkpoints; failed recovery is preserved.
''')
    print(json.dumps(read(WORK/'build_receipt.json'),indent=2))


if __name__ == '__main__':build()
