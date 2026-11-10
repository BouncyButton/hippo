"""Verify and install only this deployment's staged inputs on the cluster."""
import json
from pathlib import Path
import shutil
import sys
import zipfile

stage=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(stage))
import cluster_runner as runner
for name,sha in runner.read(stage/'manifest.json').items():
    if runner.digest(stage/name)!=sha: raise ValueError('Deployment checksum differs: '+name)
root=stage/'bundle'; root.mkdir(exist_ok=True)
with zipfile.ZipFile(stage/'sam3_126_code.zip') as z:
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()): raise ValueError('Unsafe source ZIP')
    z.extractall(root)
sys.path.insert(0,str(root))
import store
import replication_store
inputs=runner.read(stage/'input_hashes.json')
for name,sha in inputs.items():
    path=stage/'inputs'/name
    if runner.digest(path)!=sha: raise ValueError('Data archive differs: '+name)
    store.extract_zip(path,root)
replication_store.unpack(stage/'calibration_f1s2.zip',root)
weights=root/'weights/sam3.pt'; weights.parent.mkdir(exist_ok=True)
base=Path('/tmp/hippo-cross-dataset-3160552/sam3.pt')
expected=runner.read(stage/'plan.json')['base_sha256']
if runner.digest(base)!=expected: raise ValueError('Original pretrained weights differ')
if not weights.exists(): shutil.copyfile(base,weights)
if runner.digest(weights)!=expected: raise ValueError('Staged pretrained weights differ')
(root/'work').mkdir(exist_ok=True)
if not (stage/'runs').exists(): (stage/'runs').symlink_to('bundle/work',target_is_directory=True)
if not (stage/'control.json').exists(): runner.write(stage/'control.json',{'pause':False})
runner.verify(stage)
for name,sha in inputs.items():
    path=stage/'inputs'/name
    path.unlink()
runner.write(stage/'staging_cleanup.json',dict(status='verified',deleted_duplicate_archives=inputs,
    note='Original local inputs and cluster datasets retained; only verified upload ZIP copies deleted.'))
print('Cluster source, all 290 volumes, calibration and pretrained weights verified.')
