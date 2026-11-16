"""Prepare the existing MedSAM notebook for independently authenticated storage."""
from pathlib import Path


def update_storage(notebook):
    module = (Path(__file__).parent/'medsam_constraints_drive.py').read_text()
    notebook['cells'][1]['source'] = '''# Authenticate the SECOND account with Google Cloud SDK before this cell.
# The compute subscription remains attached to the notebook's account7 session.
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
ROOT=Path('/content/hippo_hfh');ROOT.mkdir(exist_ok=True)
BUNDLE=ROOT/'bundle'
STORAGE_HOME=Path('/content/hfh_storage');STORAGE_HOME.mkdir(exist_ok=True)
KIND='medsam3'
RESULTS=STORAGE_HOME/'results_medsam3_constraints_strict_v1/bootstrap';RESULTS.mkdir(parents=True,exist_ok=True)
os.environ['HFH_DRIVE_TRANSPORT']='colab_adc'
os.environ['HFH_DRIVE_CREDENTIALS']='gcloud'
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()
''' + 'storage_source = '+repr(module)+'''
storage_namespace={}
exec(compile(storage_source,'medsam_constraints_drive.py','exec'),storage_namespace)
storage=storage_namespace['client']()  # Verifies identity AND root ownership before writes.
storage.download('hfh_storage:transport_manifest.json',STORAGE_HOME/'transport_manifest.json')
probe=ROOT/'storage_probe.json'
probe.write_text(json.dumps({'purpose':'verified MedSAM storage preflight','time':time.time()}))
storage.upload(probe,'hfh_storage:results_medsam3_constraints_strict_v1/monitor/storage_probe.json')
storage.download('hfh_storage:results_medsam3_constraints_strict_v1/monitor/storage_probe.json',ROOT/'storage_probe.downloaded.json')
if sha(probe)!=sha(ROOT/'storage_probe.downloaded.json'):raise IOError('Storage round-trip differs')
print('Second-account storage upload/download verified. No training started.',flush=True)
'''
    stage = ''.join(notebook['cells'][2]['source'])
    import re
    stage = re.sub(r'\bHOME\b', 'STORAGE_HOME', stage)
    anchor = 'from concurrent.futures import ThreadPoolExecutor'
    start = stage.index(anchor) if anchor in stage else stage.index('def fetch_part(part):')
    stage = stage[:start]+'''from concurrent.futures import ThreadPoolExecutor
# Resolve only immutable input parts. Result-file ambiguity stays an error.
def select_input_part(part, rows):
    matches = [row for row in rows if row['name'] == part['name']]
    if not matches or any(int(row.get('size', -1)) != part['bytes']
                          or row.get('mimeType', '').startswith('application/vnd.google-apps.')
                          or not row.get('md5Checksum') for row in matches):
        raise ValueError('Input part missing or metadata differs: '+part['name'])
    if len({row['md5Checksum'] for row in matches}) != 1:
        raise ValueError('Conflicting input duplicates: '+part['name'])
    return min(matches, key=lambda row: row['id'])

# A fresh client also discards any interrupted HTTP response from prior staging.
storage = storage_namespace['Drive']()
input_rows = storage.children(storage_namespace['ROOT_ID'])
input_index = {part['name']: select_input_part(part,input_rows)
               for spec in transport['assets'].values() for part in spec['parts']}
(ROOT/'input_resolution.json').write_text(json.dumps(input_index,sort_keys=True,indent=2))
print('Resolved immutable inputs by manifest size and consistent server checksum; SHA256 verification retained.',flush=True)
# Each worker owns its Google API client; httplib2 transports are not shared.
download_pool = ThreadPoolExecutor(max_workers=4)
part_futures = {}
def download_part(part):
    cache=ROOT/'input_cache';cache.mkdir(exist_ok=True)
    target=cache/part['name']
    client=storage_namespace['client']()
    client.cache[(part['name'],)]=input_index[part['name']]
    client.download('hfh_storage:'+part['name'],target)
    if target.stat().st_size != part['bytes'] or sha(target) != part['sha256']:
        raise ValueError('Immutable input part SHA256 differs: '+part['name'])
    return target

def prefetch_asset(name):
    for part in transport['assets'][name]['parts']:
        part_futures[part['name']] = download_pool.submit(download_part,part)

def fetch_part(part):
    return part_futures.pop(part['name']).result()

try:
    source = STORAGE_HOME/'hfh_exploration_bundle.zip'
    local = ROOT/source.name
    prefetch_asset(source.name)
    assemble_asset(source.name, STORAGE_HOME, local, transport, fetch_part=fetch_part)
    with zipfile.ZipFile(local) as archive:
        for member in archive.infolist():
            name = Path(member.filename)
            if name.is_absolute() or '..' in name.parts or (member.external_attr>>16)&0o170000 == 0o120000:
                raise ValueError('Unsafe archive member')
        archive.extractall(ROOT)
    if sha(BUNDLE/'manifest.json') != MANIFEST_SHA:
        raise ValueError('Manifest changed')
    for name, expected in json.loads((BUNDLE/'manifest.json').read_text()).items():
        if sha(BUNDLE/name) != expected:
            raise ValueError('Bundle changed: '+name)
    if KIND == 'medsam3':
        prefetch_asset('sam3.pt')
        assemble_asset('sam3.pt', STORAGE_HOME, ROOT/'sam3.pt', transport, fetch_part=fetch_part)
    print('Local SSD inputs verified. No model training has started yet.', flush=True)
finally:
    download_pool.shutdown(wait=True, cancel_futures=True)

'''
    notebook['cells'][2]['source'] = stage
    setup = ''.join(notebook['cells'][3]['source'])
    start = setup.index('# Archive the immutable input package as complete source/split provenance.')
    end = setup.index('# HFH compact MedSAM storage overlay', start)
    setup = setup[:start]+'''# Archive the immutable input package as complete source/split provenance.
# Inputs already have immutable verified source provenance on second-account Drive.
shutil.copyfile(ROOT/'setup.log',RESULTS/'setup.log')
for path in sorted(RESULTS.rglob('*')):
    if path.is_file():
        storage.upload(path,'hfh_storage:results_medsam3_constraints_strict_v1/bootstrap/'+str(path.relative_to(RESULTS)))
print('SETUP COMPLETE; metadata verified on second-account Drive.',flush=True)

''' + setup[end:]
    notebook['cells'][3]['source'] = setup
