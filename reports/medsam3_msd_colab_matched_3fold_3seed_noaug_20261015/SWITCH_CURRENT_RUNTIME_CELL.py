# Paste this complete cell in the existing Colab notebook after uploading medsam3_msd_matched_code.zip.
# Preserves the current A100 session and all earlier results; does not launch training.

from google.colab import drive, auth
drive.mount('/content/drive')
auth.authenticate_user()
from pathlib import Path
import hashlib, json, os, shutil, signal, subprocess, sys, zipfile
HOME_DIR = Path('/content/drive/MyDrive/hippo_medsam3_msd_extension')
INPUT = HOME_DIR/'input'
RESULTS = HOME_DIR/'results_matched200_noaug_v2'
BUNDLE = Path('/content/hippo_medsam3_msd_matched_bundle')
PREVIOUS_RESULTS = HOME_DIR/'results_original200_noaug_v1'
PREVIOUS_BUNDLE = Path('/content/hippo_medsam3_msd_bundle')
RESULTS.mkdir(parents=True, exist_ok=True)


EXPECTED_UPLOADS = {'medsam3_msd_matched_code.zip': '1a6a0130083c90632f3ba392c67e9bb9940908e84ff4f4d6cfb86714f61d7239', 'medsam3_msd_data.zip': 'ad869cf8c8862697b41bda740e91cae91bea6fcfdadca364b6f3ec5e6acf95e4', 'sam3.pt': '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'}
EXPECTED_MANIFEST = '39788851ab86a8df615031a2f6fe460be09cc045c666944fb4ebe74481b74654'
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()
def checked_copy(source, target, expected):
    if target.exists() and sha256(target) == expected: return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.copying')
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected: raise ValueError('Copy failed verification: '+str(target))
    temporary.replace(target)
def safe_extract(path, destination):
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = Path(item.filename)
            if name.is_absolute() or '..' in name.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe ZIP member')
        archive.extractall(destination)
for name, expected in EXPECTED_UPLOADS.items():
    path = INPUT/name
    if not path.exists(): raise FileNotFoundError('Upload '+name+' into '+str(INPUT))
    if sha256(path) != expected: raise ValueError('Upload hash mismatch: '+name)
    print('Verified', name, flush=True)
if shutil.disk_usage('/content').free < 16*1024**3:
    raise RuntimeError('Need 16 GiB free on the runtime disk before staging')
if (BUNDLE/'bundle_manifest.json').exists() and sha256(BUNDLE/'bundle_manifest.json') != EXPECTED_MANIFEST:
    raise RuntimeError('Existing matched bundle differs; preserve its work before replacing it.')
BUNDLE.mkdir(exist_ok=True)
safe_extract(INPUT/'medsam3_msd_matched_code.zip', BUNDLE)
safe_extract(INPUT/'medsam3_msd_data.zip', BUNDLE)
base = BUNDLE/'weights/sam3.pt'
previous_base = PREVIOUS_BUNDLE/'weights/sam3.pt'
if not base.exists() and previous_base.exists() and sha256(previous_base) == EXPECTED_UPLOADS['sam3.pt']:
    os.link(previous_base, base)
checked_copy(INPUT/'sam3.pt', base, EXPECTED_UPLOADS['sam3.pt'])
if sha256(BUNDLE/'bundle_manifest.json') != EXPECTED_MANIFEST: raise ValueError('Bundle manifest differs')
(RESULTS/'source_binding.json').write_text(json.dumps(dict(uploads=EXPECTED_UPLOADS, manifest=EXPECTED_MANIFEST), indent=2))
print('Staged on the local SSD. No training has started.')


VENV = Path('/content/hippo_medsam3_venv')
PYTHON = VENV/'bin/python'
def setup(args, log_name):
    result = subprocess.run(list(map(str,args)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (RESULTS/log_name).write_text(result.stdout)
    print('\n'.join(result.stdout.splitlines()[-35:]))
    if result.returncode: raise RuntimeError(f'Command failed; full output: {RESULTS/log_name}')
if not PYTHON.exists():
    setup([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV], 'venv_setup.log')
probe = subprocess.run([str(PYTHON),'-m','pip','--version'], capture_output=True, text=True)
if probe.returncode:
    import urllib.request
    bootstrap = Path('/content/medsam3_get_pip.py')
    urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', bootstrap)
    setup([PYTHON,bootstrap], 'pip_bootstrap.log')
setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'], 'dependencies.log')
def campaign(command, *extra):
    args = [str(PYTHON),str(BUNDLE/'runtime.py'),command,'--bundle',str(BUNDLE),'--durable',str(RESULTS),*map(str,extra)]
    process = subprocess.Popen(args, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    log_path = RESULTS/(command+'.log')
    try:
        with log_path.open('a') as log:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line);log.flush()
        code = process.wait()
    except KeyboardInterrupt:
        print('Requesting clean stop. Wait for recovery upload to finish.')
        os.killpg(process.pid, signal.SIGINT);process.wait();return
    if code: raise RuntimeError(f'{command} failed. Details are printed above and saved in {log_path}')
campaign('verify')
subprocess.run(['nvidia-smi'],check=True)


campaign('reuse-preflight', '--previous-results', PREVIOUS_RESULTS, '--previous-bundle', PREVIOUS_BUNDLE)

MAX_NEW_RUNS = 126
SESSION_HOURS = 22.0
print('Matched study prepared. Next run campaign("preflight"), then campaign("run", "--max-runs", 126, "--hours", 22).')
