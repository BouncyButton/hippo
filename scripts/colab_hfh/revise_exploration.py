"""Build the versioned HFH baseline recipe and faster notebook bootstrap."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import zipfile

from scripts.colab_hfh.transport import assemble_asset

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/hfh_colab_exploration_20261111'
REVISION = EXPORT/'baseline_v2'
CAMPAIGN = 'v2_cap_schedule'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch anchor changed: '+old[:80])
    return text.replace(old, new)


def patch_core(text):
    text = replace_once(text, '    def commit():\n        payload =', '''    checkpoint_interval = float(protocol.get('checkpoint_interval_seconds', 0))
    last_checkpoint_at = time.monotonic()
    checkpoint_writes, checkpoint_seconds = 0, 0.

    def commit(force=True):
        nonlocal last_checkpoint_at, checkpoint_writes, checkpoint_seconds
        # Preserve small diagnostics at each validation without serializing the
        # full model, optimizer and best model on every fifth optimizer update.
        if not force and time.monotonic()-last_checkpoint_at < checkpoint_interval:
            if latest_validation is not None:
                base.atomic_json(args.output / "validation_latest.json", latest_validation)
            if history:
                base.write_metrics(args.output / "metrics.csv", history)
            return
        checkpoint_started = time.monotonic()
        payload =''')
    text = replace_once(text, '    def validate(lr):', '''        last_checkpoint_at = time.monotonic()
        checkpoint_writes += 1
        checkpoint_seconds += last_checkpoint_at-checkpoint_started
        base.atomic_json(args.output / 'checkpoint_io.json', dict(
            writes=checkpoint_writes, seconds=checkpoint_seconds,
            interval_seconds=checkpoint_interval))

    def validate(lr):''')
    text = replace_once(text, '        losses = []\n        commit()\n',
        '        losses = []\n        commit(force=ledger["charged_attempts"] == max_steps)\n')
    text = replace_once(text, '        elif ledger["charged_attempts"] == schedule["prefix_updates"] or step % 5 == 0:\n            commit()',
        '        elif ledger["charged_attempts"] == schedule["prefix_updates"] or step % 5 == 0:\n            commit(force=False)')
    ast.parse(text)
    return text


def build_bundle():
    bundle = REVISION/'bundle'
    if bundle.exists():
        raise ValueError('Revision already built; retain immutable bundle')
    shutil.copytree(EXPORT/'bundle', bundle,
        ignore=shutil.ignore_patterns('weights', 'upstream', 'medsam_protocols', '__pycache__', '*.pyc'))
    # Retain scientific source from the executed v1, changing only the explicit
    # controller/checkpoint patches and LR protocol below.
    source = bundle/'source'
    shutil.copyfile(ROOT/'scripts/colab_hfh/search.py', source/'scripts/colab_hfh/search.py')
    policy = source/'scripts/colab_hfh/policy.py'
    policy.write_text(replace_once(policy.read_text(),
        "MODELS = ('basic_unet16', 'nnunet', 'unetr_vit_b16', 'swinunetr')",
        "MODELS = ('nnunet', 'unetr_vit_b16', 'basic_unet16', 'swinunetr')"))
    core = source/'scripts/cross_dataset_campaign/capped_core.py'
    core.write_text(patch_core(core.read_text()))
    # Give v2 a separate VM directory as well as a separate Drive campaign.
    index = bundle/'data_index.json'
    index.write_text(index.read_text().replace('/content/hippo_hfh/', '/content/hippo_hfh_baselines_v2/'))
    changes = []
    for file in sorted((bundle/'protocols').glob('*/protocol.json')):
        p = json.loads(file.read_text())
        old = sha(file)
        original_horizon = p['lr_horizon_updates']
        updates = p['models'][p['trainer_alias']]['objectives']['dice']['max_steps']
        p['lr_horizon_updates'] = updates
        p['checkpoint_interval_seconds'] = 60
        p['learning_rate_policy'] = 'Original model-specific LR formula, horizon fitted to the unchanged HFH update cap'
        p['campaign_revision'] = CAMPAIGN
        p['hfh_v1_protocol_sha256'] = old
        p['hfh_v1_lr_horizon_updates'] = original_horizon
        p['pkl_sha256'] = sha(index)
        p = json.loads(json.dumps(p).replace('/content/hippo_hfh/', '/content/hippo_hfh_baselines_v2/'))
        write(file, p)
        changes.append(dict(model=p['model_identity'], previous_horizon=original_horizon,
                            revised_horizon=updates, max_updates=updates, budget_flops=p['budget_flops']))
    # Baseline imports do not need the MedSAM installation stack.
    keep = {'torch', 'numpy', 'scipy', 'nibabel', 'einops', 'matplotlib', 'scikit-image',
            'scikit-learn', 'pandas', 'psutil', 'tqdm', 'monai',
            'dynamic-network-architectures', 'batchgenerators', 'batchgeneratorsv2', 'acvl-utils'}
    req = bundle/'requirements-colab.txt'
    req.write_text('\n'.join(line for line in req.read_text().splitlines()
                            if line.split('==')[0] in keep)+'\n')
    write(bundle/'revision.json', dict(campaign=CAMPAIGN, changes=changes,
        scientific_change='Only learning-rate horizon; same loss, optimizer, initial LR, cap, data, folds, augmentation, precision',
        runtime_changes=['Baseline-only input/dependency package', '60-second local checkpoints plus forced final/clean-stop',
                         '2-second process-exit wait', 'Persist seed decisions and error status', 'nnU-Net and UNETR first'],
        selection='Three shared seeds >=0.5 final Dice across all four models and all three folds; reject on first completed failing cell',
        parent_manifest_sha256=sha(EXPORT/'bundle/manifest.json')))
    write(bundle/'manifest.json', {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob('*'))
                                  if p.is_file() and p.name != 'manifest.json'})
    package = REVISION/'hfh_baselines_v2_bundle.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                archive.write(p, 'bundle/'+str(p.relative_to(bundle)))
    transport = dict(version=2, campaign=CAMPAIGN, assets={package.name:dict(
        bytes=package.stat().st_size, sha256=sha(package), parts=[dict(name=package.name,
        bytes=package.stat().st_size, sha256=sha(package), offset=0)])})
    write(REVISION/'transport_baselines_v2.json', transport)
    return bundle, package


def staging_cell(package, manifest_sha, transport_name, transport_sha):
    return f'''TRANSPORT_SHA = {transport_sha!r}
PACKAGE_SHA = {sha(package)!r}
MANIFEST_SHA = {manifest_sha!r}
if sha(HOME/{transport_name!r}) != TRANSPORT_SHA:
    raise ValueError('Transport manifest differs')
transport = json.loads((HOME/{transport_name!r}).read_text())
''' + inspect.getsource(assemble_asset) + f'''
def fetch_part(part):
    cache = ROOT/'input_cache'
    cache.mkdir(exist_ok=True)
    target = cache/part['name']
    if not target.exists():
        temporary = target.with_name(target.name+'.downloading')
        helper = Path('/content/.hfh_storage_auth')
        result = subprocess.run([str(helper/'rclone'), '--config', str(helper/'rclone.conf'),
            'copyto', 'hfh_storage:/'+part['name'], str(temporary), '--retries', '3'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        if result.returncode:
            raise RuntimeError('Input download failed: '+part['name']+'; rerun this cell to retry.')
        temporary.replace(target)
    return target

source = HOME/{package.name!r}
local = ROOT/source.name
assemble_asset(source.name, HOME, local, transport, fetch_part=fetch_part)
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
    assemble_asset('sam3.pt', HOME, ROOT/'sam3.pt', transport, fetch_part=fetch_part)
print('Local SSD inputs verified. No model training has started yet.', flush=True)
'''


def build_notebooks():
    bundle, package = REVISION/'bundle', REVISION/'hfh_baselines_v2_bundle.zip'
    for kind, name in [('baselines', 'HFH_baseline_shared_seed_search.ipynb'),
                       ('medsam3', 'HFH_MedSAM3_200_3fold_3seed.ipynb')]:
        # Always rebuild from the captured pre-revision notebook.
        notebook = json.loads((EXPORT/'before_seed_recipe_v2'/name).read_text())
        baseline = kind == 'baselines'
        selected_package = package if baseline else EXPORT/'hfh_exploration_bundle.zip'
        selected_bundle = bundle if baseline else EXPORT/'bundle'
        selected_transport = REVISION/'transport_baselines_v2.json' if baseline else EXPORT/'transport_manifest.json'
        campaign = CAMPAIGN if baseline else 'v1'
        runtime_root = '/content/hippo_hfh_baselines_v2' if baseline else '/content/hippo_hfh'
        notebook['cells'][1]['source'] = (ROOT/'scripts/colab_hfh/storage.py').read_text()+f'''
KIND = {kind!r}
ROOT = Path({runtime_root!r})
ROOT.mkdir(exist_ok=True)
BUNDLE = ROOT/'bundle'
HOME, RESULTS = storage_bootstrap(KIND, campaign={campaign!r},
    transport_name={selected_transport.name!r}, transport_sha={sha(selected_transport)!r})
SESSION_HOURS = 8
CANDIDATE_COUNT = 32
'''
        notebook['cells'][2]['source'] = staging_cell(selected_package, sha(selected_bundle/'manifest.json'),
            selected_transport.name, sha(selected_transport))
        # Save logs locally during dependency installation. Publish once afterward.
        setup = notebook['cells'][3]['source']
        setup = setup.replace("    shutil.copyfile(ROOT/'setup.log',RESULTS/'setup.log')\n    if sha(ROOT/'setup.log')!=sha(RESULTS/'setup.log'):raise IOError('Setup log upload failed')\n", '')
        setup = setup.replace('    p=subprocess.run(', "    print('Setup:', ' '.join(map(str,command)), flush=True)\n    p=subprocess.run(")
        setup += "\nshutil.copyfile(ROOT/'setup.log', RESULTS/'setup.log')\nif sha(ROOT/'setup.log') != sha(RESULTS/'setup.log'): raise IOError('Setup log upload failed')\nprint('SETUP COMPLETE. Run the next cell to start training.', flush=True)\n"
        # Provenance package checksum is reused via a small receipt; no repeated
        # remote read of the whole immutable package on every setup-cell rerun.
        setup = replace_once(setup, "if not dest.exists():shutil.copyfile(local,dest)\nif sha(dest)!=PACKAGE_SHA:raise ValueError('Durable provenance mismatch')",
            "receipt=provenance/'package_verified.json'\nif not receipt.exists() or json.loads(receipt.read_text()).get('sha256') != PACKAGE_SHA or not dest.exists():\n    shutil.copyfile(local,dest)\n    if sha(dest)!=PACKAGE_SHA:raise ValueError('Durable provenance mismatch')\n    receipt.write_text(json.dumps(dict(sha256=PACKAGE_SHA,bytes=dest.stat().st_size)))")
        if baseline:
            setup = setup.replace("from scripts.colab_adni.runtime import environment;from scripts.colab_medsam3.runtime import environment as med;print(json.dumps(med() if 'medsam3'=='\"+KIND+\"' else environment()))",
                                  "from scripts.colab_adni.runtime import environment;print(json.dumps(environment()))")
        start, end = setup.index('probe=subprocess.run('), setup.index('# Archive the immutable input package')
        helper = (ROOT/'scripts/colab_hfh/gpu_runtime.py').read_text()
        setup = setup[:start]+f'''# Hardware portability is explicit; FP32 results never mix with BF16 results.
gpu_helper = ROOT/'gpu_runtime.py'
gpu_helper.write_text({helper!r})
probe = subprocess.run([str(PYTHON), str(gpu_helper), '--bundle', str(BUNDLE), '--kind', KIND],
    env=env, cwd=BUNDLE/'source', capture_output=True, text=True)
if probe.returncode:
    raise RuntimeError(probe.stderr)
runtime = json.loads(probe.stdout.strip().splitlines()[-1])
runtime_manifest_sha = runtime['manifest_sha256']
hardware = runtime['environment']
WORK_SUFFIX = ''
if hardware['precision'] == 'fp32':
    RESULTS = RESULTS.with_name(RESULTS.name+'_fp32')
    RESULTS.mkdir(exist_ok=True)
    WORK_SUFFIX = '_fp32'
binding = dict(manifest_sha256=MANIFEST_SHA, environment=hardware)
binding_path = RESULTS/'environment_binding.json'
if binding_path.exists():
    previous = json.loads(binding_path.read_text())
    old_env = previous['environment']
    if previous['manifest_sha256'] != MANIFEST_SHA or old_env.get('precision','bf16') != hardware['precision']:
        raise ValueError('Source package or precision changed; use a separate campaign')
    if any(hardware['packages'].get(k) != v for k,v in old_env['packages'].items()):
        raise ValueError('Scientific software versions changed within this campaign')
    if old_env.get('gpu') != hardware['gpu']:
        print('GPU changed:', old_env.get('gpu'), '->', hardware['gpu'], '; recording transition')
binding_path.write_text(json.dumps(binding, indent=2))
history = RESULTS/'environment_history'
history.mkdir(exist_ok=True)
(history/(str(time.time_ns())+'.json')).write_text(json.dumps(dict(at=time.time(), **runtime)))
env['MNI_MANIFEST_SHA'] = runtime_manifest_sha
env['HIPPO_RUNTIME_ENVIRONMENT'] = json.dumps(hardware)
print('Selected available GPU:', hardware['gpu'], '| precision:', hardware['precision'],
      '| VRAM GiB:', round(hardware['vram_bytes']/1024**3,1), flush=True)
for filename in ('gpu_overlay.json', 'runtime_manifest.json'):
    shutil.copyfile(BUNDLE/filename, RESULTS/filename)
shutil.copyfile(gpu_helper, RESULTS/'gpu_runtime.py')
''' + setup[end:]
        notebook['cells'][3]['source'] = setup
        notebook['cells'][4]['source'] = notebook['cells'][4]['source'].replace(
            "ROOT/('work_'+KIND)", "ROOT/('work_'+KIND+WORK_SUFFIX)")
        if baseline:
            notebook['cells'][0]['source'] = '''# HFH — shared seed exploration, revision 2

Compute account: the configured compute account. Inputs, results and recovery: the configured storage account. Use a BF16 GPU (L4/A100) with High-RAM and keep the same hardware/software within this campaign.

**Stop the old cell, reload this notebook in the existing tab, then run cells in order.** A running cell continues executing its old code. If interruption requests recovery, wait for it to finish.

This revision follows the failed fold-0 pilot. It changes only the scientific learning-rate horizon to fit each existing update allowance: BasicUNet 563, nnU-Net 90, UNETR 245, SwinUNETR 206. The cap remains 31,976,450,667,090 optimization FLOPs per run. Initial LRs, loss functions, optimizers, batch size, BF16, no augmentation, fixed subject splits and model architecture remain unchanged. Improvement is a hypothesis, not a guarantee. Earlier results remain in v1 and cannot qualify seeds for this revised recipe.

Five fixed training subjects (10 crops) and five disjoint validation subjects (10 crops) per fold. Oracle-localized 64×32×64 crops at 1×2×1 mm, padded to 64³. Metric: mean per-crop foreground Dice after removing padding. Official unlabeled test images are excluded.

Search seeds 0–31 in ascending order. Accept exactly **three shared seeds**, each reaching **at least 50% final Dice in all four models and all three folds**. Reject immediately after a completed failing cell; do not prune on intermediate validation. Errors pause the campaign and preserve recovery. For each fold, test nnU-Net then UNETR before BasicUNet and SwinUNETR to avoid spending on seeds that already fail the difficult models. Report all rejected seeds; these are validation-selected exploratory results.

Baseline setup downloads only its small dedicated package—no SAM3 weights or MedSAM dependencies. Required inputs are downloaded directly to VM SSD and checked there. Large local checkpoint saves run at most once per 60 seconds, plus mandatory final/clean-stop saves; remote recovery remains every 180 seconds and at clean exit. Abrupt VM loss remains fail-closed when charged work cannot be recovered exactly. Final results are verified before bulky artifacts are retired.

Results use **results_baselines_v2_cap_schedule** on the second account. Run only one baseline worker. SESSION_HOURS=8 is an upper session limit, not an eight-hour delay. Stop after three qualifying seeds or exhaustion of the 32 candidates. The notebook never buys compute or increases the scientific budget.
'''
        else:
            intro = notebook['cells'][0]['source']
            start = intro.index('On the first run, immutable inputs')
            end = intro.index('\n\n', start)
            intro = intro[:start]+('MedSAM3 needs the 3.45 GB base weights plus its bundle. Inputs are copied into second-account ownership only when missing. Each fresh VM downloads the required files directly to local SSD, verifies them during assembly, and reuses the local copy on cell reruns. Progress shows part number, MiB and elapsed time. The 200-update scientific recipe is unchanged.')+intro[end:]
            notebook['cells'][0]['source'] = intro
        notebook['cells'][0]['source'] += '''
GPU portability: use the best available NVIDIA GPU. Native BF16 is used when supported; older GPUs use an explicit FP32 campaign with a separate results/work folder. Model geometry, batch/accumulation and update limits are preserved. GPU changes and device memory are logged per experiment. GPU memory still limits which models can fit: an OOM is a runtime failure, never a rejected seed. No silent resolution/batch reduction. Baseline exploration pauses for recipe review if three successive rejected seeds fail the same model.
'''
        notebook['cells'][0]['source'] = notebook['cells'][0]['source'].replace(
            'Use a BF16 GPU (L4/A100) with High-RAM and keep the same hardware/software within this campaign.',
            'Use an available NVIDIA GPU with High-RAM. GPU identity is recorded; software stays pinned.').replace(
            'Select **A100 GPU and High-RAM**.', 'Select an available **GPU and High-RAM**.')
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(cell['source'])
                cell['outputs'] = []
                cell['execution_count'] = None
        write(EXPORT/name, notebook)
    write(REVISION/'build_receipt.json', dict(package_sha256=sha(package), package_bytes=package.stat().st_size,
        manifest_sha256=sha(bundle/'manifest.json'), transport_sha256=sha(REVISION/'transport_baselines_v2.json')))


if __name__ == '__main__':
    if not (REVISION/'bundle').exists():
        build_bundle()
    build_notebooks()
    print((REVISION/'build_receipt.json').read_text())
