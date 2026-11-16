"""Freeze HFH crops, inherited recipes, source, and two Colab notebooks."""
import os
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform
from scripts.colab_hfh.runtime import read, write, digest
from scripts.colab_hfh.policy import MODELS

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/hfh_colab_exploration_20261111'
ADNI = ROOT/'reports/adni_colab_seeds014_drive_optimized_20261018/package/seed4_bundle/pilot_protocols'
MSD = ROOT/'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle'
LOCAL = Path('/content/hippo_hfh')


def crop(image, label, spacing, value):
    points = np.argwhere(label == value)
    lower, upper = points.min(0), points.max(0)
    centre = (lower+upper)/2
    shape = np.array([64, 32, 64])
    target_spacing = np.array([1., 2., 1.])
    scale = target_spacing/spacing
    # Align slice indices to integer source slices: no interpolation through 2-mm gaps.
    origin = centre-(shape-1)*scale/2
    origin[1] = np.floor(origin[1])
    if np.any(lower < origin) or np.any(upper > origin+(shape-1)*scale):
        raise ValueError('Foreground does not fit predeclared 64 mm field of view')
    x = affine_transform(image.astype(np.float32), np.diag(scale), offset=origin,
        output_shape=tuple(shape), order=1, mode='constant', cval=0, prefilter=False)
    y = affine_transform((label == value).astype(np.uint8), np.diag(scale), offset=origin,
        output_shape=tuple(shape), order=0, mode='constant', cval=0, prefilter=False)
    if not np.isfinite(x).all() or y.sum() == 0:
        raise ValueError('Invalid resampled crop')
    return x, y, dict(source_origin_voxels=origin.tolist(), output_to_source_scale=scale.tolist(),
        source_label=value, source_bbox=[lower.tolist(), upper.tolist()], spacing_mm=target_spacing.tolist(),
        shape=shape.tolist(), native_foreground_voxels=len(points), resampled_foreground_voxels=int(y.sum()))


def prepare_data(bundle):
    source = ROOT/'datasets/Dataset104_HFH'
    inventory = read(ROOT/'reports/hfh_dataset_20261111/inventory.json')
    subjects = {r['case']:r for r in inventory['cases'] if r['official_partition'] == 'Train'}
    rows = []
    (bundle/'data').mkdir(parents=True, exist_ok=True)
    for case, row in sorted(subjects.items()):
        index = case.rsplit('_', 1)[1]
        image = nib.load(source/f'raw/Train/HFH_{index}.hdr')
        label = nib.load(source/f'raw/Train/Labels/HFH_{index}_Hipp_Labels.hdr')
        x, y = np.asarray(image.dataobj).squeeze(), np.asarray(label.dataobj).squeeze()
        if x.shape != y.shape or not np.allclose(image.affine, label.affine):
            raise ValueError('Source geometry differs')
        for side in (1, 2):
            xx, yy, geometry = crop(x, y, np.array(image.header.get_zooms()[:3]), side)
            name = case+f'_h{side}'
            path = bundle/'data'/(name+'.npz')
            np.savez_compressed(path, image=xx, label=yy)
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved['image'], xx) or not np.array_equal(saved['label'], yy):
                    raise ValueError('Crop serialization mismatch')
            rows.append(dict(case=name, subject=case, sha256=digest(path), **geometry,
                acquisition='fine' if x.shape[0] == 512 else 'coarse',
                source_sha256=row['source_sha256']))
    folds = read(source/'splits_final.json')
    selections = []
    for fold in range(3):
        selection = []
        for fine, count in ((False, 3), (True, 2)):
            available = sorted(c for c in folds[fold]['train'] if (subjects[c]['shape'][0] == 512) == fine)
            selection += np.random.RandomState(0).permutation(available)[:count].tolist()
        val = sorted(folds[fold]['val'])
        if len(selection) != 5 or len(val) != 5 or set(selection)&set(val):
            raise ValueError('Subject split mismatch')
        selections.append(dict(train=[s+f'_h{i}' for s in sorted(selection) for i in (1, 2)],
            val=[s+f'_h{i}' for s in val for i in (1, 2)], train_subjects=sorted(selection), val_subjects=val))
    write(bundle/'data_audit.json', dict(cases=rows, selections=selections,
        labelled_subjects=25, labelled_crops=50, official_unlabelled_test_subjects_excluded=25,
        source_archive_sha256=inventory['archive_sha256'],
        selection='Fixed subset RNG0 within the existing RNG42 stratified subject folds; 3 coarse + 2 fine subjects',
        localization='Ground-truth bounding-box centred individual hippocampi: oracle-localized exploratory benchmark',
        laterality='h1/h2 are source label IDs, not verified anatomical left/right',
        metric='Mean foreground Dice on the resampled crop grid; physical native-grid Dice is not reported',
        slice_axis='Source axis1, preserved 2 mm; MedSAM3 transposes X,Y,Z to X,Z,Y'))
    write(bundle/'saved_subject_folds.json', folds)
    write(bundle/'data_index.json', dict(data=str(LOCAL/'bundle/data'), cases=[r['case'] for r in rows]))
    return selections, rows


def prepare_protocols(bundle, selections, rows):
    for model in MODELS:
        origin = ADNI/model
        target = bundle/'protocols'/model
        target.mkdir(parents=True, exist_ok=True)
        p = read(origin/'protocol.json')
        alias = p['trainer_alias']
        p['models'][alias]['objectives'] = {'dice':p['models'][alias]['objectives']['dice']}
        p.update(name='HFH_localized_shared_seed_exploration_v1', dataset='HFH',
            folds=[0, 1, 2], seeds=list(range(32)), constraints=False, requested_arm_order=['dice'],
            train_count=10, validation_count=10, data=str(LOCAL/'bundle/data_index.json'),
            pkl_sha256=digest(bundle/'data_index.json'), sdf_policy={'dice':p['sdf_policy']['dice']},
            native_geometry={r['case']:dict(model_affine=np.diag([1., 2., 1., 1.]).tolist()) for r in rows},
            preprocessing='Linear in-plane resampling to 1x2x1 mm; nonzero z-score; symmetric padding64cubed; no augmentation',
            geometry='Source axis order retained; label-localized 64 mm field of view; no verified anatomical laterality',
            label_budget='Five training subjects of20 (25%); ten hippocampus crops; fixed across optimizer seeds and models',
            requested_fraction=.25, checkpoint_retention='Verify metrics and provenance, then retire completed checkpoints',
            calibration_policy='No constraints or calibration in this exploratory notebook',
            profile_status='Inherited measured binary64cubed dense arithmetic, same architecture/loss/optimizer',
            parent_protocol_sha256=digest(origin/'protocol.json'))
        for stale in ('baseline_reference','baseline_run_id','split_provenance_sha256'):
            p.pop(stale, None)
        write(target/'splits_10patients.json', [{k:s[k] for k in ('train','val')} for s in selections])
        shutil.copyfile(origin/'nnunet_architecture.json', target/'nnunet_architecture.json')
        p['splits_sha256'] = digest(target/'splits_10patients.json')
        p['nnunet_architecture_sha256'] = digest(target/'nnunet_architecture.json')
        write(target/'protocol.json', p)
    parent = ROOT/'reports/medsam3_mni_colab_two_subjects_20261108/bundle/protocol.json'
    for fold, selection in enumerate(selections):
        for seed in range(3):
            p = read(parent)
            for key in ('subject_order','decision_rule'):
                p.pop(key, None)
            p.update(dataset='HFH', name=f'HFH_MedSAM3_native200_f{fold}s{seed}', fold=fold, seed=seed,
                classes=['background','hippocampus'], prompts={'1':'hippocampus'},
                data=str(LOCAL/'bundle/data'), upstream=str(LOCAL/'bundle/upstream'),
                lora_weights=str(LOCAL/'bundle/weights/best_lora_weights.pt'),
                train_subjects=5, train_crops=10, support_subjects=selection['train_subjects'],
                train_cases=selection['train'], evaluation_cases=selection['val'],
                support_selection='Identical frozen five subjects as all conventional models; acquisition-stratified',
                dataset_sha256=digest(bundle/'data_audit.json'), primary_metric='Final mean foreground Dice on localized resampled crops',
                inference='Fixed hippocampus text prompt, no label-derived inference prompt; oracle crop localization is disclosed',
                geometry_units='1x1 mm in-plane, 2 mm between slices; X,Z,Y order',
                baseline_loss='Original SAM3 native detection/presence/mask objective; train medical LoRA only')
            write(bundle/'medsam_protocols'/f'f{fold}_s{seed}.json', p)


def freeze_source(bundle):
    # Source only: never copy datasets, credentials, outputs or model checkpoints.
    for folder in ('scripts', 'evaluation', 'thesis', 'baselines', 'semantic_constraints'):
        for source in (ROOT/folder).rglob('*.py'):
            if any(part.startswith('.') or part in ('__pycache__','site-packages') for part in source.relative_to(ROOT).parts):
                continue
            target = bundle/'source'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    # Published upstream assets and its compatibility shim are frozen with the old MSD package.
    for folder in ('upstream', 'compatibility'):
        for source in (MSD/folder).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts or '.git' in source.parts:
                continue
            if source.suffix in ('.pt','.pth','.ckpt','.pyc'):
                continue
            target = bundle/folder/source.relative_to(MSD/folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    (bundle/'weights').mkdir(exist_ok=True)
    weight = ROOT/'MedSam3_weights/best_lora_weights.pt'
    target = bundle/'weights/best_lora_weights.pt'
    if not target.exists():
        import os
        os.link(weight, target)
    medsam_requirements = (MSD/'requirements-colab.txt').read_text().splitlines()
    baseline_requirements = (ROOT/'reports/adni_colab_3fold_3seed_noaug_20261015/bundle/requirements-colab.txt').read_text().splitlines()
    pins = {}
    for line in medsam_requirements+baseline_requirements:
        if not line.strip() or line.startswith('#'):
            continue
        name = line.split('==')[0]
        if name in pins and pins[name] != line:
            raise ValueError('Historical dependency pins conflict: '+name)
        pins[name] = line
    requirements = '\n'.join(pins.values())+'\n'
    (bundle/'requirements-colab.txt').write_text(requirements)


def notebook(bundle, package, kind):
    cells = []
    def add(kind, source):
        source = textwrap.dedent(source).strip()+'\n'
        if kind == 'code': ast.parse(source)
        cell = dict(cell_type=kind, metadata={}, id=f'hfh_{len(cells):02}', source=source)
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    detail = ('Search optimizer seeds0–31 in ascending order, stopping after **three shared qualifying seeds**. '
        'Each must reach foreground Dice≥50% at the final FLOP budget in **all12 model/fold cells**. '
        'A completed cell below50% immediately rejects that seed and skips its remaining cells. '
        'Runtime errors pause for inspection; they are not bad seeds. Intermediate validation never rejects a seed.'
        if kind == 'baselines' else 'Run **all9 cells: seeds0,1,2 × folds0,1,2**, exactly200updates per cell. '
        'Report which seeds exceed50% in all folds; no extension to400 and no automatic seed replacement. '
        'Medical-pretrained LoRA, frozen SAM3 base, native detection/presence/mask loss, AdamW5e-5→5e-6, '
        'warmup20, accumulation2 of adjacent two-slice slabs (four slices/update), weight decay.01, clip1, no augmentation.')
    title = 'HFH — shared seeds for four baselines' if kind == 'baselines' else 'HFH — MedSAM3, 200 updates, 3 folds × 3 seeds'
    add('markdown', f'''# {title}

**Compute:** stay signed into Colab as the configured compute account (your compute units). **Storage:** the configured storage account owns the inputs, results, and recovery files. Select **A100 GPU and High-RAM**. Run cells in order.

The first cell requests a separate storage authorization JSON file (or the Colab secret `HFH_SECOND_DRIVE_TOKEN`). Run the supplied local `authorize_storage.py` helper once and choose the second account in Google’s consent page. Select the private token file when Colab prompts; never put credentials in notebook code, outputs, Drive, or chat. The login does not change Colab billing.

On the first run, immutable inputs are copied from the third account into the second account and checksummed (about 3.6 GB). Subsequent runs reuse that copy. The old inputs are retained. New results and recovery go only to the second account. Startup verifies account identity, folder ownership, quota, and an independently read-back remote test write. Existing third-account results cause a stop for recovery migration rather than a fresh run.

{detail}

Five fixed training subjects (3 coarse+2 fine), ten crops per fold; five separate validation subjects, ten crops. Subject folds0,1,2 are frozen before training. No augmentation. Official25 test scans have no labels and are excluded.

**Oracle-localized benchmark:** each hippocampus crop is centred using its annotation. Resampled spacing1×2×1mm, shape64×32×64; conventional models pad to64³. MedSAM3 slices along the original thick axis. Primary metric is mean per-crop foreground Dice on this resampled grid, not native full-brain inference. Binary HFH has no anterior/posterior subfield swaps.

This is exploratory validation-based seed selection. Keep and report all failures; the selected scores are not unbiased test estimates.

Data/training run on VM SSD. One worker preserves immediate seed rejection and limits memory/storage; this is intentionally more conservative than the historical multi-worker sweeps. BF16 and the original batch sizes stay unchanged. Baselines keep normalized input tensors on GPU after exact-copy checks, and avoid rewriting unchanged best checkpoints. MedSAM3 reuses deterministic CPU transform caching and numerically audited inference caching with fallback. Immutable recovery uploads every180seconds, plus clean-stop flush. Baseline unclean VM loss fails closed if charged work cannot be reconstructed. MedSAM3 abrupt loss can replay work since its last snapshot;200 is successful updates, not total attempted FLOPs.

Default session8hours; stop with one interrupt and wait for recovery. The stated500compute units are not a fixed GPU-hour allowance: monitor Colab's unit balance. These notebooks do not automatically buy compute or increase the training budget. Run one VM per notebook, and do not change GPU/software within a campaign.
''')
    from scripts.colab_hfh.storage import notebook_cell
    add('code', notebook_cell(kind))
    add('code', f'''
    PACKAGE_SHA={digest(package)!r}
    MANIFEST_SHA={digest(bundle/'manifest.json')!r}
    source=HOME/'hfh_exploration_bundle.zip'
    local=ROOT/source.name
    if not local.exists():shutil.copyfile(source,local)
    if sha(local)!=PACKAGE_SHA:raise ValueError('Package checksum mismatch')
    with zipfile.ZipFile(local) as archive:
        for member in archive.infolist():
            name=Path(member.filename)
            if name.is_absolute() or '..' in name.parts or (member.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe archive member')
        archive.extractall(ROOT)
    if sha(BUNDLE/'manifest.json')!=MANIFEST_SHA:raise ValueError('Manifest changed')
    for name,expected in json.loads((BUNDLE/'manifest.json').read_text()).items():
        if sha(BUNDLE/name)!=expected:raise ValueError('Bundle changed: '+name)
    if KIND=='medsam3':
        base=ROOT/'sam3.pt'
        if not base.exists():shutil.copyfile(HOME/'sam3.pt',base)
        expected='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
        if sha(base)!=expected:raise ValueError('SAM3 base checksum mismatch')
    print('Verified frozen source,50labelled crops,folds,recipes and required weights')
    ''')
    add('code', '''
    VENV=ROOT/'venv'
    PYTHON=VENV/'bin/python'
    def setup(command):
        p=subprocess.run(list(map(str,command)),text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            cwd=BUNDLE/'source',env=globals().get('env'))
        with (ROOT/'setup.log').open('a') as log:log.write(p.stdout)
        shutil.copyfile(ROOT/'setup.log',RESULTS/'setup.log')
        if sha(ROOT/'setup.log')!=sha(RESULTS/'setup.log'):raise IOError('Setup log upload failed')
        print('\\n'.join(p.stdout.splitlines()[-20:]))
        if p.returncode:raise RuntimeError('Setup failed; see setup.log')
    if not PYTHON.exists():setup([sys.executable,'-m','venv','--without-pip','--system-site-packages',VENV])
    if subprocess.run([str(PYTHON),'-m','pip','--version'],capture_output=True).returncode:
        import urllib.request
        bootstrap=ROOT/'get-pip.py'
        urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py',bootstrap)
        setup([PYTHON,bootstrap])
    setup([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',BUNDLE/'requirements-colab.txt'])
    env=dict(os.environ,PYTHONPATH=':'.join(map(str,[BUNDLE/'compatibility',BUNDLE/'source',BUNDLE/'source/scripts',BUNDLE/'source/baselines/nnUNet'])),
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',
        MPLCONFIGDIR=str(ROOT/'mpl'),MNI_MANIFEST_SHA=MANIFEST_SHA,HIPPO_HFH_GPU_CACHE='1')
    probe=subprocess.run([str(PYTHON),'-c',
        "import json;from scripts.colab_adni.runtime import environment;from scripts.colab_medsam3.runtime import environment as med;print(json.dumps(med() if 'medsam3'=='"+KIND+"' else environment()))"],
        env=env,cwd=BUNDLE/'source',capture_output=True,text=True)
    if probe.returncode:raise RuntimeError(probe.stderr)
    binding=dict(manifest_sha256=MANIFEST_SHA,environment=json.loads(probe.stdout.strip().splitlines()[-1]))
    binding_path=RESULTS/'environment_binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:
        raise ValueError('Software/GPU changed; do not mix environments in this campaign')
    binding_path.write_text(json.dumps(binding,indent=2))
    print(binding)
    # Archive the immutable input package as complete source/split provenance.
    provenance=RESULTS/'provenance'
    provenance.mkdir(exist_ok=True)
    dest=provenance/source.name
    if not dest.exists():shutil.copyfile(local,dest)
    if sha(dest)!=PACKAGE_SHA:raise ValueError('Durable provenance mismatch')
    if KIND=='baselines':
        setup([PYTHON,'-m','scripts.colab_hfh.preflight','--bundle',BUNDLE,'--output',RESULTS/'gpu_preflight.json'])
    ''')
    add('code', '''
    # Finite session; the controller resumes verified completed/recovered cells.
    command=[str(PYTHON),'-m','scripts.colab_hfh.search','--kind',KIND,'--bundle',str(BUNDLE),
        '--durable',str(RESULTS),'--work',str(ROOT/('work_'+KIND)),
        '--hours',str(SESSION_HOURS),'--candidate-count',str(CANDIDATE_COUNT)]
    process=subprocess.Popen(command,env=env,cwd=BUNDLE/'source',stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (ROOT/(KIND+'_controller.log')).open('a') as log:
        while True:
            try:
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait();break
            except KeyboardInterrupt:
                if process.poll() is None:process.send_signal(signal.SIGINT)
                print('Clean stop requested. Wait for verified recovery before disconnecting.')
    shutil.copyfile(ROOT/(KIND+'_controller.log'),RESULTS/'controller.log')
    if code:raise RuntimeError('Campaign paused by an error; inspect logs and retain recovery')
    print(json.loads((RESULTS/'search_status.json').read_text())['status'])
    ''')
    add('code', '''
    import pandas as pd
    import matplotlib.pyplot as plt
    status=json.loads((RESULTS/'search_status.json').read_text())
    table=pd.DataFrame(status['results'])
    print('Status:',status['status'],'Qualifying seeds:',status['accepted_seeds'])
    if not table.empty:
        display(table.pivot(index=['seed','fold'],columns='model',values='validation_dice')*100)
        table.to_csv(RESULTS/'all_attempted_cells.csv',index=False)
        ax=table.pivot(index=['seed','fold'],columns='model',values='validation_dice').plot.bar(figsize=(14,5),ylim=(0,1))
        ax.axhline(.5,color='red',linestyle='--');ax.set_ylabel('Final validation Dice')
        plt.tight_layout();plt.savefig(RESULTS/'exploration.png',dpi=150);plt.show()
    print('Report every attempted seed. Selected validation scores are exploratory.')
    ''')
    name = 'HFH_baseline_shared_seed_search.ipynb' if kind == 'baselines' else 'HFH_MedSAM3_200_3fold_3seed.ipynb'
    write(EXPORT/name, dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(accelerator='GPU',
        colab=dict(name=name,gpuType='A100',gpuClass='premium',machine_shape='hm'),
        kernelspec=dict(display_name='Python3',language='python',name='python3'))))


def build():
    bundle = EXPORT/'bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    if not (bundle/'data_audit.json').exists():
        selections, rows = prepare_data(bundle)
    else:
        audit = read(bundle/'data_audit.json')
        selections, rows = audit['selections'], audit['cases']
    prepare_protocols(bundle, selections, rows)
    freeze_source(bundle)
    write(bundle/'manifest.json', {str(p.relative_to(bundle)):digest(p) for p in sorted(bundle.rglob('*'))
        if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts})
    package = EXPORT/'hfh_exploration_bundle.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in [*read(bundle/'manifest.json'), 'manifest.json']:
            archive.write(bundle/name, 'bundle/'+name)
    for kind in ('baselines', 'medsam3'):
        notebook(bundle, package, kind)
    print(json.dumps(dict(export=str(EXPORT), bytes=package.stat().st_size,
        package_sha256=digest(package), crops=len(rows), folds=selections), indent=2))


if __name__ == '__main__':
    build()
