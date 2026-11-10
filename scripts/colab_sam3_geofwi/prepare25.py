"""Build the requested five-volume, 25-update derivative without replacing the original export."""
from copy import deepcopy
import json
import shutil
import zipfile

from scripts.colab_sam3_geofwi import prepare, runtime

EXPORT = prepare.ROOT/'reports/sam3_geofwi_colab_25updates_20261018'
CODE = 'sam3_geofwi_25updates_code.zip'
NOTEBOOK = 'SAM3_GeoFWI3D_5vol_25updates.ipynb'


def build():
    source = prepare.EXPORT/'bundle'
    manifest = runtime.read(source/'bundle_manifest.json')
    for name, sha in manifest.items():
        if prepare.digest(source/name) != sha:
            raise ValueError('Original export changed: '+name)
    bundle = EXPORT/'bundle'
    for name in manifest:
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    # The archived original protocols remain the authority for every changed stopping point.
    shutil.copytree(source/'protocols', bundle/'reference/protocols', dirs_exist_ok=True)
    shutil.copy2(prepare.ROOT/'scripts/colab_sam3_geofwi/runtime.py', bundle/'runtime.py')
    plan = runtime.read(source/'plan.json')
    tasks = []
    for original in plan['tasks']:
        if original['updates'] != 50:
            continue
        task = deepcopy(original)
        task.update(id=task['id'].replace('_u50_', '_u25_'), updates=25,
                    protocol=task['protocol'].replace('u50_', 'u25_', 1))
        protocol = runtime.read(source/'protocols'/original['protocol']/'protocol.json')
        protocol['updates'] = 25
        protocol['sampling'] = protocol['sampling'].replace('50 updates,', '25 updates,', 1)
        prepare.write(bundle/'protocols'/task['protocol']/'protocol.json', protocol)
        tasks.append(task)
    plan.update(name='sam3_geofwi_colab_5vol_25updates_v1', tasks=tasks, num_runs=len(tasks),
                fixed_updates=25, parent_colab_manifest_sha256=prepare.digest(source/'bundle_manifest.json'))
    prepare.write(bundle/'plan.json', plan)
    runtime.verify_plan(bundle)
    manifest = {str(f.relative_to(bundle)): prepare.digest(f) for f in sorted(bundle.rglob('*'))
                if f.is_file() and f.name != 'bundle_manifest.json' and '__pycache__' not in f.parts}
    prepare.write(bundle/'bundle_manifest.json', manifest)
    code = EXPORT/CODE
    with zipfile.ZipFile(code, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*manifest, 'bundle_manifest.json']:
            archive.write(bundle/name, name)
    uploads = runtime.read(prepare.EXPORT/'uploads.json')
    uploads.pop('sam3_geofwi_code.zip')
    uploads[CODE] = prepare.digest(code)
    if prepare.digest(prepare.EXPORT/'sam3_geofwi_101vol_data.zip') != uploads['sam3_geofwi_101vol_data.zip']:
        raise ValueError('Existing data ZIP changed')
    prepare.write(EXPORT/'uploads.json', uploads)
    nb = prepare.notebook(uploads, prepare.digest(bundle/'bundle_manifest.json'))
    for cell in nb['cells']:
        text = cell['source']
        text = text.replace('sam3_geofwi_code.zip', CODE)
        text = text.replace('results_5vol_24conditions_v1', 'results_5vol_25updates_v1')
        text = text.replace('/content/hippo_sam3_geofwi_5vol_bundle', '/content/hippo_sam3_geofwi_25updates_bundle')
        text = text.replace('MAX_NEW_RUNS=24', 'MAX_NEW_RUNS=8')
        text = text.replace('All five native stopping-point controls finish before new constraint runs.',
                            'The native 25-update control finishes before new constraint runs.')
        cell['source'] = text
    nb['cells'][0]['source'] = '''# SAM3 GeoFWI3D — five volumes, 25 updates

Eight conditions on an A100/H100 (40/80 GB), High RAM Colab runtime. Every condition uses
the same five frozen training volumes, 96 validation volumes, fold 0 / seed 0, no augmentation,
fresh zero-output LoRA, original native SAM3 loss and accumulation 2.

| Conditions | Updates |
|---|---:|
| Native baseline | 25 |
| Bands 10%, 20%, 40% | 25 |
| Bands 20% + edge 0.5%, 1%, 2%, 5% | 25 |

The learning-rate horizon remains 200 updates; learning-rate and constraint warmups remain 20.
Thus full constraint strength is reached only for the last five updates. This preserves the old
25-update comparison schedule. Percentages are sequential gradient targets, with a p95 ceiling
five times each target; they are not literal loss coefficients. These runs are update-matched.

## Switch from the old sweep
Interrupt the old run once and wait for its clean-stop cell to return before running this notebook.
Keep the old results/recovery folder intact. This notebook uses a separate local bundle and
`results_5vol_25updates_v1`, so it cannot resume a longer-budget run or overwrite old results.

Upload only `sam3_geofwi_25updates_code.zip` to
`MyDrive/hippo_sam3_geofwi_exploration/input`. Reuse the existing data ZIP and SAM3 base weights.
Mount Drive as **filippofocaccia2@gmail.com**. Run cells in order, with one VM per RESULTS folder.
The old completed 25-update baseline (65.10% validation Dice) remains a reference: this separate
bundle runs a fresh matched control and checks its own calibration and GPU environment.
'''
    nb['metadata']['colab']['name'] = NOTEBOOK
    prepare.write(EXPORT/NOTEBOOK, nb)
    print(json.dumps(dict(export=str(EXPORT), runs=len(tasks), updates=25,
                          code_bytes=code.stat().st_size, reuse_existing_data_zip=True), indent=2))


if __name__ == '__main__':
    build()
