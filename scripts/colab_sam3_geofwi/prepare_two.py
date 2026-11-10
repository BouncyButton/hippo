"""Build the requested five-volume, 25-update derivative without replacing the original export."""
from copy import deepcopy
import json
import shutil
import zipfile

from scripts.colab_sam3_geofwi import prepare, runtime

EXPORT = prepare.ROOT/'reports/sam3_geofwi_colab_two_runs_20261018'
CODE = 'sam3_geofwi_two_runs_code.zip'
NOTEBOOK = 'SAM3_GeoFWI3D_two_runs.ipynb'


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
    shutil.copy2(prepare.ROOT/'scripts/colab_sam3_geofwi/parallel.py', bundle/'parallel.py')
    plan = runtime.read(source/'plan.json')
    tasks = []
    for original in plan['tasks']:
        if original['updates'] != 50 or (original['scientific_arm'], original['bands_target'], original['edge_target']) not in [('dice_bands', .1, None), ('dice_bands_edge', .2, .01)]:
            continue
        task = deepcopy(original)
        task.update(id=task['id'].replace('_u50_', '_u25_'), updates=25,
                    protocol=task['protocol'].replace('u50_', 'u25_', 1))
        protocol = runtime.read(source/'protocols'/original['protocol']/'protocol.json')
        protocol['updates'] = 25
        protocol['sampling'] = protocol['sampling'].replace('50 updates,', '25 updates,', 1)
        if task['scientific_arm'] == 'dice_bands_edge':
            task.update(id=task['id'].replace('b20_e10', 'b10_e100'),
                        protocol=task['protocol'].replace('b20_e10', 'b10_e100'),
                        bands_target=.1, edge_target=.1)
            protocol.update(gradient_ratio_target=.1, gradient_ratio_p95_cap=.5,
                            gradient_ratio_targets=dict(bands=.1, edge=.1),
                            gradient_ratio_p95_caps=dict(bands=.5, edge=.5))
        prepare.write(bundle/'protocols'/task['protocol']/'protocol.json', protocol)
        tasks.append(task)
    plan.update(name='sam3_geofwi_colab_two_runs_v1', tasks=tasks, num_runs=len(tasks),
                fixed_updates=25, constraint_only_10pct=True, parent_colab_manifest_sha256=prepare.digest(source/'bundle_manifest.json'))
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
        text = text.replace('results_5vol_24conditions_v1', 'results_5vol_25updates_bands10_edge10_v1')
        text = text.replace('/content/hippo_sam3_geofwi_5vol_bundle', '/content/hippo_sam3_geofwi_two_runs_bundle')
        text = text.replace('MAX_NEW_RUNS=24', 'MAX_NEW_RUNS=2')
        text = text.replace('All five native stopping-point controls finish before new constraint runs.',
                            'Only bands and bands+edge are scheduled; there is no baseline task.')
        text = text.replace('MAX_WORKERS=0', 'MAX_WORKERS=2')
        text = text.replace("campaign('report')", "execute([PYTHON,BUNDLE/'runtime.py','report','--bundle',BUNDLE,'--durable',RESULTS,'--baseline-results',HOME_DIR/'results_5vol_24conditions_v1'],'report.log')")
        text = text.replace("'gain_vs_bands20_pp'", "'gain_vs_bands10_pp'")
        text = text.replace('Gains use fresh Colab native and bands20 controls at the same stopping update.',
                            'Gains use the saved 25-update Colab baseline and the new bands10 run. The baseline is read from its existing archive; it is never rerun.')
        cell['source'] = text
    nb['cells'][0]['source'] = """# SAM3 — bands and bands+edge only

Two runs, five frozen training volumes, 25 updates each, same 96 validation volumes.

| Run | Bands target | Edge target |
|---|---:|---:|
| Bands | 10% | — |
| Bands + edge | 10% | 10% |

There is no baseline task. The comparison cell reads the completed 25-update baseline
from `results_5vol_24conditions_v1` (65.10% validation Dice), without training or evaluating it again.
Targets use the existing sequential gradient calibration; the p95 ceiling is five times each target.
The fold, seed, learning-rate horizon (200), warmups (20), sampling and no-augmentation settings are unchanged.

After the old run has stopped cleanly, upload `sam3_geofwi_two_runs_code.zip` to the existing
`MyDrive/hippo_sam3_geofwi_exploration/input` folder and run this notebook in order.
Reuse the existing data ZIP and base weights. Mount Drive as **filippofocaccia2@gmail.com**.
Results go to `results_5vol_25updates_bands10_edge10_v1`; old results are preserved.
At most two workers run, subject to GPU memory and throughput checks.
"""
    nb['metadata']['colab']['name'] = NOTEBOOK
    prepare.write(EXPORT/NOTEBOOK, nb)
    print(json.dumps(dict(export=str(EXPORT), runs=len(tasks), updates=25,
                          code_bytes=code.stat().st_size, reuse_existing_data_zip=True), indent=2))


if __name__ == '__main__':
    build()
