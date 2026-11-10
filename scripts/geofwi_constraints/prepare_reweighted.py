"""Reconcile the full 468-cell suite with selected UNETR/nnU-Net strengths."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from scripts.geofwi_cluster.campaign import CAP, MODELS, read, write, sha
from scripts.geofwi_constraints.prepare import ROOT, ARMS, replace, balance_lanes
from scripts.geofwi_constraints.reweight_evidence import NAME, REPORT, REMOTE_HOME

SOURCE = ROOT / 'experiments/geofwi3d_constraints_3fold_3seed_20261020'
STAGE = ROOT / 'experiments' / NAME
REMOTE = REMOTE_HOME / NAME
SETTINGS = {'unetr_vit_b16': dict(strength='x4', target=.4, p95_cap=2.),
            'nnunet': dict(strength='x6', target=3.6, p95_cap=4.5),
            'swinunetr': dict(strength='x1', target=.1, p95_cap=.5),
            'basic_unet16': dict(strength='x1', target=.1, p95_cap=.5)}


def key(row):
    return row['model'], row['fold'], row['seed'], row['arm']


def prepare():
    assert not (STAGE / 'submission.json').exists()
    evidence = REPORT / 'prior_evidence'
    proof = read(evidence / 'evidence_inventory.json')
    receipt = read(REPORT / 'evidence_receipt.json')
    assert receipt['status'] == 'locally_verified'
    STAGE.mkdir(parents=True, exist_ok=True)
    inventory = read(SOURCE / 'manifest.json')
    common = {'baseline_runtime.py', 'worker_entry.py', 'campaign.py', 'splits.json', 'data_manifest.json',
              'baseline_results.json', 'cpu_audit_basic.json', 'cpu_audit_unetr.json', 'run.sbatch'}
    for name, digest in inventory.items():
        if name.startswith('bundles/') or name in common:
            assert sha(SOURCE / name) == digest, name
            dest = STAGE / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, dest)
    if not (STAGE / 'data').exists():
        (STAGE / 'data').symlink_to((SOURCE / 'data').resolve(), target_is_directory=True)
    replace(STAGE / 'run.sbatch', 'hippo-geofwi-constraints', 'hippo-geofwi-reweighted')
    replace(STAGE / 'baseline_runtime.py', 'f"{lane_id}_{t[\'model\']}_{t[\'arm\']}"',
            'f"{lane_id}_{t[\'model\']}_{t[\'arm\']}_{t[\'strength\']}"')
    reused, used = [], set()
    for item in proof['completed']:
        if not item['keep']:
            continue
        row = deepcopy(item['row']); k = key(row)
        assert k not in used
        used.add(k)
        src = evidence / item['evidence']
        p = read(src / 'protocol.json'); setting = SETTINGS[row['model']]
        assert p['calibration_target_ratio'] == setting['target'] and p['calibration_p95_cap'] == setting['p95_cap']
        dest = STAGE / 'reused' / row['id']
        shutil.copytree(src, dest, dirs_exist_ok=True)
        row.update(strength=setting['strength'], source_stage=item['source_stage'],
                   source=str(dest.relative_to(STAGE)), archive_manifest_sha256=sha(dest / 'archive_manifest.json'))
        reused.append(row)
    assert len(reused) == 87
    for name in {r['source_stage'] for r in reused}:
        dest = STAGE / 'ancestor_manifests' / (name + '.json')
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(evidence / name / 'source_manifest.json', dest)
    rows = read(STAGE / 'baseline_results.json')['rows']
    tasks, differences = [], {}
    for group in [('swinunetr', 'basic_unet16'), ('unetr_vit_b16', 'nnunet')]:
        for fold in range(3):
            for seed in range(3):
                for arm in ARMS:
                    for model in group:
                        if (model, fold, seed, arm) in used:
                            continue
                        identity = f'GeoFWI3D_{model}_{arm}_f{fold}s{seed}'
                        origin = SOURCE / 'protocols' / identity
                        # Original BANDS f0s0 was a reused Colab result and had no task directory.
                        if not origin.exists():
                            assert (model, fold, seed, arm) == ('unetr_vit_b16', 0, 0, 'dice_bands')
                            origin = SOURCE / 'protocols' / 'GeoFWI3D_unetr_vit_b16_dice_bands_f1s0'
                        folder = STAGE / 'protocols' / identity
                        folder.mkdir(parents=True, exist_ok=True)
                        for file in origin.iterdir():
                            if file.is_file() and file.name != 'protocol.json':
                                assert sha(file) == inventory[str(file.relative_to(SOURCE))]
                                shutil.copyfile(file, folder / file.name)
                        before = read(origin / 'protocol.json'); p = deepcopy(before)
                        setting = SETTINGS[model]
                        write(folder / 'dataset.json', dict(dataset_root=str(REMOTE / 'data'),
                              manifest=str(REMOTE / 'data/fold_0_manifest.json')))
                        baseline = next(r for r in rows if (r['model'], r['fold'], r['seed']) == (model, fold, seed))
                        p.update(name=NAME, folds=[fold], seeds=[seed], baseline_run_id=identity,
                                 strength=setting['strength'], calibration_target_ratio=setting['target'],
                                 calibration_p95_cap=setting['p95_cap'], baseline_reference=baseline,
                                 data=str(REMOTE / 'protocols' / identity / 'dataset.json'),
                                 pkl_sha256=sha(folder / 'dataset.json'))
                        if model == 'unetr_vit_b16':
                            p['rescue_arm'] = identity
                            p['pilot_initialization_sha256'] = p['reference_initializations'][str(seed)]
                        diff = {k: dict(before=before.get(k), after=v) for k, v in p.items() if before.get(k) != v}
                        assert set(diff) <= {'name', 'folds', 'seeds', 'baseline_run_id', 'strength', 'calibration_target_ratio',
                                             'calibration_p95_cap', 'baseline_reference', 'data', 'pkl_sha256',
                                             'rescue_arm', 'pilot_initialization_sha256'}
                        assert p['models'] == before['models']
                        differences[identity] = diff
                        write(folder / 'protocol.json', p)
                        sched = p['models'][p['trainer_alias']]['objectives'][arm]
                        tasks.append(dict(id=identity, kind='conventional', dataset='GeoFWI3D', model=model,
                                          fold=fold, seed=seed, arm=arm, strength=setting['strength'],
                                          bundle='unetr' if model == 'unetr_vit_b16' else 'basic',
                                          augmentation=p['augmentation'], updates=sched['max_steps']))
    tasks = balance_lanes(tasks)
    assert len(tasks) == 381 and not used & {key(t) for t in tasks}
    assert sum(t['model'] in ('swinunetr','basic_unet16') for t in tasks) == 159
    write(STAGE / 'settings.json', SETTINGS)
    write(STAGE / 'protocol_differences.json', differences)
    write(STAGE / 'plan.json', dict(name=NAME, workers=2, cap=CAP, tasks=tasks, reused_rows=reused,
          total_cells=468, new_runs=381, baseline_runs=0, baselines_reused=36, reused_constraints=87,
          remaining_original_swin_basic=159, reweighted_unetr_nnunet=222,
          source_manifest_sha256=sha(SOURCE / 'manifest.json'), evidence_sha256=receipt['sha256'],
          selection='All 13 prescribed arms, folds 0–2 and seeds 0–2 retained. UNETR x4 and nnU-Net x6 fixed by user; no replacement seeds.',
          stopping='Full per-run budget; no halfway gate or Dice-based stopping; preserve recovery on technical failure.',
          storage=dict(minimum_free_gib=54, estimated_additional_peak_gib=46, laptop_required=False,
                       archive='archives/', policy='Cluster-verified result archives before completed artifact retirement'),
          continuation='Autonomous Slurm continuation after clean allocation boundary, no collector or laptop service'))
    c = STAGE / 'campaign.py'
    replace(c, 'assert p["objective"] == t["arm"] != "dice"',
            'assert p["objective"] == t["arm"] != "dice"\n'
            '        setting = read(stage / "settings.json")[t["model"]]\n'
            '        assert p["strength"] == t["strength"] == setting["strength"]\n'
            '        assert p["calibration_target_ratio"] == setting["target"] and p["calibration_p95_cap"] == setting["p95_cap"]')
    replace(c, '    if data:\n',
            '    assert len(plan["tasks"]) == 381 and len(plan["reused_rows"]) == 87\n'
            '    assert plan["baseline_runs"] == 0\n'
            '    for row in plan["reused_rows"]:\n'
            '        folder = stage / row["source"]\n'
            '        assert sha(folder / "archive_manifest.json") == row["archive_manifest_sha256"]\n'
            '        for name, info in read(folder / "archive_manifest.json")["metadata"].items():\n'
            '            assert sha(folder / name) == info["sha256"]\n'
            '        p = read(folder / "protocol.json"); setting = read(stage / "settings.json")[row["model"]]\n'
            '        assert row["strength"] == setting["strength"]\n'
            '        assert p["calibration_target_ratio"] == setting["target"] and p["calibration_p95_cap"] == setting["p95_cap"]\n'
            '        assert row["final_dice"] == read(folder / "validation_latest.json")["summary"]["macro_hard_dice"]\n'
            '        assert row["total_flops"] == read(folder / "completion.json")["consumed_total_flops"] <= CAP\n'
            '    if data:\n')
    replace(c, 'assert usage["free_gib"] >= 24', 'assert usage["free_gib"] >= 54')
    write(STAGE / 'source_provenance.json', dict(parent_manifest_sha256=sha(SOURCE / 'manifest.json'),
          unchanged_training_sources={n:d for n,d in inventory.items() if n.startswith('bundles/') or n=='worker_entry.py'}))
    print(json.dumps(dict(stage=str(STAGE), tasks=len(tasks), reused=len(reused))))


def freeze():
    assert not (STAGE / 'submission.json').exists()
    checks = [read(STAGE / 'checks' / m / 'objective_checks.json') for m in ('unetr_vit_b16','nnunet')]
    assert all(c['status']=='passed' and c['optimizer_updates']==0 for c in checks)
    controller = read(STAGE / 'controller_checks.json'); assert controller['status']=='passed'
    write(STAGE / 'cpu_contract_checks.json', dict(status='passed', optimizer_updates=0,
          updated_models=checks, controller_tests=controller, inherited_checks=read(SOURCE / 'cpu_contract_checks.json')))
    for name in ('prepare_reweighted.py','test_reweighted.py','reweight_evidence.py','check_objectives.py','manage_reweighted.py'):
        shutil.copyfile(Path(__file__).with_name(name), STAGE / ('source_'+name))
    inventory={str(p.relative_to(STAGE)):sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file()
               and 'data' not in p.relative_to(STAGE).parts and '__pycache__' not in p.parts
               and p.relative_to(STAGE).as_posix()!='manifest.json'}
    write(STAGE / 'manifest.json', inventory)
    subprocess.run([str(ROOT / '.venv/bin/python'), str(STAGE / 'campaign.py'),'verify','--stage',str(STAGE)],check=True)
    payload=REPORT/'frozen_payload.tar.gz'
    with tarfile.open(payload,'w:gz',compresslevel=1) as tar:
        for name in sorted(inventory):tar.add(STAGE/name,arcname=name,recursive=False)
        tar.add(STAGE/'manifest.json',arcname='manifest.json',recursive=False)
    write(REPORT/'preparation.json',dict(status='verified_ready_to_submit',manifest_sha256=sha(STAGE/'manifest.json'),
          payload_sha256=sha(payload),payload_bytes=payload.stat().st_size,new_runs=381,reused_constraints=87,
          baseline_runs=0,laptop_required=False))
    for name in ('plan.json','settings.json','controller_checks.json','cpu_contract_checks.json'):
        shutil.copyfile(STAGE/name,REPORT/name)


if __name__=='__main__':
    freeze() if '--freeze' in sys.argv else prepare()
