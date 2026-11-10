"""Freeze the six requested reproductions and the original MSD runtime."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BUNDLE = ROOT/'reports/medsam3_msd_colab_matched_3fold_3seed_noaug_20261015/bundle'

def digest(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')

sam = list(csv.DictReader(open(ROOT/'reports/thesis_sam3_final_20261023/sam3_runs.csv')))
msd = list(csv.DictReader(open(ROOT/'reports/medsam3_msd_colab_final_20261019/all_108_runs.csv')))
adni = json.loads((ROOT/'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/comparison.json').read_text())['runs']
tasks = []
for dataset, rows, baseline, best, gain, score, identifier in [
    ('GeoFWI3D',sam,'dice','dice_bands_edge_depth','delta','foreground_dice','run_id'),
    ('MSD',msd,'baseline','bands_edge_sdf_depth','gain_pp','validation_dice','id'),
    ('ADNI',adni,'dice','dice_edge_sdf_depth','gain_over_baseline_percentage_points','validation_dice_percent','id')]:
    candidates=sorted([r for r in rows if r['arm']==best], key=lambda r:(float(r[gain]),int(r['fold']),int(r['seed'])))
    assert len(candidates)==9
    selected=candidates[4]
    for worker, arm in enumerate((baseline,best)):
        row=next(r for r in rows if r['arm']==arm and int(r['fold'])==int(selected['fold']) and int(r['seed'])==int(selected['seed']))
        task=dict(id=row[identifier],dataset=dataset,fold=int(row['fold']),seed=int(row['seed']),arm=arm,
                  worker=worker,role='baseline' if worker==0 else 'constrained',
                  expected_dice=float(row[score])/(100 if dataset=='ADNI' else 1),
                  updates=25 if dataset=='GeoFWI3D' else 200)
        tasks.append(task)
        if dataset=='ADNI':
            write(OUT/'payload/reference'/task['id']/'calibration.json',row['calibration'])
write(OUT/'payload/plan.json',dict(tasks=tasks,selection='Strongest mean objective per foundation-model study; median original paired gain across its nine fold/seed cells. No selection using new results.',
    purpose='Qualitative matched reproductions; retain original thesis aggregate results unchanged.',
    missing_artifacts='Final trained adapters and matched predictions absent from cluster and relevant result/recovery folders in both Drive accounts.',
    prediction_policy='Every original support and held-out volume. No annotation input to inference. Foreground, background, all-class Dice, FN/FP and A/P swaps.',
    figure_policy='Paired median-gain, smallest-gain and largest-gain held-out cases; identical slice chosen from reference foreground occupancy. Descriptive examples, not new independent replication.',
    retraining_authorized=True,max_workers=2,estimated_peak_additional_gib=8))
cal=json.loads((ROOT/'reports/medsam3_msd_colab_final_20261019/calibration.json').read_text())['0']
write(OUT/'payload/reference/msd_fold0_calibration.json',cal)
manifest=json.loads((BUNDLE/'bundle_manifest.json').read_text())
with tarfile.open(OUT/'payload/msd_bundle.tar.gz','w:gz') as tar:
    for rel,sha in manifest.items():
        p=BUNDLE/rel
        assert digest(p)==sha, rel
        if rel.startswith('weights/'):
            continue
        tar.add(p,arcname='msd/'+rel,recursive=False)
    for rel in ['bundle_manifest.json','data_manifest.json']:
        if rel not in manifest:tar.add(BUNDLE/rel,arcname='msd/'+rel)
    for rel,sha in json.loads((BUNDLE/'data_manifest.json').read_text()).items():
        assert digest(BUNDLE/'data'/rel)==sha, rel
        tar.add(BUNDLE/'data'/rel,arcname='msd/data/'+rel)
write(OUT/'payload/msd_payload_sha256.json',{'sha256':digest(OUT/'payload/msd_bundle.tar.gz')})
print(json.dumps({'tasks':tasks,'msd_payload_bytes':(OUT/'payload/msd_bundle.tar.gz').stat().st_size},indent=2))
