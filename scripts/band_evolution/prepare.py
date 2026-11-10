"""Freeze a finite independent-target/band sweep and inherited training protocol."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from collections import Counter
from .matrix import experiment_matrix

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/experiments/band_evolution_20261010_recovery'
OUT=ROOT/'evaluation_output/band_evolution_20261010_recovery'
NAME='band_evolution_fold0_seed0_5pct_20261010_01'
PREVIOUS='/mnt/beegfsstudents/home/3160552/band_chamfer_fold0_seed0_5pct_20261008_01'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    oldpath=ROOT/'docs/experiments/band_chamfer_20261008/protocol.json'
    old=json.loads(oldpath.read_text());p=copy.deepcopy(old)
    for key in ('identity_sha','source_sha256','chamfer_policy','ap_training_audit_sha256'):
        p.pop(key,None)
    prefix=json.loads((ROOT/'docs/experiments/band_chamfer_20261008/provenance.json').read_text())
    entries=experiment_matrix()
    p.update(name=NAME,runs=entries,main_runs=len(entries),prior_protocol_sha256=sha(oldpath),
        predecessor_root=PREVIOUS,predecessor_job='690825',minimum_free_gib=3.,estimated_additional_peak_gib=1.5,
        shared_warmup=dict(folder=PREVIOUS+'/shared_warmup',protocol_identity=old['identity_sha'],
            artifacts=prefix['artifacts'],calibration=old['shared_warmup']['calibration'],prefix_epochs=5),
        resources=dict(partition='stud',gres='gpu:4g.40gb:1',cpus=4,memory='24G',time='23:50:00'),
        geometry_policy='Annotation-derived native 1 mm outer boundaries. Original controls unchanged. Independent scales use oriented crossings; whole intervening side segments must remain in class and native FOV. Separate-support arms retain valid sides independently. Common-support arms match oriented faces. Endpoint steps are measured from immediate inside/outside voxel centers; actual separation is inside_steps + outside_steps + 1 mm.',
        calibration_policy='Verified shared epoch-5 Dice-only prefix. Bands calibrated median logit-gradient RMS 0.1 and p95 <= 0.5 relative to Dice. Extras jointly calibrated at 0.1 relative to Dice+bands, except explicit 0.05/0.15 budget arms; fixed p95 cap 0.5. Training-only calibration on all 10 cases; coefficients fixed, original epoch6-10 ramp.',
        loss_definitions=dict(pair='Independent Brier targets, side-specific scale mixtures, side mass, endpoint eligibility/offsets and reductions. Alternate BCE, power, Huber, probability-error dead-zone and 50/50 mean/top-tail penalties.',
            bands='Canonical original or physical regional BCE/Brier. Optional predeclared shell weights, side balance, exponential distance decay, class-balanced surface association or top-tail emphasis.',
            surface='Existing distance/Chamfer losses or squared local foreground-fraction mismatch in 3/5-voxel boxes, valid-FOV normalized within 3 mm of annotated boundary. Local volume can cancel opposite errors; standalone endpoints remain active.'),
        storage_budget='Retain three controls and current top five best checkpoints, one active resume checkpoint, and common warmup. Verify all completed artifacts before retiring other new-run checkpoints. Keep every prediction, metric, history, config, source and calibration. No deletion from previous campaigns.',
        interpretation=f'Exploratory post-hoc follow-up: {len(entries)} predeclared arms, one fold/seed, 10 training and 52 validation cases. Hypotheses informed by prior validation errors; same validation set is development data. Report every result and paired differences; independent seed/fold confirmation required. No automatic outcome-based additions.')
    original=json.loads((ROOT/'docs/experiments/band_evolution_20261009/protocol.json').read_text())
    assert p['runs']==original['runs'] and p['splits']==original['splits']
    p['recovery']=dict(original_job='691444',original_protocol_identity=original['identity_sha'],
        reason='CUDA preflight failed before training: nested lists of sampling weights were not transferred to GPU. Recursive device transfer repaired; scientific matrix and training settings unchanged.')
    files={ROOT/n for n in old['source_sha256']}
    files.update(f for f in (ROOT/'scripts/band_evolution').glob('*') if f.suffix in ('.py','.sh'))
    p['source_sha256']={}
    for f in sorted(files):
        if f.suffix=='.py':ast.parse(f.read_text(),str(f))
        p['source_sha256'][str(f.relative_to(ROOT))]=sha(f)
    p['identity_sha']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    (DOC/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    lines=['# Independent targets and bands follow-up','',f'{len(entries)} ordinary SwinUNETR arms; 5%, fold 0, seed 0; no auxiliary branch.',
        'Same epoch-5 prefix, 75-epoch cap, optimizer, early stopping and raw A/P Dice checkpoint selection as job 690825.','',
        p['geometry_policy'],'',p['calibration_policy'],'',p['storage_budget'],'',p['interpretation'],'',
        '| Experiment | Family | Comparison | Hypothesis |','|---|---|---|---|']
    lines += [f"| {e['id']} | {e['family']} | {e['compare']} | {e['idea']} |" for e in entries]
    (DOC/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(identity=p['identity_sha'],runs=len(entries),families=dict(Counter(e['family'] for e in entries)))))
    return p


if __name__=='__main__':build()
