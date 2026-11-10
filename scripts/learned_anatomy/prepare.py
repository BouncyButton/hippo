"""Freeze the single authorized experiment and package verified source."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/experiments/learned_anatomy_20261008'
OUT=ROOT/'evaluation_output/learned_anatomy_20261008'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old=json.loads((ROOT/'docs/experiments/tissue_profiles_20261007/protocol.json').read_text())
    keys=['dataset','fold','seed','max_epochs','baseline_loss','feature_size','spatial_size','batch_size',
        'drop_rate','activation_checkpointing','pretrained','training_augmentation','precision','learning_rate',
        'weight_decay','step_size','lr_gamma','early_stopping','selection','geometry','required_monai','pkl','pkl_sha256','calibration']
    p={key:old[key] for key in keys}
    names=old['splits']['5pct']['train']+old['splits']['5pct']['val']
    p.update(name='learned_anatomy_fold0_seed0_5pct_20261008_01',main_runs=1,model='MONAI SwinUNETR',
        splits={'5pct':old['splits']['5pct']},native_geometry={n:old['native_geometry'][n] for n in names},
        guide_atlas=old['prior_root']+'/atlas/5pct',atlas_protocol_sha256=old['atlas_protocol_sha256'],
        encoder=dict(seed=0,steps=1000,lr=.001,channels=[8,12,16],inputs=['candidate_probabilities','guide_roi','original_fov'],
            training_tasks=['candidate reconstruction','clean reconstruction','typed corruption detection'],
            task_weights=[1,1,1],clean_fraction=.25,corruption_radii_mm=[1.5,3.,4.5],confidence_range=[.6,.99],
            selection='fixed final step; no validation selection'),
        gate=dict(seed=20261008,radii_mm=[1.,2.5,4.],confidences=[.8,.95],helpful_fraction=.9,
            minimum_probe_helpful_fraction=.75,repair_path_fraction=.95,mass_direction_fraction=.95),
        learned_loss='Equal mean of three FOV-masked squared feature differences, divided by training-only feature second moments; no raw-label identity feature.',
        minimum_free_gib=2.,storage_budget_gib=1.,allowed_losses=['dice','learned_spatial_feature_distance'],
        inference='Unchanged one-channel native SwinUNETR; prior encoder and ROI are used only for training loss.',
        baseline_source='Saved completed seed-0 Dice baseline; no completed segmentation run is retrained.')
    reuse=DOC/'reused_encoder.json'
    if reuse.exists():
        p['reused_encoder']=json.loads(reuse.read_text())
        p['refinement']=dict(seed=17,steps=300,lr=.0003,representation_weight=.1,
            purpose='Correct observed background leakage during A/P swaps without relaxing audit thresholds.')
        p['gate']['constructions']=['isolated','uncertain']
    refined=DOC/'reused_refinement.json'
    if refined.exists():
        p['reused_refinement']=json.loads(refined.read_text())
        p['mass_refinement']=dict(seed=29,steps=600,lr=.0003,representation_weight=.1,mass_weight=2.,swap_margin=.1,
            purpose='Fit the actual probability-step direction and foreground preservation; encoder training only.')
        p['gate']['seed']=20261009
    source=[ROOT/'baselines/swin_unetr/swin_unetr.py']
    source+=sorted((ROOT/'scripts/depth_supervision').glob('*.py'))
    source+=sorted((ROOT/'scripts/learned_anatomy').glob('*.py'))
    source+=[ROOT/'scripts/learned_anatomy/run.sh']
    p['source_sha256']={str(f.relative_to(ROOT)):sha(f) for f in source}
    p['identity_sha']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    DOC.mkdir(exist_ok=True,parents=True)
    (DOC/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    unit=json.loads((OUT/'unit_checks.json').read_text());smoke=json.loads((OUT/'swin_smoke.json').read_text())
    assert unit['passed'] and smoke['passed'] and smoke['encoder_frozen']
    stage=OUT/'payload';stage.mkdir(exist_ok=True)
    for f in source:
        dest=stage/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for name in ['README.md','protocol.json']:shutil.copy2(DOC/name,stage/name)
    prior=ROOT/'docs/experiments/unwrapped_followup_20261007/final_results/evidence/unwrapped_swin_fold0_seed0_20261006_02/runs/5pct/dice'
    completion=json.loads((prior/'completion.json').read_text())
    assert completion['identity']['train']==p['splits']['5pct']['train'] and completion['identity']['val']==p['splits']['5pct']['val']
    assert completion['artifacts']['validation_best.json']==sha(prior/'validation_best.json')
    shutil.copy2(prior/'validation_best.json',stage/'baseline_validation.json')
    shutil.copy2(prior/'completion.json',stage/'baseline_completion.json')
    checks=dict(passed=True,unit=unit,swin=smoke,source_sha256=p['source_sha256'],
        note='Local CPU PyTorch 2.2.2/MONAI 1.4; actual production CUDA/MONAI 1.6 backward and trained-encoder audits required inside allocation.')
    (stage/'local_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    shutil.copy2(stage/'local_checks.json',DOC/'local_checks.json')
    manifest={str(f.relative_to(stage)):sha(f) for f in stage.rglob('*') if f.is_file() and f.name!='payload_manifest.json'}
    (stage/'payload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(OUT/'payload.tar.gz','w:gz') as t:
        for f in sorted(stage.rglob('*')):
            if f.is_file():t.add(f,arcname=str(f.relative_to(stage)))
    receipt=dict(archive=str(OUT/'payload.tar.gz'),sha256=sha(OUT/'payload.tar.gz'),bytes=(OUT/'payload.tar.gz').stat().st_size,protocol_identity=p['identity_sha'])
    (OUT/'payload_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
