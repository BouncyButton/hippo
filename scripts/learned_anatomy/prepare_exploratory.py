"""Package one explicitly authorized 5% run with the current frozen encoder."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
from .prepare import ROOT, DOC, OUT, sha


def main():
    archive=OUT/'frozen_evidence_689705.zip'
    assert sha(archive)=='9de3d7abc16c4750eb42dce19d46b49f4810e09c35bd4f1c1331df0dc1fe52a3'
    evidence=OUT/'evidence_689705';evidence.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence_manifest.json'))
        for name,expected in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==expected
            dest=evidence/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    audit=json.loads((evidence/'gradient_gate.json').read_text())
    assert audit['passed'] is False and audit['encoder_sha256']==sha(evidence/'encoder/frozen.pt')
    original=json.loads((evidence/'protocol.json').read_text())
    assert audit['protocol_sha256']==original['identity_sha']
    assert sorted(k for k,v in audit['groups'].items() if not v['passed'])==['swap_uncertain']
    p=dict(original);p.pop('identity_sha')
    p.update(name='learned_anatomy_fold0_seed0_5pct_20261008_02',parent_protocol_sha256=original['identity_sha'],
        source_root='/mnt/beegfsstudents/home/3160552/'+original['name'],encoder_updates_this_job=0,
        stages=['frozen','preflight','swin'],
        exploratory_audit_acceptance=dict(user_authorized=True,scope='one_5pct_swin_run',
            user_request='Ok meanwhile submit the 5% run with the current frozen encoder',date='2026-10-08',
            audit_protocol_sha256=audit['protocol_sha256'],audit_sha256=sha(evidence/'gradient_gate.json'),
            encoder_sha256=audit['encoder_sha256'],accepted_failed_groups=['swap_uncertain'],
            interpretation='Exploratory segmentation run after a failed tissue-preservation audit; the audit remains failed.'))
    source=[ROOT/'baselines/swin_unetr/swin_unetr.py']
    source+=sorted((ROOT/'scripts/depth_supervision').glob('*.py'))
    source+=sorted((ROOT/'scripts/learned_anatomy').glob('*.py'))
    source+=[ROOT/'scripts/learned_anatomy/run.sh']
    p['source_sha256']={str(f.relative_to(ROOT)):sha(f) for f in source}
    p['identity_sha']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    doc=DOC/'exploratory_5pct';doc.mkdir(parents=True,exist_ok=True)
    (doc/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    (doc/'README.md').write_text('''# Exploratory 5% SwinUNETR with the current frozen anatomy encoder

The user explicitly requested submission on 2026-10-08 after discussing the failed tissue-preservation audit. This job performs zero encoder updates and reuses the final checkpoint from job 689705 byte for byte. The original campaign and its failed audits remain intact in the `_01` root; this one segmentation run uses a separate `_02` root.

MSD fold 0, seed 0, the same ten training and 52 validation patients; native one-channel SwinUNETR, all segmentation weights trainable, feature size 24, maximum 75 epochs, early stopping minimum 25 / patience 8 / delta 0.0005. The objective is Dice plus the frozen learned spatial feature loss. Five Dice-only epochs precede training-only gradient calibration (median ratio 0.1, p95 cap 0.5) and a five-epoch ramp. No depth, SDF, focal, profile, topology or additional MRI branch is introduced. Inference uses only SwinUNETR.

The audit remains **failed**: all missing/extra and isolated A/P groups pass, but only 95/120 uncertain A/P probes satisfy the mass test; 21 failures reduce foreground probability too much and four increase it too much relative to correct relabeling. Every probe increases the correct-label probability. The authorization is bound to the exact audit and encoder hashes. Changed evidence, changed weights, absent authorization or an unexpected failure group blocks training. No threshold is relaxed and the failed audit is not rerun to seek a pass.

The allocation verifies the frozen inputs, performs a production CUDA backward/optimizer check, then trains exactly one segmentation model. The frozen loss encoder receives no parameter updates. Compare final Dice, missing/extra foreground, A/P swaps, surfaces and depth bands against the preserved 74.166740% Dice baseline, paying particular attention to the false-positive / missing-tissue trade-off. A good result is exploratory evidence in one fold/seed, not proof of tissue preservation or equivalence to ROI fusion. All previous encoder fitting is additional compute.
''')
    # Keep readable audit evidence beside the protocol; model binaries stay in ignored output storage.
    for name in ['gradient_gate.json','encoder/mass_refinement_completion.json']:
        shutil.copy2(evidence/name,doc/Path(name).name)
    (doc/'input_evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    stage=OUT/'exploratory_payload';stage.mkdir(exist_ok=True)
    for f in source:
        dest=stage/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for name in ['protocol.json','README.md']:shutil.copy2(doc/name,stage/name)
    names=['gradient_gate.json','roi_manifest.json','encoder/frozen.pt','encoder/mass_refinement_completion.json']
    names+=['roi/'+n+'.npz' for n in p['splits']['5pct']['train']]
    for name in names:
        dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(evidence/name,dest)
    baseline=ROOT/'docs/experiments/unwrapped_followup_20261007/final_results/evidence/unwrapped_swin_fold0_seed0_20261006_02/runs/5pct/dice'
    completion=json.loads((baseline/'completion.json').read_text())
    assert completion['identity']['train']==p['splits']['5pct']['train'] and completion['identity']['val']==p['splits']['5pct']['val']
    assert completion['artifacts']['validation_best.json']==sha(baseline/'validation_best.json')
    for src,dst in [('validation_best.json','baseline_validation.json'),('completion.json','baseline_completion.json')]:
        shutil.copy2(baseline/src,stage/dst)
    unit=json.loads((OUT/'exploratory_unit_checks.json').read_text())
    smoke=json.loads((OUT/'swin_smoke.json').read_text())
    assert unit['passed'] and smoke['passed'] and smoke['encoder_frozen']
    checks=dict(passed=True,unit=unit,swin=smoke,source_sha256=p['source_sha256'],
        note='Learned loss and native network are unchanged; production CUDA preflight is required before segmentation training.')
    (stage/'local_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    shutil.copy2(stage/'local_checks.json',doc/'local_checks.json')
    files={str(f.relative_to(stage)):sha(f) for f in stage.rglob('*') if f.is_file() and f.name!='payload_manifest.json'}
    (stage/'payload_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    dest=OUT/'exploratory_payload.tar.gz'
    with tarfile.open(dest,'w:gz') as t:
        for f in sorted(stage.rglob('*')):
            if f.is_file():t.add(f,arcname=str(f.relative_to(stage)))
    receipt=dict(archive=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,protocol_identity=p['identity_sha'])
    (OUT/'exploratory_payload_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
