"""Freeze the broad post-screen follow-up without changing prior campaigns."""
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from .matrix import experiment_matrix

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/experiments/band_focus_20261011_recovery'
OUT=ROOT/'evaluation_output/band_focus_20261011_recovery'
NAME='band_focus_fold0_seed0_5pct_20261011_02'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    oldpath=ROOT/'docs/experiments/band_evolution_20261010_recovery/protocol.json'
    old=json.loads(oldpath.read_text());p=copy.deepcopy(old)
    for key in ('identity_sha','source_sha256','recovery'):
        p.pop(key,None)
    entries=experiment_matrix()
    failedpath=ROOT/'docs/experiments/band_focus_20261011/protocol.json'
    assert entries==json.loads(failedpath.read_text())['runs'], 'Recovery must preserve the frozen scientific matrix'
    p.update(name=NAME,runs=entries,main_runs=len(entries),prior_protocol_sha256=sha(oldpath),
        recovery=dict(failed_job='694088',failed_protocol_sha256=sha(failedpath),
            failed_root='/mnt/beegfsstudents/home/3160552/band_focus_fold0_seed0_5pct_20261011_01',
            reason='Startup compared JSON lists against Python tuples in six multiscale entries. Use JSON-native lists; all 193 scientific configurations are unchanged.'),
        predecessor_root='/mnt/beegfsstudents/home/3160552/band_evolution_fold0_seed0_5pct_20261010_01',
        predecessor_job='692240',minimum_free_gib=5.,estimated_additional_peak_gib=3.5,
        calibration_policy='Same verified epoch-5 prefix. Each band targets median gradient RMS 0.1 times band_budget relative to Dice. All extra terms share ONE joint 0.1 times extra_budget relative to Dice+weighted band, with fixed p95 cap 0.5. Training-only calibration on all ten cases; fixed coefficients. Curricula calibrate their final objective and gradually approach it.',
        loss_definitions=dict(bands='Regional error on actual GT outer-boundary bands; mean plus top-tail, smooth detached difficulty weights, side-specific fractions, local error ranking, spatial tiles, nested bands or shells. Rank squared/CE errors, never signed residuals; neighborhoods normalize by valid same-side support. No validation geometry or guide inputs.',
             pair='Existing independent foreground targets; exact controls retained.',
             ap='Conditional anterior/posterior probability errors on actual GT foreground near the actual interface; balanced A/P masses, difficult-quarter emphasis. Shares extra gradient budget with pairs.',
             schedule='190 arms retain the exact 75-epoch schedule. Three predeclared 150-epoch arms use StepLR40 and minimum75/patience16, with paired Dice/bands controls. Report separately.'),
        storage_budget='Retain five fixed controls and current top five, one active resume checkpoint and common warmup. Retain all predictions, metrics, configs, calibration and sources. Before staging, verified lower-ranked checkpoints from completed predecessor campaigns may be retired under explicit user authorization.',
        interpretation='Exploratory 193-arm single-job follow-up, ordinary SwinUNETR, 5%, fold0 seed0. Target is to investigate whether loss-only training can approach the historical auxiliary-branch 81-82%, not a guaranteed outcome. Fixed finite matrix; no outcome-driven additions or early screen truncation. Reused validation is development data. Confirm selected formulations on new seeds/folds before generalization claims.',
        reference_controls={'bands':'bands','independent13':'independent13','focus25':'band_focus_top25_in50','strong_inside':'budget_protect_inside_1p5'})
    files={ROOT/n for n in old['source_sha256']}
    files.update(f for f in (ROOT/'scripts/band_focus').glob('*') if f.suffix in ('.py','.sh'))
    p['source_sha256']={}
    for f in sorted(files):
        if f.suffix=='.py':ast.parse(f.read_text(),str(f))
        p['source_sha256'][str(f.relative_to(ROOT))]=sha(f)
    p['identity_sha']=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    (DOC/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    lines=['# Focused-band and neighborhood follow-up','',
           f'{len(entries)} predeclared arms, ordinary SwinUNETR, fold 0 seed 0, same 10 training/52 validation cases.',
           '',p['interpretation'],'',p['calibration_policy'],'',p['loss_definitions']['schedule'],'',p['storage_budget'],'',
           '## Families','', '| Family | Arms |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in Counter(x['family'] for x in entries).items()]
    lines += ['', '## Frozen order','', '| Arm | Family | Idea |','|---|---|---|']
    lines += [f"| {x['id']} | {x['family']} | {x['idea']} |" for x in entries]
    (DOC/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(identity=p['identity_sha'],runs=len(entries),families=dict(Counter(x['family'] for x in entries)))))
    return p


if __name__=='__main__':build()
