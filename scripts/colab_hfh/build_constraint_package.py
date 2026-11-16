"""Package the constraint runtime and update the existing four-model notebook."""
import hashlib
import argparse
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('bands_strict','constraints_pilot','constraints_screen','constraints_matrix','constraints_auto','nnunet_lower_budget'),default='constraints_pilot')
    parser.add_argument('--models',nargs='+',choices=('nnunet','swinunetr','basic_unet16','unetr_vit_b16'),default=['nnunet','swinunetr','basic_unet16','unetr_vit_b16'])
    parser.add_argument('--optimize-mean',action='store_true')
    parser.add_argument('--objectives',nargs='+')
    args=parser.parse_args()
    if args.objectives and not args.optimize_mean:
        parser.error('Objective scope requires --optimize-mean')
    root = Path(__file__).resolve().parents[2]
    export = root/'reports/hfh_colab_exploration_20261111'
    source = root/'scripts/colab_hfh'
    lower=json.loads((export/'lower_budget_plan.json').read_text()) if args.mode=='nnunet_lower_budget' else None
    lower_version=lower['version'] if lower else 'nnunet_budget75_v1'
    files = {name: export/name for name in ('constraint_matrix_plan.json','bands_v1_payload.json','online_sdf_arithmetic_audit.json')}
    if args.mode=='nnunet_lower_budget':
        files['lower_budget_plan.json']=export/'lower_budget_plan.json'
    for name in ('constraint_calibration.py','constraint_geometry.py','constraint_schedule.py',
                 'constraint_worker.py','constraint_archive.py','constraint_campaign.py','constraint_search.py',
                 'bands_campaign.py','bands_variants.py','constraint_mean_search.py','constraint_report.py','constraint_drive.py','constraint_diagnostics.py','parameter_gradient_diagnostics.py','constraint_online_sdf.py'):
        files['bundle/source/scripts/colab_hfh/'+name] = source/name
    for name in ('lower_budget_protocol.py','lower_budget_baseline_campaign.py','lower_budget_search.py'):
        files['bundle/source/scripts/colab_hfh/'+name]=source/name
    package = export/'hfh_constraint_assets.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            info=zipfile.ZipInfo(name,(2026,11,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
    digest=hashlib.sha256(package.read_bytes()).hexdigest()
    notebook_path=export/'HFH_baseline_shared_seed_search.ipynb'
    notebook=json.loads(notebook_path.read_text())
    strict_hash=hashlib.sha256((export/'hfh_band_strict_assets.zip').read_bytes()).hexdigest()
    notebook['cells'][4]['source'] = f'''# Same notebook: BAND search, constraint pilot, screening, or frozen full matrix.
RUN_MODE = '{args.mode}'
SESSION_HOURS = 8
MODEL_SCOPE = {args.models!r}
OBJECTIVE_SCOPE = {args.objectives!r}
OPTIMIZE_MEAN = {args.optimize_mean!r}
import psutil,hashlib,zipfile,time,json,subprocess
from pathlib import Path
BAND_STRICT_ASSET_SHA256 = '{strict_hash}'
CONSTRAINT_ASSET_SHA256 = '{digest}'
if RUN_MODE not in ('bands_strict','constraints_pilot','constraints_screen','constraints_matrix','constraints_auto','nnunet_lower_budget'):
    raise ValueError('Unknown run mode')
is_band = RUN_MODE == 'bands_strict'
is_lower = RUN_MODE == 'nnunet_lower_budget'
module = 'scripts.colab_hfh.lower_budget_search' if is_lower else 'scripts.colab_hfh.constraint_search'
entry = str(ROOT/'bands_strict_search.py') if is_band else module
work = ROOT/('{lower_version}' if is_lower else ('bands_strict_search' if is_band else 'constraint_search'))
active=[]
for item in psutil.process_iter(['cmdline']):
    try:
        command=item.info['cmdline'] or []
        if str(ROOT/'bands_strict_search.py') in command or any(m in command for m in ('scripts.colab_hfh.constraint_search','scripts.colab_hfh.lower_budget_search')):active.append(item)
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
if len(active)>1:raise RuntimeError('Multiple experiment controllers; inspect before continuing')
def alive(p):
    try:return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
log_path=ROOT/('{lower_version}.log' if is_lower else ('bands_strict_search.log' if is_band else 'constraint_search.log'))
log_start=0
if active:
    process=active[0]
    if entry not in process.cmdline():raise RuntimeError('Finish the other controller using its STOP file before switching modes')
    print('Attached to existing controller',process.pid,flush=True)
else:
    filename='hfh_band_strict_assets.zip' if is_band else 'hfh_constraint_assets.zip'
    expected=BAND_STRICT_ASSET_SHA256 if is_band else CONSTRAINT_ASSET_SHA256
    package=ROOT/filename
    if not package.exists() or hashlib.sha256(package.read_bytes()).hexdigest()!=expected:
        private=Path('/content/.hfh_storage_auth')
        subprocess.run([str(private/'rclone'),'--config',str(private/'rclone.conf'),'copyto',
            'hfh_storage:'+filename,str(package),'--checksum'],check=True,timeout=300)
    if hashlib.sha256(package.read_bytes()).hexdigest()!=expected:raise ValueError('Asset checksum mismatch')
    with zipfile.ZipFile(package) as zipped:
        for item in zipped.infolist():
            relative=Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe package path')
            target=ROOT/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(zipped.read(item))
    work.mkdir(exist_ok=True);(work/'STOP').unlink(missing_ok=True)
    phase='all' if RUN_MODE=='constraints_auto' else ('matrix' if RUN_MODE=='constraints_matrix' else 'screen')
    command=[str(PYTHON),entry,'--root',str(ROOT)] if is_band else [str(PYTHON),'-m',module,'--root',str(ROOT),'--hours',str(SESSION_HOURS)]
    if not is_band and not is_lower:command += ['--phase',phase]
    if RUN_MODE=='constraints_pilot':command.append('--pilot')
    if not is_band and not is_lower:
        command += ['--models',*MODEL_SCOPE]
        if OPTIMIZE_MEAN:command.append('--optimize-mean')
        if OBJECTIVE_SCOPE:command += ['--objectives',*OBJECTIVE_SCOPE]
    log_start=log_path.stat().st_size if log_path.exists() else 0
    with log_path.open('a') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    process=psutil.Process(child.pid)
with log_path.open() as log:
    log.seek(log_start)
    try:
        while alive(process):
            content=log.read()
            if content:print(content,end='',flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        (work/'STOP').touch()
        print('Stop requested after the current candidate has been archived.',flush=True)
    print(log.read(),end='',flush=True)
status_path=work/'search_status.json'
if status_path.exists():
    result=json.loads(status_path.read_text());print('Experiment status:',result['status'])
else:print('Inspect startup log above')
'''
    notebook['cells'][0]['source'] = '''# HFH — adapted constraint recipes and complete comparison matrix

Compute: the configured compute account. Storage: the configured storage account. Use an available BF16-capable GPU; comparisons retain the frozen BF16 controls and native training recipes.

The same notebook now supports the strict BAND search and the subsequent **14 objectives × 4 models × 3 seeds × 3 folds = 504 conditions**. Seeds and folds are fixed at 0, 1 and 2. The 36 native controls and 18 verified BasicUNet/UNETR BAND results can be reused. The current target is **mean gain >=+1.0 Dice percentage point across all nine fixed pairs**, with all final Dice >=50%. Small individual regressions are allowed; every pair and the worst case are reported.

**Active mean recovery:** preserve scientific protocols and reuse completed pairs. Finish the active candidate across all nine comparisons without per-pair improvement rejection. Then prioritize the declared diagnosis-led nnU-Net candidates; remaining candidates use the common f0s1 pilot, never unequal partial averages. Freeze each model/objective as soon as a complete recipe meets the mean target; launch no further trials for that arm. Prioritize nnU-Net, then unresolved SwinUNETR arms. BasicUNet/UNETR are outside this exploration scope. One common recipe is used across all nine pairs. Preparing the 504-condition matrix is not completing it; missing cells remain blank. Adaptive validation selection is exploratory.

**nnU-Net diagnosis-led follow-up:** 14 appended hypotheses cover the 13 unresolved constraint objectives. BAND tests fixed inner/outer weights0.9/0.1 at the original and later onset. Combinations use a later calibration anchor, BAND target0.3 and small secondary targets0.01, with joint target0.4/cap0.75. Objectives without BAND test a weak later target0.03/cap0.15. Later onset is at least45 updates, increased only when required to retain all90 native updates under the existing cap. These hypotheses respond to missed foreground, late foreground learning and measured gradient-strength drift; they are not established improvements. No existing candidate ID, native loss, optimizer, LR, sample or control changes.

Transferred lessons: keep BasicUNet's delayed onset and UNETR's established ramp as starting hypotheses; recalibrate each actual objective; check joint gradient caps for every ramp subset at the training-only calibration anchor; use physical 1×2×1 mm geometry for SDF/depth; activate SDF only after the early ramp. Standard recipes keep coefficients fixed: their calibration caps are not per-update clipping. A post-training probe measures their final strength and reports a separate diagnostic FLOPs allowance. Two explicitly labelled SwinUNETR SDF pilots instead track a training-only FP32 logit-gradient ratio each active update; the extra probe arithmetic is charged **inside** the optimization cap, with the native loss path preserved and a 64× anchor-coefficient ceiling. Their recorded traces are audited before accepting results. Optimizer-reset gains remain separate diagnostics.

Four additional nnU-Net/SwinUNETR BAND pilots test **training-case adaptive inner/outer weighting**, separately labelled from canonical BAND. Each case's detached inner/outer BCE determines its inner weight, clipped to [0.1,0.9]; the outer weight is its complement. The same rule applies in calibration and training, with no validation input and no gradient through the weight. An additional 64 scalar FLOPs per active update is charged inside the existing cap. These are hypotheses, not verified improvements.

Cell 4 selects `bands_strict`, `constraints_pilot`, `constraints_screen`, `constraints_matrix`, or `constraints_auto`. Auto mode screens, freezes one recipe per model/objective, then completes all nine pairs, including failed recipes, within the same eight-hour session limit. It resumes saved progress after a clean pause. Finish the active candidate before changing mode. The native optimization cap remains **31,976,450,667,090 FLOPs/run**; exact updates are recomputed per objective. Calibration is charged separately. Native optimizer, loss, LR horizon, samples, initialization and no-augmentation policy remain frozen.
'''
    notebook['cells'][0]['source'] += (f'\n**Active exploration:** {", ".join(args.models)}; mode `{args.mode}`. '
        + ('Freeze already qualified model/objective recipes and work only on unresolved arms. '
           if args.optimize_mean else '')
        + 'Models outside this scope retain their saved results and receive no new recipe trials.\n')
    if args.optimize_mean:
        notebook['cells'][0]['source'] += ('\n**Current user priority: nnU-Net BAND and BAND+depth.** '
            'Skip further standalone SDF tuning. The focused phase tests the remaining late-onset BAND '
            'recipe, then a moderate fixed inner/outer balance0.75/0.25 for BAND and BAND+depth. '
            'The latter keeps BAND target0.3 and depth target0.01, joint target0.4/cap0.75. '
            'These append new candidates without changing existing IDs or controls. The same recipe '
            'runs across all nine pairs; no case-specific validation tuning. Other objectives are '
            'deferred during this focused phase, with their completed results preserved.\n')
        if args.objectives:
            notebook['cells'][0]['source'] += '\n**Active objective scope:** '+', '.join(args.objectives)+'.\n'
        notebook['cells'][0]['source'] += ('\nNew nnU-Net runs at fold0/seed1 and fold2/seed1 also collect a read-only '
            'parameter-gradient diagnostic after final evaluation: two fixed training subjects, native total, '
            'three active deep-supervision heads and the weighted auxiliary objective. It changes no training '
            'update or coefficient, preserves model/RNG/gradient state, and records a separate conservative '
            '3,552,203,953,190-FLOP diagnostic allowance per probed run. This tests whether full-resolution '
            'logit calibration reflects the actual parameter-gradient balance. Existing completed runs are not rerun.\n')
    if args.mode=='nnunet_lower_budget':
        lower=json.loads((export/'lower_budget_plan.json').read_text())
        notebook['cells'][0]['source'] += ('\n**CURRENT PHASE: nnU-Net lower-budget comparison.** '
            f'Run9 new native controls and18 matched constrained runs at **{lower["budget_flops"]:,} optimization FLOPs** '
            f'per run ({lower["native_updates"]} native updates;{100*lower["ratio"][0]/lower["ratio"][1]:g}% of the full cap). '
            +lower['matching']+' '
            'Use the best independently verified complete full-budget BAND and BAND+depth recipes; '
            'recalibrate coefficients using only training crops. Preserve all9 results even if Dice falls below50%. '
            'Compare absolute Dice and deltas against both the new lower-budget control and the original '
            'full-budget control. These results use separate folders and never enter the original504-cell matrix. '
            'This is a budget diagnostic, not seed selection or evidence that a deliberately weaker baseline is preferable.\n')
    notebook['cells'][5]['source'] = '''# Complete matrix: blank cells remain pending; screening failures are separate.
import sys,json,pandas as pd
sys.path.insert(0,str(ROOT/'bundle/source'))
from scripts.colab_hfh.constraint_report import write_report
summary,matrix_path,summary_path=write_report(ROOT)
print(f"Verified final-matrix cells: {summary['verified']}/504; pending: {summary['missing']}")
print(summary['interpretation'])
display(pd.DataFrame(summary['groups']))
status_path=ROOT/'constraint_search/search_status.json'
if status_path.exists():
    status=json.loads(status_path.read_text())
    records=[]
    for name,attempt in status['attempts'].items():
        completed=attempt.get('results',[])
        records.append(dict(recipe=name,status=attempt['status'],pairs=len(completed),
            mean_delta_pp=sum(r['delta_dice_pp'] for r in completed)/9 if len(completed)==9 else None,
            minimum_delta_pp=min((r['delta_dice_pp'] for r in completed),default=None)))
    display(pd.DataFrame(records))
print('Full 504-cell CSV:',matrix_path)
'''
    notebook_path.write_text(json.dumps(notebook,indent=1)+'\n')
    print(json.dumps(dict(package=str(package),sha256=digest)))


if __name__=='__main__':main()
