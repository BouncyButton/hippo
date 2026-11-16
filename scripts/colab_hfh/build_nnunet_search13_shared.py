"""Freeze the shared-cap search and deployment cell without changing old runs."""
import os
import ast
import json
from pathlib import Path
import shutil
import zipfile
from scripts.colab_hfh.nnunet_search13_shared_policy import NAME,CAP,ARMS,ACTIVITY,SCHEDULE,candidates,make_protocol,recost
from scripts.colab_hfh.constraint_schedule import training_flops
from scripts.colab_hfh.runtime import read,write,digest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'
WORK = REPORT/NAME
REMOTE = Path('/content/hippo_'+NAME)


def extract_reference(archive,destination):
    with zipfile.ZipFile(archive) as z:
        for item in z.namelist():
            if item.startswith(('results/','protocol/')):
                target=destination/item;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(item))


def build():
    WORK.mkdir(parents=True,exist_ok=True)
    package = WORK/'package';package.mkdir(exist_ok=True)
    source = package/'source/scripts/colab_hfh';source.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'scripts/colab_hfh').glob('*.py'):shutil.copyfile(p,source/p.name)
    extract_reference(REPORT/'nnunet_composition_a100_v1/results/native.zip',package/'reference_a100_native')
    extract_reference(REPORT/'nnunet_seed0_repair/kickstart_plateau007.zip',package/'reference_shared_native')
    native = package/'protocols/native'
    shutil.copytree(package/'reference_shared_native/protocol',native,dirs_exist_ok=True)
    p = read(native/'protocol.json')
    if p['budget_flops'] != CAP or p['seed0_plateau_schedule'] != SCHEDULE:
        raise ValueError('Shared-cap native reference differs')
    p.update(name=NAME+'__native',folds=[0],seeds=[0],shared_cap_search=True,
             native_reference_folder='reference_shared_native',hardware_campaign=NAME)
    write(native/'protocol.json',p)
    budgets=[]
    for recipe in candidates():
        arm = recipe['arm']
        directory = next(REPORT/d/arm/'protocol' for d in ('nnunet_long_matrix_v1','nnunet_long_constraints_v1')
                         if (REPORT/d/arm/'protocol/protocol.json').exists())
        target = package/'protocols'/recipe['id']
        shutil.copytree(directory,target,dirs_exist_ok=True)
        p = make_protocol(read(directory/'protocol.json'),recipe,REMOTE)
        write(target/'protocol.json',p)
        s = p['models']['nnunet']['objectives'][arm]
        if not training_flops(s['max_steps'],s) <= CAP < training_flops(s['max_steps']+1,s):
            raise ValueError('Candidate does not exhaust shared cap exactly')
        budgets.append(dict(id=recipe['id'],arm=arm,prefix=recipe['prefix'],ramp=recipe['ramp'],
            updates=s['max_steps'],active_updates=s['max_steps']-recipe['prefix'],
            full_strength_updates=s['max_steps']-recipe['prefix']-recipe['ramp']+1,
            optimization_flops=training_flops(s['max_steps'],s),cap=CAP))
    design=dict(name=NAME,fold=0,seed=0,arms=list(ARMS),candidates=candidates(),
        candidates_count=len(candidates()),total_runs=1+len(candidates()),optimization_cap=CAP,
        original_cross_model_cap=CAP,max_workers=12,initial_verification_workers=3,threads_per_worker=4,
        worker_module='scripts.colab_hfh.nnunet_search13_shared_worker',activity_policy=ACTIVITY,
        native_schedule=SCHEDULE,native_updates=90,matched_prefixes=[20,30,45,60],
        selection='Final Dice among activity-verified candidates; all included raw penalties also reported; exploratory f0/s0',
        accounting='Each run charges full native prefix, all active updates and per-update activity probe within shared cap; calibration and post-training diagnostics separate as in original cross-model protocol',
        storage_account=os.environ.get('HIPPO_STORAGE_EMAIL', ''),compute_account=os.environ.get('HIPPO_COMPUTE_EMAIL', ''),
        no_other_fold_or_seed=True,prior_long_search='Superseded after first three pilots; excluded from shared-cap comparison')
    write(package/'search_design.json',design)
    write(package/'budget_verification.json',dict(status='all_fit_shared_cap',candidates=budgets))
    manifest={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*'))
              if p.is_file() and p.name!='manifest.json'}
    write(package/'manifest.json',manifest)
    archive=WORK/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*manifest,'manifest.json']:z.write(package/name,name)
    sha=digest(archive)
    old=REPORT/'nnunet_search13_a100_f0s0_v1'
    old_sha=read(old/'build_receipt.json')['package_sha256']
    launch=(old/'launch_cell.py').read_text()
    launch=launch.replace('nnunet_search13_a100_f0s0_v1',NAME).replace(old_sha,sha)
    launch=launch.replace('scripts.colab_hfh.nnunet_search13_campaign','scripts.colab_hfh.nnunet_search13_shared_campaign')
    launch=launch.replace('scripts.colab_hfh.test_nnunet_search13','scripts.colab_hfh.test_nnunet_search13_shared')
    launch=launch.replace('78 candidates + native','104 candidates + native | shared31.98T cap')
    launch=launch.replace('threads_per_worker=1','threads_per_worker=4')
    launch=launch.replace("OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'", "OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'")
    ast.parse(launch);(WORK/'launch_cell.py').write_text(launch)
    write(WORK/'build_receipt.json',dict(status='prepared_not_started',package_sha256=sha,
        bytes=archive.stat().st_size,files=len(manifest),candidates=len(candidates()),
        cap=CAP,native_updates=90,constraint_updates=sorted({r['updates'] for r in budgets}),
        active_updates_range=[min(r['active_updates'] for r in budgets),max(r['active_updates'] for r in budgets)]))
    (WORK/'README.md').write_text('''# Shared-cap nnU-Net search: fold0 / seed0

User explicitly selected the original cross-model cap: 31,976,450,667,090 optimization FLOPs per run. Native uses the already studied kickstart/plateau/decay90 recipe, unchanged native Dice+CE and deep supervision, SGD momentum0.95, no augmentation, BF16. Initial20 updates use original polynomial LR; plateau0.07 until update60, then decay to update90. No doubled-budget endpoint is used as comparator.

104 candidates cover all13 combinations. Strengths0.05/0.10/0.20 at prefix45/ramp9; early0.10 at prefix30/ramp18; earliest gentle0.05 at prefix20/ramp20; late reference0.10 at prefix60/ramp6. Additional allocations emphasize each included component2x at prefix45/ramp9/strength0.10. The early training model predicts almost no foreground before update40 in the previous native trace; early activation is therefore an explicitly tested hypothesis, not an assumption of readiness. Exact native model hashes at updates20/30/45/60 are required before each candidate activates.

Every active training update measures native and individual auxiliary logit gradients. At full ramp each included component has at least1% of native RMS and10% of the sum of component norms; the sum is capped at50%. During ramp the floors scale with the shared ramp. Keep frozen anchor weights when feasible, otherwise project contributions into this bounded envelope; coefficients may not exceed64x their anchor. No component is silently dropped. These are operational activity guarantees in full-resolution logit space, not hard anatomical guarantees.

The probe allowance is charged inside the shared optimization cap on every active update. Every prefix update is also charged even if reusable in principle. Calibration and final read-only diagnostics are separately reported, consistently with the original cross-model optimization-only scope. Final Dice, per-case FP/FN and all four raw canonical penalty values are archived for native and candidates; A/P swaps do not apply to this binary task. Selection is exploratory on one validation fold and seed; no independent validation claim. Up to12 workers with measured resource headroom; verify three initial runs before expansion. Verify remote archives before deleting completed checkpoints; retain failed recovery. No automatic other-seed/fold expansion.
''')
    print(json.dumps(read(WORK/'build_receipt.json'),indent=2))


if __name__=='__main__':build()
