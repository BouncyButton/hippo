"""Update the existing SAM3 notebook to the eleven missing established arms."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

from scripts.colab_sam3_geofwi.missing_constraints import ARMS

ROOT=Path(__file__).resolve().parents[2]
EXPORT=ROOT/'reports/sam3_geofwi_colab_two_runs_20261018'


def build():
    nb=json.loads((EXPORT/'notebook_before_missing_constraints.ipynb').read_text())
    for cell in nb['cells']:
        if isinstance(cell['source'],list):cell['source']=''.join(cell['source'])
        if cell['cell_type']=='code':cell.update(outputs=[],execution_count=None)
        cell.get('metadata',{}).pop('executionInfo',None)
    cells=nb['cells']
    labels=[a.removeprefix('dice_').replace('_',' + ') for a in ARMS]
    cells[0]['source']='''# SAM3 — complete the missing constraint matrix

**Refresh this notebook, then Run all.** GeoFWI3D, **fold0 / seed0**, **five training volumes**,
**25 total updates per arm**, no augmentation. Each final model is evaluated once on the same
five training-support and96 validation volumes. There are **11 new runs**:

'''+''.join(f'- {label}\n' for label in labels)+'''
Native SAM3 training loss is included in every arm; legacy `dice_` identifiers do not mean Dice-only loss.
Baseline, bands, original bands+edge and adapted-calibration bands+edge are reused from their saved
archives. This notebook does not repeat them or launch another edge-only tuning experiment.

### What we learned and what this experiment tests

The original bands+edge scored52.00% Dice, adapted calibration62.53%, and bands alone68.41%.
The adapted edge recipe improved a failed result, but still increased missed foreground and did not
beat bands. The purpose is to evaluate constraints fairly and preserve negative findings, not to force
one constraint to win.

- Initial gradient calibration can become inappropriate as SAM3 adapts.
- A10% median target /50% p95 calibration cap is **not a persistent gradient bound**.
- Positive gradient alignment does not establish validation benefit.
- Five diagnostic anchors are not the same distribution as the20 calibration slabs.
- Matching the recipe does not guarantee bitwise-identical GPU trajectories; CPU tests do not prove that.

### Common recipe for the missing arms

Updates1–5 use native+bands for arms containing bands, or native loss for arms without bands.
The saved bands coefficient is retained wherever bands is present. After update5, all other selected
terms are calibrated sequentially in the established order **edge → SDF → depth** on the original20
training slabs, targeting10% median with50% p95 cap. Updates6–25 use the resulting fixed coefficients
and the original global constraint ramp. No hard online cap, gradient projection or validation-driven tuning
is added. Monitoring at updates10 and25 records individual and combined auxiliary gradient strength
on the same five anchors. Compare those five anchors across stages, not the20-slab median to a5-slab median.

Existing controls used earlier calibration policies, so this is a transparent exploratory extension,
not a fully matched factorial comparison. Report all arms, including worse outcomes.

One worker runs at a time. Every completed arm is archived and its checkpoint removed before continuing.
Use the same GPU/software as the preceding runs. All new outputs go to
`results_5vol_25updates_missing11_warm5_v1`; earlier folders are preserved.
'''
    old="RESULTS=HOME_DIR/'results_5vol_25updates_warm5_edge10_v1'"
    assert old in cells[1]['source']
    cells[1]['source']=cells[1]['source'].replace(old,
        "WARM_RESULTS=HOME_DIR/'results_5vol_25updates_warm5_edge10_v1'\nRESULTS=HOME_DIR/'results_5vol_25updates_missing11_warm5_v1'")
    cells[6]['source']='''## Stage the missing-arm runner

The existing code/data/weights bundle is reused. The new runner activates the existing SDF and depth
geometry for each applicable arm and checks that the required fields are finite and nonzero.
All four reference archives are verified before new training starts.
'''
    files={}
    for source,dest in [('missing_constraints.py','sam3_missing_constraints_v1.py'),('diagnose.py','sam3_edge_diagnose_v1.py')]:
        shutil.copy2(ROOT/'scripts/colab_sam3_geofwi'/source,EXPORT/dest)
        files[dest]=hashlib.sha256((EXPORT/dest).read_bytes()).hexdigest()
    cells[7]['source']=textwrap.dedent(f'''
        CAMPAIGN_FILES={files!r}
        for name,expected in CAMPAIGN_FILES.items():
            source=INPUT/name
            if sha(source)!=expected:raise ValueError('Campaign source differs: '+name+'; refresh the notebook')
            destination=Path('/content')/name
            shutil.copyfile(source,destination)
            if sha(destination)!=expected:raise ValueError('Campaign copy failed: '+name)
        CAMPAIGN_SCRIPT=Path('/content/sam3_missing_constraints_v1.py')
        print('Ready:11 missing arms, fold0 seed0,5 volumes,25 updates each; four saved references reused.')
        ''').strip()+'\n'
    cells[8]['source']='''## Run or resume the missing arms

This cell runs every unfinished arm sequentially. It prioritizes interrupted work, skips verified completed
arms, and updates the comparison after each completion. Each child process releases its GPU memory before
the next model loads. Calibration uses training support only; final evaluation is the slowest phase.

Interrupt once and wait for the clean-stop message before disconnecting. Local checkpoints are saved
after each update, with verified Drive snapshots periodically and at calibration/monitoring boundaries.
An abrupt VM loss with unsaved charged updates stops for recovery rather than silently repeating updates.
Audited evaluation feature reuse falls back to the original implementation when its checks fail.
'''
    cells[9]['source']=textwrap.dedent('''
        execute([PYTHON,CAMPAIGN_SCRIPT,'--bundle',BUNDLE,'--source-results',SOURCE_RESULTS,
                 '--previous-results',PREVIOUS_RESULTS,'--warm-results',WARM_RESULTS,
                 '--durable',RESULTS],'missing_constraints.log')
        ''').strip()+'\n'
    cells[10]['source']='''## All completed results and gradient monitoring

The table keeps saved controls and every completed new arm, regardless of ranking. Dice is the same
per-volume mean used previously; FN/FP are foreground voxel counts. Monitoring compares the identical
five anchors at updates5,10,25. The full20-slab calibration and selected coefficients remain in each
arm's `calibration.json`. Gradient norms are not shares of the Adam optimizer update.
'''
    cells[11]['source']=textwrap.dedent('''
        import pandas as pd
        import numpy as np
        path=RESULTS/'comparison.json'
        if path.exists():
            result=json.loads(path.read_text())
            print(f"New arms completed: {result['new_completed']}/{result['new_total']}; references reused: {result['reused_runs']}")
            table=pd.DataFrame(result['rows']);display(table)
            table.to_csv(RESULTS/'comparison.csv',index=False)
            print(result['interpretation'])
        else:print('No results yet; run the campaign cell.')
        summaries=[]
        for path in sorted((RESULTS/'runs').glob('*/calibration.json')):
            cal=json.loads(path.read_text())
            anchors={(r['case'],r['anchor']) for r in cal['monitoring']['10']}
            groups=[(5,[r for r in cal['rows'] if (r['case'],r['anchor']) in anchors])]
            groups += [(int(step),rows) for step,rows in cal['monitoring'].items()]
            for step,rows in sorted(groups):
                for name in cal['names'][1:]:
                    full=[r['full_strength'][name]['relative_norm'] for r in rows]
                    scheduled=[r['scheduled_strength'][name]['relative_norm'] for r in rows]
                    summaries.append(dict(run=path.parent.name,after_update=step,term=name,slabs=len(rows),
                        coefficient=cal['weights'][name],full_median=float(np.median(full)),full_max=max(full),
                        scheduled_median=float(np.median(scheduled)),
                        conflicting_slabs=sum(r['full_strength'][name]['cosine'] is not None and r['full_strength'][name]['cosine']<0 for r in rows),
                        combined_auxiliary_over_native_median=float(np.median([r['full_strength']['combined_auxiliary_over_native'] for r in rows]))))
        if summaries:
            monitoring=pd.DataFrame(summaries);display(monitoring)
            monitoring.to_csv(RESULTS/'constraint_monitoring.csv',index=False)
        print('Results and recovery:',RESULTS)
        ''').strip()+'\n'
    for cell in cells:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    nb['metadata'].setdefault('colab',{})['name']='SAM3_GeoFWI3D_two_runs.ipynb'
    output=EXPORT/'SAM3_GeoFWI3D_missing_constraints.ipynb'
    output.write_text(json.dumps(nb,indent=2)+'\n')
    lesson=dict(scope='GeoFWI3D SAM3 fold0 seed0;5 training volumes;25 updates',
        missing_arms=list(ARMS),reused=['native baseline','bands','original bands+edge','adapted-calibration bands+edge'],
        lesson='Calibrate after adaptation; monitor gradient magnitude and alignment; do not equate calibrated targets with persistent bounds or alignment with validation benefit; preserve negative results.',
        controls='Earlier calibration policies are reused as references, not claimed fully matched controls.',
        policy='5 prefix updates; fixed saved bands if present; sequential10% median/50% p95 calibration of new terms; no online cap; no tuning on validation.')
    (EXPORT/'missing_constraints_lessons.json').write_text(json.dumps(lesson,indent=2)+'\n')
    print(json.dumps(dict(notebook=str(output),files=files,arms=list(ARMS)),indent=2))


if __name__=='__main__':build()
