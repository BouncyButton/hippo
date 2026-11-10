"""Build the single adapted-calibration retry notebook from the current notebook."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

ROOT=Path(__file__).resolve().parents[2]
EXPORT=ROOT/'reports/sam3_geofwi_colab_two_runs_20261018'


def build():
    notebook=json.loads((EXPORT/'notebook_before_warm_edge.ipynb').read_text())
    for cell in notebook['cells']:
        if isinstance(cell['source'],list):cell['source']=''.join(cell['source'])
        if cell['cell_type']=='code':cell.update(outputs=[],execution_count=None)
        cell.get('metadata',{}).pop('executionInfo',None)
    cells=notebook['cells']
    cells[0]['source']='''# SAM3 — one bands+edge retry with adapted calibration

**Run all after refreshing this notebook.** One model, five training volumes, **25 total optimizer updates**.

1. Updates **1–5:** native SAM3 loss + bands, using the saved bands coefficient.
2. After update **5:** calibrate only edge to a **10% median gradient target**, with the original
   **50% p95 calibration cap**, on the original **20 training-support slabs**. Keep bands fixed.
3. Updates **6–25:** native SAM3 loss + bands + the newly calibrated edge term.
4. Evaluate this final model once on the original **5 training + 96 validation volumes**.

The original learning-rate schedule, global constraint ramp, seeds, sampling and loss definitions
are retained. Calibration and monitoring do not perform optimizer updates. Edge strength is measured
again on five fixed slabs after updates **10 and 25**; these measurements do not change the weights.
The target and cap apply to calibration, not a guaranteed bound throughout training.

Saved baseline, bands and previous bands+edge results are reused; none is retrained or reevaluated.
The completed diagnostic notebook is backed up separately. New results are written to
`results_5vol_25updates_warm5_edge10_v1`. Use the same GPU/software as before.
'''
    cells[1]['source']=cells[1]['source'].replace("RESULTS=HOME_DIR/'diagnostics_edge_10updates_v1'",
        "PREVIOUS_RESULTS=HOME_DIR/'results_5vol_24conditions_v1'\nRESULTS=HOME_DIR/'results_5vol_25updates_warm5_edge10_v1'")
    cells[6]['source']='''## Stage the retry code

Reuse the original frozen bundle and saved references. The runner verifies the original initialization,
training volumes, baseline archive and completed constraint runs before starting.
'''
    files={}
    for source,dest in [('warm_edge.py','sam3_warm_edge_v1.py'),('diagnose.py','sam3_edge_diagnose_v1.py')]:
        shutil.copy2(ROOT/'scripts/colab_sam3_geofwi'/source,EXPORT/dest)
        files[dest]=hashlib.sha256((EXPORT/dest).read_bytes()).hexdigest()
    cells[7]['source']=textwrap.dedent(f'''
        RETRY_FILES={files!r}
        for name,expected in RETRY_FILES.items():
            source=INPUT/name
            if sha(source)!=expected:raise ValueError('Retry source differs: '+name+'; refresh the notebook')
            destination=Path('/content')/name
            shutil.copyfile(source,destination)
            if sha(destination)!=expected:raise ValueError('Retry copy failed: '+name)
        RETRY_SCRIPT=Path('/content/sam3_warm_edge_v1.py')
        print('Ready: ONE retry; 5 bands-only updates + 20 bands+edge updates; original controls reused.')
        ''').strip()+'\n'
    cells[8]['source']='''## Run or resume the single retry

The checkpoint is saved locally after every update, with periodic verified Drive recovery snapshots
and forced snapshots after calibration and monitoring. Interrupt once and wait for a clean stop
before disconnecting. Rerun this cell to resume. A hard VM loss with unsaved charged updates
raises an error rather than silently repeating them.

Final evaluation is the slow part: only this new model is evaluated. Audited evaluation feature
reuse is enabled with automatic fallback to the original implementation. Completed runs are skipped.
Checkpoints are removed only after the complete metrics archive is verified.
'''
    cells[9]['source']=textwrap.dedent('''
        execute([PYTHON,RETRY_SCRIPT,'--bundle',BUNDLE,'--source-results',SOURCE_RESULTS,
                 '--previous-results',PREVIOUS_RESULTS,'--durable',RESULTS],'retry.log')
        ''').strip()+'\n'
    cells[10]['source']='''## Results and effective edge strength

The comparison includes the three saved references and this one new run. Dice is measured on
the same validation cases. Gradient ratios use LoRA parameter gradients and are not Adam update shares.
The calibration row summarizes 20 slabs; later monitoring rows summarize five fixed slabs.
'''
    cells[11]['source']=textwrap.dedent('''
        import pandas as pd
        import numpy as np
        comparison=RESULTS/'comparison.json'
        if comparison.exists():
            table=pd.DataFrame(json.loads(comparison.read_text())['rows'])
            display(table)
            table.to_csv(RESULTS/'comparison.csv',index=False)
        else:
            print('Retry is incomplete; rerun the training cell to resume.')
        candidates=list((RESULTS/'runs').glob('*/calibration.json'))
        if candidates:
            cal=json.loads(candidates[0].read_text())
            print('Final calibrated coefficients:',cal['weights'])
            groups=[(5,cal['rows'])]+[(int(step),rows) for step,rows in cal['monitoring'].items()]
            rows=[]
            for step,items in sorted(groups):
                full=[r['full_strength']['edge_over_native_bands'] for r in items]
                scheduled=[r['scheduled_strength']['edge_over_native_bands'] for r in items]
                rows.append(dict(after_update=step,slabs=len(items),full_strength_median=float(np.median(full)),
                    full_strength_p95=float(np.quantile(full,.95)),full_strength_max=max(full),
                    next_update_schedule_median=float(np.median(scheduled)),
                    conflicting_slabs=sum(r['full_strength']['cosine_edge_native_bands'] is not None and
                        r['full_strength']['cosine_edge_native_bands']<0 for r in items)))
            monitoring=pd.DataFrame(rows);display(monitoring)
            monitoring.to_csv(RESULTS/'edge_monitoring.csv',index=False)
        print('Saved results:',RESULTS)
        ''').strip()+'\n'
    for cell in cells:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    notebook['metadata'].setdefault('colab',{})['name']='SAM3_GeoFWI3D_two_runs.ipynb'
    out=EXPORT/'SAM3_GeoFWI3D_warm_edge.ipynb'
    out.write_text(json.dumps(notebook,indent=2)+'\n')
    print(json.dumps(dict(notebook=str(out),files=files),indent=2))


if __name__=='__main__':build()
