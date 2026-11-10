"""Replace the current Colab notebook's experiment launch with a bounded diagnostic."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

ROOT=Path(__file__).resolve().parents[2]
EXPORT=ROOT/'reports/sam3_geofwi_colab_two_runs_20261018'


def build():
    before=json.loads((EXPORT/'notebook_before_diagnosis.ipynb').read_text())
    notebook=deepcopy(before)
    for cell in notebook['cells']:
        if isinstance(cell['source'],list):cell['source']=''.join(cell['source'])
        if cell['cell_type']=='code':cell.update(outputs=[],execution_count=None)
        cell.get('metadata',{}).pop('executionInfo',None)
    cells=notebook['cells']
    cells[0]['source']=textwrap.dedent('''
        # SAM3 — diagnose the edge constraint

        One short **10-update native-loss adaptation**, using the original five training volumes.
        Probe the same five adjacent-slice slabs at **updates 0, 5, and 10**. No baseline evaluation,
        validation inference, constraint-training sweep, or new hyperparameter search is run.

        The probe measures the native/bands/edge LoRA gradients, their alignment, the influence of
        the existing coefficients, and gradients into mask, query-confidence and presence logits.
        RNG and model buffers are restored after each probe so probing does not change adaptation.
        The old calibration and completed results are read from their existing Drive folder.

        **Run all** after refreshing this notebook. The original code/data/weights are reused.
        All new files go to `diagnostics_edge_10updates_v1`. Completed diagnostics are skipped;
        interrupt once and wait for a clean stop to resume safely. Do not run the old sweep concurrently.
        Use the same GPU/software as the completed run (A100 80 GB in that run).

        The previous notebook, including its outputs, is preserved separately. Existing validation
        Dice: baseline 65.10%, bands 68.41%, bands+edge 52.00%.
        ''').strip()+'\n'
    old="RESULTS=HOME_DIR/'results_5vol_25updates_bands10_edge10_v1'"
    assert old in cells[1]['source']
    cells[1]['source']=cells[1]['source'].replace(old,
        "SOURCE_RESULTS=HOME_DIR/'results_5vol_25updates_bands10_edge10_v1'\nRESULTS=HOME_DIR/'diagnostics_edge_10updates_v1'")
    source=ROOT/'scripts/colab_sam3_geofwi/diagnose.py'
    dest=EXPORT/'sam3_edge_diagnose_v1.py';shutil.copy2(source,dest)
    sha=hashlib.sha256(dest.read_bytes()).hexdigest()
    cells[6]['source']='''## Use the existing calibration

No full recalibration is run. The diagnostic verifies the completed archive, original initialization,
training support, saved coefficients, and GPU/software identity before adapting.
'''
    cells[7]['source']=textwrap.dedent(f'''
        DIAGNOSTIC_SOURCE=INPUT/'sam3_edge_diagnose_v1.py'
        DIAGNOSTIC_SCRIPT=Path('/content/sam3_edge_diagnose_v1.py')
        if sha(DIAGNOSTIC_SOURCE)!={sha!r}:raise ValueError('Diagnostic source differs; refresh the notebook')
        shutil.copyfile(DIAGNOSTIC_SOURCE,DIAGNOSTIC_SCRIPT)
        if sha(DIAGNOSTIC_SCRIPT)!={sha!r}:raise ValueError('Diagnostic copy failed')
        print('Ready: 10 native updates; 5 support slabs at updates 0, 5, 10; no validation inference.')
        ''').strip()+'\n'
    cells[8]['source']='''## Run or resume the diagnostic

One worker runs at a time. The five fixed probes cover one slab from each training volume,
using anchors from the original calibration. Each probe reuses its forward graph for three
loss-gradient measurements. Checkpoints stay local each update and are copied to Drive at
updates 0, 5 and 10, or on a clean stop. Measurements are saved after every slab.

Interrupt once and wait for the cell to finish saving. A hard VM loss that loses charged updates
raises an error rather than silently repeating them. On completion, the verified small diagnostic
archive is retained and diagnostic recovery checkpoints are removed.
'''
    cells[9]['source']=textwrap.dedent('''
        execute([PYTHON,DIAGNOSTIC_SCRIPT,'--bundle',BUNDLE,
                 '--source-results',SOURCE_RESULTS,'--durable',RESULTS],
                'diagnostic.log')
        ''').strip()+'\n'
    cells[10]['source']='''## Interpret the diagnosis

- **Edge/native+bands ratio:** the saved edge coefficient applied to the current gradients.
  The main table shows hypothetical full-strength ratios, so different probe stages are comparable.
  Per-slab JSON also records the original next-update warmup factor.
- **Conflicting slabs:** edge has a negative gradient dot product with native+bands.
- **Confidence derivatives:** a positive presence/query logit gradient pushes that confidence
  downward under direct gradient descent. These output-head derivatives are not additive
  percentages of the LoRA update; Adam and shared parameters can change the realized response.
- **Candidate weight:** what the original median/p95 rule suggests on these five probes with
  bands fixed. It is a diagnostic comparison, not a replacement for 20-slab production calibration.

This observes a **native-only adaptation trajectory**, not a replay of the failing edge trajectory.
Ratios growing during adaptation would support the calibration-drift hypothesis. Stable ratios
would weaken it. Five training slabs cannot establish validation benefit or prove causation.
'''
    cells[11]['source']=textwrap.dedent('''
        import pandas as pd
        import matplotlib.pyplot as plt
        if not (RESULTS/'summary.json').exists():
            print('No probes saved yet; run or resume the diagnostic cell.')
        else:
            summary=pd.DataFrame(json.loads((RESULTS/'summary.json').read_text()))
            display(summary)
            summary.to_csv(RESULTS/'summary.csv',index=False)
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for name,label in [('edge_ratio_median','Median'),('edge_ratio_p95','95th percentile'),('edge_ratio_max','Maximum')]:
                axes[0].plot(summary['update'],summary[name],marker='o',label=label)
            axes[0].axhline(.1,color='gray',linestyle='--',label='10% target')
            axes[0].axhline(.5,color='gray',linestyle=':',label='50% calibration p95 cap')
            axes[0].set(xlabel='Native adaptation updates',ylabel='Weighted edge / native+bands gradient norm')
            axes[0].legend()
            for name,label in [('mean_foreground_probability','Foreground probability'),('mean_presence_probability','Presence confidence')]:
                axes[1].plot(summary['update'],summary[name],marker='o',label=label)
            axes[1].set(xlabel='Native adaptation updates',ylabel='Probability',ylim=(0,1))
            axes[1].legend();fig.tight_layout();fig.savefig(RESULTS/'diagnostic.png',dpi=140);plt.show()
            status='complete' if (RESULTS/'completion.json').exists() else 'partial — resume the diagnostic cell'
            print('Diagnosis:',status)
            print('Detailed per-slab gradients and head derivatives:',RESULTS/'measurements.json')
        ''').strip()+'\n'
    notebook['metadata'].setdefault('colab',{})['name']='SAM3_GeoFWI3D_two_runs.ipynb'
    for cell in cells:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    out=EXPORT/'SAM3_GeoFWI3D_edge_diagnosis.ipynb'
    out.write_text(json.dumps(notebook,indent=2)+'\n')
    print(json.dumps(dict(notebook=str(out),diagnostic_source_sha256=sha,cells=len(cells)),indent=2))


if __name__=='__main__':build()
