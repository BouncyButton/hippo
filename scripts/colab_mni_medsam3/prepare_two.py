"""Revise the existing Colab notebook for one accelerated two-subject pilot."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

from scripts.colab_mni_medsam3 import prepare as first
from scripts.colab_mni_medsam3.data import select_support
from scripts.cross_dataset_campaign.common import digest, read, write

ROOT=first.ROOT
EXPORT=ROOT/'reports/medsam3_mni_colab_two_subjects_20261108'
ORIGIN=ROOT/'reports/medsam3_mni_colab_pilot_20261108/bundle'


def build():
    bundle=EXPORT/'bundle'
    if bundle.exists():raise ValueError('Preserve existing frozen export')
    for name,sha in read(ORIGIN/'manifest.json').items():
        if digest(ORIGIN/name)!=sha:raise ValueError('Original pilot changed: '+name)
    shutil.copytree(ORIGIN,bundle,ignore=shutil.ignore_patterns('__pycache__'))
    module=bundle/'source/scripts/colab_mni_medsam3'
    for name in ('optimized.py','supervisor.py','prepare_two.py'):
        shutil.copy2(Path(__file__).parent/name,module/name)
    framework=bundle/'source/scripts/colab_medsam3'
    framework.mkdir(exist_ok=True)
    provenance={}
    for name in ('__init__.py','parallel.py','runtime.py','fast_eval.py'):
        source=ROOT/'scripts/colab_medsam3'/name
        shutil.copy2(source,framework/name)
        provenance[name]=digest(source)
    # Three fixed MNI text prompts; same cache, numerical audit and fallback.
    fast=framework/'fast_eval.py'
    text=fast.read_text()
    if text.count('if len(self.text) < 2:')!=1:raise ValueError('Unexpected cache source')
    fast.write_text(text.replace('if len(self.text) < 2:','if len(self.text) < 3:'))
    p=read(ORIGIN/'protocol.json')
    selected=select_support(read(ORIGIN/'saved_folds.json'),2)
    p.update(selected)
    p.update(name='MNI_MedSAM3_n2_f0s0_native200',train_subjects=2,train_crops=4,
        support_selection='First two subjects of the same fixed RandomState(0) MNI fold0 ordering; both L/R crops. Nested extension of the completed n1 pilot.',
        decision_rule='Only the two-subject fold0 seed0 run; inspect before any sample-size or seed/fold extension.')
    for key in ('data','upstream','lora_weights'):
        p[key]=p[key].replace('/content/hippo_mni_medsam3','/content/hippo_mni_medsam3_n2_optimized')
    write(bundle/'protocol.json',p)
    audit=read(bundle/'data_audit.json');audit['selection']=selected;write(bundle/'data_audit.json',audit)
    write(bundle/'plan.json',dict(tasks=[dict(id=p['name'],arm='dice',subjects=2,fold=0,seed=0)],
        optimization='MSD async immutable recovery and audited image/text evaluation reuse; original native training retained',
        concurrency='Only one task is authorized; worker count cannot exceed the listed task count'))
    write(bundle/'source_provenance.json',dict(parent_manifest_sha256=digest(ORIGIN/'manifest.json'),
        parent_protocol_sha256=digest(ORIGIN/'protocol.json'),framework_sources=provenance,
        adaptations=['MNI four-class evaluator bridge','Text cache capacity2→3 for three fixed subfield prompts',
                     'Single-task supervisor using MSD resource admission, telemetry and clean-stop handling'],
        training_changed=False,scientific_change='Training support expanded from1subject/2crops to2subjects/4crops',
        gpu_validation='Pending Colab; full-volume numerical and speed audit gates evaluation cache use'))
    write(bundle/'manifest.json',{str(f.relative_to(bundle)):digest(f) for f in sorted(bundle.rglob('*'))
        if f.is_file() and f!=bundle/'manifest.json' and '__pycache__' not in f.parts})
    package=EXPORT/'medsam3_mni_n2_optimized.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:z.write(f,'bundle/'+str(f.relative_to(bundle)))
    first.EXPORT=EXPORT
    first.notebook(package,bundle)
    path=EXPORT/'MedSAM3_MNI_one_subject_pilot.ipynb'
    notebook=read(path)
    notebook['cells'][0]['source']=textwrap.dedent('''
    # MNI MedSAM3: two-subject pilot
    Updated in the same notebook. Sign in as **filippofocaccia2@gmail.com**, select **A100** (80GB preferred), and run all cells.

    **Exactly one experiment:** subjects s23+s03, four L/R crops; fold0, seed0; same ten validation crops,
    native MedSAM3 loss, medical LoRA initialization, 200 updates, accumulation2, warmup20, LR5e-5→5e-6, no augmentation.
    The completed one-subject results remain available for the comparison at the end.

    Reuses the MSD/ADNI framework: deterministic CPU transform caching, immutable background recovery uploads,
    memory admission checks, live GPU telemetry, and checkpoint-specific evaluation feature reuse.
    Fast evaluation must match the original full-volume predictions and log-probabilities and show a measured
    speed gain; each subsequent volume has a checked slice. Failure automatically uses the original evaluator.
    Three prompts share one image encoding per slice during audited inference. Training's batch size and GPU graph are unchanged.

    Only this n2 run is scheduled, so the resource-aware worker limit is one. No extra subjects, seeds or folds are launched.
    Run only one VM for this study. Interrupt the Run cell once and wait for all recovery uploads before disconnecting.
    Background recovery snapshots are attempted every180seconds; abrupt VM loss can replay work since the latest verified snapshot.
    New MNI acceleration has CPU tests; its A100 equivalence and speed are checked when you run it.
    ''').strip()+'\n'
    for cell in notebook['cells'][1:]:
        cell['source']=cell['source'].replace('/content/hippo_mni_medsam3','/content/hippo_mni_medsam3_n2_optimized')
        cell['source']=cell['source'].replace('medsam3_mni_pilot.zip','medsam3_mni_n2_optimized.zip')
        cell['source']=cell['source'].replace('results_n1_f0s0_native200_v1','results_n2_f0s0_native200_optimized_v1')
    staging=notebook['cells'][2]
    staging['source']=staging['source'].replace("BASE = ROOT/'sam3.pt'", "BASE = ROOT/'sam3.pt'\nprevious_base = Path('/content/hippo_mni_medsam3/sam3.pt')\nif previous_base.exists() and sha(previous_base)==protocol['base_sha256']: BASE=previous_base")
    setup=notebook['cells'][3]
    setup['source']=setup['source'].replace("VENV = ROOT/'venv'", "VENV = ROOT/'venv'\nprevious_venv=Path('/content/hippo_mni_medsam3/venv')\nif (previous_venv/'bin/python').exists(): VENV=previous_venv")
    setup['source']=setup['source'].replace("['protocol.json','saved_folds.json'", "['protocol.json','plan.json','saved_folds.json'")
    notebook['cells'][4]['source']='## Run the two-subject pilot\n\nThe existing MSD GPU controller prints usage and progress every30seconds. Run the results cell afterward to compare against n1 and complete verified cleanup.\n'
    notebook['cells'][5]['source']=textwrap.dedent('''
    command=[str(PYTHON),'-m','scripts.colab_mni_medsam3.supervisor',
        '--root',str(ROOT),'--durable',str(RESULTS),'--base',str(BASE),'--max-workers','0','--hours','12']
    process=subprocess.Popen(command,cwd=BUNDLE/'source',env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (RESULTS/'supervisor.log').open('a') as log:
        while True:
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait();break
            except KeyboardInterrupt:
                print('Clean stop requested; wait for verified Drive recovery.')
                if process.poll() is None:process.send_signal(signal.SIGINT)
    if code:raise RuntimeError('Run stopped or failed. Inspect the logs; preserve recovery. Rerun to resume after resolving the cause.')
    done=json.loads((RESULTS/'completion.json').read_text())
    print('Completed two-subject pilot:',done)
    ''').strip()+'\n'
    results=notebook['cells'][6]
    results['source']=results['source'].replace("('.json','.csv','.png','.log')", "('.json','.csv','.png','.log','.jsonl')")
    results['source']=results['source'].replace("print('Inspect this pilot before choosing a target or scheduling more subjects/seeds/folds.')",textwrap.dedent('''
    # Retire only the verified final snapshot for this completed experiment.
    import tarfile
    recovery=RESULTS/'recovery'
    marker=recovery/'latest.json'
    if marker.exists():
        record=json.loads(marker.read_text());name=record['file']
        if Path(name).name!=name or not name.startswith('recovery-'):raise ValueError('Unsafe recovery path')
        archive=recovery/name
        if sha(archive)!=record['sha256']:raise ValueError('Recovery checksum mismatch; retained')
        with tarfile.open(archive) as tar:
            snapshot=json.load(tar.extractfile('snapshot_manifest.json'))
        if snapshot['checkpoint_latest.pt']!=done['checkpoint_sha256']:raise ValueError('Recovery is from another checkpoint; retained')
        cleanup=json.loads(cleanup_path.read_text())
        cleanup['deleted'].append(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))
        cleanup_path.write_text(json.dumps(cleanup,indent=2))
        archive.unlink();marker.unlink()
    previous=HOME/'results_n1_f0s0_native200_v1/summary.csv'
    if previous.exists():
        old=pd.read_csv(previous,index_col=0)
        comparison=pd.DataFrame({'one_subject':old['mean_subfield_dice'],'two_subjects':summary['mean_subfield_dice']})*100
        comparison['gain_percentage_points']=comparison['two_subjects']-comparison['one_subject']
        display(comparison)
        comparison.to_csv(RESULTS/'comparison_with_one_subject.csv')
    if (RESULTS/'evaluation_acceleration.json').exists():
        acceleration=json.loads((RESULTS/'evaluation_acceleration.json').read_text())
        print('Evaluation backend:',acceleration['status'],'measured slice speedup:',acceleration.get('observed_slice_speedup'),'fallback:',acceleration.get('fallback_reason'))
    print('Inspect the n2 result before scheduling any other experiment.')
    ''').strip())
    results['source'] += "\n# Include the derived n1/n2 comparison in the final durable archive.\nmetadata={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.png','.log','.jsonl') and p.name not in ('archive_manifest.json','cleanup.json')}\nmanifest_path.write_text(json.dumps(metadata,indent=2))\nfor name, expected_sha in metadata.items():\n    if sha(RESULTS/name)!=expected_sha: raise ValueError('Final archive changed')\n"
    # Preserve the user's GPU inventory cell using standard Python syntax.
    notebook['cells'].insert(1,dict(cell_type='code',id='mni_gpu_inventory',metadata={},
        execution_count=None,outputs=[],source="import subprocess\nsubprocess.run(['nvidia-smi'],check=True)\n"))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    notebook['metadata']['colab']['name']='MedSAM3_MNI_sample_size_pilot.ipynb'
    write(path,notebook)
    write(EXPORT/'build_verification.json',dict(package_sha256=digest(package),
        notebook_sha256=digest(path),manifest_sha256=digest(bundle/'manifest.json'),
        training_cases=p['train_cases'],validation_cases=p['evaluation_cases'],runs=1,gpu_test=False))
    print(json.dumps(read(EXPORT/'build_verification.json'),indent=2))


if __name__=='__main__':build()
