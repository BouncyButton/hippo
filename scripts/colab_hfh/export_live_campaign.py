"""Preserve exact executed source and final campaign status on the storage account."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root
    suffix='' if json.loads((root/'bundle/gpu_overlay.json').read_text())['precision']=='bf16' else '_fp32'
    status_path=root/('consistent_recipes_status'+suffix+'.json')
    status=json.loads(status_path.read_text())
    if status['status']!='all_four_models_passed_same_nine_cells':
        raise RuntimeError('Campaign not complete; leave active evidence untouched')
    assert len(status['selected_recipes'])==4
    for model,recipe in status['selected_recipes'].items():
        rows=[r for r in status['attempts'] if r['model']==model and r['lr']==recipe['initial_lr']]
        assert {(r['fold'],r['seed']) for r in rows}=={(f,s) for f in range(3) for s in range(3)}
        assert all(.5<=r['validation_dice']<=1 for r in rows)
    target=root/'campaign_evidence.zip'
    sources=[root/n for n in ('consistent_recipes.py','recipe_probe.py','gpu_runtime.py',
        'consistent_recipes.log',status_path.name)]
    sources += [root/'bundle'/n for n in ('manifest.json','runtime_manifest.json','gpu_overlay.json')]
    sources += [p for p in (root/'bundle/source').rglob('*') if p.is_file()
                and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.sh')]
    sources += [p for p in (root/'work_baselines/nnunet_f0_s0').glob('*')
                if p.is_file() and p.suffix in ('.json','.csv','.log')]
    manifest={}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            if p.exists():
                name=str(p.relative_to(root));payload=p.read_bytes()
                z.writestr(name,payload);manifest[name]=hashlib.sha256(payload).hexdigest()
        z.writestr('evidence_sha256.json',json.dumps(manifest,indent=2))
    private=Path('/content/.hfh_storage_auth')
    rc=[str(private/'rclone'),'--config',str(private/'rclone.conf'),
        '--tpslimit','2','--tpslimit-burst','2','--retries','3']
    for local,name in ((status_path,'campaign_status.json'),(target,target.name)):
        remote='hfh_storage:results_recipe_diagnostics_v3'+suffix+'/'+name
        subprocess.run(rc+['copyto',str(local),remote],check=True,timeout=600)
        digest=subprocess.check_output(rc+['md5sum',remote],text=True,timeout=120).split()[0]
        assert digest==hashlib.md5(local.read_bytes()).hexdigest()
        print('VERIFIED',name,flush=True)


if __name__=='__main__':
    main()
