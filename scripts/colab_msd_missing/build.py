"""Build a self-contained missing-only campaign from the completed recovery audit."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'reports/msd_colab_seeds0220_recovery_20261018'
OUT=ROOT/'reports/msd_missing_324_second_account_20261018'
PARENT=ROOT/'reports/msd_colab_3fold_3seed_noaug_20261015'
sys.path.insert(0,str(HERE))
from notebook import notebook


def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def build():
    OUT.mkdir(exist_ok=True)
    package=OUT/'package'
    if package.exists():shutil.rmtree(package)  # Generated staging only.
    shutil.copytree(OLD/'package',package,ignore=shutil.ignore_patterns(
        '__pycache__','preserved_results.zip','recovery_evidence.zip','package_manifest.json'))
    payload=package/'seed20_bundle/data/msd_hippocampus_full.pkl'
    if payload.exists() or payload.is_symlink():payload.unlink()  # Dataset is supplied once, in the separate data ZIP.
    with zipfile.ZipFile(PARENT/'msd_replication_code.zip') as z:z.extractall(package/'original_bundle')
    audit=read(OLD/'recovery_plan.json')
    all_tasks=[t for t in read(package/'original_bundle/plan.json')['tasks'] if t['seed'] in (0,2)]
    all_tasks += read(package/'seed20_bundle/plan.json')['tasks']
    assert len(all_tasks)==504 and len({t['id'] for t in all_tasks})==504
    kept={r['id'] for r in audit['records'] if r['action'] in ('preserve_verified','preserve_recovered')}
    missing=[t for t in all_tasks if t['id'] not in kept]
    excluded=[t for t in all_tasks if t['id'] in kept]
    assert len(kept)==len(excluded)==180 and len(missing)==324
    assert {t['seed'] for t in all_tasks}=={0,2,20}
    scope=dict(schema=1,audit_sha256=sha(OLD/'recovery_plan.json'),
        all_tasks=all_tasks,missing_tasks=missing,excluded_completed_tasks=excluded,
        original_total=504,missing_count=324,excluded_count=180,
        seeds=[0,2,20],folds=[0,1,2],augmentation='none',
        policy='Fixed exclusion list from the existing audit. No original-account folder listing or old run verification.')
    write(package/'missing_scope.json',scope)
    reference=[r['reported_result'] for r in audit['records'] if r['id'] in kept]
    assert {r['id'] for r in reference}==kept
    write(package/'historical_reference.json',reference)
    # One existing incomplete checkpoint was copied server-side to the new account.
    for row in audit['interrupted']:
        if row.get('available_checkpoint'):
            assert row['id']=='MSD_unetr_vit_b16_dice_bands_f1s2'
            row['available_checkpoint'].update(id='1wQMNczpf3vSG5NJVWn_koQs8q1s8s1Gf',
                name='resume_unetr_f1s2_bands_acfa846b.tar',
                account='filippofocaccia2@gmail.com')
    write(package/'recovery_plan.json',audit)
    for phase in ('02','20'):
        folder=package/'bindings'/phase;folder.mkdir(parents=True)
        for name in ('environment_binding.json','preflight.json'):
            shutil.copyfile(OLD/'preserved'/phase/name,folder/name)
    cloud=package/'cloud'
    # Always package the reviewed live storage code, not an older frozen copy of
    # the orchestrator. Scientific source bundles above remain byte-identical.
    for name in ('store.py','hooks.py','driver.py','local_recovery.py'):
        shutil.copyfile(ROOT/'scripts/colab_msd_cloud'/name, cloud/name)
    shutil.copyfile(cloud/'driver.py',cloud/'recovery_driver.py')
    (cloud/'driver.py').unlink()
    for name in ('missing.py','traffic.py'):shutil.copyfile(HERE/name,cloud/name)
    shutil.copyfile(HERE/'scope.py',package/'scope.py')
    store=(cloud/'store.py').read_text()
    store=replace_once(store,"ACCOUNT = 'focacciafilippo@gmail.com'","ACCOUNT = 'filippofocaccia2@gmail.com'")
    store=replace_once(store,"CAMPAIGN = '1ipls98K9dmRQ5rZwfbepw1JyWVL3Yq0P'","CAMPAIGN = '1fErhcTI519LV2wpPtbzLjiHd5dzxRUQu'")
    store=replace_once(store,"STORE_NAME = 'results_seeds0220_server_verified_v2'","STORE_NAME = 'results_noaug_v1'")
    store=replace_once(store,'        creating = file_id is None',
        "        if path.stat().st_size > 4*1024**2:\n            from traffic import reserve_upload\n            reserve_upload(self, path, properties)\n        creating = file_id is None")
    (cloud/'store.py').write_text(store)
    hooks=(cloud/'hooks.py').read_text()
    hooks=replace_once(hooks,'            self.last_signature = None',
        '            self.last = time.monotonic()  # Avoid an immediate large upload for a short run.\n            self.last_signature = None')
    (cloud/'hooks.py').write_text(hooks)
    campaign=(package/'campaign.py').read_text()
    start=campaign.index('def filtered_plan(');end=campaign.index('\n\ndef comparison',start)
    campaign=campaign[:start]+'''def filtered_plan(plan, seeds):
    from scope import restrict
    return restrict(plan,seeds,read(Path(__file__).resolve().parent/'missing_scope.json'))
'''+campaign[end:]
    campaign=replace_once(campaign,"        plan = dict(read(args.parent_bundle/'plan.json'), tasks=combined_tasks)",
        "        from scope import merged_rows\n        scope = read(package/'missing_scope.json')\n        new_completed = len(combined_rows)\n        combined_rows = merged_rows(combined_rows,read(package/'historical_reference.json'),scope)\n        plan = dict(read(args.parent_bundle/'plan.json'), tasks=scope['all_tasks'])")
    campaign=replace_once(campaign,"        runtime.write(args.durable/'comparison.json', value)",
        "        value.update(new_completed=new_completed,new_total=324,historical_fixed_reference_count=180,\n                     historical_reverified=False)\n        runtime.write(args.durable/'comparison.json', value)")
    campaign=replace_once(campaign,"        print(f\"Verified seeds 0/2/20 results: {value['completed']}/504\", flush=True)",
        "        print(f\"New completed: {new_completed}/324; combined with fixed historical reference: {value['completed']}/504\", flush=True)")
    campaign=campaign.replace('PHASE: verify/resume requested seeds','PHASE: missing tasks for seeds')
    (package/'campaign.py').write_text(campaign)
    phase=(package/'phase.py').read_text()
    phase=replace_once(phase,"    sweep = scheduler.Sweep(args.bundle, args.durable, args.control, args.workers,",
        "    from scope import historical_baselines\n    import json\n    fixed_scope = json.loads((Path(__file__).resolve().parent/'missing_scope.json').read_text())\n    reference_baselines = historical_baselines(fixed_scope,args.seeds)\n    original_candidates = scheduler.candidates\n    scheduler.candidates = lambda pending,resumed,done: original_candidates(pending,resumed,[*done,*reference_baselines])\n    sweep = scheduler.Sweep(args.bundle, args.durable, args.control, args.workers,")
    phase=replace_once(phase,'    verify_original = sweep.r.verify_bundle',
        "    guarded_launch = sweep.launch\n    allowed_ids = {t['id'] for t in fixed_scope['missing_tasks']}\n    def launch_missing(task):\n        if task['id'] not in allowed_ids: raise RuntimeError('Task excluded by the fixed missing manifest')\n        return guarded_launch(task)\n    sweep.launch = launch_missing\n    verify_original = sweep.r.verify_bundle")
    phase=phase.replace('Verified requested seeds','Completed missing tasks for seeds')
    (package/'phase.py').write_text(phase)
    for part in ('02','20'):
        control=package/('control'+part)
        path=control/'scheduler.py';s=path.read_text()
        s=s.replace('listing saved run folders on Drive','listing only this new campaign’s local result index')
        s=s.replace('saved run folders; GPU workers start after verification','new-campaign results; historical archives are excluded')
        s=s.replace('Preserved {len(ids)}/{self.total_tasks} verified runs','Completed {len(ids)}/{self.total_tasks} missing-list tasks')
        path.write_text(s)
        manifest=read(control/'control_manifest.json')
        manifest['files']['scheduler.py']=sha(path)
        manifest['missing_scope_sha256']=sha(package/'missing_scope.json')
        write(control/'control_manifest.json',manifest)
    # Preserve the source/initialization recipe exactly. Only orchestration changed.
    with zipfile.ZipFile(PARENT/'msd_data_verified.zip') as data:
        for bundle in ('original_bundle','seed20_bundle'):
            root=package/bundle
            for name,expected in read(root/'bundle_manifest.json').items():
                if (root/name).is_file():actual=sha(root/name)
                else:
                    assert name in data.namelist(),name
                    actual=hashlib.sha256(data.read(name)).hexdigest()
                assert actual==expected,name
    # Reconstruct the combined reference byte-for-byte in Colab instead of
    # transferring its 52 MiB historical portion twice. This is storage only.
    initial=read(package/'original_bundle/initialization_reference.json')
    extra=read(package/'seed20_bundle/seed20_initialization_reference.json')
    for model,values in extra['models'].items():initial['models'][model].update(values)
    combined=(json.dumps(initial,indent=2,allow_nan=False)+'\n').encode()
    reference_file=package/'seed20_bundle/initialization_reference.json'
    assert combined==reference_file.read_bytes()
    reference_file.unlink()
    files={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
           if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}
    write(package/'package_manifest.json',files)
    code_zip=OUT/'msd_missing_324_code.zip'
    with zipfile.ZipFile(code_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(package)))
    data_zip=OUT/'msd_missing_data.zip';shutil.copyfile(PARENT/'msd_data_verified.zip',data_zip)
    book=notebook(sha(code_zip),sha(package/'package_manifest.json'),sha(data_zip))
    for cell in book['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    for p in package.rglob('*.py'):
        if 'source' not in p.parts:ast.parse(p.read_text())
    write(OUT/'MSD_MISSING_324_NO_AUGMENTATION.ipynb',book)
    summary=dict(account='filippofocaccia2@gmail.com',missing=324,excluded=180,
        by_seed=dict(Counter(t['seed'] for t in missing)),
        code_zip=code_zip.name,code_sha256=sha(code_zip),data_zip=data_zip.name,data_sha256=sha(data_zip),
        notebook='MSD_MISSING_324_NO_AUGMENTATION.ipynb',folder_id='1fErhcTI519LV2wpPtbzLjiHd5dzxRUQu',
        existing_checkpoint_copy_id='1wQMNczpf3vSG5NJVWn_koQs8q1s8s1Gf',
        no_historical_run_scans=True,augmentation='none',gpu_tested=False)
    write(OUT/'export_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':build()
