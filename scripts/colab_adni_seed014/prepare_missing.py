"""Build the self-contained ADNI 175-task continuation, excluding historical archives."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
PRIOR = ROOT/'reports/adni_colab_seeds014_drive_optimized_20261018'
OUT = ROOT/'reports/adni_colab_missing_only_20261018'
AUDIT = ROOT/'reports/adni_disconnect_audit_20261018_2035'
ORIGINAL = ROOT/'reports/adni_colab_3fold_3seed_noaug_20261015'
ZIP_NAME = 'adni_missing175_noaug_v4_code.zip'
BOOK_NAME = 'ADNI_MISSING_175_RESUME_v4_NO_AUGMENTATION.ipynb'


def read(path): return json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def split_upload():
    """Connector uploads are limited to 100 MiB; Colab rejoins exact ZIP bytes."""
    parts = []
    with (OUT/ZIP_NAME).open('rb') as source:
        index = 1
        while data := source.read(64*1024**2):
            path = OUT/(ZIP_NAME+f'.part{index:02d}')
            path.write_bytes(data)
            parts.append(dict(name=path.name, size=len(data), sha256=sha(path)))
            index += 1
    book = read(OUT/BOOK_NAME)
    setup = book['cells'][2]['source']
    setup = replace(setup, '            shutil.copyfile(INPUT/name,local)\n',
        f"            if name=={ZIP_NAME!r} and not (INPUT/name).is_file():\n"
        "                temporary=local.with_suffix(local.suffix+'.partial')\n"
        "                with temporary.open('wb') as joined_zip:\n"
        f"                    for suffix in {[f'.part{i:02d}' for i in range(1,len(parts)+1)]!r}:\n"
        "                        with (INPUT/(name+suffix)).open('rb') as chunk:\n"
        "                            shutil.copyfileobj(chunk,joined_zip,8*1024**2)\n"
        "                assert sha256(temporary)==expected, 'Incomplete/corrupt code transfer; training remains blocked.'\n"
        "                temporary.replace(local)\n"
        "            else:\n                shutil.copyfile(INPUT/name,local)\n")
    ast.parse(setup)
    book['cells'][2]['source'] = setup
    book['cells'][11]['source'] += ('\nThe code ZIP is stored on Drive in two upload parts to fit the connector limit. '
        'Setup automatically joins them on SSD and verifies the full ZIP SHA256; no manual assembly is needed.\n')
    write(OUT/BOOK_NAME,book)
    summary = read(OUT/'export_summary.json')
    summary.update(upload_parts=parts,notebook_sha256=sha(OUT/BOOK_NAME))
    write(OUT/'export_summary.json',summary)


def prepare():
    package = OUT/'package'
    shutil.copytree(PRIOR/'package', package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'data'))
    for name in ('missing_scope.py', 'bootstrap_missing.py'):
        shutil.copyfile(SOURCE/name, package/name)
    with zipfile.ZipFile(ORIGINAL/'adni_replication_code.zip') as z:
        assert z.testzip() is None
        z.extractall(package/'original_bundle')
    shutil.copyfile(AUDIT/'retained_seeds01.json', package/'historical_snapshot.json')
    recovery = {}
    static = {}
    for folder in ('results_noaug_v1', 'results_seeds014_noaug_v1'):
        for p in (OUT/'source_copy'/folder).glob('*.json'):
            relative = folder+'/'+p.name
            target = package/'import_state'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            static[relative] = sha(target)
    sizes = {'nnunet': 328939520, 'basic_unet16': 23244800, 'unetr_vit_b16': 1483192320}
    for folder in (OUT/'source_copy/results_noaug_v1/recovery').iterdir():
        task = next(t for t in read(AUDIT/'missing_seed01_tasks.json') if t['id'] == folder.name)
        recovery[folder.name] = dict(pointer=read(folder/'latest.json'), ledger=read(folder/'charged_ledger.json'),
                                    size=sizes[task['model']])
    shutil.copytree(OUT/'source_copy/baselines', package/'baselines', dirs_exist_ok=True)
    baselines = {p.parent.name: sha(p) for p in (package/'baselines').glob('*/initialization.json')}
    scope = dict(version=4, dataset='ADNI', target_account='filippofocaccia2@gmail.com',
        target_folder_id='1cEfUypMBV9Ay9hNW5nE1_XTJfnGVR9Dm',
        historical_completed=329, historical_snapshot_sha256=sha(package/'historical_snapshot.json'),
        historical_archive_policy='Saved prior inventory trusted at user request; do not scan/import historical archives.',
        seed01_tasks=read(AUDIT/'missing_seed01_tasks.json'), seed4_tasks=read(package/'seed4_bundle/plan.json')['tasks'],
        recovery=recovery, baseline_initializations=baselines, static_receipts=static,
        augmentation='none', new_experiments=0, continuation_tasks=175)
    assert len(scope['seed01_tasks']) == 7 and len(scope['seed4_tasks']) == 168 and len(recovery) == 4
    write(package/'continuation.json', scope)
    phase = (PRIOR/'package/phase.py').read_text()
    phase = replace(phase, 'from campaign import comparison, filtered_plan\n',
        'from campaign import comparison, filtered_plan\nfrom missing_scope import missing_plan, install\n')
    phase = replace(phase, 'verified_plan = filtered_plan(verify_original(root), args.seeds)',
        'verified_plan = missing_plan(verify_original(root), Path(__file__).resolve().parent, args.seeds)')
    phase = replace(phase, '    original_stop = sweep.check_stop\n',
        '    install(sweep, Path(__file__).resolve().parent, args.seeds)\n    original_stop = sweep.check_stop\n')
    phase = replace(phase, '        value = comparison(rows, verified_plan, args.seeds,',
        "        rows = [dict(row, archive_validation='verified_in_continuation') for row in rows]\n"
        '        value = comparison(rows, verified_plan, args.seeds,')
    (package/'phase.py').write_text(phase)
    campaign = (PRIOR/'package/campaign.py').read_text()
    campaign = replace(campaign, '    package = Path(__file__).resolve().parent\n',
        "    package = Path(__file__).resolve().parent\n"
        "    from missing_scope import verify_package\n"
        "    verify_package(package)\n")
    campaign = replace(campaign, '        combined_rows = []\n',
        "        combined_rows = [dict(row, archive_validation='historical_saved_snapshot_not_rescanned')\n"
        "                         for row in read(package/'historical_snapshot.json')['runs']]\n")
    campaign = campaign.replace("'retained_seeds01.json'", "'continuation_seed01.json'")
    campaign = replace(campaign, "        # All data just passed the full archive checks in both phases. Publish only\n"
        "        # now; failed/partial verification never replaces the prior combined report.\n",
        "        # Historical rows are explicitly labelled as saved, not rescanned.\n"
        "        value['historical_saved_count'] = 329\n"
        "        value['verified_continuation_count'] = len(combined_rows)-329\n"
        "        value['continuation_total'] = 175\n"
        "        value['historical_archives_rescanned'] = False\n")
    campaign = campaign.replace('Verified seeds 0/1/4 results:', 'Saved historical + verified continuation results:')
    (package/'campaign.py').write_text(campaign)
    files = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*'))
             if p.is_file() and p.name != 'package_manifest.json' and '__pycache__' not in p.parts}
    write(package/'package_manifest.json', files)
    for name in files:
        if name.endswith('.py'): ast.parse((package/name).read_text())
    with zipfile.ZipFile(OUT/ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in [*files, 'package_manifest.json']: z.write(package/name, name)
    book = read(PRIOR/'ADNI_SEEDS_0_1_4_RESUME_DRIVE_v3_NO_AUGMENTATION.ipynb')
    book['metadata']['colab']['name'] = BOOK_NAME
    book['cells'][0]['source'] = '''# ADNI: only the 175 missing runs — no augmentation

Use **filippofocaccia2@gmail.com** for both Drive mounting and Google authentication.
Code, dataset and four recovery TARs have been copied into that account's
`My Drive/hippo_adni_3fold_3seed`. This is ADNI, not MSD.

The fixed allowlist contains **4 interrupted runs to resume + 3 untouched seed-1 runs + 168 seed-4 runs**.
The 329 completed target runs are excluded from scheduling and archive scans. Their saved comparison
snapshot is included for context and labelled as historical, not freshly reverified. Original results
remain untouched in the source account. No seed-2 runs or new experiments/benchmarks are added.

The existing 5% subject splits, folds 0/1/2, seeds 0/1/4, initialization, BF16, batch size,
losses, calibration, optimizers, schedules and 31,976,450,667,090 optimization-FLOP cap are unchanged.
nnUNet keeps native Dice+CE/SGD and 60% constraint targets; the other models keep Dice/AdamW and 10%.
**No augmentation.** Seed replacement after observing results remains exploratory/post-selection.

Use the previously validated **A100-SXM4-80GB** and the saved software versions. Source, environment,
preflight, checkpoint hashes and FLOP ledgers remain mandatory checks. Missing or stale recovery stops
the affected continuation; it cannot silently restart. Run this notebook from the top in the new account.
Do not run the old account's notebook in parallel: the accounts cannot share controller locks.

The v3 speed and Drive improvements are retained: local SSD data/caches/status, adaptive worker slots,
validated acceleration, lossless recovery compression, duplicate-transfer suppression, paced writes,
serialized bulk transfers, bounded quota retries and comparison uploads at most once per five minutes.
Every update still has its durable FLOP charge/readback. This reduces traffic; it cannot guarantee that
Google will never throttle the account. Never repeatedly interrupt a recovery upload.
'''
    setup = book['cells'][2]['source']
    setup = setup.replace("PACKAGE=Path('/content/hippo_adni_seeds014_drive_v3')", "PACKAGE=Path('/content/hippo_adni_missing175_v4')")
    start = setup.index("with phase('check original results and ownership'):")
    setup = setup[:start] + '''with phase('confirm this is the destination Drive mount'):
    assert json.loads((DRIVE_ROOT/'destination_account.json').read_text()) == {
        'account':'filippofocaccia2@gmail.com',
        'campaign_folder_id':'1cEfUypMBV9Ay9hNW5nE1_XTJfnGVR9Dm',
        'continuation':'adni_missing175_20261018'}, 'Mount filippofocaccia2@gmail.com; source account must not be used.'
    for folder in (ORIGINAL_RESULTS,RESULTS):
        for name in ('.adni_controller_lock','.adni_seeds014_controller_lock'):
            assert not (folder/name).exists(), 'Existing controller ownership: preserve and inspect it.'
with ExitStack() as locks:
    for root in (ORIGINAL_BUNDLE,BUNDLE):
        (root/'work').mkdir(parents=True,exist_ok=True)
        lock=locks.enter_context((root/'work/sweep.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
        try: words=cmdline.read_bytes().decode(errors='replace').split('\\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        for root in (ORIGINAL_BUNDLE,BUNDLE):
            assert not (str(root/'entry.py') in words or (str(root) in words and any(w.endswith(('/worker.py','/phase.py')) for w in words))), 'An ADNI worker is still active.'
    stage_zip(''' + repr(ZIP_NAME) + ',' + repr(sha(OUT/ZIP_NAME)) + ''',PACKAGE)
    verify_files(PACKAGE,PACKAGE/'package_manifest.json')
    with phase('stage frozen science and dataset on local SSD'):
        for root, supplied, expected in (
            (ORIGINAL_BUNDLE,PACKAGE/'original_bundle','e0b87c00231bb355aa66531f6684c361a632b3f8e393f6afbfb4939a37d0caeb'),
            (BUNDLE,PACKAGE/'seed4_bundle','88349a82b03b4f93884def610f5ef12077968ef3b1b75192a3472f395266a148'),
        ):
            marker=root/'bundle_manifest.json'
            if marker.exists():
                assert sha256(marker)==expected, 'Preserve changed source; do not overwrite it.'
            else:
                assert not any((root/'work').glob('*/budget_ledger.json')), 'Existing work without manifest requires inspection.'
                shutil.copytree(supplied,root,dirs_exist_ok=True)
            if root==ORIGINAL_BUNDLE:
                data=root/'data/adni_hippocampus_full.pkl'
                if not data.exists():
                    stage_zip('adni_data_verified.zip','e05bb92b0a6f5024dc9c823283a63020dba8f9cbef70141949a854f6bad6e709',root)
            else:
                (root/'data').mkdir(exist_ok=True)
                data=root/'data/adni_hippocampus_full.pkl'
                if not data.exists():data.symlink_to(ORIGINAL_BUNDLE/'data/adni_hippocampus_full.pkl')
            verify_files(root,marker)
print('Continuation results:',ORIGINAL_RESULTS,RESULTS)
print('Historical result archives will not be scanned or copied.')
'''
    # Reuse SSD copies on reruns; verify bytes before extracting either way.
    setup = replace(setup, "        shutil.copyfile(INPUT/name,local)\n",
        "        if not local.is_file() or sha256(local)!=expected:\n            shutil.copyfile(INPUT/name,local)\n")
    book['cells'][2]['source'] = setup
    environment = book['cells'][4]['source']
    environment = replace(environment, "with phase('authenticate Drive quota API'):auth.authenticate_user()\n",
        "with phase('authenticate destination account and quota API'):\n"
        "    auth.authenticate_user()\n"
        "    import google.auth\n"
        "    from googleapiclient.discovery import build\n"
        "    credentials,_=google.auth.default()\n"
        "    service=build('drive','v3',credentials=credentials,cache_discovery=False)\n"
        "    account=service.about().get(fields='user(emailAddress)').execute()['user']['emailAddress']\n"
        "    assert account=='filippofocaccia2@gmail.com', 'Authenticate the destination account.'\n")
    environment += "\nwith phase('import gate receipts and recovery pointers once; never reset accounting'):\n    run_logged([PYTHON,PACKAGE/'bootstrap_missing.py','--drive-root',DRIVE_ROOT],'bootstrap')\n"
    book['cells'][4]['source'] = environment
    book['cells'][7]['source'] = '''## 4. Resume four interrupted runs, then finish the allowlist

Only results produced by these 175 tasks are checked and skipped on subsequent sessions.
The original 329 completed archives are never opened. Full checkpoint hash/readability/charge checks
remain mandatory for the four resumed runs. They have priority over untouched work.

Live output and SSD status show phases, workers, updates, GPU activity and Drive waits every 30 seconds.
Worker count is resource-adaptive; GPU RAM occupancy is not GPU utilization. No profiling trials are added.
Use one controller only. Existing STOP/ownership files are respected. Interrupt once, wait for verified
recovery uploads, then rerun this notebook to continue. The 8-hour default is a clean-stop target, not a
Colab runtime guarantee. Hard runtime deletion before upload can still require a recovery audit.
'''
    book['cells'][8]['source'] = "MAX_NEW_RUNS=175  # Fixed missing-only scope; verified continuation results are skipped.\nMAX_WORKERS=0    # Existing CPU/resource-supported adaptive scheduler.\nSESSION_HOURS=8.0\ncampaign('run','--max-runs',MAX_NEW_RUNS,'--workers',MAX_WORKERS,'--hours',SESSION_HOURS)\n"
    book['cells'][9]['source'] = '''## 5. Results: saved historical snapshot + verified continuation

This cell starts no training. It checks only continuation archives, never old-account archives.
The 329 historical rows retain their saved metrics and are labelled `historical_saved_snapshot_not_rescanned`.
New rows are labelled `verified_in_continuation`. Gains use matched model/fold/seed baselines.
This reporting provenance must stay visible; the notebook does not claim a fresh audit of the old study.
'''
    book['cells'][10]['source'] = "campaign('report','--hours',2)\nimport pandas as pd\nresult=json.loads((RESULTS/'comparison.json').read_text())\nprint('Historical saved (not rescanned):',result['historical_saved_count'])\nprint('Verified continuation:',result['verified_continuation_count'],'/175')\ndisplay(pd.DataFrame(result['aggregate']))\ndisplay(pd.DataFrame(result['runs']))\n"
    book['cells'][11]['source'] = '''Files needed on the destination account are already staged in
`My Drive/hippo_adni_3fold_3seed`: this notebook's matching v4 code ZIP and `adni_data_verified.zip`
in `input/`, plus the four copied TARs under `results_noaug_v1/recovery/`.
Do not replace or remove recovery files, charge ledgers, results, STOP files or ownership markers.
The notebook restores only missing immutable setup metadata on its first launch; later launches never
reimport old checkpoint pointers over newer progress. Original source-account results are preserved.

Local logs: `/content/adni_seeds014_*.log`; live status: each bundle's `parallel_status/`;
Drive counters: `/content/hippo_adni_drive_io/`. Full checkpoint verification happens before resumed
updates. Scheduling/import/accounting failure handling were tested locally; this release adds no GPU
benchmark or scientific run and has not been exercised end-to-end on Colab.
'''
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell.update(outputs=[], execution_count=None)
    write(OUT/BOOK_NAME, book)
    write(OUT/'export_summary.json', dict(status='packaged_pending_tests', notebook=BOOK_NAME,
        code=ZIP_NAME, code_sha256=sha(OUT/ZIP_NAME), notebook_sha256=sha(OUT/BOOK_NAME),
        missing=175, resumptions=4, fresh=171, historical_excluded=329,
        source_account='focacciafilippo@gmail.com', target_account='filippofocaccia2@gmail.com',
        data_file=str(ORIGINAL/'adni_data_verified.zip'), data_sha256=sha(ORIGINAL/'adni_data_verified.zip'),
        gpu_tested=False, full_large_recovery_hashes_checked_locally=False))
    split_upload()
    print(OUT)


if __name__ == '__main__': prepare()
