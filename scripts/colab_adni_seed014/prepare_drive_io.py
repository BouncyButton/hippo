"""Package transport-only changes on top of the immutable v2 Swin repair."""
import ast
import shutil
import zipfile

from scripts.colab_adni_seed014 import prepare_initfix as previous

old = previous.old
OUT = old.ROOT/'reports/adni_colab_seeds014_drive_optimized_20261018'
ZIP_NAME = 'adni_seeds014_resume_drive_v3_code.zip'
BOOK_NAME = 'ADNI_SEEDS_0_1_4_RESUME_DRIVE_v3_NO_AUGMENTATION.ipynb'
replace = previous.replace_once


def prepare():
    OUT.mkdir(exist_ok=True)
    package = OUT/'package'
    shutil.copytree(previous.OUT/'package', package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'data'))
    for name in ('control01', 'control04'):
        control = package/name
        prior = previous.OUT/'package'/name
        shutil.copyfile(old.SOURCE/'drive_io.py', control/'drive_io.py')
        worker = (prior/'worker.py').read_text()
        worker = replace(worker, '    original_write, original_read = entry.write, entry.read\n',
            "    import runtime\n"
            "    from drive_io import install as install_drive_io\n"
            "    recovery = Path(os.environ['HIPPO_COLAB_RECOVERY']).resolve()\n"
            "    drive_io = install_drive_io(runtime, recovery.parents[2])\n"
            "    entry.write, entry.read, entry.RecoveryWriter = runtime.write, runtime.read, runtime.RecoveryWriter\n"
            "    stats['drive_io'] = drive_io.stats\n"
            "    original_write, original_read = entry.write, entry.read\n")
        (control/'worker.py').write_text(worker)
        scheduler = (prior/'scheduler.py').read_text()
        scheduler = replace(scheduler, '        self.stop_lock = threading.RLock()\n',
            '        self.stop_lock = threading.RLock()\n'
            '        from drive_io import install as install_drive_io\n'
            '        self.drive_io = install_drive_io(self.r, self.durable.parent)\n')
        scheduler = replace(scheduler, '    def check_stop(self):\n',
            '    def check_stop(self):\n'
            "        if self.drive_io.quota_event():\n"
            "            self.request_stop('Drive rate limit: stop launches and preserve verified recovery')\n")
        scheduler = replace(scheduler, '        # A controller blocked in Drive cannot delay a requested clean stop.\n',
            '        self.live.update(drive_io=dict(self.drive_io.stats))\n'
            '        if self.drive_io.quota_event():\n'
            "            self.request_stop('Drive rate limit: stop launches and preserve verified recovery')\n"
            '        # A controller blocked in Drive cannot delay a requested clean stop.\n')
        scheduler = scheduler.replace("'save updated comparison on Drive'", "'update local comparison; publish to Drive at most every five minutes'")
        (control/'scheduler.py').write_text(scheduler)
        manifest = old.read(prior/'control_manifest.json')
        for file in ('worker.py', 'scheduler.py', 'drive_io.py'):
            manifest['files'][file] = old.sha(control/file)
        manifest.setdefault('compatible_profile_control_sha256', []).append(old.sha(prior/'control_manifest.json'))
        manifest.setdefault('compatible_profile_worker_sha256', []).append(old.sha(prior/'worker.py'))
        manifest['drive_transport'] = dict(version=3, source_sha256=old.sha(control/'drive_io.py'),
            recovery_interval_seconds=180, compression='gzip level1 lossless; original payload bytes',
            charge_policy='unchanged durable write and readback BEFORE EACH original update',
            extra_scientific_runs=0, quota_guarantee=False)
        old.write(control/'control_manifest.json', manifest)
    phase = (previous.OUT/'package/phase.py').read_text()
    phase = replace(phase, '    def report(root, durable, rows=None):\n',
        "    from drive_io import ReportPublisher\n"
        "    publisher = ReportPublisher(sweep.r, sweep.local_audit/'comparison_local.json', args.snapshot)\n"
        '    def report(root, durable, rows=None):\n')
    phase = replace(phase, '        sweep.r.write(args.snapshot, value)\n', '        publisher.publish(value)\n')
    phase = replace(phase, '    sweep.loop()\n', '    try:\n        sweep.loop()\n    finally:\n        publisher.flush()\n')
    (package/'phase.py').write_text(phase)
    campaign = (previous.OUT/'package/campaign.py').read_text()
    campaign = replace(campaign, "    package = Path(__file__).resolve().parent\n",
        "    package = Path(__file__).resolve().parent\n"
        "    transport = load(package/'control01/drive_io.py', 'coordinator_drive_io')\n"
        "    transport_started = time.time()\n")
    campaign = replace(campaign, '            if index == 1:\n',
        "            if transport.quota_since(args.parent_results.parent, transport_started):\n"
        "                print('Drive throttling observed; no next seed phase will start. Existing recovery retained.', flush=True)\n"
        "                return\n"
        '            if index == 1:\n')
    (package/'campaign.py').write_text(campaign)
    files = {str(p.relative_to(package)): old.sha(p) for p in sorted(package.rglob('*'))
             if p.is_file() and p.name != 'package_manifest.json' and '__pycache__' not in p.parts}
    old.write(package/'package_manifest.json', files)
    with zipfile.ZipFile(OUT/ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in [*files, 'package_manifest.json']:
            archive.write(package/name, name)
    book = old.read(previous.OUT/previous.BOOK_NAME)
    prior_zip_sha, new_zip_sha = old.sha(previous.OUT/previous.ZIP_NAME), old.sha(OUT/ZIP_NAME)
    for cell in book['cells']:
        text = cell['source'].replace(previous.ZIP_NAME, ZIP_NAME)
        text = text.replace(prior_zip_sha, new_zip_sha)
        text = text.replace('/content/hippo_adni_seeds014_initfix_v2', '/content/hippo_adni_seeds014_drive_v3')
        if cell['cell_type'] == 'code':
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cell['source'] = text
    first = book['cells'][0]['source']
    first = first[:first.index('**Initialization repair v2:**')]
    # Do not claim the last audited controller state is current at install time.
    first = first.replace('**Before resuming:** the checked controller has already stopped. If you started another controller\nsince that check, stop it once and wait for verified recovery uploads. Do not run',
        '**Before switching:** if any old controller is running, stop it once and wait for verified recovery uploads. Do not run')
    first += """**Drive transport v3 (includes the Swin initialization repair):** replace the notebook and put
the matching v3 code ZIP in the SAME `hippo_adni_3fold_3seed/input` folder. Keep all existing code/data
ZIPs, `results_noaug_v1`, `results_seeds014_noaug_v1`, checkpoints, ledgers and ownership/STOP files.
Run this notebook from the top after a clean stop. No new dataset upload is required: a fresh VM
copies the existing data from Drive to SSD. Do not delete a VM with unverified local recovery.

Recovery snapshots use lossless gzip compression; every checkpoint/metadata payload retains its exact
SHA256. The 180-second snapshot cadence, forced final upload, pre-update durable FLOP charge and
readback, original preflights/environment gates and all archive/cleanup checks remain mandatory.
An identical snapshot already verified on Drive is not uploaded again. Existing `.tar` recovery
archives remain readable. Recovery transfer failures retain local staging as well as prior Drive recovery.

Bulk uploads are serialized across workers on this VM; small ledger writes have a separate lane,
with globally spaced mutation starts. Throttling triggers bounded backoff and a clean stop request;
no new tasks launch after the controller observes it. This reduces bursts and duplicate traffic,
but cannot guarantee Google's shared account quotas or automatic recovery after a hard VM deletion.
If Drive remains blocked, keep the runtime and retained local files until recovery is verified.

Progress and I/O counters live on SSD, including bytes uploaded/avoided, compression time and retries
in `/content/hippo_adni_drive_io`. Derived comparisons publish at most once per five minutes and at
phase exit; EACH completed experiment is still immediately archived and fully verified. A stale
comparison never makes an unverified experiment complete. Normal startup verifies the actual archives.

The selected 504-run seeds-0/1/4 study is unchanged; no benchmarks or experiments are added.
Completed runs are verified and skipped, interrupted work is prioritized, and existing charge gaps
still block continuation rather than being reset. The strict Swin CPU compatibility preflight is
bound to this new package; original passed GPU preflights and scientific manifests stay unchanged.
Transport/recovery/failure tests were run locally; this release has no new GPU or live-Drive validation.
"""
    book['cells'][0]['source'] = first
    book['metadata']['colab']['name'] = BOOK_NAME
    old.write(OUT/BOOK_NAME, book)
    summary = old.read(previous.OUT/'export_summary.json')
    summary.update(status='packaged_pending_local_tests', notebook=BOOK_NAME, code_zip=ZIP_NAME,
                   code_zip_sha256=old.sha(OUT/ZIP_NAME), notebook_sha256=old.sha(OUT/BOOK_NAME),
                   gpu_tested=False, live_drive_tested=False, extra_scientific_runs=0)
    old.write(OUT/'export_summary.json', summary)
    print(OUT)


if __name__ == '__main__':prepare()
