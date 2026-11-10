"""Build storage-only updates of the two latest missing-only continuations."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT/'reports/colab_drive_low_writes_20261018'
ADNI = ROOT/'reports/adni_colab_missing_only_20261018'
MSD = ROOT/'reports/msd_missing_324_second_account_20261018'


def read(path): return json.loads(path.read_text())
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def replace(text, before, after):
    assert text.count(before)==1, (before, text.count(before))
    return text.replace(before, after)
def source(cell):
    return ''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']


def package_manifest(package):
    files={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
           if p.is_file() and p.name!='package_manifest.json' and '__pycache__' not in p.parts}
    write(package/'package_manifest.json', files)
    for name in files:
        if name.endswith('.py'): ast.parse((package/name).read_text())


def compress(package, path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(package)))


def adni():
    target=OUT/'adni';package=target/'package'
    shutil.copytree(ADNI/'package',package,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('adni_low_write.py','request_gate.py'):shutil.copyfile(SOURCE/name,package/name)
    for part in ('control01','control04'):
        control=package/part; prior=ADNI/'package'/part
        shutil.copyfile(SOURCE/'adni_low_write.py',control/'adni_low_write.py')
        script=(control/'drive_io.py').read_text()
        script=replace(script,"    runtime.write, runtime.read, runtime.copy_verified = io.write, io.read, io.copy\n",
            "    from adni_low_write import install as install_low_write\n"
            "    install_low_write(io)\n"
            "    runtime.write, runtime.read, runtime.copy_verified = io.write, io.read, io.copy\n")
        script=replace(script,'def __init__(self, runtime, local, durable, interval=300):',
                       'def __init__(self, runtime, local, durable, interval=900):')
        script=replace(script,"                    if getattr(self, '_verified_signature', None) == signature:\n",
            "                    if previous and getattr(self, '_verified_signature', None) is None:\n"
            "                        from adni_low_write import prior_snapshot_matches\n"
            "                        existing = self.durable/io.r.safe_member(previous['file'])\n"
            "                        if prior_snapshot_matches(io.r, existing, previous, manifest):\n"
            "                            self._verified_signature, self._verified_pointer = signature, previous\n"
            "                    if getattr(self, '_verified_signature', None) == signature:\n")
        (control/'drive_io.py').write_text(script)
        script=(control/'scheduler.py').read_text()
        script=replace(script,"            manifest = self.r.read(self.control/'control_manifest.json')\n"
            "            for name in (*manifest['files'], 'control_manifest.json'):\n"
            "                self.r.copy_verified(self.control/name, self.audit_dir/name)\n",
            "            from adni_low_write import publish_control\n"
            "            publish_control(self.r, self.control, self.durable, self.audit_dir)\n")
        script=script.replace('publish to Drive at most every five minutes','publish to Drive at most every fifteen minutes')
        (control/'scheduler.py').write_text(script)
        manifest=read(prior/'control_manifest.json')
        for name in ('drive_io.py','scheduler.py','adni_low_write.py'):manifest['files'][name]=sha(control/name)
        manifest.setdefault('compatible_profile_control_sha256',[]).append(sha(prior/'control_manifest.json'))
        # Worker bytes and acceleration candidate stay identical.
        manifest['low_write_transport']=dict(version=5, scientific_changes=False,
            charge_before_each_update=True,recovery_interval_seconds=180,
            report_interval_seconds=900,quota_snapshots='local_ssd',
            control_provenance='one verified immutable directory per control SHA')
        write(control/'control_manifest.json',manifest)
    script=(package/'campaign.py').read_text()
    script=replace(script,'    verify_package(package)\n',
        "    verify_package(package)\n"
        "    if args.command == 'run':\n"
        "        from adni_low_write import server_save_probe\n"
        "        server_save_probe('1cEfUypMBV9Ay9hNW5nE1_XTJfnGVR9Dm', 'filippofocaccia2@gmail.com')\n")
    (package/'campaign.py').write_text(script)
    package_manifest(package)
    code=target/'adni_missing175_noaug_v5_code.zip';compress(package,code)
    parts=[]
    with code.open('rb') as stream:
        for i in range(1,10):
            data=stream.read(64*1024**2)
            if not data:break
            path=Path(str(code)+f'.part{i:02d}');path.write_bytes(data)
            parts.append(dict(name=path.name,sha256=sha(path),bytes=len(data)))
    assert len(parts)==2
    book=read(OUT/'backups/adni_before.ipynb')
    for cell in book['cells']:
        s=source(cell).replace('adni_missing175_noaug_v4_code.zip',code.name)
        s=s.replace(sha(ADNI/'adni_missing175_noaug_v4_code.zip'),sha(code))
        s=s.replace('/content/hippo_adni_missing175_v4','/content/hippo_adni_missing175_v5')
        s=s.replace('at most once per five minutes','at most once per fifteen minutes')
        s=s.replace('matching v4 code ZIP','matching v5 code ZIP')
        cell['source']=s
    book['cells'][0]['source']+='''\n**Lower Drive traffic v5:** volatile quota snapshots stay on SSD; live quota checks still run.
Controller source is uploaded once per verified control version, then referenced by later sessions.
An unchanged restored recovery snapshot is reused only after its full archive and member hashes verify.
Derived comparisons publish at most every 15 minutes and on exit; per-run archives stay immediate.
A real server write/read check must pass before training. Required per-update FLOP charges and the
180-second checkpoint interval are unchanged. Existing results/checkpoints/STOP files stay in place.
This reduces avoidable I/O, but cannot lift an active Drive quota or Colab session restriction.
'''
    name='ADNI_MISSING_175_LOW_WRITES_v5.ipynb'
    finish_book(book,target/name)
    return dict(notebook=str(target/name),code=str(code),code_sha256=sha(code),parts=parts,
                existing_notebook_id='1_-wJXpDp3rCVgVrnSefPvJq1baJcRI0O',tasks=175)


def msd():
    target=OUT/'msd';package=target/'package'
    shutil.copytree(MSD/'package',package,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('msd_low_write.py','request_gate.py'):shutil.copyfile(SOURCE/name,package/'cloud'/name)
    path=package/'cloud/store.py';script=path.read_text()
    script=replace(script,'        from googleapiclient.errors import HttpError\n        for attempt in range(6):',
        '        from googleapiclient.errors import HttpError\n        from request_gate import pace, cooldown\n        for attempt in range(6):')
    script=replace(script,'                return request.execute(num_retries=0)',
        '                pace()\n                return request.execute(num_retries=0)')
    script=replace(script,'                time.sleep(min(2**(attempt+1)+random.random(), 32))',
        '                cooldown(attempt)')
    script=replace(script,'                    progress, result = request.next_chunk(num_retries=5)',
        '                    from request_gate import request_call\n'
        '                    progress, result = request_call(lambda: request.next_chunk(num_retries=0))')
    script=replace(script,'        found=self.execute(self.service().files().list(q=',
        '        from msd_low_write import find_verified, remember\n'
        '        cached = find_verified(self, path, name)\n'
        '        if cached is not None: return cached\n'
        '        found=self.execute(self.service().files().list(q=')
    script=replace(script,'            return dict(found[0], local_sha256=sha)\n'
        "        return self.upload(path, name, {'kind':kind,'task':task,'phase':phase,'sha256':sha})",
        '            return remember(self, name, dict(found[0], local_sha256=sha))\n'
        "        return remember(self, name, self.upload(path, name, {'kind':kind,'task':task,'phase':phase,'sha256':sha}))")
    path.write_text(script)
    path=package/'cloud/hooks.py';script=path.read_text()
    start=script.index('    archive = output.parent/',script.index('def bundle_results'))
    end=script.index('\n\ndef commit_result',start)
    script=script[:start]+"    from msd_low_write import stable_zip\n    return stable_zip(output, manifest)\n"+script[end:]
    # A server-confirmed commit can be reused by the parent without rebuilding
    # or uploading its identical result ZIP. Hash every archived payload first.
    script=replace(script,"    archive = bundle_results(output, manifest)\n",
        "    existing = output.parent/(output.name+'_server_commit.json')\n"
        "    if existing.exists():\n"
        "        prior = json.loads(existing.read_text())\n"
        "        if prior.get('manifest_sha256') == digest(output/'archive_manifest.json'):\n"
        "            remote = cloud.metadata(prior['id'])\n"
        "            if (remote.get('parents') != [cloud.folder] or remote.get('size') != prior.get('size')\n"
        "                    or remote.get('md5Checksum') != prior.get('md5Checksum')\n"
        "                    or remote.get('appProperties',{}).get('sha256') != prior.get('local_sha256')):\n"
        "                raise ValueError('Committed result object changed; preserve all local work')\n"
        "            return prior, manifest\n"
        "    archive = bundle_results(output, manifest)\n")
    script=replace(script,"    atomic(output.parent/(output.name+'_server_commit.json'), receipt)\n",
        "    receipt = dict(receipt, manifest_sha256=digest(output/'archive_manifest.json'))\n"
        "    atomic(output.parent/(output.name+'_server_commit.json'), receipt)\n")
    path.write_text(script)
    package_manifest(package)
    code=target/'msd_missing324_low_writes_v2_code.zip';compress(package,code)
    book=read(OUT/'backups/msd_before.ipynb')
    for cell in book['cells']:
        s=source(cell).replace('msd_missing_324_code.zip',code.name)
        s=s.replace(sha(MSD/'msd_missing_324_code.zip'),sha(code))
        s=s.replace(sha(MSD/'package/package_manifest.json'),sha(package/'package_manifest.json'))
        s=s.replace('/content/hippo_msd_missing_package','/content/hippo_msd_missing_package_v2')
        cell['source']=s
    book['cells'][0]['source']+='''\n**Lower Drive traffic v2:** stable metadata ZIPs avoid new uploads caused only by filesystem
timestamps. The controller reuses the worker's result commit only after checking the server object
and every local manifest payload. Immutable-object references are cached on SSD and revalidated
against server checksums. Workers share request pacing and retry cooldowns, including bulk uploads.
Every FLOP charge still reaches Drive before its update. Completed results, checkpoints and recovery
accounting stay in the same folder. The existing server write/read probe must pass before training.
These changes do not lift an active Google quota or session limit. No new experiments or benchmarks.
'''
    name='MSD_MISSING_324_LOW_WRITES_v2.ipynb';finish_book(book,target/name)
    return dict(notebook=str(target/name),code=str(code),code_sha256=sha(code),
                existing_notebook_id='15VOVc2RRROzfqTFZmIdU-z9isZ92zDkB',tasks=324)


def finish_book(book,path):
    book['metadata'].setdefault('colab',{})['name']=path.name
    for cell in book['cells']:
        if cell['cell_type']=='code':
            ast.parse(source(cell));cell.update(outputs=[],execution_count=None)
    write(path,book)


def main():
    OUT.mkdir(exist_ok=True)
    summary=dict(adni=adni(),msd=msd(),gpu_tested=False,extra_experiments=0,
                 results_paths_changed=False,checkpoint_interval_changed=False,charges_coalesced=False)
    write(OUT/'export_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
