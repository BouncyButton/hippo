"""Build one uploadable validator; it does not replace the running notebook."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/adni_colab_acceleration_validation_20261015'
SCIENCE = ROOT/'reports/adni_colab_3fold_3seed_noaug_20261015/bundle'


def prepare():
    EXPORT.mkdir(parents=True, exist_ok=True)
    names = ('acceleration_candidate.py', 'validate_acceleration.py')
    sources = {name:Path(__file__).with_name(name).read_text() for name in names}
    manifest = dict(diagnostic_only=True,
        scientific_bundle_sha256=hashlib.sha256((SCIENCE/'bundle_manifest.json').read_bytes()).hexdigest(),
        files={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()})
    sources['validation_manifest.json'] = json.dumps(manifest, indent=2)+'\n'
    launcher = textwrap.dedent('''\
    """ADNI acceleration equivalence validator. Upload this whole file to Colab.

    Run with /content/hippo_adni_venv/bin/python after the current campaign has
    stopped cleanly. Existing data/results/environment are reused, never reset.
    This contains the candidate and its checks; no installation or downloads.
    """
    import argparse
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    import zipfile

    SOURCES = __SOURCES__

    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--bundle', type=Path, default=Path('/content/hippo_adni_bundle'))
        parser.add_argument('--results', type=Path, default=Path('/content/drive/MyDrive/hippo_adni_3fold_3seed/results_noaug_v1'))
        args = parser.parse_args()
        if not (args.bundle/'bundle_manifest.json').is_file():
            raise RuntimeError('Use the existing prepared ADNI runtime; the validator will not replace its bundle or install packages.')
        output = Path('/content')/('adni_acceleration_validation_'+str(time.time_ns()))
        logfile = output.with_suffix('.log')
        result_zip = output.with_suffix('.zip')
        code = 1
        print('Bounded equivalence checks only: at most 24 disposable optimizer steps, zero campaign runs.', flush=True)
        print('Output report:', result_zip, flush=True)
        with tempfile.TemporaryDirectory(prefix='adni_validation_code_', dir='/content') as directory:
            package = Path(directory)
            for name, source in SOURCES.items():
                (package/name).write_text(source)
            command = [sys.executable, '-u', str(package/'validate_acceleration.py'),
                       '--bundle', str(args.bundle), '--results', str(args.results), '--output', str(output)]
            with logfile.open('w') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, start_new_session=True)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush()
                    code = process.wait()
                except KeyboardInterrupt:
                    print('Stopping diagnostics; campaign results were not modified.', flush=True)
                    os.killpg(process.pid, signal.SIGINT)
                    code = process.wait()
            # Also return a useful log when a safety/environment check blocked
            # the diagnostic before the model phase created its report folder.
            with zipfile.ZipFile(result_zip, 'a', zipfile.ZIP_DEFLATED) as archive:
                archive.write(logfile, 'validation.log')
                archive.writestr('launcher.json', json.dumps(dict(exit_code=code, extra_scientific_runs=0)))
                for name, source in SOURCES.items():
                    archive.writestr('validator_source/'+name, source)
        print('SEND THIS ZIP BACK IN CHAT:', result_zip, flush=True)
        print('No campaign preflight receipt was replaced; no acceleration was enabled for training.', flush=True)
        return code

    if __name__ == '__main__':
        raise SystemExit(main())
    ''').replace('__SOURCES__', repr(sources))
    ast.parse(launcher)
    path = EXPORT/'Run_ADNI_ACCELERATION_VALIDATION.py'
    path.write_text(launcher)
    (EXPORT/'validation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name in names:
        (EXPORT/name).write_text(sources[name])
    print(path)


if __name__ == '__main__':
    prepare()
