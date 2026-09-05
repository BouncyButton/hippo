"""Run on the cluster: copy exactly the official source, then submit two seeds."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HOME = Path('/mnt/beegfsstudents/home/3160552')
REPO = HOME / 'hippo'
ROOT = HOME / 'equivariance_family_b_20260905_01'
BASELINE = HOME / 'matched_control_20260902/msd_fold0_none_matched_50epoch_seed0'
SEED1 = HOME / 'seed1_replicate_20260903/msd_fold0_none_seed1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    config = json.loads((BASELINE / 'config.json').read_text())
    other = json.loads((SEED1 / 'config.json').read_text())
    manifest = config['source_provenance']['files']
    assert other['source_provenance']['files'] == manifest
    assert not (ROOT / 'submission.json').exists(), 'Already submitted'
    source = ROOT / 'source'
    source.mkdir()  # Fail closed if an earlier preparation needs inspection.
    for relative, expected in manifest.items():
        assert sha(REPO / relative) == expected, relative
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
        assert sha(target) == expected
        target.chmod(0o444)
    for key in ('pkl', 'splits_json'):
        assert sha(Path(config[key])) == config[key + '_sha256']
    record = {'source_sha256': config['source_provenance']['sha256'],
              'source_files': manifest, 'controls': [str(BASELINE), str(SEED1)],
              'runs': []}
    (ROOT / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    for seed in (0, 1):
        args = ['sbatch', '--parsable', '--chdir=' + str(source),
                '--output=' + str(ROOT / '%x-%j.out'),
                '--error=' + str(ROOT / '%x-%j.err')]
        job = subprocess.check_output(args + ['--job-name=equivB-s' + str(seed),
              str(ROOT / 'run.sbatch'), str(seed)], text=True).strip().split(';')[0]
        row = {'seed': seed, 'job_id': job}
        record['runs'].append(row)
        (ROOT / 'submission.json').write_text(json.dumps(record, indent=2) + '\n')
    # The account permits two submitted jobs. Reserve both slots for training;
    # run.sbatch accepts the predecessor ID for a later checkpoint continuation.
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
