"""Stage and run four disjoint full-coverage ADNI medical controls on the cluster."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile


HERE = Path(__file__).resolve().parent
KEYS = ['ADNI_medsam3_f1_s2', 'ADNI_medsam3_f3_s0', 'ADNI_medsam3_f3_s1', 'ADNI_medsam3_f3_s2']
SUFFIX = '_coverage_control'


def build(parent, output):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    files = {}
    with zipfile.ZipFile(parent) as source, zipfile.ZipFile(HERE / 'ADNI_medical_protocols.zip') as medical:
        for n in source.namelist():
            if n.startswith('source/') and not n.endswith('/'):
                files[n] = source.read(n)
        for key in KEYS:
            ref = manifest[key]['archived_protocol']
            raw = (HERE.parents[1] / ref['file']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == ref['sha256']
            original = json.loads(raw)
            config = json.loads(medical.read('protocols/' + key + '.json'))
            for n in ('data', 'upstream', 'lora_weights'):
                config[n] = original[n]
            config.update(name=key + SUFFIX, sampling='Native crop and anchor sequence with duplicate draws replaced when needed for full coverage.')
            assert (config['updates'], config['accumulation'], len(config['train_cases']),
                    len(config['evaluation_cases'])) == (200, 2, 216, 54)
            assert config['evaluation_cases'] == original['evaluation_cases']
            files['protocols/' + key + '.json'] = json.dumps(config, indent=2).encode()
            files['references/' + key + '.json'] = raw
    for n in ('cluster_medical.py', 'covered_medical.py', 'cluster_medical.sbatch'):
        files[n] = (HERE / n).read_bytes()
    checks = ''.join(hashlib.sha256(b).hexdigest() + '  ' + n + '\n' for n, b in sorted(files.items()))
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for n, data in files.items():
            archive.writestr(n, data)
        archive.writestr('SHA256SUMS', checks)
    print(output, hashlib.sha256(output.read_bytes()).hexdigest())


def train(key):
    from scripts.colab_mni_medsam3 import optimized
    from scripts.cross_dataset_campaign import foundation
    from scripts.cross_dataset_campaign.common import digest
    from scripts.mni_medsam3_cluster import worker
    from scripts.train_medsam3_constraint_few_shot import Slabs
    from covered_medical import covered_schedule

    spec = HERE / 'protocols' / (key + '.json')
    config = json.loads(spec.read_text())
    base = (Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())) / 'sam3_f013_s012_cluster_tail_20261022/bundle/weights/sam3.pt')
    assert digest(config['data']) == config['dataset_sha256']
    assert digest(config['lora_weights']) == config['lora_sha256'] and digest(base) == config['base_sha256']
    assert config['dataset'] == 'ADNI' and config['name'] == key + SUFFIX
    optimized.Volumes = foundation.Volumes
    optimized.SupportSlabs = foundation.SupportSlabs
    native_schedule = Slabs.schedule
    Slabs.schedule = lambda self, p: covered_schedule(native_schedule(self, p), self.indices, p)
    out = HERE / 'work' / key
    write = foundation.write
    final_checkpoint_saved = False

    def save_case_progress(path, value):
        nonlocal final_checkpoint_saved
        write(path, value)
        path = Path(path)
        if path.name == 'evaluation.json':
            if not final_checkpoint_saved:
                weights = path.parent / 'checkpoint_latest.pt'
                # Training has finished. Let the last asynchronous training copy close first.
                while (path.parent / 'mirror_snapshot.pt').exists():
                    time.sleep(1)
                worker.copy_verified(weights, out / weights.name)
                final_checkpoint_saved = True
            worker.copy_verified(path, out / path.name)

    foundation.write = save_case_progress
    sys.argv = [sys.argv[0], '--protocol', str(spec), '--output', str(out), '--arm', 'dice',
                '--base-weights', str(base), '--record', str(out / 'gpu_worker.json')]
    worker.main()


def prepare_archive(key):
    from scripts.cross_dataset_campaign.common import digest, read, write
    from scripts.cross_dataset_campaign.storage import prepare
    out = HERE / 'work' / key
    config = read(HERE / 'protocols' / (key + '.json'))
    assert read(out / 'protocol.json') == dict(config, arm='dice')
    schedule = read(out / 'slab_schedule.json')
    assert len(schedule) == 400 and {c for c, _ in schedule} == set(config['train_cases'])
    assert len(read(out / 'evaluation.json')['records']) == 270
    write(out / 'cluster_protocol.json', dict(key=key, dataset='ADNI', train_cases=216, validation_cases=54,
        updates=200, charged_slab_draws=400, allocation=os.getenv('SLURM_JOB_ID'),
        input_manifest=digest(HERE / 'SHA256SUMS'), protocol_sha256=digest(HERE / 'protocols' / (key + '.json')),
        reference_sha256=digest(HERE / 'references' / (key + '.json')),
        training='ADNI reader and native adapter objective with the corrected full-coverage schedule'))
    prepare(out, dict(id=key, dataset='ADNI', kind='foundation', fold=config['fold'], seed=config['seed']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    stage = sub.add_parser('build')
    stage.add_argument('parent', type=Path)
    stage.add_argument('output', type=Path)
    for name in ('train', 'archive', 'release-local'):
        sub.add_parser(name).add_argument('key', choices=KEYS)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.parent, args.output)
    elif args.command == 'train':
        train(args.key)
    elif args.command == 'archive':
        prepare_archive(args.key)
    else:
        from scripts.cross_dataset_campaign.common import digest, read
        dest = HERE / 'work' / args.key
        cleanup = read(dest / 'cleanup.json')
        assert cleanup['status'] == 'complete' and read(dest / 'archive_acknowledgment.json')['status'] == 'verified'
        local = Path(read(dest / 'worker_timing.json')['local_scratch'])
        expected = Path('/tmp') / f"hippo-mni-{os.getuid()}-{os.environ['SLURM_JOB_ID']}" / args.key
        if local == expected and local.is_dir():
            checkpoint = local / 'checkpoint_latest.pt'
            if checkpoint.is_file():
                assert digest(checkpoint) == read(dest / 'completion.json')['checkpoint_sha256']
                checkpoint.unlink()
