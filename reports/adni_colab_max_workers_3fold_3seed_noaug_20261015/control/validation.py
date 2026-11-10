"""Export existing local observations only; no model calls, Drive reads or updates."""
import argparse
import json
from pathlib import Path
import zipfile

try:
    from .scheduler import verify_control, sha
except ImportError:
    from scheduler import verify_control, sha


def export(root, control, output, log=None):
    control_sha, manifest = verify_control(control)
    if sha(root/'bundle_manifest.json') != manifest['scientific_bundle_sha256']:
        raise ValueError('Scientific bundle does not match the controller')
    files, sessions, observations = {}, [], []
    folder = root/'parallel_status'
    # Local-only inventory; no scans of Drive or dataset/checkpoint payloads.
    for path in sorted(folder.glob('*/controller.json')):
        controller = json.loads(path.read_text())
        if controller.get('control_sha256') != control_sha:
            continue
        sessions.append(controller)
        for name in ('controller.json', 'status.json', 'timing.jsonl', 'environment_binding.json'):
            item = path.parent/name
            if item.is_file():
                data = item.read_bytes()
                files[str(item.relative_to(folder))] = data
                if name == 'status.json':
                    observations.append(json.loads(data))
                elif name == 'timing.jsonl':
                    observations.extend(json.loads(line) for line in data.splitlines() if line.strip())
    worker_count = 0
    for path in sorted(folder.glob('*.json')):
        data = path.read_bytes()
        value = json.loads(data)
        if value.get('control_sha256') == control_sha:
            files['workers/'+path.name] = data
            worker_count += 1
    for path in sorted((root/'work').glob('*/local_storage.json')):
        files['storage/'+path.parent.name+'.json'] = path.read_bytes()
    if log is not None and log.is_file():
        files['controller_run.log'] = log.read_bytes()
    files['control_manifest.json'] = (control/'control_manifest.json').read_bytes()
    summary = dict(
        status='observations_only_not_a_speedup_claim', control_sha256=control_sha,
        scientific_bundle_sha256=manifest['scientific_bundle_sha256'],
        sessions=len(sessions), worker_reports=worker_count,
        peak_workers=max((max(x.get('peak_workers', 0), x.get('workers', 0)) for x in observations), default=0),
        targets=sorted({x['worker_target'] for x in sessions}),
        observed_gpu_utilization_percent=[x['gpu']['util'] for x in observations if 'util' in x.get('gpu', {})],
        launch_blockers=sorted({x['launch_gate'] for x in observations if x.get('launch_gate')}),
        extra_scientific_runs=0, optimizer_calls=0, gpu_benchmarks=0, drive_reads=0,
        interpretation='Only ordinary pending-run telemetry. No numerical acceleration enabled or validated by this export.')
    files['validation_summary.json'] = (json.dumps(summary, indent=2)+'\n').encode()
    # Fail before opening the destination if the inputs were invalid.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    temporary.replace(output)
    print(json.dumps(summary, indent=2))
    print('Validation ZIP:', output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    export(args.bundle, args.control, args.output, args.log)


if __name__ == '__main__':
    main()
