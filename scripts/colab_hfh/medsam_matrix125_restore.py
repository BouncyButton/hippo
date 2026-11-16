"""Restore the exact matrix inputs/results after Colab discards a runtime."""
import json
import os
from pathlib import Path
import zipfile

from scripts.colab_hfh.medsam_constraints_drive import client
from scripts.colab_hfh.medsam_constraints_campaign import archived_control, zip_read
from scripts.colab_hfh.medsam_archive import verify_zip
from scripts.colab_hfh.medsam_matrix125_campaign import ALL_ARMS, RECIPE, Matrix125
from scripts.colab_hfh.medsam_recovery import restore
from scripts.colab_hfh.runtime import digest, read, write


def extract_metadata(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        for item in stream.infolist():
            path = Path(item.filename)
            if (len(path.parts) == 2 and path.parts[0] == 'results'
                    and path.suffix in ('.json', '.csv', '.log')):
                target = output/path.name
                data = stream.read(item)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError('Existing restored metadata differs: '+str(target))
                target.write_bytes(data)


def main():
    root = Path('/content/hippo_hfh')
    api = client()
    import torch
    if torch.__version__ != '2.12.1+cu130':
        raise ValueError('Restore requires the actual original PyTorch 2.12.1+cu130; got '+torch.__version__)
    if torch.cuda.get_device_name() != 'NVIDIA A100-SXM4-40GB':
        raise ValueError('Restore requires the original A100-SXM4-40GB')
    if digest(root/'bundle/manifest.json') != '8d6a299b8ca9dc6937d8613899ae51f2ee8d89b9090f42887f71ff5b14560005':
        raise ValueError('Immutable package identity changed')
    old = root/'medsam_fixed125_v1'
    hashes = {
        'u125_strict_native__dice__f0_s0': 'e89b8f73111126319ce748f3d12180993227930f97ea7394bcb5273255246df6',
        'u125_strict_gentle_t001_r20__dice_bands__f0_s0': '3bf06f1eaf80a155976c219e6fa83b2d5a2f5eb86a2f80ebed84492d5277718d',
    }
    for name, expected in hashes.items():
        target = old/'compact_archives'/(name+'.zip')
        api.download('hfh_storage:results_medsam3_fixed125_v1/compact_archives/'+target.name, target)
        if digest(target) != expected:
            raise ValueError('Original paired archive changed: '+name)
        verify_zip(target)
        extract_metadata(target, old/'runs'/name)
    native = archived_control(old/'compact_archives/u125_strict_native__dice__f0_s0.zip')
    original = root/'recovery_inputs/strict_native__dice__f0_s0.zip'
    api.download('hfh_storage:results_medsam3_constraints_strict_v1/compact_archives/'+original.name, original)
    reference = archived_control(original)
    if (reference['protocol']['updates'], reference['protocol']['seed'], reference['protocol']['fold']) != (200, 0, 0):
        raise ValueError('Wrong original trajectory reference')
    write(root/'medsam_constraints_strict_v1/protocols/strict_native__dice__f0_s0.json', reference['protocol'])
    (root/'medsam_constraints_strict_v1/STOP').touch()
    api.download('hfh_storage:results_medsam3_fixed125_v1/spatial_audit/analysis.json', root/'medsam_spatial125_v1/analysis.json')

    # Cache and verify every completed matrix archive; no completed arm is relaunched.
    remote = 'hfh_storage:results_medsam3_matrix125_f0s0_v1/compact_archives'
    cached = root/'medsam_matrix125_f0s0_v1/compact_archives'
    entries = api.children(api.resolve(remote)['id'])
    completed = []
    for item in entries:
        name = item['name']
        if not name.endswith('.zip'):
            continue
        if Path(name).name != name:
            raise ValueError('Unsafe matrix archive name')
        target = cached/name
        api.download(remote+'/'+name, target)
        row = verify_zip(target)
        if (row['seed'], row['fold'], row['updates']) != (0, 0, 125) or row['arm'] not in ALL_ARMS:
            raise ValueError('Wrong matrix archive identity')
        completed.append(row['arm'])
    if len(completed) != len(set(completed)) or len(completed) < 10:
        raise ValueError('Previously verified completed matrix entries are missing')
    if 'dice_sdf_depth' in completed:
        raise ValueError('Recovery scope changed: SDF+DEPTH is already complete; inspect before proceeding')

    from types import SimpleNamespace
    context = SimpleNamespace(root=root, work=root/'medsam_matrix125_f0s0_v1',
        durable=Path('/content/hfh_storage/results_medsam3_matrix125_f0s0_v1'),
        controls={(0, 0): native}, state={'geometry': dict(tensor_spacing_mm=[2., 1., 1.],
            bands='two six-connected voxel steps', sdf='Physical EDT clipped8mm',
            depth='Physical spacing; unverified source-axis orientation')})
    job = Matrix125.prepare(context, RECIPE, 'dice_sdf_depth', 0, 0)
    if not (job['output']/'checkpoint_latest.pt').exists():
        restore(job['output'], job['remote']/'recovery')
    checkpoint = job['output']/'checkpoint_latest.pt'
    if not checkpoint.exists():
        raise ValueError('Saved SDF+DEPTH checkpoint is required; restarting from zero is forbidden')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not 0 < state['step'] <= 125 or len(state['history']) != state['step']:
        raise ValueError('Incomplete saved checkpoint history')
    expected = {**job['protocol'], 'arm': 'dice_sdf_depth'}
    if read(job['output']/'protocol.json') != expected or state['protocol_sha256'] != digest(job['output']/'protocol.json'):
        raise ValueError('Saved checkpoint does not match the exact paired matrix protocol')
    receipt = dict(status='ready', completed_arms=completed, completed_reused=len(completed),
        resumed_arm='dice_sdf_depth', saved_update=state['step'], checkpoint_sha256=digest(checkpoint),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), optimizer_updates_performed=0,
        note='BAND+EDGE+SDF had no uploaded optimizer checkpoint; only its unfinished calibration may repeat.')
    write(root/'medsam_matrix125_f0s0_v1/runtime_recovery.json', receipt)
    api.upload(root/'medsam_matrix125_f0s0_v1/runtime_recovery.json',
               'hfh_storage:results_medsam3_matrix125_f0s0_v1/monitor/runtime_recovery.json')
    print('MATRIX_RECOVERY_VERIFIED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    os.environ.update(HFH_DRIVE_TRANSPORT='colab_adc', HFH_DRIVE_CREDENTIALS='gcloud')
    main()
