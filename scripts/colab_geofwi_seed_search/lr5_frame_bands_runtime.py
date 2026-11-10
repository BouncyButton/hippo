"""One full-budget frame-excluded BANDS trial against frozen LR5 references."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ARM = 'lr5_frame_bands'


def run(root, durable, hours):
    sys.path.insert(0, str(root))
    import lr3_runtime as campaign
    io = campaign.io
    root = io.activate(root)
    campaign.verify(root)
    references = {}
    for name, score in [('baseline', .5061980031645187), ('original_bands', .5060704595521789)]:
        folder = root / ('reference_lr5_' + name)
        manifest = io.read(folder / 'archive_manifest.json')
        for filename, info in manifest['metadata'].items():
            if io.digest(folder / filename) != info['sha256'] or (folder / filename).stat().st_size != info['bytes']:
                raise ValueError('Archived reference changed: ' + filename)
        result = campaign.engine.validate_output(root, folder, manifest['task'])
        if abs(result['final_dice'] - score) > 1e-12:
            raise ValueError('Reference score differs')
        references[name] = result
    work = root / 'work'; work.mkdir(exist_ok=True)
    # Share a campaign-wide lock with the former bundle in the existing runtime.
    lock_path = Path('/content/hippo_geofwi_unetr_lr3_bands_v1/work/augmentation.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.preflight(root, durable, ARM)
        total = campaign.settings(root, ARM)[2]
        if total != 407:
            raise ValueError('Expected capped 407-update schedule')
        authorization = dict(instruction='Test frame-excluded BANDS using the highest-Dice LR5 recipe',
            only_training_change='Exclude outer two-voxel frame from BANDS support; recalibrate this loss',
            peak_lr=5e-4, halfway_gate=False, frame_width=2, cap_flops=campaign.CAP,
            maximum_new_runs=1, target_updates=total, references=references,
            driver_sha256=io.digest(__file__), diagnostic_preflight='Three separately reported updates, established convention')
        io.write(durable/'frame_bands_authorization.json', authorization)
        result = campaign.final_result(root, durable, ARM)
        if result is None:
            out = work/campaign.task(ARM)['id']; out.mkdir(parents=True, exist_ok=True)
            io.write(out/'experiment_authorization.json', authorization)
            io.write(durable/'frame_bands_status.json', dict(status='running', **authorization))
            print('RUN frame-excluded BANDS with LR 5e-4 through 407 updates; no halfway gate', flush=True)
            if not campaign.execute(root, durable, ARM, total, time.time()+hours*3600):
                io.write(durable/'frame_bands_status.json', dict(status='paused', **authorization))
                return
            result = campaign.final_result(root, durable, ARM)
        if not result:
            raise RuntimeError('Full-budget archive not verified')
        output = durable/'runs'/campaign.task(ARM)['id']
        current = {r['case_name']:r for r in io.read(output/'validation_latest.json')['cases']}
        comparisons = {}
        for name, reference in references.items():
            previous = io.read(root/('reference_lr5_'+name)/'validation_latest.json')['cases']
            rows = []
            for old in previous:
                new = current[old['case_name']]; a,b=old['confusion'],new['confusion']
                if sum(a[1]) != sum(b[1]): raise ValueError('Ground truth differs')
                rows.append(dict(case_name=old['case_name'], reference_dice=old['macro_hard_dice'],
                    frame_bands_dice=new['macro_hard_dice'], delta=new['macro_hard_dice']-old['macro_hard_dice'],
                    reference_fp=a[0][1], frame_bands_fp=b[0][1], reference_fn=a[1][0], frame_bands_fn=b[1][0]))
            comparisons[name] = dict(delta_percentage_points=100*(result['final_dice']-reference['final_dice']),cases=rows)
        io.write(durable/'frame_bands_comparisons.json',comparisons)
        status=dict(status='complete', result=result,references=references,
            deltas_percentage_points={k:v['delta_percentage_points'] for k,v in comparisons.items()},
            threshold_passed=campaign.qualifies(result), scope='One development fold/seed; no further runs authorized by this driver',
            cap_flops=campaign.CAP, halfway_gate=False, frame_width=2)
        io.write(durable/'frame_bands_status.json',status)
        io.write(durable/'FRAME_BANDS_FINISHED.json',status)
        print(json.dumps(status,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--durable',type=Path,required=True)
    p.add_argument('--hours',type=float,default=2)
    args=p.parse_args()
    if not 0 < args.hours <= 2:p.error('Session limit must be in (0,2] hours')
    run(args.bundle.resolve(),args.durable.resolve(),args.hours)
