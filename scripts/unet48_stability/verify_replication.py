"""Fail before selecting a new budget unless all three stability seeds qualify."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    home = Path('/mnt/beegfsstudents/home/3160552')
    records = []
    for seed, job in [(0,684114),(1,684116),(2,684116)]:
        run = home/f'unet48_stability_{job}/linear_bg'/f'seed{seed}'
        done = json.loads((run/'completion.json').read_text())
        assert (done['status'], done['seed'], done['updates'], done['discarded_attempts']) == ('complete',seed,3000,0)
        for key, filename in [('metrics_sha256','metrics.csv'),('config_sha256','config.json'),
                              ('checkpoint_best_sha256','checkpoint_best.pt'),('checkpoint_latest_sha256','checkpoint_latest.pt'),
                              ('budget_ledger_sha256','budget_ledger.json'),('calibration_sha256','calibration.json'),
                              ('validation_best_sha256','validation_best.json')]:
            assert digest(run/filename) == done[key], filename
        curve = list(csv.DictReader((run/'metrics.csv').open()))
        final = curve[-1]
        at2000 = next(row for row in curve if int(row['step'])==2000)
        assert min(float(at2000[key]) for key in ['anterior_hard_dice','posterior_hard_dice']) >= .70
        assert min(float(final[key]) for key in ['anterior_hard_dice','posterior_hard_dice']) >= .75
        late = [float(row['macro_hard_dice']) for row in curve if int(row['step'])>=2000]
        maximum_regression = max(a-b for i,a in enumerate(late) for b in late[i:])
        assert maximum_regression <= .05
        records.append({'seed':seed,'job':job,'completion_sha256':digest(run/'completion.json'),
                        'final_macro_dice':float(final['macro_hard_dice']),
                        'each_foreground_at2000':[float(at2000[k]) for k in ['anterior_hard_dice','posterior_hard_dice']],
                        'maximum_late_macro_regression':maximum_regression})
    assert max(r['final_macro_dice'] for r in records)-min(r['final_macro_dice'] for r in records)<=.05
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status':'qualified','seeds':records,'scope':'Same fixed fold0 patients, initialization and augmentation seeds0/1/2'},indent=2)+'\n')
    print('STABILITY_REPLICATION_QUALIFIED',flush=True)


if __name__ == '__main__':
    main()
