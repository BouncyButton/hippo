"""Aggregate paired A/P errors on identical, correctly detected tissue.

Run beside the dataset on the cluster. Export only aggregate statistics and
artifact hashes; no case identifiers, medical images, or predictions.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_counts(truth, baseline, candidate):
    assert truth.shape == baseline.shape == candidate.shape
    assert all(np.isin(x, (0, 1, 2)).all() for x in (truth, baseline, candidate))
    tissue = truth > 0
    shared = tissue & (baseline > 0) & (candidate > 0)
    lost = tissue & (baseline > 0) & (candidate == 0)
    gained = tissue & (baseline == 0) & (candidate > 0)
    wrong_b, wrong_c = baseline != truth, candidate != truth
    b = int((shared & wrong_b).sum())
    c = int((shared & wrong_c).sum())
    corrected = int((shared & wrong_b & ~wrong_c).sum())
    introduced = int((shared & ~wrong_b & wrong_c).sum())
    assert c-b == introduced-corrected
    row = dict(shared_voxels=int(shared.sum()), baseline_shared_swaps=b,
               candidate_shared_swaps=c, corrected=corrected, introduced=introduced,
               lost_tissue=int(lost.sum()), lost_baseline_swaps=int((lost & wrong_b).sum()),
               gained_tissue=int(gained.sum()), gained_candidate_swaps=int((gained & wrong_c).sum()))
    total_b = int((tissue & (baseline > 0) & wrong_b).sum())
    total_c = int((tissue & (candidate > 0) & wrong_c).sum())
    assert total_c-total_b == c-b+row['gained_candidate_swaps']-row['lost_baseline_swaps']
    for cls, label in ((1, 'anterior'), (2, 'posterior')):
        mask = shared & (truth == cls)
        row[label+'_shared_voxels'] = int(mask.sum())
        row[label+'_baseline_swaps'] = int((mask & wrong_b).sum())
        row[label+'_candidate_swaps'] = int((mask & wrong_c).sum())
    return row


def aggregate(records):
    result = {key:sum(r[key] for r in records) for key in records[0]}
    rates = [(r['candidate_shared_swaps']-r['baseline_shared_swaps'])/r['shared_voxels']
             for r in records if r['shared_voxels']]
    result.update(cases=len(records), cases_with_shared_tissue=len(rates),
        cases_improved=sum(r['candidate_shared_swaps'] < r['baseline_shared_swaps'] for r in records),
        cases_worsened=sum(r['candidate_shared_swaps'] > r['baseline_shared_swaps'] for r in records),
        cases_equal=sum(r['candidate_shared_swaps'] == r['baseline_shared_swaps'] for r in records),
        shared_swap_delta=result['candidate_shared_swaps']-result['baseline_shared_swaps'],
        mean_case_shared_error_rate_delta=float(np.mean(rates)) if rates else None)
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def audit(root, output, include_targeted=False):
    from baselines.swin_unetr.swin_unetr import _load_pkl_dataframe
    from scripts.depth_supervision.train import Patients
    from .ap82_matrix import ANCHOR
    p = json.loads((root/'protocol.json').read_text())
    assert sha(p['pkl']) == p['pkl_sha256']
    patients = Patients(_load_pkl_dataframe(p['pkl']), p['splits']['5pct']['val'], p['native_geometry'])
    assert len(patients) == 52
    table = {r['objective']:r for r in csv.DictReader((root/'results.csv').open())}
    scope = json.loads((root/'ap82_scope.json').read_text())
    entries = list(scope['stage1'])
    if include_targeted:
        entries += json.loads((root/'ap82_targeted_scope.json').read_text())['entries']
    assert all(e['id'] in table for e in entries), 'Finish all requested comparisons before freezing evidence'
    report = dict(protocol_identity=p['identity_sha'], stage1_scope_sha256=sha(root/'ap82_scope.json'),
        source_sha256=sha(__file__), definition='GT foreground AND baseline foreground AND candidate foreground; identical voxels within each paired comparison.',
        interpretation='Development-set diagnostic, not independent replication or proof of causality. Pairwise supports differ across candidates; compare each against its own matched baseline.',
        validation_cases=52, rows=[], artifacts={})
    for bg, reference in (('winner', ANCHOR), ('dice', 'dice')):
        path = root/'runs'/reference/'validation_predictions.npz'
        record = json.loads((path.parent/'completion.json').read_text())
        assert sha(path) == record['artifacts'][path.name]
        report['artifacts'][str(path.relative_to(root))] = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            baseline = {item['case_name']:archive[item['case_name']] for item in patients.items}
        b = table[reference]
        for e in entries:
            if e['background'] != bg: continue
            name = e['id']; path = root/'runs'/name/'validation_predictions.npz'
            record = json.loads((path.parent/'completion.json').read_text())
            assert sha(path) == record['artifacts'][path.name]
            report['artifacts'][str(path.relative_to(root))] = sha(path)
            records = []
            with np.load(path, allow_pickle=False) as archive:
                assert set(archive.files) == set(baseline)
                for item in patients.items:
                    records.append(paired_counts(item['label'][0].numpy(), baseline[item['case_name']], archive[item['case_name']]))
            row = dict(objective=name, background=bg, baseline=reference, **aggregate(records))
            for key in ('macro_dice','foreground_dice','surface_distance_mean_mm','hd95_mm',
                        'ap_interface_chamfer_mm','ap_interface_hd95_mm','ap_best_plane_error_mm'):
                row[key] = float(table[name][key])
                row['delta_'+key] = row[key]-float(b[key])
            for key in ('missed','extra','swaps'):
                row[key] = int(table[name][key]); row['delta_'+key] = row[key]-int(b[key])
            row['delta_outer_errors'] = row['delta_missed']+row['delta_extra']
            assert row['delta_swaps'] == row['shared_swap_delta']+row['gained_candidate_swaps']-row['lost_baseline_swaps']
            report['rows'].append(row)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), comparisons=len(report['rows']), validation_cases=52)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-targeted',action='store_true');args=parser.parse_args()
    audit(args.root,args.output,args.include_targeted)
