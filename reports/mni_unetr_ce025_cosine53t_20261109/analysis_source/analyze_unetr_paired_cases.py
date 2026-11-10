"""Describe crop and subject gains without treating repeated subjects as independent."""
import argparse
import json
import re
from pathlib import Path
from statistics import mean

from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.cross_dataset_campaign.storage import verify_archive
from scripts.unetr_subfields.contracts import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--result-file', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    assert '/' not in args.name and '/' not in args.result_file
    report = ROOT/'reports'/args.name
    result = read(report/args.result_file)
    rows = []
    for pair in result['rows']:
        if pair['candidate'] != args.candidate:
            continue
        folders = [report/'runs'/pair[k] for k in ('baseline_id', 'id')]
        evaluations, hashes = [], []
        for folder in folders:
            verify_archive(folder, read(folder/'archive_manifest.json'))
            assert read(folder/'cleanup.json')['status'] == 'complete'
            values = read(folder/'validation_latest.json')
            by_name = {v['case_name']:v for v in values['cases']}
            assert len(by_name) == len(values['cases']) == 10
            assert abs(mean(v['macro_hard_dice'] for v in by_name.values())
                       - values['summary']['macro_hard_dice']) < 1e-12
            evaluations.append(by_name)
            hashes.append(digest(folder/'archive_manifest.json'))
        baseline, constrained = evaluations
        assert baseline.keys() == constrained.keys()
        crops, subjects = [], {}
        for name in sorted(baseline):
            match = re.fullmatch(r'(hippocampus_mni_s\d+)_([LR])', name)
            assert match, name
            gain = 100*(constrained[name]['macro_hard_dice']-baseline[name]['macro_hard_dice'])
            subfields = {key:100*(constrained[name][key]-baseline[name][key])
                         for key in ('CA1_3_hard_dice', 'subiculum_hard_dice', 'CA4_DG_hard_dice')}
            crops.append(dict(case_name=name, subject=match[1], side=match[2],
                              gain_pp=gain, subfield_gain_pp=subfields))
            subjects.setdefault(match[1], []).append(crops[-1])
        assert len(subjects) == 5
        assert all({c['side'] for c in group} == {'L', 'R'} for group in subjects.values())
        subject_rows = [dict(subject=name, gain_pp=mean(c['gain_pp'] for c in group))
                        for name,group in sorted(subjects.items())]
        assert abs(mean(c['gain_pp'] for c in crops)-pair['gain_pp']) < 1e-10
        rows.append(dict(seed=pair['seed'], fold=pair['fold'], gain_pp=pair['gain_pp'],
            improved_crops=sum(c['gain_pp']>0 for c in crops), total_crops=10,
            improved_subjects=sum(s['gain_pp']>0 for s in subject_rows), total_subjects=5,
            minimum_crop_gain_pp=min(c['gain_pp'] for c in crops),
            maximum_crop_gain_pp=max(c['gain_pp'] for c in crops),
            crops=crops, subjects=subject_rows, archive_sha256=hashes))
    output = dict(candidate=args.candidate, rows=rows,
        input_result_sha256=digest(report/args.result_file), source_sha256=digest(Path(__file__)),
        scope='Descriptive paired final-budget crop and two-side subject means. The acceptance rule remains the nine seed/fold macro-Dice comparisons.',
        caveat='Repeated seeds revisit subjects and folds overlap in training. These counts are not independent replications, confidence intervals, or untouched-test evidence.')
    write(report/(args.candidate+'_paired_cases.json'), output)
    print(json.dumps([{k:v for k,v in row.items() if k not in ('crops','subjects','archive_sha256')}
                      for row in rows]))


if __name__ == '__main__':
    main()
