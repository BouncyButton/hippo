"""Audit fold-0 A/P errors from frozen checkpoint confusion matrices.

No training, inference, or changes to source experiments. Standard library only.
"""
import hashlib
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = Path('/Users/filippofocaccia/.codex/worktrees/5063/hippo/experiments')
ARMS = ('baseline', 'bands')
CHECKPOINTS = ('best', 'latest')


def source(schedule, seed, arm, checkpoint):
    if schedule == 75:
        base = EXP / 'final_four_fold_report_20260918/verified/results'
        base /= f'low_data_noaug_seed{seed}_20260917_01/bands_audit'
    else:
        suffix = 'seed0' if seed == 0 else 'seeds12'
        name = f'low_data_noaug_5pct_200ep_es_fold0_{suffix}_20260918'
        base = EXP / name / 'verified/results' / (name + '_01') / f'audits/seed{seed}'
    return base / f'{arm}_{checkpoint}.json'


def dice(cm):
    return st.mean(2 * cm[k][k] / (sum(cm[k]) + sum(r[k] for r in cm))
                   for k in (1, 2))


def profile(cm):
    fg = sum(cm[1]) + sum(cm[2])
    fn = cm[1][0] + cm[2][0]
    ap, pa = cm[1][2], cm[2][1]
    oracle = [row[:] for row in cm]
    oracle[1][1] += ap
    oracle[2][2] += pa
    oracle[1][2] = oracle[2][1] = 0
    return dict(swaps=ap + pa, a_to_p=ap, p_to_a=pa, fn=fn,
                fp=cm[0][1] + cm[0][2], gt_fg=fg, retained_gt_fg=fg - fn,
                swap_pct_gt=100 * (ap + pa) / fg,
                swap_pct_retained=100 * (ap + pa) / (fg - fn),
                a_to_p_pct_gt_a=100 * ap / sum(cm[1]),
                p_to_a_pct_gt_p=100 * pa / sum(cm[2]),
                net_anterior_overassignment_on_retained_gt=pa - ap,
                pooled_dice_pct=100 * dice(cm),
                pooled_swap_oracle_dice_pct=100 * dice(oracle))


records = []
lookup = {}
gt_counts = {}
case_sets = {}
for schedule in (75, 200):
    for seed in range(3):
        for checkpoint in CHECKPOINTS:
            for arm in ARMS:
                path = source(schedule, seed, arm, checkpoint)
                raw = path.read_bytes()
                data = json.loads(raw)
                for split in ('train', 'validation'):
                    d = data[split]
                    cases = []
                    summed = [[0] * 3 for _ in range(3)]
                    for c in d['cases']:
                        name, cm = c['case_name'], c['confusion']
                        rows = tuple(sum(row) for row in cm)
                        key = split, name
                        if key in gt_counts:
                            assert gt_counts[key] == rows, (path, key)
                        gt_counts[key] = rows
                        p = profile(cm)
                        assert abs(p['pooled_dice_pct'] - 100 * c['dice_hard']) < 1e-4
                        assert p['swaps'] == c['ap_swap_voxels']
                        cases.append(dict(case_name=name, **p))
                        for i in range(3):
                            for j in range(3):
                                summed[i][j] += cm[i][j]
                    assert summed == d['confusion']
                    names = frozenset(c['case_name'] for c in cases)
                    assert len(names) == len(cases) == d['case_count']
                    if split in case_sets:
                        assert names == case_sets[split]
                    case_sets[split] = names
                    p = profile(summed)
                    assert p['swaps'] == d['ap_swap_voxels']
                    mean_dice = st.mean(c['pooled_dice_pct'] for c in cases)
                    assert abs(mean_dice - 100 * d['dice_hard']) < 1e-4
                    oracle = st.mean(c['pooled_swap_oracle_dice_pct'] for c in cases)
                    r = dict(schedule=schedule, seed=seed, checkpoint=checkpoint,
                             arm=arm, split=split, epoch=data['epoch'],
                             source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                             **p, case_count=len(cases), cases_with_swaps=sum(c['swaps'] > 0 for c in cases),
                             mean_case_swap_pct_retained=st.mean(c['swap_pct_retained'] for c in cases),
                             median_case_swap_pct_retained=st.median(c['swap_pct_retained'] for c in cases),
                             cases_p_to_a_dominant=sum(c['p_to_a'] > c['a_to_p'] for c in cases),
                             cases_a_to_p_dominant=sum(c['a_to_p'] > c['p_to_a'] for c in cases),
                             dice_pct=mean_dice, swap_oracle_dice_pct=oracle,
                             swap_oracle_gain_pp=oracle - mean_dice,
                             top5_swap_share_pct=100 * sum(sorted((c['swaps'] for c in cases), reverse=True)[:5]) / p['swaps'] if p['swaps'] else 0,
                             cases=cases)
                    records.append(r)
                    lookup[schedule, seed, checkpoint, arm, split] = r
assert case_sets['train'].isdisjoint(case_sets['validation'])

pairs = []
for schedule in (75, 200):
    for seed in range(3):
        for checkpoint in CHECKPOINTS:
            for split in ('train', 'validation'):
                b, a = [lookup[schedule, seed, checkpoint, arm, split] for arm in ARMS]
                bc = {c['case_name']: c for c in b['cases']}
                ac = {c['case_name']: c for c in a['cases']}
                changes = [dict(case_name=name, swap_delta=ac[name]['swaps'] - bc[name]['swaps'],
                                retained_rate_delta_pp=ac[name]['swap_pct_retained'] - bc[name]['swap_pct_retained'],
                                fn_delta=ac[name]['fn'] - bc[name]['fn']) for name in sorted(bc)]
                pairs.append(dict(schedule=schedule, seed=seed, checkpoint=checkpoint, split=split,
                                  swap_delta=a['swaps'] - b['swaps'],
                                  swap_relative_change_pct=100 * (a['swaps'] - b['swaps']) / b['swaps'] if b['swaps'] else None,
                                  retained_rate_delta_pp=a['swap_pct_retained'] - b['swap_pct_retained'],
                                  cases_fewer_swaps=sum(c['swap_delta'] < 0 for c in changes),
                                  cases_equal_swaps=sum(c['swap_delta'] == 0 for c in changes),
                                  cases_more_swaps=sum(c['swap_delta'] > 0 for c in changes),
                                  cases_lower_retained_rate=sum(c['retained_rate_delta_pp'] < 0 for c in changes),
                                  cases_higher_retained_rate=sum(c['retained_rate_delta_pp'] > 0 for c in changes),
                                  changes=changes))

out = dict(scope='Fold 0, same 10 training and 52 validation cases; 24 checkpoint files, 48 split profiles. No spatial predictions, training, or inference.',
           definitions=dict(confusion='Rows are ground truth; columns are predictions; 0=background, 1=anterior, 2=posterior.',
                            swaps='C[1,2]+C[2,1]', retained_rate='100*swaps/(GT foreground - foreground misses)',
                            oracle='Correct A/P swaps only, leave all foreground/background mistakes unchanged; reconstruct per-case class-mean Dice and average over cases.'),
           records=records, paired_comparisons=pairs)
(HERE / 'AP_SWAPS_ANALYSIS.json').write_text(json.dumps(out, indent=2) + '\n')

lines = ['# Fold-0 A/P swaps: training versus validation', '',
         'Generated from frozen checkpoint evaluations by `analyze_ap_swaps.py`. No jobs, inference, or original experiment modifications.', '',
         '## Scope and interpretation', '',
         'Ten identical training cases and 52 identical validation cases across all seeds, arms and schedules. GT foreground totals are 32,517 training and 174,350 validation voxels. Source hashes and every case-level measurement are in AP_SWAPS_ANALYSIS.json. Assertions check case identities, class totals, confusion-matrix sums and reconstruction of reported Dice.', '',
         'A/P swaps are true anterior predicted posterior or true posterior predicted anterior, excluding foreground misses and background false positives. The retained-foreground rate divides swaps by GT foreground minus foreground misses. This helps expose changes in foreground retention, but uses a different voxel support for each model; it is not a paired same-voxel cut comparison.', '',
         'The maximum-200 runs stopped early: final epochs are shown below. Best checkpoints were selected for overall validation Dice, not minimum swaps. These are retrospective descriptive comparisons, not an independent held-out test of A/P model selection.', '']
for checkpoint in CHECKPOINTS:
    lines += [f'## {checkpoint.capitalize()} checkpoints', '',
              'Arrows are baseline → bands. Rate is validation swaps / retained true foreground.', '',
              '| Max epochs | Seed | Epochs | Train swaps | Validation swaps | Δ swaps | Validation rate (%) | Cases fewer / tied / more swaps |',
              '|---|---|---|---|---|---|---|---|']
    for schedule in (75, 200):
        for seed in range(3):
            b, a = [lookup[schedule, seed, checkpoint, arm, 'validation'] for arm in ARMS]
            bt, at = [lookup[schedule, seed, checkpoint, arm, 'train'] for arm in ARMS]
            p = next(p for p in pairs if (p['schedule'], p['seed'], p['checkpoint'], p['split']) == (schedule, seed, checkpoint, 'validation'))
            lines.append(f"| {schedule} | {seed} | {b['epoch']} → {a['epoch']} | {bt['swaps']:,} → {at['swaps']:,} | {b['swaps']:,} → {a['swaps']:,} | {p['swap_delta']:+,} | {b['swap_pct_retained']:.3f} → {a['swap_pct_retained']:.3f} | {p['cases_fewer_swaps']} / {p['cases_equal_swaps']} / {p['cases_more_swaps']} |")
    lines.append('')

lines += ['## Every checkpoint: directions, coverage and diagnostic headroom', '',
          'Swap-only oracle gain is a theoretical relabeling diagnostic, not a forecast of an achievable improvement. It leaves FP/FN unchanged and does not reconstruct or score an anatomical surface. All Dice gains below use the original macro-over-cases metric, not pooled-voxel Dice.', '',
          '| Schedule | Seed | Checkpoint | Arm | Split | A→P | P→A | FN | Swap / GT (%) | Swap / retained GT (%) | Dice (%) | Swap-only oracle gain (pp) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in records:
    lines.append(f"| {r['schedule']} | {r['seed']} | {r['checkpoint']} | {r['arm']} | {r['split']} | {r['a_to_p']:,} | {r['p_to_a']:,} | {r['fn']:,} | {r['swap_pct_gt']:.4f} | {r['swap_pct_retained']:.4f} | {r['dice_pct']:.3f} | {r['swap_oracle_gain_pp']:.3f} |")

lines += ['', '## Findings and limits', '',
          '- Training A/P assignment is already almost perfect on retained GT foreground; every final maximum-200 checkpoint has zero training swaps. This is not equivalent to perfect overall training segmentation.',
          '- At the original 75-epoch best checkpoints, bands reduces validation swaps for all three seeds. However, it also increases foreground misses for all three. Lower retained-foreground rates do not rule out selective removal of difficult-to-label voxels.',
          '- At maximum 200, best-checkpoint bands effects are mixed: seed 0 worsens both raw count and retained-foreground rate; seed 1 increases the raw count slightly while improving the rate and retaining more foreground; seed 2 improves both A/P measures but misses more foreground. Final checkpoints confirm that seed 0 is worse and seed 2 better; seed 1 has fewer swaps and a better rate.',
          '- P→A exceeds A→P for every maximum-200 validation checkpoint. This is a net anterior overassignment within recovered true foreground, not proof of a consistently posterior-shifted anatomical plane. Missing foreground and false positives can alter full predicted class volumes.',
          '- Longer training does not uniformly reduce A/P confusion after accounting for retained foreground: seed 2 baseline best-checkpoint retained rates rise from 3.364% at maximum 75 to 3.570% at maximum 200; bands rises from 3.207% to 3.283%. Both raw counts fall, but many more GT foreground voxels are missed. Different schedules and selected epochs prevent attributing this change to duration alone.',
          '- Direction varies by case. For example, maximum-200 seed 2 baseline has more P→A than A→P swaps in 35 validation cases, but the reverse in 17. An aggregate directional imbalance is not evidence that a single global cut shift would fix all cases.',
          '- The archived bands loss supervises grouped foreground versus background and is invariant under exchanging A/P labels. Any A/P improvement is indirect, through the shared network or changed foreground support; these experiments do not show a direct cut-placement penalty.',
          '- Confusion matrices cannot establish cut displacement, tilt, roughness, disconnected islands, or error distance from the true interface. The local audited archives contain no spatial prediction volumes. Spatial claims require predictions and GT in the same physical coordinate system.',
          '- Next diagnostic: evaluate A/P mistakes on GT foreground predicted as foreground by BOTH arms; separately track swap→correct, swap→background and background→swap transitions. Inspect all 52 cases, not only selected successes. With spatial masks, report signed AP displacement and surface distances of the internal interface, coverage/missing-interface failures, and errors near versus far from the true interface. Use plane-fit offset/tilt only if a plane is justified by the labeling protocol.',
          '- Report paired case-level changes and seed-specific effects. Voxels are not independent samples; repeated seeds do not multiply the number of independent cases. Case identifiers alone do not establish independent subjects.', '',
          '## Validation case concentration at best checkpoints', '',
          '| Schedule | Seed | Arm | Cases with swaps | Top 5 cases’ share of swaps (%) | Median case retained rate (%) |',
          '|---|---|---|---|---|---|']
for r in records:
    if r['checkpoint'] == 'best' and r['split'] == 'validation':
        lines.append(f"| {r['schedule']} | {r['seed']} | {r['arm']} | {r['cases_with_swaps']}/{r['case_count']} | {r['top5_swap_share_pct']:.2f} | {r['median_case_swap_pct_retained']:.3f} |")
(HERE / 'AP_SWAPS_ANALYSIS.md').write_text('\n'.join(lines) + '\n')
print(f'Checked {len(records)} split profiles and {sum(len(r["cases"]) for r in records)} case profiles.')
for r in records:
    if r['checkpoint'] == 'best' and r['split'] == 'validation':
        print(json.dumps({k: r[k] for k in ('schedule', 'seed', 'arm', 'swaps', 'fn', 'swap_pct_retained', 'swap_oracle_gain_pp', 'top5_swap_share_pct')}))
