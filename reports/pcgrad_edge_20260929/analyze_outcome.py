"""Describe completed PCGrad results without changing experiment outputs."""
import hashlib
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
BASE = OUT / 'results'
s = json.loads((BASE / 'SUMMARY.json').read_text())
assert json.loads((BASE / 'LAUNCHER_EXIT.json').read_text())['exit_code'] == 0
assert s['status'] == 'complete'
rows, cases, tails = [], [], []
for seed in range(3):
    for arm in ('sum', 'pcgrad'):
        p = BASE / f'fold0/seed{seed}' / arm
        c = json.loads((p / 'completion.json').read_text())
        b = (p / 'selected_cases.json').read_bytes()
        assert hashlib.sha256(b).hexdigest() == c['selected_cases_sha256']
        data = json.loads(b)
        row = dict(seed=seed, arm=arm, selected_epoch=c['selected_epoch'],
                   stopped_epoch=c['stopped_epoch'], checkpoint_sha256=c['checkpoint_sha256'])
        for split, n in [('train', 10), ('validation', 52)]:
            group = [r for r in data if r['split'] == split]
            assert len(group) == n
            row[split] = {k: st.mean(r['metrics'][k] for r in group)
                          for k in group[0]['metrics']}
            cases.extend(dict(seed=seed, arm=arm, **r) for r in group)
        rows.append(row)
        if arm == 'pcgrad':
            epochs = [json.loads(f.read_text()) for f in sorted((p/'epochs').glob('*.json'))]
            q = epochs[-10:]
            x, y = [r['epoch'] for r in q], [r['val_dice_hard'] for r in q]
            slope = sum((a-st.mean(x))*(b-st.mean(y)) for a,b in zip(x,y))/sum((a-st.mean(x))**2 for a in x)
            tails.append(dict(seed=seed, epoch_start=x[0], epoch_end=x[-1],
                validation_macro_dice_slope_pp_per_epoch=100*slope,
                final_validation_macro_dice=y[-1],
                initial_train_soft_dice_including_background=1-q[0]['train_dice'],
                final_train_soft_dice_including_background=1-q[-1]['train_dice']))

out = dict(scope='Fold 0, same 10 training and 52 validation cases; three seeds; original 75-epoch-cap stopping policy.',
    averaging='Cases within each seed, then seeds. Descriptive repeated-case/seed summaries, not independent replications.',
    selected=rows, final_ten_epoch_trends=tails,
    validation_comparison=s['comparisons']['selected_validation'],
    epoch60_comparison=s['comparisons']['epoch60_validation'])
(OUT/'ANATOMY_SUMMARY.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
(OUT/'selected_cases.json').write_text(json.dumps(cases, indent=2, allow_nan=False)+'\n')

lines=['# PCGrad: completed anatomical-generalization assessment', '',
    '**The tested symmetric PCGrad configuration follows a different trajectory, but within this budget it does not generalize hippocampal anatomy better.** It improves inner-pair correctness and recall while substantially overpredicting foreground, losing correct boundary transitions, and producing more disconnected components. This pattern occurs in all three seeds.', '',
    'The six runs completed successfully (job 676122; 50 minutes 30 seconds). No optimizer updates were skipped. The comparison uses the same five weighted task gradients, summed normally in the control or projected with symmetric PCGrad. Data, initialization, original separated-rule coefficients, schedule, and stopping policy are matched. The fresh sum control is the primary comparator; the earlier separated experiment is a secondary reference.', '',
    '## Selected checkpoints', '',
    'Hard macro Dice averages anterior/posterior Dice within each case, then cases. These are post-training audits of selected checkpoints; they are not the logged training Dice loss or necessarily the final training epoch.', '',
    '| Seed | Arm | Selected epoch | Stopped epoch | Train Dice % | Validation Dice % | Gap (pp) |',
    '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    t,v=r['train']['macro_dice'],r['validation']['macro_dice']
    lines.append(f"| {r['seed']} | {r['arm']} | {r['selected_epoch']} | {r['stopped_epoch']} | {100*t:.4f} | {100*v:.4f} | {100*(t-v):+.4f} |")
for arm in ('sum','pcgrad'):
    rr=[r for r in rows if r['arm']==arm]
    t,v=[st.mean(r[sp]['macro_dice'] for r in rr) for sp in ('train','validation')]
    lines.append(f'| Mean | {arm} | — | — | {100*t:.4f} | {100*v:.4f} | {100*(t-v):+.4f} |')
lines += ['', '## Validation anatomy and boundary measurements', '',
    'Case means within seed, then the three seeds. Percentages are used for correctness/disagreement/Dice; counts are mean counts per case. Inner/outer pair correctness requires both endpoints to be correct. Crossing correctness is the correctly oriented ground-truth transition. Agreement alone can be coherently wrong.', '',
    '| Metric | Sum control | PCGrad | PCGrad minus control | Seeds improved |',
    '|---|---:|---:|---:|---:|']
metrics=[('Macro Dice %','macro_dice',100),('Foreground-union Dice %','union_dice',100),
    ('False-positive voxels','all_fp',1),('False-negative voxels','all_fn',1),
    ('Inner pair correctness %','inner_both_correct',100),('Outer pair correctness %','outer_both_correct',100),
    ('Correct boundary transitions %','cross_correct_transition',100),
    ('Inner pair disagreement %','inner_disagree',100),('Outer pair disagreement %','outer_disagree',100),
    ('Connected components (6-neighbor)','foreground_components_6',1),
    ('Foreground voxels outside largest component','foreground_outside_largest_6',1)]
for label,k,m in metrics:
    r=s['comparisons']['selected_validation'][k]
    lines.append(f"| {label} | {m*r['sum']:.4f} | {m*r['pcgrad']:.4f} | {m*r['delta']:+.4f} | {r['seeds_improved']}/3 |")
e=s['comparisons']['epoch60_validation']['macro_dice']
lines += ['',f"At the common epoch-60 endpoint, mean validation Dice is {100*e['sum']:.4f}% control versus {100*e['pcgrad']:.4f}% PCGrad. The decline is therefore not just a consequence of different selected epochs.", '',
    '## Does this support better anatomical learning or a need for more updates?', '',
    'The higher inner correctness and fewer false negatives are real validation improvements. They occur alongside a much larger foreground region and sharply worse outer/boundary correctness; these observations are consistent with increased foreground coverage rather than improved anatomical specificity. Lower neighboring disagreement does not imply better global connectivity: disconnected-component and outside-largest-component counts worsen in every seed. These measurements do not establish an internal anatomical representation.', '',
    'PCGrad also performs poorly on the training cases. Its smaller macro-Dice train–validation gap is therefore not evidence of better generalization. The ordinary sum controls fit and generalize substantially better with the same data/update budget. Limited data or insufficient updates may interact with PCGrad, but this experiment cannot isolate those explanations.', '',
    'All PCGrad runs used patience-based stopping under the 75-epoch cap, selecting epochs 61/58/62 and stopping at 69/66/70. The final-ten-epoch validation Dice slopes are descriptive only:', '',
    '| Seed | Final window | Validation Dice slope (pp/epoch) |', '|---|---|---:|']
for r in tails:
    lines.append(f"| {r['seed']} | {r['epoch_start']}–{r['epoch_end']} | {r['validation_macro_dice_slope_pp_per_epoch']:+.4f} |")
lines += ['', 'Training soft Dice continues improving over these windows, but it is the minibatch soft score including background, measured during updates; it is not the audited hard macro Dice. Validation hard Dice declines over each final window. This provides no observed late validation recovery, while leaving longer-budget behavior unknown. No longer run or new experiment has been launched.', '',
    'The ideal foreground probabilities p=y jointly satisfy all three edge rules. Gradient competition here is an optimization issue, not a logical inconsistency of the desired constraints. Symmetric PCGrad can redirect the supervised gradient even when the conflicting constraint has a small positive scalar weight; preserving the pre-projection loss coefficients does not bound the effect of projection.', '',
    'These conclusions apply to this symmetric five-task PCGrad implementation, reused validation fold and limited budget; they do not establish that every PCGrad variant or longer schedule would fail. Case/seed repetitions are not independent patient replications. Full original result summaries, per-update projection records, per-case audits and checkpoint hashes are retained.', '',
    f'![Validation learning curves]({OUT}/validation_learning_curves.png)', '']
(OUT/'INTERPRETATION.md').write_text('\n'.join(lines))

fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,h in zip(axes,s['learning_curves']):
    for arm,color,label in [('sum','#2367a5','Matched control'),('pcgrad','#d66818','PCGrad'),('previous_separated','#777777','Earlier separated')]:
        q=sorted((int(k),v*100) for k,v in h['histories'][arm].items())
        ax.plot([a for a,b in q],[b for a,b in q],color=color,label=label,linewidth=2 if arm!='previous_separated' else 1,alpha=1 if arm!='previous_separated' else .6)
    for r in rows:
        if r['seed']==h['seed']:
            ax.scatter(r['selected_epoch'],r['validation']['macro_dice']*100,marker='*',s=80,color='#2367a5' if r['arm']=='sum' else '#d66818',zorder=5)
    ax.set_title(f"Seed {h['seed']}");ax.set_xlabel('Epoch');ax.grid(alpha=.2)
axes[0].set_ylabel('Validation hard macro Dice (%)');axes[0].legend(fontsize=8)
fig.suptitle('Completed PCGrad pilot: same data, three initializations, 75-epoch cap')
fig.tight_layout();fig.savefig(OUT/'validation_learning_curves.png',dpi=160);fig.savefig(OUT/'validation_learning_curves.pdf');plt.close(fig)
print(json.dumps({'complete':True,'runs':len(rows),'cases':len(cases),'report':str(OUT/'INTERPRETATION.md')}))
