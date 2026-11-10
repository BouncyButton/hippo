"""Compare verified fold-0/seed-0 ADNI and MSD baseline metadata; no training."""
from pathlib import Path
import csv
import hashlib
import io
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hippo-adni-plot')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUDGET = 31976450667090
MODELS = {'swinunetr':'SwinUNETR', 'unetr_vit_b16':'UNETR',
          'basic_unet16':'3D U-Net', 'nnunet':'nnU-Net'}
COLORS = {'ADNI':'#007C91', 'MSD':'#B87928'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    snapshot = json.loads((HERE/'results_snapshot.json').read_text())
    assert snapshot['summary']['status'] == 'complete'
    curves, summaries, matching = {}, [], {}
    for model in MODELS:
        curves[model] = {}
        configurations, recipes = {}, {}
        for dataset in ('ADNI','MSD'):
            if dataset == 'ADNI':
                files = snapshot['models'][model]
            else:
                directory = ROOT/'reports/four_models_six_optimization_31976_20261006/runs_metadata'/model/'dice/fold0/seed0/run'
                files = {p.name:p.read_bytes().decode('utf-8') for p in directory.iterdir() if p.suffix in ('.json','.csv')}
            done, config, recipe = [json.loads(files[f]) for f in ('completion.json','config.json','runtime_recipe.json')]
            for key, filename in [('config_sha256','config.json'),('metrics_sha256','metrics.csv'),
                                  ('budget_ledger_sha256','budget_ledger.json'),('validation_best_sha256','validation_best.json')]:
                assert sha(files[filename].encode()) == done[key], (dataset,model,filename)
            assert done['status'] == 'complete' and done['discarded_attempts'] == 0
            assert config['budget_flops'] == BUDGET and config['fold'] == config['seed'] == 0
            configurations[dataset], recipes[dataset] = config, recipe
            rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(io.StringIO(files['metrics.csv']))]
            steps = {int(r['step']):r for r in rows}
            cost = int(rows[-1]['estimated_training_flops']) // int(rows[-1]['charged_attempts'])
            assert done['updates'] == BUDGET // cost
            assert done['updates'] * cost <= BUDGET < (done['updates']+1) * cost
            eighty = steps[(BUDGET*4//5)//cost]
            best_row = max(rows,key=lambda r:r['macro_hard_dice'])
            assert abs(best_row['macro_hard_dice']-done['best_validation_hard_dice']) < 1e-10
            summaries.append(dict(dataset=dataset,model=model,updates=done['updates'],
                final_dice_pct=rows[-1]['macro_hard_dice']*100,
                best_dice_pct=best_row['macro_hard_dice']*100,
                best_step=done['selected_step'], best_compute_fraction=best_row['compute_fraction'],
                reference_budget_fraction=.8, dice_at_80pct=eighty['macro_hard_dice']*100,
                last_20pct_gain_pp=(rows[-1]['macro_hard_dice']-eighty['macro_hard_dice'])*100,
                best_to_final_drop_pp=(best_row['macro_hard_dice']-rows[-1]['macro_hard_dice'])*100,
                final_soft_dice_pct=rows[-1]['macro_soft_dice']*100,
                final_loss=rows[-1]['mean_train_loss'],
                final_lr=rows[-1]['learning_rate'],
                consumed_gflops=rows[-1]['estimated_training_flops']/1e9,
                completion_sha256=sha(files['completion.json'].encode()),
                metrics_sha256=sha(files['metrics.csv'].encode())))
            curves[model][dataset] = rows
        keys = ['fold','seed','torch','monai','augmentation','precision','initial_lr','max_steps','budget_flops']
        if model == 'nnunet':
            keys += ['nnunet_sgd_momentum','nnunet_warmup_fraction']
        for key in keys:
            assert configurations['ADNI'][key] == configurations['MSD'][key], (model,key)
        assert recipes['ADNI']['lr_horizon_updates'] == configurations['MSD']['lr_horizon_updates']
        adni_steps = {int(r['step']):r['learning_rate'] for r in curves[model]['ADNI']}
        msd_steps = {int(r['step']):r['learning_rate'] for r in curves[model]['MSD']}
        for step in set(adni_steps) & set(msd_steps):
            assert abs(adni_steps[step]-msd_steps[step]) < 1e-12
        matching[model] = dict(verified_equal_fields=keys,lr_horizon_updates=recipes['ADNI']['lr_horizon_updates'],
                               common_step_lrs_match=True,
                               adni_training_crops=len(configurations['ADNI']['train_cases']),
                               msd_training_cases=len(configurations['MSD']['train_cases']))
    # Exact target checkpoints, rounded down to each model's affordable update.
    # Extra MSD observations remain in the plot, but not in matched-grid summaries.
    common = []
    for model in MODELS:
        for dataset, rows in curves[model].items():
            cost = int(rows[-1]['estimated_training_flops'])//int(rows[-1]['step'])
            by_step = {int(r['step']):r for r in rows}
            for k in range(1,21):
                step=(k*BUDGET//20)//cost
                row=by_step[step]
                common.append(dict(model=model,dataset=dataset,target_fraction=k/20,
                    actual_fraction=row['compute_fraction'],step=step,
                    dice_pct=row['macro_hard_dice']*100,loss=row['mean_train_loss'],
                    lr=row['learning_rate']))
    for name, records in [('comparison_summary.csv',summaries),('matched_compute_checkpoints.csv',common)]:
        with (HERE/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    evidence = dict(job_id=698735,status='complete',elapsed='00:11:31',exit_code='0:0',
        budget_gflops=BUDGET/1e9,comparison_scope='fold 0, seed 0, baseline objectives only',
        matched_recipe_checks=matching,results=summaries,
        caveats=['ADNI is binary whole-hippocampus Dice; MSD is anterior/posterior macro Dice.',
                 'ADNI has five training subjects / ten paired crops; MSD has ten named training cases.',
                 'Absolute training losses differ with class count and objective; nnU-Net uses Dice+CE.',
                 'Loss means span the interval since validation; MSD has extra validation checkpoints.',
                 'One seed/fold per dataset; late gains are descriptive, not proof of convergence or a shared optimization basin.',
                 'MSD metadata lacks foreground-union intersection/counts or saved predictions; whole-hippocampus Dice cannot be reconstructed from class Dice alone.'],
        snapshot_sha256=sha((HERE/'results_snapshot.json').read_bytes()))
    (HERE/'comparison_summary.json').write_text(json.dumps(evidence,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    for metric,filename,title,ylabel in [
        ('macro_hard_dice','adni_vs_msd_learning_curves','Validation learning curves at the same compute cap','Validation Dice (%)'),
        ('mean_train_loss','adni_vs_msd_training_loss','Training-loss trajectories','Training objective (raw value)')]:
        fig,axes=plt.subplots(2,2,figsize=(12.8,8.3),sharex=True,sharey=metric=='macro_hard_dice')
        for ax,(model,label) in zip(axes.flat,MODELS.items()):
            ax.axvspan(.8*BUDGET/1e12,BUDGET/1e12,color='#EDEFF1',zorder=0)
            for dataset in ('MSD','ADNI'):
                rows=curves[model][dataset]
                x=np.array([r['estimated_training_flops']/1e12 for r in rows])
                y=np.array([r[metric] for r in rows])*(100 if metric=='macro_hard_dice' else 1)
                ax.plot(x,y,color=COLORS[dataset],linewidth=2,label=dataset,
                        marker='o',markersize=3,alpha=.95)
                if metric=='macro_hard_dice':
                    best=int(np.argmax(y))
                    ax.scatter(x[best],y[best],s=68,facecolors='none',edgecolors=COLORS[dataset],linewidth=1.3,zorder=5)
                    other='MSD' if dataset=='ADNI' else 'ADNI'
                    other_final=curves[model][other][-1][metric]*100
                    offset=8 if y[-1]>=other_final else -13
                    ax.annotate(f'{y[-1]:.1f}%',(x[-1],y[-1]),xytext=(-3,offset),
                                textcoords='offset points',ha='right',color=COLORS[dataset],weight='bold',fontsize=9)
            ax.set_title(f'{label}  ·  {int(curves[model]["ADNI"][-1]["step"])} updates',loc='left',pad=10)
            ax.set_xlim(0,34);ax.set_xticks([0,8,16,24,32]);ax.grid(axis='y',color='#E4E7EA',linewidth=.7)
            if metric=='macro_hard_dice':
                ax.set_ylim(0,100);ax.set_yticks([0,20,40,60,80,100])
            ax.set_xlabel('Training compute (thousand GFLOPs)')
            ax.set_ylabel(ylabel)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles[::-1],labels[::-1],loc='upper right',bbox_to_anchor=(.975,.982),ncol=2,frameon=False)
        fig.suptitle(title,x=.065,y=.975,ha='left',fontsize=19,weight='bold')
        fig.text(.065,.923,'Fold 0 · seed 0 · 5% training subset · 31,976.45 GFLOPs per model',fontsize=10.5,color='#495563')
        note=('ADNI: whole hippocampus. MSD: anterior/posterior macro Dice. Absolute scores measure different targets.\n'
              'Shading: final 20% of the allowance. Open circles: best observed checkpoints. Lines join recorded observations.') if metric=='macro_hard_dice' else (
              'Raw losses are not directly comparable across datasets or objectives. MONAI models: Dice; nnU-Net: Dice+CE.\n'
              'Values are interval means; MSD has additional validation checkpoints and thus shorter averaging windows.')
        fig.text(.065,.035,note,fontsize=9,color='#495563',linespacing=1.5)
        fig.subplots_adjust(left=.075,right=.97,bottom=.14,top=.86,hspace=.35,wspace=.22)
        fig.savefig(HERE/(filename+'.png'),dpi=170)
        fig.savefig(HERE/(filename+'.pdf'))
        plt.close(fig)
    print(json.dumps({'status':'verified_and_plotted','results':summaries},indent=2))


if __name__ == '__main__':
    main()
