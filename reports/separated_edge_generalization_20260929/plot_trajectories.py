"""Plot saved validation audits and explicitly limited training-probe trajectories."""
import os
from pathlib import Path
import json
import statistics as st
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=OUT.parent/'separated_edge_20260929/results'
COLORS={'pooled':'#2369a6','separated':'#d2691e'}
COMMON=[5,15,30,45,60]

def read(p):return json.loads(p.read_text())

def main():
    data=[]
    for arm in COLORS:
        for seed in range(3):
            p=BASE/f'fold0/seed{seed}/{arm}'
            for f in sorted((p/'coherence').glob('epoch_*.json')):
                epoch=int(f.stem.split('_')[-1]);rows=read(f);probe=read(p/f'gradients/epoch_{epoch:03d}.json')
                data.append(dict(arm=arm,seed=seed,epoch=epoch,validation={k:st.mean(r['metrics'][k] for r in rows) for k in rows[0]['metrics']},train={k:st.mean(r['losses'][k] for r in probe['cases']) for k in ['inner','outer','cross']}))
    fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for ax,(key,title) in zip(axs.flat,[('inner_both_correct','Inner pair correctness'),('outer_both_correct','Outer pair correctness'),('cross_correct_transition','Crossing correct transition'),('macro_dice','Macro Dice')]):
        for arm,color in COLORS.items():
            for seed in range(3):
                q=sorted([r for r in data if r['arm']==arm and r['seed']==seed],key=lambda r:r['epoch'])
                ax.plot([r['epoch'] for r in q],[r['validation'][key]*100 for r in q],color=color,alpha=.23,lw=1)
                p=BASE/f'fold0/seed{seed}/{arm}';selected=read(p/'selected_cases.json');ep=read(p/'completion.json')['selected_epoch']
                ax.scatter([ep],[st.mean(r['metrics'][key] for r in selected if r['split']=='validation')*100],color=color,marker='*',s=60,zorder=4)
            ax.plot(COMMON,[st.mean(r['validation'][key] for r in data if r['arm']==arm and r['epoch']==ep)*100 for ep in COMMON],color=color,label=arm.capitalize(),lw=2.3,marker='o',ms=4)
        ax.set_title(title);ax.set_ylabel('%');ax.grid(alpha=.2);ax.set_xlim(3,77)
    for ax in axs[-1]:ax.set_xlabel('Epoch')
    axs[0,0].legend(frameon=False)
    fig.suptitle('Validation trajectories: regional correctness and Dice',fontsize=13)
    fig.text(.5,.01,'52 validation cases per seed. Faint lines: individual seeds. Bold: three-seed means at common epochs. Stars: selected checkpoints.\nCorrectness is distinct from same-side agreement and soft constraint loss.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95))
    fig.savefig(OUT/'validation_trajectories.png',dpi=180);fig.savefig(OUT/'validation_trajectories.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axs,['inner','outer','cross']):
        for arm,color in COLORS.items():
            ax.plot(COMMON,[st.mean(r['train'][key] for r in data if r['arm']==arm and r['epoch']==ep) for ep in COMMON],color=color,ls='--',marker='o',ms=3,label=arm.capitalize()+' train (2 cases)')
            ax.plot(COMMON,[st.mean(r['validation'][key+'_soft_squared_error'] for r in data if r['arm']==arm and r['epoch']==ep) for ep in COMMON],color=color,lw=2,marker='o',ms=3,label=arm.capitalize()+' validation (52 cases)')
        ax.set_title(key.capitalize()+' edge MSE');ax.set_xlabel('Epoch');ax.set_ylabel('Mean squared error');ax.grid(alpha=.2)
    axs[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Limited soft-error trajectories: fixed training probes versus full validation',fontsize=13)
    fig.text(.5,.01,'Case means first, then three seeds. Training probes use FP32 eval mode; validation uses CUDA AMP.\nThis is not a full-training hard-satisfaction trajectory; lower MSE is better.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.94));fig.savefig(OUT/'soft_probe_trajectories.png',dpi=180);fig.savefig(OUT/'soft_probe_trajectories.pdf');plt.close(fig)

if __name__=='__main__':main()
