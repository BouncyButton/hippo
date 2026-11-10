"""Matched examples selected by a recorded rule from all held-out predictions."""
import argparse
import csv
import json
from pathlib import Path
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from driver import read,write,digest

def render(stage):
    tasks=read(stage/'plan.json')['tasks'];folder=stage/'figures';folder.mkdir(exist_ok=True)
    summaries=[]
    for dataset in ('GeoFWI3D','MSD','ADNI'):
        base=next(t for t in tasks if t['dataset']==dataset and t['role']=='baseline')
        aux=next(t for t in tasks if t['dataset']==dataset and t['role']=='constrained')
        paths=[stage/'runs'/t['id'] for t in (base,aux)]
        results=[{r['case']:r for r in read(p/'evaluation.json')['records'] if r['split']=='validation'} for p in paths]
        assert results[0].keys()==results[1].keys()
        paired=[]
        for case in sorted(results[0]):
            a,b=results[0][case],results[1][case]
            assert a['truth_sha256']==b['truth_sha256'] and a['shape']==b['shape']
            am,bm=a['qualitative_metrics'],b['qualitative_metrics']
            row=dict(case=case,baseline_dice=am['foreground_dice'],constrained_dice=bm['foreground_dice'],
                delta_pp=100*(bm['foreground_dice']-am['foreground_dice']))
            for k in ('background_dice','all_class_dice','foreground_fn','foreground_fp','ap_swaps'):
                row['baseline_'+k]=am[k];row['constrained_'+k]=bm[k]
            paired.append(row)
        with (folder/(dataset+'_all_cases.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
        ordered=sorted(paired,key=lambda r:(r['delta_pp'],r['case']))
        selected=[('Median paired change',ordered[len(ordered)//2]),('Smallest paired change',ordered[0]),('Largest paired change',ordered[-1])]
        fig,axes=plt.subplots(3,4,figsize=(12,9),layout='constrained')
        cmap=ListedColormap(['#2878b5','#e3812e'])
        metadata=[]
        for line,(description,row) in enumerate(selected):
            case=row['case'];data=[np.load(p/results[i][case]['artifact'],allow_pickle=False) for i,p in enumerate(paths)]
            assert np.array_equal(data[0]['truth'],data[1]['truth']) and np.array_equal(data[0]['image'],data[1]['image'])
            image,truth=data[0]['image'],data[0]['truth'];z=int(np.argmax((truth>0).sum(axis=(0,1))))
            masks=[None,truth[:,:,z].T,data[0]['prediction'][:,:,z].T,data[1]['prediction'][:,:,z].T]
            for col,mask in enumerate(masks):
                ax=axes[line,col];ax.imshow(image[:,:,z].T,cmap='gray',vmin=0,vmax=1,origin='lower',interpolation='nearest')
                if mask is not None:
                    ax.imshow(np.ma.masked_where(mask==0,mask),cmap=cmap,vmin=1,vmax=2,alpha=.6,
                        origin='lower',interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                if line==0:ax.set_title(['Input','Reference','Baseline','Constrained'][col],fontsize=12)
            axes[line,0].set_ylabel(description+'\n'+case+'; slice '+str(z),fontsize=9)
            axes[line,2].set_xlabel('Volume Dice '+format(100*row['baseline_dice'],'.2f')+'%',fontsize=10)
            axes[line,3].set_xlabel('Volume Dice '+format(100*row['constrained_dice'],'.2f')+'%; '+format(row['delta_pp'],'+.2f')+' pp',fontsize=10)
            metadata.append(dict(case=case,slice_axis=2,slice_zero_based=z,selection=description,**{k:v for k,v in row.items() if k!='case'}))
            for d in data:d.close()
        names=['Anterior','Posterior'] if dataset=='MSD' else ['Salt' if dataset=='GeoFWI3D' else 'Hippocampus']
        fig.legend(handles=[Patch(color=['#2878b5','#e3812e'][i],label=name) for i,name in enumerate(names)],loc='outside lower center',ncols=len(names),frameon=False)
        fig.suptitle(dataset+' — matched qualitative reproduction',fontsize=15)
        fig.savefig(folder/(dataset+'_paired.png'),dpi=220)
        fig.savefig(folder/(dataset+'_paired.svg'));plt.close(fig)
        summary=dict(dataset=dataset,baseline=base,constrained=aux,held_out_cases=len(paired),examples=metadata,
            mean_paired_gain_pp=float(np.mean([r['delta_pp'] for r in paired])),
            positive_cases=sum(r['delta_pp']>0 for r in paired),
            qualification='Fresh reproduction for visualization. Original nine-cell study remains the source of aggregate claims.',
            slice_rule='First source-axis-2 slice attaining maximum reference foreground area; identical image and slice for all panels.',
            original_comparisons=[read(p/'original_comparison.json') for p in paths])
        summaries.append(summary)
        write(folder/(dataset+'_selection.json'),summary)
    write(folder/'summary.json',summaries)
    manifest={str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()}
    write(folder/'archive_manifest.json',dict(files=manifest))
    archive=stage/'exports/figures.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for p in folder.rglob('*'):
            if p.is_file():t.add(p,arcname='figures/'+str(p.relative_to(folder)),recursive=False)
    write(stage/'exports/figures.tar.ready.json',dict(file=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,status='ready_for_off_cluster_archive'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    render(parser.parse_args().stage)
