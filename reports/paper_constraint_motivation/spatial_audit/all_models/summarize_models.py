"""Audit coverage and summarize each architecture/dataset before averaging models."""
import json
import sys
from pathlib import Path
import pandas as pd
from collect import check
from render_results import write_coverage, write_report

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from summarize import rates

jobs=json.loads((HERE/'jobs.json').read_text())
done=check()
tables=[]
for f in HERE.parent.glob('*_f*_s1741_u*.csv'):
    tables.append(pd.read_csv(f).assign(model='basic_unet16'))
for key in done:
    for t in (5,50):
        tables.append(pd.read_csv(HERE/'results'/key/f'u{t}.csv'))
cases=pd.concat(tables,ignore_index=True)
assert not cases.duplicated(['model','dataset','fold','seed','step','case']).any()
records=[]
for (m,d,f,t),c in cases.groupby(['model','dataset','fold','step']):
    records.append(dict(model=m,dataset=d,fold=f,step=t,**rates(c)))
runs=pd.DataFrame(records)
runs.to_csv(HERE/'per_run.csv',index=False)
cells=runs.drop(columns='fold').groupby(['model','dataset','step'])
means=cells.mean().rename(columns={'crop_evaluations':'mean_crops_per_fold'})
means.insert(0,'partitions',cells.size())
means.to_csv(HERE/'model_dataset_means.csv')
write_coverage(means)
if len(done)==len(jobs):
    assert (means.partitions==3).all() and len(means)==40
    selected=means.drop(columns=['partitions','mean_crops_per_fold'])
    selected.groupby(['model','step']).mean().to_csv(HERE/'model_means.csv')
    selected.groupby(['dataset','step']).mean().to_csv(HERE/'dataset_means.csv')
    selected.groupby(['step']).mean().to_csv(HERE/'overall_means.csv')
    write_report(means, runs)
    print('Complete: 60 runs, five models, four datasets, three partitions, two updates.')
else:
    print(len(done),'of 48 additional runs verified. Previous 12 runs included.')
