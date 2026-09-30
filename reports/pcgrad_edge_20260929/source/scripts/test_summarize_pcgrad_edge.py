import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_pcgrad_edge import summarize,KEYS


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_paired_seed_aggregation_projection_and_historical_drift(tmp_path):
    root=tmp_path/'new';old=tmp_path/'old'
    put(root/'CONFLICT_GATE.json',dict(passed=True,seeds=[0,1,2],counts=[10]*3))
    put(old/'SUMMARY.json',{'comparisons':{'selected_validation':{}}})
    gram=[[float(i==j) for j in range(5)] for i in range(5)]
    for seed in range(3):
        pair=root/f'fold0/seed{seed}'
        put(pair/'completion.json',dict(status='complete'))
        put(old/f'fold0/seed{seed}/separated/epochs/060.json',dict(epoch=60,val_dice_hard=.7))
        for arm in ('sum','pcgrad'):
            d=(seed+1)*.01 if arm=='pcgrad' else 0
            rows=[dict(split=split,case_name=f'{split}{i}',metrics={k:.1+d for k in KEYS}) for split,n in [('train',10),('validation',52)] for i in range(n)]
            put(pair/arm/'selected_cases.json',rows)
            put(pair/arm/'coherence/epoch_060.json',[r for r in rows if r['split']=='validation'])
            put(pair/arm/'completion.json',dict(stopped_epoch=60,selected_epoch=60))
            put(pair/arm/'epochs/060.json',dict(epoch=60,val_dice_hard=.7+d))
            step=dict(finite=True,optimizer_step_skipped=False,task_gram=gram,
                projection_count=int(arm=='pcgrad'),projected_over_original_norm=.8 if arm=='pcgrad' else 1.,cosine_with_original=.9 if arm=='pcgrad' else 1.)
            put(pair/arm/'updates/060.json',dict(steps=[step]*600))
    result=summarize(root,old)
    v=result['comparisons']['selected_validation']['macro_dice']
    assert v['delta']==pytest.approx(.02)
    assert v['interval95']==pytest.approx([.02,.02])
    assert v['seed_deltas']==pytest.approx([.01,.02,.03])
    assert all(r['fraction_updates_projected']==(r['arm']=='pcgrad') for r in result['update_summaries'])
    assert all(c['sum_control_vs_previous_max_absolute_dice_drift']==0 for c in result['learning_curves'])
