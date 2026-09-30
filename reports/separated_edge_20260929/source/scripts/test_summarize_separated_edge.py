import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from summarize_separated_edge import summarize,KEYS


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_seed_averaging_and_common_epoch_original_comparison(tmp_path):
    original=[]
    for seed in range(3):
        root=tmp_path/f'fold0/seed{seed}'
        put(root/'completion.json',dict(status='complete'))
        historic={m:dict(rows=[dict(epoch=60,val_dice_hard=.7),dict(epoch=75,val_dice_hard=.71)]) for m in ('dice','bands','edge')}
        put(root/'historical_curves.json',historic)
        for method in ('dice','bands','edge'):
            for split,n in [('train',10),('validation',52)]:
                original += [dict(method=method,split=split,seed=seed,case_name=f'{split}{i}',metrics={k:.1 for k in KEYS}) for i in range(n)]
        for arm in ('pooled','separated'):
            difference=(seed+1)*.01 if arm=='separated' else 0
            rows=[dict(split=split,case_name=f'{split}{i}',metrics={k:.1+difference for k in KEYS})
                  for split,n in [('train',10),('validation',52)] for i in range(n)]
            put(root/arm/'selected_cases.json',rows)
            put(root/arm/'coherence/epoch_060.json',[r for r in rows if r['split']=='validation'])
            put(root/arm/'completion.json',dict(stopped_epoch=75))
            for e in (60,75):
                put(root/arm/f'epochs/{e:03d}.json',dict(epoch=e,val_dice_hard=.7+difference))
    put(tmp_path/'original_coherence.json',dict(rows=original,provenance={}))
    result=summarize(tmp_path)
    r=result['comparisons']['selected_validation']['inner_disagree']
    assert r['delta']==pytest.approx(.02)
    assert r['interval95']==pytest.approx([.02,.02])
    assert r['seed_deltas']==pytest.approx([.01,.02,.03])
    assert result['learning_curves'][0]['common_last_epoch']==75
    assert result['historical_validation']['edge']['all_fn']==pytest.approx(.1)
    assert (tmp_path/'REPORT.md').exists()
