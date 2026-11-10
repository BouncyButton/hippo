"""Synthetic selection-state tests; these are not experiment results."""
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.cross_dataset_campaign.common import write
from scripts.mni_constraints.unetr_delayed_combinations import ARMS,PAIRS,PILOTS,tasks,selection,eligible


def checks():
    with TemporaryDirectory(prefix='unetr-selection-') as directory:
        stage=Path(directory);plan={'tasks':tasks()}
        write(stage/'controls.json',[dict(seed=s,fold=f,validation=.60) for s,f in PAIRS])

        def finish(task,gain):
            folder=stage/'runs'/task['id']
            write(folder/'validation_latest.json',{'summary':{'macro_hard_dice':.60+gain/100}})
            write(folder/'cleanup.json',{'status':'synthetic_test_only'})

        pilots=[t for t in tasks() if t['pilot']]
        pilot_gain={ARMS[0]:[.4,.6,.5,.7],ARMS[1]:[.2,2.,2.,2.],ARMS[2]:[-.1,3.,3.,3.]}
        for task in pilots[:-1]:
            index=PILOTS.index((task['seed'],task['fold']))
            finish(task,pilot_gain[task['arm']][index])
        state=selection(stage,plan)
        assert not state['complete'] and eligible(stage,plan,state)==[pilots[-1]]
        finish(pilots[-1],3.)
        state=selection(stage,plan)
        assert state['pilot_ranking']==list(ARMS[:2]), 'Worst-pair ranking must outrank a larger mean'
        assert state['selected']['arm']==ARMS[0] and not state['complete']
        pending=eligible(stage,plan,state)
        assert len(pending)==5 and all(t['arm']==ARMS[0] and not t['pilot'] for t in pending)
        for i,task in enumerate(pending):finish(task,-.05 if i==0 else .3)
        state=selection(stage,plan)
        assert not state['complete'] and state['selected']['arm']==ARMS[1]
        assert state['selected']['winner'] is None, 'One negative pair must prevent success'
        for task in eligible(stage,plan,state):finish(task,.1)
        state=selection(stage,plan)
        assert state['complete'] and state['outcome']=='nine_pair_success'
        assert state['selected']['winner']==ARMS[1] and not eligible(stage,plan,state)
    with TemporaryDirectory(prefix='unetr-exhausted-') as directory:
        stage=Path(directory);write(stage/'controls.json',[dict(seed=s,fold=f,validation=.6) for s,f in PAIRS])
        for task in tasks():
            if not task['pilot']:continue
            folder=stage/'runs'/task['id']
            write(folder/'validation_latest.json',{'summary':{'macro_hard_dice':.6}})
            write(folder/'cleanup.json',{'status':'synthetic_test_only'})
        state=selection(stage,{'tasks':tasks()})
        assert state['complete'] and state['outcome']=='exhausted' and state['selected']['winner'] is None
        assert not eligible(stage,{'tasks':tasks()},state)
    return dict(status='passed',scope='Synthetic state-machine tests only; no training or validation results.',
                checks=['finish all pilots before confirmation','negative pilot disqualifies candidate',
                        'rank by worst pair before mean','confirm five remaining pairs',
                        'negative confirmation prevents success and advances candidate',
                        'require all nine positive','zero-gain pilot exhaustion has no winner'])


if __name__=='__main__':
    import json
    print(json.dumps(checks()))
