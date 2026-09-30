import json
from pathlib import Path
import sys
import pytest
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_train_separated_edge as small
import train_pcgrad_50cases as experiment


@pytest.fixture(autouse=True)
def one_cpu_thread():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def policy():
    return dict(name='ReduceLROnPlateau',monitor='val_dice_hard',factor=.5,
                patience=3,threshold=.0005,min_lr=1e-6)


def test_plateau_schedule_waits_four_checks_and_respects_floor():
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=1e-4)
    scheduler=experiment.make_scheduler(optimizer,dict(lr_policy=policy()))
    scheduler.step(.5)
    for _ in range(3):scheduler.step(.5)
    assert optimizer.param_groups[0]['lr']==1e-4
    scheduler.step(.5)
    assert optimizer.param_groups[0]['lr']==5e-5
    for _ in range(40):scheduler.step(.5)
    assert optimizer.param_groups[0]['lr']==1e-6


def test_fifty_case_training_and_exact_resume_across_lr_reduction(monkeypatch,tmp_path):
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    small.setup(monkeypatch)
    for name in ('build_model','objectives','verify_payload'):
        monkeypatch.setattr(experiment,name,getattr(small.experiment,name))
    class Fifty(small.TinyData):
        def __init__(self):self.data=[dict(case_name=f'case_{i:02d}') for i in range(50)]
        def __len__(self):return 50
    def loaders(c):
        dg=torch.Generator().manual_seed(0);ag=torch.Generator().manual_seed(1)
        return DataLoader(Fifty(),batch_size=1,shuffle=True,generator=dg),DataLoader(small.TinyData(),batch_size=1),dg,ag
    monkeypatch.setattr(experiment,'loaders',loaders)
    monkeypatch.setattr(trainer,'evaluate_validation_metrics',lambda *a,**kw:dict(val_dice_soft=.25,val_dice_hard=.25))
    c=small.config();c['run'].update(epochs=6,gradient_method='pcgrad',projection_seed=20260930,lr_policy=policy())
    c['cohort']=dict(original_train_10=[f'case_{i:02d}' for i in range(10)])
    full=tmp_path/'full';full.mkdir();resumed=tmp_path/'resumed';resumed.mkdir()
    experiment.fit(c,full,torch.device('cpu'))
    original=experiment.save
    def interrupt(path,value):
        if path.parent.name=='epochs' and path.name=='004.json':raise RuntimeError('interrupt')
        original(path,value)
    monkeypatch.setattr(experiment,'save',interrupt)
    with pytest.raises(RuntimeError,match='interrupt'):experiment.fit(c,resumed,torch.device('cpu'))
    assert torch.load(resumed/'checkpoint_latest.pt',weights_only=True)['epoch']==3
    monkeypatch.setattr(experiment,'save',original)
    experiment.fit(c,resumed,torch.device('cpu'))
    a=torch.load(full/'checkpoint_latest.pt',weights_only=True)
    b=torch.load(resumed/'checkpoint_latest.pt',weights_only=True)
    assert a['epoch']==b['epoch']==6
    assert a['optimizer']['param_groups'][0]['lr']==5e-5
    assert a['scheduler']==b['scheduler']
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert torch.equal(a['projection_rng'],b['projection_rng'])
    assert json.loads((full/'updates/006.json').read_text())==json.loads((resumed/'updates/006.json').read_text())
    assert len(json.loads((full/'updates/006.json').read_text())['steps'])==50
    audit=json.loads((full/'coherence/epoch_005.json').read_text())
    assert sum(r['split']=='train' for r in audit)==50
    assert sum(r['split']=='validation' for r in audit)==10
    assert len(json.loads((full/'selected_cases.json').read_text()))==60
