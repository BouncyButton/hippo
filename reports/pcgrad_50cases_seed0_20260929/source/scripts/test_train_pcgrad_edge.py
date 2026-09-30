import json
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_train_separated_edge as small
import train_pcgrad_edge as experiment


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def setup(monkeypatch,method):
    small.setup(monkeypatch)
    for name in ('loaders','build_model','objectives','verify_payload'):
        monkeypatch.setattr(experiment,name,getattr(small.experiment,name))
    c=small.config()
    c['run'].update(gradient_method=method,projection_seed=20260930)
    return c


@pytest.mark.parametrize('method',['sum','pcgrad'])
def test_training_and_exact_midrun_resume(monkeypatch,tmp_path,method):
    config=setup(monkeypatch,method)
    full=tmp_path/'full';full.mkdir()
    resumed=tmp_path/'resumed';resumed.mkdir()
    experiment.fit(config,full,torch.device('cpu'))
    save=experiment.save
    def stop(path,value):
        if path.parent.name=='epochs' and path.name=='002.json':
            raise RuntimeError('interrupt before second checkpoint')
        save(path,value)
    monkeypatch.setattr(experiment,'save',stop)
    with pytest.raises(RuntimeError,match='interrupt'):
        experiment.fit(config,resumed,torch.device('cpu'))
    cp=torch.load(resumed/'checkpoint_latest.pt',weights_only=True)
    assert cp['epoch']==1 and 'projection_rng' in cp
    monkeypatch.setattr(experiment,'save',save)
    experiment.fit(config,resumed,torch.device('cpu'))
    a=torch.load(full/'checkpoint_best.pt',weights_only=True)
    b=torch.load(resumed/'checkpoint_best.pt',weights_only=True)
    assert a['epoch']==b['epoch']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert json.loads((full/'updates/002.json').read_text())==json.loads((resumed/'updates/002.json').read_text())
    assert len(json.loads((full/'selected_cases.json').read_text()))==20
    reports=json.loads((full/'updates/002.json').read_text())['steps']
    assert len(reports)==10 and all(r['finite'] for r in reports)
    if method=='sum':assert all(r['projection_count']==0 for r in reports)
    else:assert any(r['projection_count']>0 for r in reports)
