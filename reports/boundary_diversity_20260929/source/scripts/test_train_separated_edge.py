import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import pytest
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_separated_edge as experiment


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class TinyData(Dataset):
    def __init__(self):
        self.data = [{'case_name':f'case_{i:02d}'} for i in range(10)]
    def __len__(self):
        return 10
    def __getitem__(self,i):
        y=torch.zeros(1,9,9,9,dtype=torch.long)
        y[:,2:7,2:7,2:7]=1
        y[:,5:7,2:7,2:7]=2
        return dict(image=y.float()/2 + i*.01,label=y,case_name=self.data[i]['case_name'])


class TinyBands:
    config = SimpleNamespace(equivariance_weight=0., bands_weight=.1)
    def __call__(self,model,images,logits,labels):
        p=logits.float().softmax(1)[:,1:].sum(1,keepdim=True)
        return {'loss':.1*(p-(labels>0).float()).square().mean()}


def config():
    return {'run':dict(seed=0,fold=0,epochs=2,drop_rate=0.,activation_checkpointing=False,
        spatial_size=[9,9,9],optimizer_mode='adamw_0.01',adamw_gamma=.5,step_size=20,
        learning_rate=1e-4,weight_decay=1e-5,amp=False,early_stopping_patience=8,
        early_stopping_min_delta=.0005,early_stopping_min_epochs=2,
        constraint_warmup_epochs=5,edge_arm='separated',edge_weight=.1)}


def setup(monkeypatch):
    from monai.losses import DiceLoss
    monkeypatch.setattr(experiment,'objectives',lambda c:(DiceLoss(to_onehot_y=True,softmax=True),TinyBands()))
    monkeypatch.setattr(experiment,'verify_payload',lambda:None)
    monkeypatch.setattr(experiment,'build_model',lambda c,d:torch.nn.Conv3d(1,3,1).to(d))
    def build(c):
        data=TinyData()
        dg=torch.Generator().manual_seed(c['run']['seed'])
        ag=torch.Generator().manual_seed(c['run']['seed']+1)
        return DataLoader(data,batch_size=1,shuffle=True,generator=dg),DataLoader(data,batch_size=1),dg,ag
    monkeypatch.setattr(experiment,'loaders',build)


def test_monitor_has_no_rng_parameter_or_gradient_side_effects(monkeypatch,tmp_path):
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    setup(monkeypatch)
    c=config();train,_,dg,ag=experiment.loaders(c)
    model=experiment.build_model(c,torch.device('cpu'))
    model.train()
    before=copy.deepcopy(model.state_dict())
    state=trainer.rng_state(dg,ag)
    experiment.monitor(model,train.dataset,c,5,torch.device('cpu'),dg,ag,tmp_path)
    after=trainer.rng_state(dg,ag)
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    for k,v in before.items():
        torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    for k in ('torch','data_generator','translation_generator'):
        assert torch.equal(state[k],after[k])
    assert state['python']==after['python']
    assert torch.equal(state['numpy']['keys'],after['numpy']['keys'])
    rows=json.loads((tmp_path/'gradients/epoch_005.json').read_text())['cases']
    assert [r['case_name'] for r in rows]==['case_00','case_01']
    assert all(r['parameters']['norms']['edge']>0 for r in rows)


def test_two_epoch_training_checkpoint_audit_and_resume(monkeypatch,tmp_path):
    setup(monkeypatch)
    c=config()
    original_save=experiment.save
    def interrupt_before_final_audit(path,value):
        if path.name=='selected_cases.json':
            raise RuntimeError('Simulated interruption after final training checkpoint')
        original_save(path,value)
    monkeypatch.setattr(experiment,'save',interrupt_before_final_audit)
    import pytest
    with pytest.raises(RuntimeError,match='Simulated interruption'):
        experiment.fit(c,tmp_path,torch.device('cpu'))
    assert (tmp_path/'checkpoint_latest.pt').exists()
    cp=torch.load(tmp_path/'checkpoint_latest.pt',weights_only=True)
    assert cp['epoch']==2 and {'optimizer','scheduler','scaler','rng','stopping_state'}<=cp.keys()
    monkeypatch.setattr(experiment,'save',original_save)
    experiment.fit(c,tmp_path,torch.device('cpu'))
    done=json.loads((tmp_path/'completion.json').read_text())
    assert done['status']=='complete' and done['stopped_epoch']==2
    assert (tmp_path/'checkpoint_best.pt').exists() and not (tmp_path/'checkpoint_latest.pt').exists()
    rows=json.loads((tmp_path/'selected_cases.json').read_text())
    assert len(rows)==20
    assert (tmp_path/'gradients/epoch_002.json').exists()
