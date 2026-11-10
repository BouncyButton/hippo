"""CPU checks for the rescue loss and the actual staged trainer continuation."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'reports/geofwi3d_unetr_xy_aug_20261019/bundle'
sys.path.insert(0,str(BUNDLE))
import storage_runtime as io
io.activate(BUNDLE)
import torch
import rescue_objective as objective
import augmentation_runtime as runtime
from xy_augmentation import apply_transform, augment, transform_id
from scripts.cross_dataset_campaign import conventional as exp


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(2)
    for label_kind in ('mixed','empty','full'):
        torch.manual_seed(7)
        logits=torch.randn(1,2,4,4,4,requires_grad=True)
        y=(torch.rand(1,1,4,4,4)>.7).long()
        if label_kind=='empty':y.zero_()
        if label_kind=='full':y.fill_(1)
        actual=objective.training_loss(logits,y,'dice',{},.2)
        prob=logits.softmax(1)[:,1];target=y[:,0].float()
        expected=1-2*(prob*target).sum()/(prob.square().sum()+target.square().sum()+1e-6)
        expected+=torch.nn.functional.cross_entropy(logits,y[:,0])
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,logits,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,logits)[0]
        torch.testing.assert_close(ga,ge)
        assert torch.isfinite(ga).all()
    cost,half,full=runtime.settings(BUNDLE)
    assert (half,full)==(207,414) and full*cost<=runtime.TOTAL_CAP
    rows=[dict(step=(runtime.CAP*i//10)//cost,macro_hard_dice=v) for i,v in [(1,.99),(3,.2),(4,.3),(5,.4)]]
    assert abs(runtime.screening_score(rows,cost)[0]-.35)<1e-10
    assert runtime.gate(dict(selection_score=.399999))['decision']=='prune'
    assert runtime.gate(dict(selection_score=.4))['decision']=='continue'
    x=torch.arange(96**3).reshape(1,1,96,96,96)
    y=(x%7==0).long();rng=torch.get_rng_state();transforms=[]
    for i in range(8):
        a,b=apply_transform(x,y,i)
        assert a.shape==x.shape and a.is_contiguous()
        assert torch.equal((a%7==0).long(),b)
        assert torch.equal(b.sum((2,3)),y.sum((2,3))), 'Depth profile changed'
        assert torch.equal(torch.sort(a.flatten()).values,x.flatten()), 'Voxels changed'
        transforms.append(a)
    assert len({v.numpy().tobytes() for v in transforms})==8
    assert torch.equal(rng,torch.get_rng_state()), 'Global RNG changed'
    assert equal(augment(x,y,0,200),augment(x,y,0,200))
    assert len({transform_id(0,i) for i in range(414)})==8
    # Exercise the orchestrator decision paths without GPU or training spend.
    from unittest.mock import patch
    for score,expected_limits in [(.39,[207]),(.4,[207,414])]:
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)/'bundle';durable=Path(work)/'results';root.mkdir();durable.mkdir()
            io.write(durable/'preflight.json',dict(status='passed'))
            state={'screen':None,'status':'in_progress','calls':[]}
            def execute(root,durable,arm,limit,deadline):
                state['calls'].append(limit)
                state['screen']=dict(selection_score=score)
                if limit==414:state['status']='complete'
                return True
            def prune(*args):state['status']='pruned'
            with patch.object(io,'binding',return_value={}), patch.object(runtime,'settings',return_value=(cost,207,414)), patch.object(runtime,'publish',side_effect=lambda *a,**k:dict(status=state['status'])), patch.object(runtime,'read_screen',side_effect=lambda *a:state['screen']), patch.object(runtime,'execute',side_effect=execute), patch.object(runtime,'prune',side_effect=prune), patch.object(io,'completed',return_value=False):
                runtime.run(root,durable,1);runtime.run(root,durable,1)
            assert state['calls']==expected_limits, state

    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);folder=tmp/'protocol';folder.mkdir()
        p=io.read(BUNDLE/'templates/unetr_vit_b16/protocol.json')
        exp.configure(p,'dice')
        exp.trainer.REPO=BUNDLE/'source'
        tiny=lambda:torch.nn.Sequential(torch.nn.Conv3d(1,4,1),torch.nn.Dropout(.1),torch.nn.Conv3d(4,2,1))
        exp.base.build_model=lambda *args:tiny()
        exp.base.build_loss_and_optimizer=lambda alias,model,protocol:(None,torch.optim.AdamW(model.parameters(),lr=.01))
        exp.trainer.training_loss=objective.training_loss
        names=['case'+str(i) for i in range(20)]
        class Data:
            def __init__(self,df,names):self.names=names
            def __len__(self):return len(self.names)
            def __getitem__(self,i):
                g=torch.Generator().manual_seed(int(self.names[i][4:]))
                return dict(case_name=self.names[i],image=torch.rand(1,4,4,4,generator=g),label=(torch.rand(1,4,4,4,generator=g)>.5).long())
        exp.base.PreparedPatients=Data;exp.base._load_pkl_dataframe=lambda *a:None
        def evaluate(model,data,device,amp):
            model.eval();torch.rand(5) # Deliberately consume RNG to test screening isolation.
            with torch.no_grad():score=float(model(data[0]['image'][None]).softmax(1)[:,1].mean())
            return dict(summary=dict(macro_hard_dice=score),cases=[])
        exp.base.evaluate=evaluate
        p.update(budget_flops=40,rescue_validation_steps=[1,2,3,4],native_geometry={n:dict(model_affine=torch.eye(4).tolist()) for n in names})
        p['models']['unet']['parameters']=sum(v.numel() for v in tiny().parameters())
        sch=p['models']['unet']['objectives']['dice']
        for key in ('baseline_flops_per_update','early_flops_per_update','flops_per_update'):sch[key]=10
        sch['max_steps']=4
        io.write(folder/'splits_10patients.json',[dict(train=names[:19],val=names[19:])])
        io.write(folder/'nnunet_architecture.json',{})
        io.write(folder/'data.json',{})
        p.update(splits_sha256=io.digest(folder/'splits_10patients.json'),nnunet_architecture_sha256=io.digest(folder/'nnunet_architecture.json'),pkl_sha256=io.digest(folder/'data.json'))
        io.write(folder/'protocol.json',p)
        args=argparse.Namespace(model='unet',objective='dice',fold=0,seed=0,pkl=folder/'data.json',output=tmp/'full',protocol_dir=folder,device='cpu',threads=2,resume=True,smoke_steps=0)
        os.environ['HIPPO_STAGE_LIMIT']='4';exp.trainer.run(args)
        args.output=tmp/'staged';os.environ['HIPPO_STAGE_LIMIT']='2'
        try:exp.trainer.run(args)
        except SystemExit as exc:assert exc.code==76
        else:raise AssertionError('Half-budget run did not stop')
        assert not (args.output/'completion.json').exists()
        assert io.read(args.output/'budget_ledger.json')['charged_attempts']==2
        os.environ['HIPPO_STAGE_LIMIT']='4';exp.trainer.run(args)
        a=torch.load(tmp/'full/checkpoint_latest.pt',weights_only=False)
        b=torch.load(tmp/'staged/checkpoint_latest.pt',weights_only=False)
        assert equal(a,b), 'Staged continuation differs from uninterrupted training'
        assert io.read(args.output/'completion.json')['consumed_training_flops']==40
    receipt=dict(status='passed',loss_value_and_gradient_cases=3,actual_trainer_stage_resume_exact=True,selection_and_budget=True,xy_symmetry_and_depth_preservation=True,gate_and_no_duplicate_execution=True)
    io.write(BUNDLE.parent/'local_validation.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':main()
