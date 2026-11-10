"""CPU checks for the rescue loss and the actual staged trainer continuation."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'reports/geofwi3d_unetr_ce2_bands_20261019/bundle'
sys.path.insert(0,str(BUNDLE))
import storage_runtime as io
io.activate(BUNDLE)
import torch
import rescue_objective as objective
import ce2_runtime as runtime
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
        expected+=2*torch.nn.functional.cross_entropy(logits,y[:,0])
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual,logits,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,logits)[0]
        torch.testing.assert_close(ga,ge)
        assert torch.isfinite(ga).all()
    cost,half,full=runtime.settings(BUNDLE)
    assert (half,full)==(207,414)
    assert runtime.settings(BUNDLE,'ce2_bands')[2]==407
    for arm in runtime.ARMS:
        sch=runtime.schedule(BUNDLE,arm);n=sch['max_steps']
        assert runtime.total_cost(n,sch)<=runtime.CAP<runtime.total_cost(n+1,sch)
    rows=[dict(step=(runtime.CAP*i//10)//cost,macro_hard_dice=v) for i,v in [(1,.99),(3,.2),(4,.3),(5,.4)]]
    assert abs(runtime.screening_score(rows,[165,207])[0]-.35)<1e-10
    assert runtime.gate(dict(arm='ce2',selection_score=.459999))['decision']=='prune'
    assert runtime.gate(dict(arm='ce2',selection_score=.46))['decision']=='continue'
    assert not runtime.qualifies(dict(final_dice=.5,best_dice=.99))
    assert not runtime.qualifies(dict(final_dice=.4482058253085583,best_dice=.4507538557514277))
    assert runtime.qualifies(dict(final_dice=.500001))
    # Verify calibration against the actual CE2 gradient, and the added BANDS term.
    z=torch.randn(1,2,12,12,12,requires_grad=True);label=torch.zeros(1,1,12,12,12,dtype=torch.long)
    label[:,:,3:9,3:9,3:9]=1
    row=objective.calibration_case(z,label,'dice_bands')
    reference=objective.training_loss(z,label,'dice',{},0.)
    grad=torch.autograd.grad(reference,z,retain_graph=True)[0]
    assert abs(row['dice_rms']-float(grad.square().mean().sqrt()))<1e-10
    band=objective.bands()(z,label).loss
    actual=objective.training_loss(z,label,'dice_bands',{'bands':.02},.4)
    torch.testing.assert_close(actual,reference+.008*band)
    torch.testing.assert_close(torch.autograd.grad(actual,z,retain_graph=True)[0],torch.autograd.grad(reference+.008*band,z)[0])
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
    # Exercise actual orchestrator branching, strict final-only trigger and idempotence.
    from unittest.mock import patch
    from contextlib import ExitStack
    scenarios=[(.459,.9,[('ce2',207)]),(.46,.5,[('ce2',207),('ce2',414)]),
        (.47,.51,[('ce2',207),('ce2',414),('ce2_bands',207),('ce2_bands',414)])]
    for score,final_score,expected in scenarios:
        with tempfile.TemporaryDirectory() as work, ExitStack() as stack:
            root=Path(work)/'bundle';durable=Path(work)/'results';root.mkdir();durable.mkdir()
            io.write(durable/'ce2_preflight.json',dict(status='passed'))
            state=dict(screens={},finals={},calls=[],terminal=False)
            def execute(root,durable,arm,limit,deadline):
                state['calls'].append((arm,limit))
                state['screens'][arm]=dict(arm=arm,selection_score=score)
                if limit==414:
                    state['finals'][arm]=dict(final_dice=final_score,best_dice=.99)
                    if arm=='ce2_bands' or final_score<=.5:state['terminal']=True
                return True
            def prune(*args):state['terminal']=True
            bindings=[(io,'binding',lambda *a:{}),(runtime,'settings',lambda *a:(cost,207,414)),
                (runtime,'publish',lambda *a,**k:dict(status='complete' if state['terminal'] else 'in_progress')),
                (runtime,'read_screen',lambda r,d,a:state['screens'].get(a)),
                (runtime,'execute',execute),(runtime,'prune',prune),(runtime,'preflight',lambda *a:None),
                (runtime,'final_result',lambda r,d,a:state['finals'].get(a)),
                (io,'completed',lambda d,t:t['protocol'] in state['finals'])]
            for target,name,fn in bindings:stack.enter_context(patch.object(target,name,side_effect=fn))
            runtime.run(root,durable,1);runtime.run(root,durable,1)
            assert state['calls']==expected,state

    for arm in ('dice','dice_bands'):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);folder=tmp/'protocol';folder.mkdir()
            p=io.read(BUNDLE/'templates/unetr_vit_b16/protocol.json')
            exp.configure(p,arm)
            exp.trainer.REPO=BUNDLE/'source'
            tiny=lambda:torch.nn.Sequential(torch.nn.Conv3d(1,4,1),torch.nn.Dropout(.1),torch.nn.Conv3d(4,2,1))
            exp.base.build_model=lambda *args:tiny()
            exp.base.build_loss_and_optimizer=lambda alias,model,protocol:(None,torch.optim.AdamW(model.parameters(),lr=.01))
            exp.trainer.training_loss=objective.training_loss
            exp.trainer.calibration_case=objective.calibration_case
            names=['case'+str(i) for i in range(20)]
            class Data:
                def __init__(self,df,names):self.names=names
                def __len__(self):return len(self.names)
                def __getitem__(self,i):
                    g=torch.Generator().manual_seed(int(self.names[i][4:]))
                    return dict(case_name=self.names[i],image=torch.rand(1,12,12,12,generator=g),label=label[0].clone())
            exp.base.PreparedPatients=Data;exp.base._load_pkl_dataframe=lambda *a:None
            def evaluate(model,data,device,amp):
                model.eval();torch.rand(5) # Deliberately consume RNG to test screening isolation.
                with torch.no_grad():score=float(model(data[0]['image'][None]).softmax(1)[:,1].mean())
                return dict(summary=dict(macro_hard_dice=score),cases=[])
            exp.base.evaluate=evaluate
            p.update(budget_flops=45 if arm=='dice_bands' else 40,rescue_validation_steps=[1,2,3,4],native_geometry={n:dict(model_affine=torch.eye(4).tolist()) for n in names})
            p['models']['unet']['parameters']=sum(v.numel() for v in tiny().parameters())
            sch=p['models']['unet']['objectives'][arm]
            for key in ('baseline_flops_per_update','early_flops_per_update','flops_per_update'):sch[key]=10
            sch.update(max_steps=4,prefix_updates=1 if arm=='dice_bands' else 0,sdf_prefix_updates=1 if arm=='dice_bands' else 0,calibration_flops=5 if arm=='dice_bands' else 0)
            io.write(folder/'splits_10patients.json',[dict(train=names[:19],val=names[19:])])
            io.write(folder/'nnunet_architecture.json',{})
            io.write(folder/'data.json',{})
            p.update(splits_sha256=io.digest(folder/'splits_10patients.json'),nnunet_architecture_sha256=io.digest(folder/'nnunet_architecture.json'),pkl_sha256=io.digest(folder/'data.json'))
            io.write(folder/'protocol.json',p)
            args=argparse.Namespace(model='unet',objective=arm,fold=0,seed=0,pkl=folder/'data.json',output=tmp/'full',protocol_dir=folder,device='cpu',threads=2,resume=True,smoke_steps=0)
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
            done=io.read(args.output/'completion.json')
            assert done['consumed_total_flops']==p['budget_flops']
            if arm=='dice_bands':
                report=io.read(args.output/'calibration.json')
                assert report['anchor_successful_updates']==1 and len(report['cases'])==19
                assert set(report['weights'])=={'bands'} and done['calibration_attempts']==1
            # Completed reruns cannot perform another optimizer step or calibration.
            ledger=io.read(args.output/'budget_ledger.json');exp.trainer.run(args)
            assert io.read(args.output/'budget_ledger.json')==ledger
    # Real archive/verification/retirement path, including authorized augmentation.
    from scripts.cross_dataset_campaign import storage
    for arm in runtime.ARMS:
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as directory:
            temp=Path(directory);out=temp/'output';durable=temp/'durable';out.mkdir();durable.mkdir()
            task=runtime.task(arm)
            io.write(durable/'environment_binding.json',dict(bundle_sha256=io.digest(BUNDLE/'bundle_manifest.json')))
            config=dict(augmentation='xy_d4_v1',budget_flops=100,train_cases=['train'],validation_cases=['val'])
            io.write(out/'config.json',config)
            for name in ['checkpoint_best.pt','checkpoint_latest.pt']:(out/name).write_bytes(b'test fixture only')
            done=dict(status='complete',consumed_training_flops=90,consumed_total_flops=95,discarded_attempts=0,
                checkpoint_best_sha256=io.digest(out/'checkpoint_best.pt'),checkpoint_latest_sha256=io.digest(out/'checkpoint_latest.pt'),selected_step=1,updates=2)
            io.write(out/'completion.json',done)
            case=dict(case_name='val',confusion=[[10,1],[2,3]],foreground_fn=2)
            for name in ['best','latest']:io.write(out/f'validation_{name}.json',dict(cases=[case]))
            io.write(out/'training_final.json',dict(cases=[dict(case,case_name='train')]))
            io.archive_completed(BUNDLE,out,durable,temp/'recovery',task)
            assert io.completed(durable,task)
            assert not (out/'checkpoint_latest.pt').exists()
            archived=durable/'runs'/task['id']/'completion.json'
            archived.write_text('{}')
            try:io.completed(durable,task)
            except ValueError:pass
            else:raise AssertionError('Corrupted archive accepted')
    receipt=dict(status='passed',loss_values_and_gradients=True,calibration_uses_ce2=True,
        both_actual_trainers_resume_exactly=True,calibration_and_training_cap=True,
        same_46pct_gate=True,strict_final_trigger_and_idempotence=True,
        xy_geometry_preserved=True,archive_retirement_end_to_end=True)
    io.write(BUNDLE.parent/'local_validation.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
