"""Synthetic native all-constraint training: interruption/resume versus uninterrupted."""
import argparse
from types import SimpleNamespace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from campaign90 import BUDGET, MODELS, ROOT, digest, write_json
from compute_constraint_objectives import scheduled_costs
import train_campaign90_core as core
from train_campaign90 import configure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    work=args.work
    if work.exists():raise ValueError('Synthetic smoke requires a fresh scratch directory')
    work.mkdir(parents=True)
    protocol_dir=work/'protocol'
    protocol_dir.mkdir()
    origin=ROOT/'experiments/nnunet_pair95_20261005/baseline'
    protocol=json.loads((origin/'protocol.json').read_text())
    shutil.copyfile(origin/'nnunet_architecture.json',protocol_dir/'nnunet_architecture.json')
    names=[f'synthetic_{i}' for i in range(11)]
    split=[{'train':names[:10],'val':names[10:]}]
    write_json(protocol_dir/'splits_10patients.json',split)
    rng=np.random.default_rng(7)
    target=np.zeros((64,64,64),dtype=np.uint8)
    target[20:32,22:42,25:39]=1
    target[32:44,22:42,25:39]=2
    frame=pd.DataFrame([{'subject_id':name,'image_data':rng.normal(size=(64,64,64)).astype(np.float32),
                         'label_data':target.copy()} for name in names])
    pkl=work/'synthetic.pkl'
    frame.to_pickle(pkl,compression='gzip')
    costs=json.loads((ROOT/'reports/four_models_six_31976_20261006/profiles/nnunet.json').read_text())['costs']
    arm='dice_bands_edge_sdf'
    s=scheduled_costs(BUDGET,MODELS['nnunet']['baseline_flops'],costs[arm]['update_flops'],costs[arm]['calibration_flops'],1)
    s.update(flops_per_update=costs[arm]['update_flops'],baseline_flops_per_update=MODELS['nnunet']['baseline_flops'])
    protocol.update(model_identity='nnunet',trainer_alias='nnunet',lr_horizon_updates=174,budget_flops=BUDGET,
                    models={'nnunet':{'parameters':MODELS['nnunet']['parameters'],'objectives':{arm:s}}},
                    splits_sha256=digest(protocol_dir/'splits_10patients.json'),pkl_sha256=digest(pkl),
                    folds=[0],seeds=[0],constraint_warmup_updates=1,validation_milestones=2,
                    validation_target_flops=[MODELS['nnunet']['baseline_flops'],2*MODELS['nnunet']['baseline_flops']],
                    learning_rate_policy='Synthetic smoke; fixed174 horizon, one-update prefix/ramp to exercise all code paths.')
    write_json(protocol_dir/'protocol.json',protocol)
    configure(protocol,arm)
    resume=work/'interrupted'
    uninterrupted=work/'uninterrupted'
    save=core.base.save_checkpoint
    class SyntheticInterruption(Exception):pass
    def stop_after_committed_checkpoint(path,payload):
        save(path,payload)
        if path.name=='checkpoint_latest.pt' and payload['step']==1:
            raise SyntheticInterruption('Simulated interruption after durable update1 checkpoint')
    def arguments(output,resume_flag):
        return SimpleNamespace(output=output,protocol_dir=protocol_dir,model='nnunet',objective=arm,
                               fold=0,seed=0,pkl=pkl,smoke_steps=2,device='cpu',threads=2,resume=resume_flag)
    core.base.save_checkpoint=stop_after_committed_checkpoint
    try:
        core.run(arguments(resume,False))
        raise ValueError('Did not exercise interruption')
    except SyntheticInterruption:
        pass
    finally:
        core.base.save_checkpoint=save
    core.run(arguments(resume,True))
    core.run(arguments(uninterrupted,False))
    a=torch.load(resume/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(uninterrupted/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    if a['step']!=2 or b['step']!=2 or a['history']!=b['history']:
        raise ValueError('Resumed validation trajectory differs')
    if any(not torch.equal(a['model'][key],b['model'][key]) for key in a['model']):
        raise ValueError('Resumed model parameters differ')
    for filename in ['budget_ledger.json','calibration.json']:
        if json.loads((resume/filename).read_text())!=json.loads((uninterrupted/filename).read_text()):
            raise ValueError('Resumed calibration or ledger differs')
    for output in [resume,uninterrupted]:
        done=json.loads((output/'completion.json').read_text())
        if done['status']!='smoke_complete' or done['updates']!=2 or done['calibration_attempts']!=1:
            raise ValueError('Incomplete synthetic smoke')
        for key,filename in core.ARTIFACTS.items():
            if digest(output/filename)!=done[key]:raise ValueError('Smoke artifact hash differs')
    write_json(args.output,{'status':'passed','model':'nnunet','objective':arm,
               'synthetic_only':True,'updates':2,'calibration_cases':10,
               'interruption_after_update1_exercised':True,'resumed_model_exactly_matches_uninterrupted':True,
               'resumed_validation_calibration_and_ledger_identical':True,
               'torch':torch.__version__,'monai':core.monai.__version__,
               'source_sha256':{name:digest(ROOT/name) for name in ['scripts/smoke_campaign90_resume.py',
                                'scripts/train_campaign90_core.py','scripts/train_campaign90.py',
                                'scripts/nnunet_multi_objective.py','scripts/campaign90.py']}})
    print('Synthetic all-constraint interruption/resume smoke passed',flush=True)


if __name__=='__main__':
    main()
