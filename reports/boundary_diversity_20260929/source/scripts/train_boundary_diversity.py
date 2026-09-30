"""Intervene on unique case count with fixed update and actual LR budgets."""
from pathlib import Path
import argparse
import copy
import json
import os
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import train_boundary_causal as causal
from train_edge_lowdata import data_arguments
from probe_edge_consistency import digest,save


class ReplayLR:
    def __init__(self,optimizer,values):
        self.optimizer,self.values,self.cursor=optimizer,list(values),0
        self.set_lr()
    def set_lr(self):
        for group in self.optimizer.param_groups:group['lr']=self.values[min(self.cursor,len(self.values)-1)]
    def step(self,metric):
        self.cursor+=1
        self.set_lr()
    def state_dict(self):return dict(values=self.values,cursor=self.cursor)
    def load_state_dict(self,state):
        assert state['values']==self.values
        self.cursor=state['cursor'];self.set_lr()


def repeated_loaders(config):
    from torch.utils.data import DataLoader,ConcatDataset
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    dg,ag,_=trainer.build_experiment_generators(config['run']['seed'])
    old,val,_,nt,nv=trainer.build_data(data_arguments(config),dg)
    assert (nt,nv)==(10,52)
    assert {r['case_name'] for r in old.dataset.data}==set(config['cohort']['original_train_10'])
    assert {r['case_name'] for r in val.dataset.data}==set(config['cohort']['validation'])
    train=DataLoader(ConcatDataset([old.dataset]*5),batch_size=1,shuffle=True,num_workers=0,generator=dg)
    return train,val,dg,ag


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--base',type=Path,default=Path('/mnt/beegfsstudents/home/3160552'))
    args=p.parse_args()
    causal.verify_payload()
    ref=json.loads((args.reference/'config.json').read_text())
    done=json.loads((args.reference/'completion.json').read_text())
    assert done['status']=='complete' and done['run']==ref['run']
    assert digest(args.reference/'checkpoint_best.pt')==done['checkpoint_sha256']
    assert ref['run']['causal_arm']=='sum' and ref['run']['seed']==0
    old=args.base/'pcgrad_edge_20260929_01/fold0/seed0/sum/config.json'
    old_config=json.loads(old.read_text())
    config=copy.deepcopy(ref)
    for k in ('splits_json','splits_json_sha256'):config[k]=old_config[k]
    for k in ('pkl','splits_json'):assert digest(config[k])==config[k+'_sha256']
    config['cohort']['train']=config['cohort']['original_train_10']
    curve=[json.loads(p.read_text()) for p in sorted((args.reference/'epochs').glob('*.json'))]
    assert [r['epoch'] for r in curve]==list(range(1,done['stopped_epoch']+1))
    lrs=[r['learning_rate'] for r in curve]
    config['run'].update(training_examples=10,updates_per_pass=50,repeats_per_case=5,
        epochs=len(curve),causal_arm='sum10_repeated',splits_json_sha256=config['splits_json_sha256'],
        constraint_set='causal_sum10_repeated',lr_policy=dict(name='replay_50case_sum',values=lrs,
            reference=str(args.reference),reference_completion_sha256=digest(args.reference/'completion.json')))
    config.update(train_samples=10,experiment_payload_sha256=digest(REPO/'PAYLOAD.json'),
        training_script_sha256=digest(__file__),actual_core_driver_sha256=digest(REPO/'scripts/train_boundary_causal.py'))
    import torch
    from torch.utils.data import DataLoader
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from run_degree_bands_ab import check_quota
    torch.set_num_threads(4)
    device=torch.device('cuda')
    assert torch.cuda.is_available()
    assert trainer.collect_runtime_provenance()==ref['runtime_provenance']
    execution=trainer.collect_execution_provenance(device)
    trainer.validate_calibration_execution(ref['execution_provenance'],execution,allow_mig_profile_change=True)
    config['execution_provenance']=execution
    output=args.root/'seed0/sum10_repeated';output.mkdir(parents=True,exist_ok=True)
    check_quota(output,'start',1.1)
    save(output/'config.json',config)
    original_audit=causal.audit
    def unique_audit(model,cohorts,*a,**kw):
        cohorts=dict(cohorts)
        cohorts['train']=DataLoader(cohorts['train'].dataset.datasets[0],batch_size=1,shuffle=False)
        return original_audit(model,cohorts,*a,**kw)
    causal.audit=unique_audit
    causal.loaders=repeated_loaders
    causal.make_scheduler=lambda opt,run:ReplayLR(opt,run['lr_policy']['values'])
    causal.fit(config,output,device)
    save(args.root/'completion.json',dict(status='complete',job_id=os.getenv('SLURM_JOB_ID'),
        original_train_cases=10,updates_per_pass=50,reference=str(args.reference)))


if __name__=='__main__':main()
