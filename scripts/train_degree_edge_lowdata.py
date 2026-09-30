#!/usr/bin/env python3
"""Add inner-normalized degree weighting to the completed low-data edge run."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'scripts'))
from probe_edge_consistency import digest, save


def weighted_config(reference):
    """Only the declared A weights change; preserve both scalar coefficients."""
    config = copy.deepcopy(reference)
    run = config['run']
    assert run['constraint_set'] == 'bands_edge'
    assert run['constraint_config'].get('bands_degree_alpha', 0) == 0
    assert run['epochs'] == 75 and run['training_augmentation'] == 'none'
    assert run['early_stopping_patience'] == 8 and run['early_stopping_min_epochs'] == 60
    run['constraint_set'] = 'degree_bands_edge_A'
    run['constraint_config'].update(bands_degree_alpha=2.0, bands_degree_normalization='inner')
    config['experiment_note'] = 'Fixed-coefficient ablation of A/inner degree weighting. Existing per-seed bands and edge coefficients unchanged; no recalibration.'
    config['checkpoint_storage'] = 'Best and latest model weights/metadata only, no optimizer state or duplicate export. Not resumable.'
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    parser.add_argument('--minimum-free-gib', type=float, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'run.lock').open('x') as f:f.write(str(os.getenv('SLURM_JOB_ID')))
    try:
        import torch
        from torch.utils.data import DataLoader
        from thesis.new_constraints import train_swinunetr_constraints as trainer
        from train_edge_lowdata import fit
        from train_edge_consistency_experiment import audit
        from audit_edge_generalization import compare_splits
        from edge_lowdata_metrics import compare_histories
        from run_degree_bands_ab import check_quota
        for name,sha in json.loads((REPO/'PAYLOAD.json').read_text())['files'].items():
            assert digest(REPO/name)==sha,name
        check_quota(args.root,'start',args.minimum_free_gib)
        assert torch.cuda.is_available()
        device=torch.device('cuda')
        runtime,execution=trainer.collect_runtime_provenance(),trainer.collect_execution_provenance(device)
        base=Path('/mnt/beegfsstudents/home/3160552')
        old=base/f'low_data_noaug_seed{args.seed}_20260917_01/runs'
        edge_root=base/f'edge_lowdata_20260924_01/seed{args.seed}'
        assert json.loads((edge_root/'completion.json').read_text())['status']=='complete'
        directories={'dice':old/f'baseline_5pct_seed{args.seed}_noaug',
                     'bands':old/f'bands_5pct_seed{args.seed}_noaug','edge':edge_root/'run'}
        configs,histories,bindings={},{},{}
        for name,directory in directories.items():
            config=json.loads((directory/'config.json').read_text())
            completed=json.loads((directory/'completion_manifest.json').read_text())
            assert completed['status']=='complete' and completed['run']==config['run']
            assert config['run']['seed']==args.seed and config['run']['fold']==0
            assert config['runtime_provenance']==runtime
            trainer.validate_calibration_execution(config['execution_provenance'],execution,allow_mig_profile_change=True)
            if name=='edge':assert config['execution_provenance']==execution
            assert config['source_provenance']['files']['baselines/swin_unetr/swin_unetr.py']==digest(REPO/'baselines/swin_unetr/swin_unetr.py')
            for key in ('pkl','splits_json'):assert digest(config[key])==config[key+'_sha256']
            export=directory/'MSD_fold0/model.pt'
            assert digest(export)==completed['artifacts']['MSD_fold0/model.pt']
            cp=torch.load(directory/'checkpoint_best.pt',map_location='cpu',weights_only=True)
            assert cp['epoch']==completed['selected_epoch'] and json.loads(json.dumps(cp['run']))==config['run']
            weights=torch.load(export,map_location='cpu',weights_only=True)
            assert weights.keys()==cp['model'].keys()
            assert all(torch.equal(v,weights[k]) for k,v in cp['model'].items())
            del cp,weights
            configs[name]=(directory,config)
            histories[name]=list(csv.DictReader((directory/'metrics.csv').open()))
            bindings[name]={'directory':str(directory),'config_sha256':digest(directory/'config.json'),
                'checkpoint_sha256':digest(directory/'checkpoint_best.pt'),'metrics_sha256':digest(directory/'metrics.csv')}
        reference=configs['edge'][1]
        for _,config in configs.values():
            for key in ('seed','fold','epochs','batch_size','spatial_size','resize','optimizer_mode','learning_rate',
                        'weight_decay','step_size','adamw_gamma','amp','initial_checkpoint','training_augmentation',
                        'supervised_loss','plain_tensors','drop_rate','activation_checkpointing',
                        'early_stopping_patience','early_stopping_min_delta','early_stopping_min_epochs','splits_json_sha256'):
                assert config['run'][key]==reference['run'][key],key
        assert reference['run']['splits_json_sha256']=='7b69e54dd720b38f58f11e73302899e626ab3985d5c15d5c72c5daeb420ce8be'
        assert digest(reference['run']['edge_calibration'])==reference['run']['edge_calibration_sha256']
        assert digest(reference['run']['bands_calibration'])==reference['run']['bands_calibration_sha256']
        config=weighted_config(reference)
        source=trainer.collect_source_provenance()
        config['run'].update(source_sha256=source['sha256'],runtime_sha256=trainer.canonical_sha256(runtime),
                             execution_sha256=trainer.canonical_sha256(execution))
        config.update(source_provenance=source,runtime_provenance=runtime,execution_provenance=execution,
                      training_script_sha256=digest(__file__),payload_sha256=digest(REPO/'PAYLOAD.json'))
        output=args.root/'run';output.mkdir(exist_ok=False)
        save(output/'config.json',config);save(args.root/'reference_bindings.json',bindings)
        print(json.dumps({'starting_seed':args.seed,'alpha':2,'normalization':'inner',
            'fixed_bands_weight':config['run']['constraint_config']['bands_weight'],
            'fixed_edge_weight':config['run']['edge_weight']}),flush=True)
        start=time.perf_counter()
        train,val=fit(config,output,device,source,compact_checkpoints=True)
        elapsed=time.perf_counter()-start
        histories['weighted']=list(csv.DictReader((output/'metrics.csv').open()))
        save(args.root/'histories.json',histories)
        learning=compare_histories(histories,candidate='weighted',references=('dice','bands','edge'))
        # The primary contrast gets its own common horizon, independent of older controls.
        learning['primary_weighted_vs_edge']=compare_histories(
            {k:histories[k] for k in ('edge','weighted')},candidate='weighted',references=('edge',))
        learning.update(seed=args.seed,training_including_io_seconds=elapsed,calibration_seconds=0,
            timing_scope='Primary weighted-versus-edge uses matching 3g hardware and epochs/updates. Compact checkpoints reduce I/O; do not attribute total elapsed-time differences to loss speed. epoch_seconds excludes checkpoint writing. Dice/bands historical controls used 4g.')
        save(args.root/'learning_curves.json',learning)
        configs['weighted']=(output,config)
        for name,loader in [('train',DataLoader(train.dataset,batch_size=1,shuffle=False)),('validation',val)]:
            audit(args.root/name,configs,device,loader,candidate='weighted',references=('dice','bands','edge'))
        records={name:json.loads((args.root/name/'audit/cases.json').read_text()) for name in ('train','validation')}
        save(args.root/'generalization.json',compare_splits(records['train'],records['validation'],
                                                          candidate='weighted',references=('dice','bands','edge')))
        save(args.root/'completion.json',{'status':'complete','seed':args.seed,'job_id':os.getenv('SLURM_JOB_ID'),
             'trained_runs':1,'reused_baselines':['dice','bands','edge'],'train_cases':10,'validation_cases':52})
    except BaseException:
        import traceback
        save(args.root/'failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
