"""Audit the binary loss graphs, frozen data, and follow-on execution contracts."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, write, digest
from scripts.cross_dataset_campaign.followon_job import verify, check_predecessor, command_for
from scripts.cross_dataset_campaign import binary_ce_constraints as obj
from scripts.cross_dataset_campaign.probe_other_losses import paired_loss
from nnunet_band_objective import native_loss


def numerical(stage):
    torch.set_num_threads(2);torch.manual_seed(0);obj.bind()
    for dataset,size in [('ADNI',64),('GeoFWI3D',96)]:
        axis=torch.arange(size);x,y,z=torch.meshgrid(axis,axis,axis,indexing='ij')
        labels=((((x-size/2)/(size*.28))**2+((y-size/2)/(size*.18))**2+
                 ((z-size/2)/(size*.15))**2)<1).long()[None,None]
        outputs=[torch.randn(1,2,n,n,n,requires_grad=True) for n in (size,size//2,size//4,size//8)]
        sdf=obj.binary.signed_distance_maps(labels)
        for identity in MODELS:
            p=read(stage/'protocols'/f'{dataset}_{identity}'/'protocol.json')
            criterion,optimizer=obj.experiment.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'],torch.nn.Linear(1,1),p)
            native=criterion if identity=='nnunet' else None;logits=outputs if native else outputs[0]
            if native:
                if criterion.loss.dc.do_bg or not np.allclose(criterion.weight_factors,[1/1.75,.5/1.75,.25/1.75,0]):
                    raise ValueError('Native foreground Dice/deep supervision changed')
                baseline=native_loss(logits,labels,criterion)
            else:
                if not criterion.include_background:raise ValueError('Established Dice background handling changed')
                baseline=paired_loss(logits,labels,'dice',{},0.,weight_ce=1)
            if not torch.equal(baseline,obj.training_loss(logits,labels,'dice',{},0.,criterion=native)):
                raise ValueError('Binary Dice+CE baseline mismatch')
            affine=np.eye(4);affine[:3,:3]=np.array([[0,1,0],[1,0,0],[0,0,1]])
            depth=obj.binary.geometry_weights(labels,affine,p['depth_geometry'])
            terms=dict(bands=obj.binary.bands()(outputs[0],labels).loss,
                edge=obj.binary.BinaryEdge()(outputs[0],labels),sdf=obj.binary.sdf_loss(outputs[0],labels,sdf),
                depth=obj.depth_loss(outputs[0],labels,depth))
            gradients={'dice':torch.autograd.grad(baseline,outputs[0],retain_graph=True)[0]}
            gradients.update({k:torch.autograd.grad(v,outputs[0],retain_graph=True)[0] for k,v in terms.items()})
            for arm in ARMS[1:]:
                names=[n for n in ('bands','edge','sdf','depth') if n in arm];weights={n:.13 for n in names}
                policy=p['sdf_policy'][arm];delayed='sdf' in arm and policy=='delayed_5_of_9'
                scale=1. if delayed else .4
                actual=obj.training_loss(logits,labels,arm,weights,scale,sdf,native,depth,.2,sdf_policy=policy)
                expected=baseline+sum(weights[n]*scale*(.2 if delayed and n=='sdf' else 1.)*terms[n] for n in names)
                if not torch.allclose(actual,expected,atol=1e-6):raise ValueError('Binary auxiliary graph differs')
                gs=torch.autograd.grad(actual,outputs[:3] if native else outputs[:1],retain_graph=True)
                if any(not g.isfinite().all() for g in gs):raise ValueError('Nonfinite gradients')
                row=obj.calibration_case(logits,labels,arm,sdf,native,depth)
                if not np.isclose(row['dice_rms'],float(gradients['dice'].square().mean().sqrt()),rtol=1e-6):
                    raise ValueError('Calibration baseline is not joint Dice+CE')
                selected,stats=obj.experiment.select_at_target([deepcopy(row) for _ in range(10)],arm,.6,.75)
                joint=gradients['dice'].clone()
                for name in names:
                    ratio=float((selected[name]*gradients[name]).square().mean().sqrt()/joint.square().mean().sqrt())
                    if not np.isclose(ratio,.6,rtol=1e-5) or stats[name]['weighted_p95_ratio']>.75000001:
                        raise ValueError('Sequential gradient calibration mismatch')
                    joint+=selected[name]*gradients[name]
            print('Numerical loss and calibration checks:',dataset,identity,flush=True)


def foundation_check(stage):
    from scripts.cross_dataset_campaign.foundation import Volumes, SupportSlabs
    p=read(stage/'protocols/ADNI_MedSAM3/protocol.json')
    p={**p,'arm':'dice_bands_edge_sdf_depth'}
    slabs=SupportSlabs(Volumes(p),p,None)
    if len(slabs.inventory)!=5 or len(slabs.schedule(p))!=400 or len(slabs.calibration_schedule(4))!=20:
        raise ValueError('Incorrect foundation support/sampling')
    for case in p['train_cases']:
        if not slabs.sdf[case].isfinite().all() or not slabs.depth[case].isfinite().all():
            raise ValueError('Invalid foundation geometry')
        if not np.isclose(float(slabs.depth[case].sum()),1.,atol=1e-6):
            raise ValueError('Foundation depth weights are not normalized')
    write(stage/'foundation_cpu_preflight.json',dict(status='passed',inventory=slabs.inventory,
        slab_samples=400,calibration_samples=20,gpu='Actual SAM3 forward/backward checked in allocated baseline; not executed on login node'))


def preflight(stage):
    plan=verify(stage);check_predecessor(plan);checked=set();data=set();splits={}
    for task in plan['tasks']:
        if task['protocol'] in checked:continue
        checked.add(task['protocol']);folder=stage/'protocols'/task['protocol'];p=read(folder/'protocol.json')
        if task['kind']=='conventional':
            for name,key in [('splits_10patients.json','splits_sha256'),('nnunet_architecture.json','nnunet_architecture_sha256'),('profile.json','profile_sha256')]:
                if digest(folder/name)!=p[key]:raise ValueError('Protocol artifact differs: '+name)
            profile=read(folder/'profile.json')
            if (digest(folder/'reference_profile.json')!=profile['reference_profile_sha256'] or
                profile['objective_sha256']!=digest(stage/'source/scripts/cross_dataset_campaign/binary_ce_constraints.py') or
                profile['shared_objective_sha256']!=digest(stage/'source/scripts/cross_dataset_campaign/msd_ce_constraints.py')):
                raise ValueError('Profile provenance changed')
            split=read(folder/'splits_10patients.json')[0]
            expected=(10,54) if p['dataset']=='ADNI' else (19,96)
            if (len(set(split['train'])),len(set(split['val'])))!=expected or set(split['train'])&set(split['val']):
                raise ValueError('Wrong cohort or leakage')
            if p['dataset'] in splits and split!=splits[p['dataset']]:raise ValueError('Models use different data')
            splits[p['dataset']]=split
            if p['data'] not in data:
                if digest(p['data'])!=p['pkl_sha256']:raise ValueError('Dataset changed')
                if p['dataset']=='GeoFWI3D':
                    descriptor=read(p['data']);manifest=read(descriptor['manifest'])
                    if split!={'train':manifest['splits']['training_subsets']['5pct'],'val':manifest['splits']['validation']}:
                        raise ValueError('Seismic saved split differs')
                    for name in split['train']+split['val']:
                        row=manifest['cases'][name]
                        for kind in ('image','label'):
                            if digest(Path(descriptor['dataset_root'])/row[kind])!=row[kind+'_sha256']:
                                raise ValueError('Seismic volume changed')
                data.add(p['data'])
        else:
            if not set(p['train_cases'])<=set(splits['ADNI']['train']) or p['evaluation_cases']!=splits['ADNI']['val']:
                raise ValueError('Foundation support differs from saved ADNI split')
            if len(set(p['support_subjects']))!=5 or set(p['train_cases'])&set(p['evaluation_cases']):
                raise ValueError('Invalid support subjects or leakage')
            for path,sha in [(p['data'],p['dataset_sha256']),(p['lora_weights'],p['lora_sha256']),
                             (plan['base_staging'],p['base_sha256'])]:
                if digest(path)!=sha:raise ValueError('Foundation source changed: '+path)
        if not Path(command_for(stage,plan,task,Path(plan['base_staging']))[0]).exists():raise ValueError('Runtime missing')
    numerical(stage)
    subprocess.run([sys.executable,'-m','unittest','scripts.cross_dataset_campaign.tests',
        'scripts.cross_dataset_campaign.test_resume','scripts.cross_dataset_campaign.test_archive_handoff',
        'scripts.cross_dataset_campaign.test_followon','-v'],check=True)
    subprocess.run([plan['foundation_python'],'-c',
        'from pathlib import Path; from scripts.cross_dataset_campaign.preflight_followon import foundation_check; foundation_check(Path('+repr(str(stage))+'))'],check=True)
    subprocess.run(['bash','-n',str(stage/'run.sbatch')],check=True)
    write(stage/'preflight.json',dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),runs=126,
        checked_protocols=sorted(checked),checks=['Binary 64/96 loss and gradient checks','Exact split and data hashes',
        'All profile/source identities','Optimizer/RNG/sampling resume equivalence','Archive corruption protection',
        'MSD-before-follow-on and shared allocation deadline','Five-crop foundation geometry and sampling'],
        gpu_validation='Actual MedSAM3 forward/backward checked in allocated baseline; not claimed as GPU-tested before submission'))
    print('Follow-on preflight passed',flush=True)
