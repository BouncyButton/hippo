"""Unaugmented binary ADNI/seismic constraints on matched Dice+CE baselines."""
import argparse
from functools import partial
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.cross_dataset_campaign import msd_ce_constraints as objective
from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, write, digest
from scripts.adni_constraints import objectives as binary


def depth_loss(logits, labels, weights):
    if logits.ndim != 5 or logits.shape[:2] != (1,2) or labels.shape != logits[:, :1].shape:
        raise ValueError('Expected a binary batch-one volume')
    if weights is None or tuple(weights.shape) != tuple(logits.shape[2:]):
        raise ValueError('Depth geometry shape differs')
    return (F.cross_entropy(logits.float(),labels[:,0].long(),reduction='none')[0]*weights).sum()


def bind():
    binary.bind()
    objective.original.depth_loss = depth_loss


bind()
training_loss = objective.training_loss
calibration_case = objective.calibration_case


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir',type=Path,required=True)
    parser.add_argument('--objective',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args()
    p=read(args.protocol_dir/'protocol.json')
    if (p['dataset'] not in ('ADNI','GeoFWI3D') or p['model_identity'] not in MODELS
            or p['augmentation']!='none' or p['weight_ce']!=1 or args.objective not in ARMS
            or p['calibration_target_ratio']!=.6 or p['calibration_p95_cap']!=.75):
        raise ValueError('Unexpected binary target60 recipe')
    experiment.configure(p,args.objective);bind()
    experiment.base.build_loss_and_optimizer=experiment.ORIGINAL_LOSS_OPTIMIZER
    experiment.trainer.training_loss=partial(training_loss,sdf_policy=p['sdf_policy'][args.objective])
    experiment.trainer.calibration_case=calibration_case
    previous=experiment.trainer.calibrate
    def calibrate(*args,**kwargs):
        report=previous(*args,**kwargs)
        report.update(base_gradient='Dice+CE',scope='Training-only sequential full-resolution logit gradients: bands, edge, SDF, depth; native deep-supervision weighting retained')
        return report
    experiment.trainer.calibrate=calibrate
    builder=experiment.base.build_model
    def build_model(name,architecture):
        model=builder(name,architecture)
        h=hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        initialization=dict(model_state_sha256=h.hexdigest())
        baseline=args.output.parent/p['baseline_run_id']/'initialization.json'
        if args.objective!='dice' and (not baseline.exists() or read(baseline)!=initialization):
            raise ValueError('Model initialization differs from this campaign baseline')
        write(args.output/'initialization.json',initialization)
        return model
    experiment.base.build_model=build_model
    write(args.output/'runtime_recipe.json',dict(dataset=p['dataset'],model=p['model_identity'],
        objective=args.objective,weight_dice=1,weight_ce=1,dice_include_background=p['model_identity']!='nnunet',
        augmentation='none',constraints=args.objective.split('_')[1:],calibration_base='Dice+CE',
        target=.6,p95_cap=.75,sdf_policy=p['sdf_policy'][args.objective],
        primary_checkpoint='final_budget',protocol_sha256=digest(args.protocol_dir/'protocol.json')))
    experiment.trainer.main()


if __name__=='__main__':main()
