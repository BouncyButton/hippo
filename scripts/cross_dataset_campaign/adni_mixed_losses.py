"""Unaugmented ADNI: native nnUNet Dice+CE, established Dice for other models."""
import argparse
from functools import partial
import hashlib
from pathlib import Path

from scripts.cross_dataset_campaign import conventional as experiment
from scripts.cross_dataset_campaign.common import ARMS, MODELS, read, write, digest
from scripts.adni_constraints import objectives as binary

training_loss = binary.training_loss
calibration_case = binary.calibration_case


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    p = read(args.protocol_dir/'protocol.json')
    native = p['model_identity']=='nnunet'
    target, ceiling = (.6,.75) if native else (.1,.5)
    if (p['dataset']!='ADNI' or p['model_identity'] not in MODELS or p['augmentation']!='none'
            or p['weight_ce']!=int(native) or args.objective not in ARMS
            or p['calibration_target_ratio']!=target or p['calibration_p95_cap']!=ceiling
            or p.get('runtime_acceleration',{}).get('enabled')):
        raise ValueError('Unexpected mixed-loss ADNI recipe')
    experiment.configure(p,args.objective)
    binary.bind()
    experiment.base.build_loss_and_optimizer = experiment.ORIGINAL_LOSS_OPTIMIZER
    experiment.trainer.training_loss = partial(training_loss,policy=p['sdf_policy'][args.objective])
    experiment.trainer.calibration_case = calibration_case
    previous = experiment.trainer.calibrate
    def calibrate(*args, **kwargs):
        result = previous(*args,**kwargs)
        result.update(base_gradient='Dice+CE' if native else 'Dice',
            scope='All ten training crops only; sequential bands, edge, SDF and depth against accumulated weighted logit gradients; native deep supervision retained')
        return result
    experiment.trainer.calibrate = calibrate
    builder = experiment.base.build_model
    def build_model(name, architecture):
        model = builder(name,architecture)
        h = hashlib.sha256()
        for key,value in sorted(model.state_dict().items()):
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        initialization = dict(model_state_sha256=h.hexdigest())
        baseline = args.output.parent/p['baseline_run_id']/'initialization.json'
        if args.objective!='dice' and (not baseline.exists() or read(baseline)!=initialization):
            raise ValueError('Initialization differs from the fresh matched ADNI baseline')
        write(args.output/'initialization.json',initialization)
        return model
    experiment.base.build_model = build_model
    write(args.output/'runtime_recipe.json',dict(dataset='ADNI',model=p['model_identity'],
        objective=args.objective,weight_dice=1,weight_ce=int(native),dice_include_background=not native,
        native_deep_supervision=native,augmentation='none',constraints=args.objective.split('_')[1:],
        calibration_base='Dice+CE' if native else 'Dice',target=target,p95_cap=ceiling,
        sdf_policy=p['sdf_policy'][args.objective],primary_checkpoint='final_budget',
        ap_swaps_applicable=False,protocol_sha256=digest(args.protocol_dir/'protocol.json')))
    experiment.trainer.main()


if __name__=='__main__':main()
