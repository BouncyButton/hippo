"""Profile new categorical bands against the exact historical logit graph."""
from copy import deepcopy
import torch
import monai
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import profile, objectives, class_aware_bands
from scripts.unetr_subfields.contracts import ROOT

NAME = 'mni_class_aware_bands_cap2_f0s0_20261107'


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__ == '1.6.0'
    torch.set_num_threads(4)
    original = objectives.term
    for identity in ('swinunetr', 'unetr_vit_b16'):
        parent = ROOT / 'reports/mni_constraints14_cap8_20261107/profiles' / f'{identity}.json'
        old = read(parent); rows = []
        for variant in ('original', 'class_aware'):
            objectives.term = original
            if variant == 'class_aware':
                class_aware_bands.install()
            for geometry in ('ellipsoid', 'box'):
                label = profile.target(geometry)
                for calibration in (False, True):
                    torch.manual_seed(0)
                    z = torch.randn(old['output_shapes'][0], requires_grad=True)
                    def call():
                        if calibration:
                            return objectives.calibration_case(z, label, 'dice_bands', identity=identity)
                        value = objectives.training_loss(z, label, 'dice_bands', {'bands': .04}, 1., identity=identity)
                        value.backward()
                        assert torch.isfinite(value)
                        return float(value)
                    row, _ = profile.audit_call(call)
                    prior = next(r for r in old['profiles'] if r['arm']=='dice_bands' and r['geometry']==geometry and r['calibration']==calibration)
                    if variant == 'original':
                        assert row['flops'] == prior['flops'], (identity, geometry, calibration, row['flops'], prior['flops'])
                    row.update(variant=variant, geometry=geometry, calibration=calibration,
                               composed_flops=prior['composed_flops']+row['flops']-prior['flops'])
                    rows.append(row)
        costs = deepcopy(old['costs'])
        training = {r['composed_flops'] for r in rows if r['variant']=='class_aware' and not r['calibration']}
        calibration = {r['composed_flops'] for r in rows if r['variant']=='class_aware' and r['calibration']}
        assert len(training)==len(calibration)==1
        costs['dice_bands'] = dict(update_flops=training.pop(), calibration_flops=10*calibration.pop()+4096)
        result = dict(status='profiled', model=identity, parameters=old['parameters'],
            output_shapes=old['output_shapes'], costs={k:costs[k] for k in ('dice','dice_bands')}, profiles=rows,
            parent_profile_sha256=digest(parent), objective_sha256=digest(ROOT/'scripts/mni_constraints/objectives.py'),
            categorical_source_sha256=digest(ROOT/'scripts/mni_constraints/class_aware_bands.py'),
            runtime=dict(torch=torch.__version__, monai=monai.__version__),
            accounting='Exact original full-objective/logit-backward reproduction, then measured categorical replacement delta. Network backward/optimizer unchanged. Same dense MAC=2 convention, mean/unreduced CE supplement, ten eval-mode calibration forwards and logit gradients plus4096 host selection FLOPs. Scalar coefficient product added by scheduler.')
        write(ROOT/'reports'/NAME/'profiles'/f'{identity}.json', result)
        print(identity, result['costs'], flush=True)
    objectives.term = original


if __name__ == '__main__':
    main()
