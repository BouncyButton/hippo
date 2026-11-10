"""Measure both dynamic objectives against the reproduced historical graph."""
from copy import deepcopy
import torch
import monai
from scripts.cross_dataset_campaign.common import read, write, digest
from scripts.mni_constraints import profile, objectives, adaptive_bands
from scripts.unetr_subfields import baseline_runtime
from scripts.unetr_subfields.contracts import ROOT

NAME = 'mni_adaptive_outer_bands_cap2_f0s0_20261107'
MODELS = ('nnunet','swinunetr','unetr_vit_b16','basic_unet16')


def main():
    assert torch.__version__.startswith('2.2.2') and monai.__version__ == '1.6.0'
    torch.set_num_threads(4)
    original = objectives.term
    for identity in MODELS:
        parent = ROOT/'reports/mni_constraints14_cap8_20261107/profiles'/f'{identity}.json'
        old = read(parent)
        protocol = read(ROOT/'experiments/mni_shared_cap2_baseline_bands_f0s0_20261107/protocols'/f'{identity}_dice_bands/protocol.json')
        criterion = None
        if identity == 'nnunet':
            criterion, _ = baseline_runtime.conventional.ORIGINAL_LOSS_OPTIMIZER(identity, torch.nn.Linear(1,1), protocol)
        for variant in ('original','weighted','adaptive'):
            objectives.term = original
            if variant != 'original': adaptive_bands.install(variant)
            rows=[]
            for geometry in ('ellipsoid','box'):
                label=profile.target(geometry)
                for regime in (('random',) if variant=='original' else ('random','background','perfect')):
                    for calibration in (False, True):
                        torch.manual_seed(0)
                        outputs=[torch.randn(shape,requires_grad=True) for shape in old['output_shapes']]
                        if regime!='random':
                            with torch.no_grad():
                                outputs[0].zero_()
                                if regime=='background': outputs[0][:,0]=5
                                else: outputs[0].scatter_(1,label,5.)
                        if identity!='nnunet': outputs=outputs[0]
                        def call():
                            if calibration:
                                return objectives.calibration_case(outputs,label,'dice_bands',criterion=criterion,identity=identity)
                            value=objectives.training_loss(outputs,label,'dice_bands',{'bands':.04},1.,criterion=criterion,identity=identity)
                            value.backward(); assert torch.isfinite(value)
                            return float(value)
                        row,_=profile.audit_call(call)
                        prior=next(r for r in old['profiles'] if r['arm']=='dice_bands' and r['geometry']==geometry and r['calibration']==calibration)
                        if variant=='original': assert row['flops']==prior['flops'],(identity,row['flops'],prior['flops'])
                        row.update(variant=variant,geometry=geometry,regime=regime,calibration=calibration,
                                   composed_flops=prior['composed_flops']+row['flops']-prior['flops'])
                        rows.append(row)
            if variant=='original': continue
            training={r['composed_flops'] for r in rows if not r['calibration']}
            calibration={r['composed_flops'] for r in rows if r['calibration']}
            assert len(training)==len(calibration)==1, (identity,variant,training,calibration)
            costs=deepcopy(old['costs'])
            costs['dice_bands']=dict(update_flops=training.pop(),calibration_flops=10*calibration.pop()+4096)
            result=dict(status='profiled',model=identity,variant=variant,parameters=old['parameters'],output_shapes=old['output_shapes'],
                costs={k:costs[k] for k in ('dice','dice_bands')},profiles=rows,parent_profile_sha256=digest(parent),
                objective_sha256=digest(ROOT/'scripts/mni_constraints/objectives.py'),
                adaptive_source_sha256=digest(ROOT/'scripts/mni_constraints/adaptive_bands.py'),
                runtime=dict(torch=torch.__version__,monai=monai.__version__),
                accounting='Original full objective/backward exactly reproduced; measured replacement delta includes dynamic masks, weight arithmetic, convolutional dilation and loss/backward. Same dense MAC=2 convention; comparisons, boolean/index operations are tracked operators but not floating point operations. Calibration separately charged. Evaluation/spatial/gradient probes separately reported, outside optimization cap. Scalar coefficient product added by scheduler.')
            write(ROOT/'reports'/NAME/'profiles'/f'{identity}_{variant}.json',result)
            print(identity,variant,result['costs'],flush=True)
    objectives.term=original


if __name__=='__main__': main()
