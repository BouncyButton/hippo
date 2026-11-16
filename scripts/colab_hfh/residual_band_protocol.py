"""Freeze the HFH residual-band ablation and charge its conditional arithmetic."""
import hashlib
from pathlib import Path
from scripts.colab_hfh.residual_band import policy_dict, EXTRA_UPDATE_FLOPS, VERSION
from scripts.colab_hfh.constraint_schedule import make_payload, training_flops


def payload(base,plan):
    result=make_payload(base,'nnunet','dice_bands',plan['cost_references']['nnunet']['canonical_costs'],
        prefix=45,ramp=9,target=.1,cap=.5,screening=False)
    p=result['protocols']['nnunet']['protocol.json']
    p.update(residual_band_policy=policy_dict(),name=VERSION,
        constraint_calibration='conditional_residual_band_envelope_v1',
        constraint_definition='HFH physical residual-band BCE; readiness and coverage gates use training cases only; fixed native nnU-Net loss, optimizer, LR and sample order',
        constraint_geometry_units='physical_mm_hfh_1x2x1_residual_band',
        endpoint='Original full cap; final checkpoint; separate exploratory conditional-BAND ablation')
    p['residual_band_source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('residual_band.py','residual_band_worker.py','residual_band_archive.py',
                     'residual_band_protocol.py','residual_band_campaign.py','residual_band_pilot.py')}
    s=p['models']['nnunet']['objectives']['dice_bands']
    s['early_flops_per_update']+=EXTRA_UPDATE_FLOPS
    s['flops_per_update']+=EXTRA_UPDATE_FLOPS
    s['calibration_flops']+=10*EXTRA_UPDATE_FLOPS
    s['residual_band_extra_flops_per_active_update']=EXTRA_UPDATE_FLOPS
    steps=0
    while training_flops(steps+1,s)<=p['budget_flops']:steps+=1
    used=training_flops(steps,s)
    s.update(max_steps=steps,consumed_training_flops=used,consumed_flops=used,
        consumed_total_flops=used+s['calibration_flops'],unused_flops=p['budget_flops']-used,
        patient_pass_epochs=steps/10)
    if steps!=90 or p['lr_horizon_updates']!=90:
        raise ValueError('Residual BAND must preserve all 90 native updates within the original cap')
    result.update(version=VERSION,folds=[0],seeds=[0,1,2],
        interpretation='New conditional-BAND ablation; not a replacement for canonical matrix results')
    return result
