"""Remeasure the full UNETR rescue graph in the original CPU accounting runtime."""
import json
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import rescue_objective as recipe
from scripts.cross_dataset_campaign import profile_seismic as full
from scripts.cross_dataset_campaign import conventional

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/geofwi3d_unetr_rescue_20261019'
SOURCE=ROOT/'reports/geofwi3d_unetr_ce_pilot_20261019/bundle'


def supplement(func,args,kwargs,result):
    value,category=full.intermediate_supplement(func,args,kwargs,result)
    name=str(func).split('.')[1] if str(func).startswith('aten.') else ''
    if category=='no_supplement' and name in ('native_dropout','native_dropout_backward'):
        # Fused mask multiplication and inverse keep-probability scaling.
        return 2*args[0].numel(),'dropout_mask_and_scale'
    return value,category


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch=full.audit.torch
    assert torch.__version__.startswith('2.2.2') and full.audit.monai.__version__=='1.6.0'
    torch.set_num_threads(2)
    full.audit.reference.supplementary_flops=supplement
    full.build_model=recipe.build_model
    full.obj.training_loss=recipe.training_loss
    full.conventional.dice_only_optimizer=conventional.ORIGINAL_LOSS_OPTIMIZER
    p=json.loads((SOURCE/'templates/unetr_vit_b16/protocol.json').read_text())
    arch=json.loads((SOURCE/'templates/unetr_vit_b16/nnunet_architecture.json').read_text())
    # Profiling helper supplies unused auxiliary weights; discard those for this strict baseline.
    full.obj.training_loss=lambda outputs,labels,arm,weights,*args:recipe.training_loss(outputs,labels,arm,{},*args)
    rows=[]
    for geometry in ('ellipsoid','box'):
        row=full.measure('unetr_vit_b16',arch,p,'dice',geometry,False)
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('geometry','flops','parameters')}),flush=True)
        (OUT/'profile.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert len({r['flops'] for r in rows})==1 and all(r['parameters']==92802274 for r in rows)
    result=dict(status='profiled',profiles=rows,costs={'dice':{'update_flops':rows[0]['flops'],'calibration_flops':0}},
        parameters=92802274,dropout_rate=.1,loss_arguments=recipe.LOSS_ARGUMENTS,
        recipe_sha256=hashlib.sha256(Path(recipe.__file__).read_bytes()).hexdigest(),
        runtime={'torch':torch.__version__,'monai':full.audit.monai.__version__},
        accounting='Established full forward/loss/backward/AdamW arithmetic, MAC=2. Dropout mask multiplication/scaling included; random sampling, comparisons, validation and I/O excluded as in the original arithmetic convention.',
        profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'profile.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
