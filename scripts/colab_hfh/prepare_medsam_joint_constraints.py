"""Prepare fold1/fold2 candidate protocols from saved controls; launch nothing."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.colab_hfh import medsam_joint_constraints as joint


def prepare(archive, arm, routing):
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('results/archive_manifest.json'))
        for name,expected in manifest.items():
            if hashlib.sha256(z.read('results/'+name)).hexdigest()!=expected:
                raise ValueError('Control archive differs: '+name)
        read=lambda name:json.loads(z.read('results/'+name+'.json'))
        p=deepcopy(read('protocol'))
        if p['arm']!='dice' or read('completion')['updates']!=125:
            raise ValueError('Expected a completed 125-update native control')
        history=read('training_history')
        schedule=read('slab_schedule')[:250]
        if len(history)!=125 or len(schedule)!=250:
            raise ValueError('Incomplete control')
        if [r['lr'] for r in history]!=p['frozen_learning_rates'][:125]:
            raise ValueError('Saved LR schedule differs')
    terms=joint.active_terms(arm)
    p.pop('arm')
    p['name']=f'joint_candidate_{routing}_{arm}_f{p["fold"]}_s0'
    p['requested_arms']=[arm]
    p['baseline_archive_sha256']=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    p['joint_constraints']={**joint.DEFAULT,'auxiliary_field':routing}
    p['joint_constraint_source_sha256']=hashlib.sha256(Path(joint.__file__).read_bytes()).hexdigest()
    p['constraint_recipe']=dict(name=joint.FORMULATION,target=.01,ramp=20,numerics='strict_empty_grid_v1')
    p['frozen_sample_schedule']=schedule
    p['gradient_ratio_target']=.01
    p['gradient_ratio_p95_cap']=.05
    p['gradient_ratio_targets']={k:.01/len(terms) for k in terms}
    p['gradient_ratio_p95_caps']={k:.05/len(terms) for k in terms}
    p['calibration_policy']=dict(sampler='Same deterministic 200 augmented training slabs as recalibration v1',
        slabs_per_crop=20,zero_gradients='retained',native_preflight_anchors_per_crop=4,
        reference='Every term relative to native; actual combined Gram norm checked',
        online='Per-LoRA-module cap after two-slab accumulation, before total clipping/AdamW')
    p['constraint_geometry']=dict(tensor_spacing_mm=[2.,1.,1.],
        bands='Unchanged two six-connected voxel steps, equal inner/outer mass',
        edge='Unchanged signed residual differences; no anatomical wrap',
        sdf='Whole-crop physical EDT, half-minimum-voxel offset, clipped at 8mm',
        depth='Unchanged 8mm ROI, spatial-layer weights; anatomical orientation unverified')
    p['candidate_status']='prepared_not_validated; no training launched'
    p['comparison']='Original gain/native controls reused; exact samples, LR, initialization, 125 updates. Additional backwards are not FLOP matched.'
    joint.check_protocol(p)
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arms',nargs='+',choices=joint.ARMS,default=['dice_bands_edge_sdf_depth'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for fold in (1,2):
        archive=args.controls/f'gain125__dice__f{fold}_s0.zip'
        for arm in args.arms:
            for route in ('semantic','mask_grad'):
                p=prepare(archive,arm,route)
                path=args.output/(p['name']+'.json')
                path.write_text(json.dumps(p,indent=2)+'\n')
                records.append(dict(protocol=path.name,fold=fold,arm=arm,routing=route,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output/'preparation.json').write_text(json.dumps(dict(status='prepared_only',launched=0,
        real_model_preflight='pending; worker refuses training if native gradient equivalence fails',
        effectiveness='unknown; these are hypothesis comparisons, not selected winners',protocols=records),indent=2)+'\n')
    print(json.dumps(dict(prepared=len(records),launched=0,output=str(args.output))))


if __name__=='__main__':main()
