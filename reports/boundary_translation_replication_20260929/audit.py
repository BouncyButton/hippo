"""Exact integer-translation intervention on frozen selected models."""
import argparse
import json
import hashlib
from pathlib import Path
import os
import sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arms',nargs='+',default=['dice_aug','sum_aug','dice','sum'])
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    src=args.pilot/'source'
    for name,h in read(src/'PAYLOAD.json')['files'].items():assert sha(src/name)==h
    sys.path[:0]=[str(src),str(src/'scripts')]
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from train_boundary_causal import spatial_metrics
    from train_separated_edge import build_model,objectives
    from train_pcgrad_50cases import loaders
    from audit_edge_coherence import geometry
    from thesis.new_constraints import train_swinunetr_constraints as trainer
    from thesis.new_constraints.equivariance import translate_3d,restore_translation,translation_valid_mask
    from thesis.new_constraints.teacher.translation_teacher import DEFAULT_TEACHER_SHIFTS
    torch.set_num_threads(4);device=torch.device('cuda');assert torch.cuda.is_available()
    bindings=[]
    for arm in args.arms:
        folder=args.pilot/f'seed{args.seed}'/arm;config=read(folder/'config.json');done=read(folder/'completion.json')
        assert done['status']=='complete' and config['run']==done['run']
        assert sha(folder/'checkpoint_best.pt')==done['checkpoint_sha256']
        assert sha(folder/'selected_cases.json')==done['selected_cases_sha256']
        for k in ('pkl','splits_json'):assert sha(config[k])==config[k+'_sha256']
        assert trainer.collect_runtime_provenance()==config['runtime_provenance']
        trainer.validate_calibration_execution(config['execution_provenance'],trainer.collect_execution_provenance(device),allow_mig_profile_change=True)
        trainer.seed_everything(config['run']['seed'])
        train,val,_,_=loaders(config)
        cp=torch.load(folder/'checkpoint_best.pt',map_location='cpu',weights_only=True)
        assert cp['epoch']==done['selected_epoch'] and cp['run']==config['run']
        model=build_model(config,device);model.load_state_dict(cp['model'],strict=True);model.eval();del cp
        _,obj=objectives(config)
        old={(r['split'],r['case_name']):r['metrics'] for r in read(folder/'selected_cases.json')}
        rows=[]
        for split,loader in [('train',DataLoader(train.dataset,batch_size=1,shuffle=False)),('validation',val)]:
            for batch in loader:
                image=batch['image'].to(device);label=batch['label'].to(device)
                truth=batch['label'][0,0].numpy().astype(np.uint8);y,inner,outer=geometry(truth);shell=inner|outer
                ap_faces=0
                ap_endpoints=np.zeros_like(y)
                outer_endpoints=np.zeros_like(y)
                for axis in range(3):
                    a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None);a,b=tuple(a),tuple(b)
                    am=(truth[a]>0)&(truth[b]>0)&(truth[a]!=truth[b]);om=y[a]!=y[b]
                    ap_faces+=int(am.sum())
                    ap_endpoints[a]|=am;ap_endpoints[b]|=am;outer_endpoints[a]|=om;outer_endpoints[b]|=om
                gt_geometry=dict(label_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),ap_faces=ap_faces,
                    ap_endpoints=int(ap_endpoints.sum()),outer_endpoints=int(outer_endpoints.sum()))
                with torch.inference_mode(),torch.autocast('cuda',enabled=True):base_logits=model(image)
                base_p=base_logits.float().softmax(1)
                probability_sum=base_p.clone();counts=torch.ones_like(base_p[:,:1]);views=[]
                base_pred=base_p[0].argmax(0).cpu().numpy().astype(np.uint8)
                for shift in DEFAULT_TEACHER_SHIFTS:
                    valid=translation_valid_mask(image.shape[-3:],shift,device=device)
                    # Record clipping; the intervention does not use GT to select a view.
                    lost_input_voxels=int(((image!=0)&~valid.bool()).sum())
                    lost_gt_voxels=int(((label>0)&~valid.bool()).sum())
                    with torch.inference_mode(),torch.autocast('cuda',enabled=True):z=model(translate_3d(image,shift))
                    aligned=restore_translation(z.float().softmax(1),shift)
                    probability_sum.add_(aligned*valid);counts.add_(valid)
                    h=aligned[0].argmax(0).cpu().numpy().astype(np.uint8);v=valid[0,0].cpu().numpy().astype(bool)
                    views.append(dict(shift=list(shift),lost_input_nonzero_voxels=lost_input_voxels,
                        lost_gt_foreground_voxels=lost_gt_voxels,
                        boundary_union_disagreement_voxels=int(((h>0)!=(base_pred>0))[shell&v].sum()),
                        common_union_errors=int(((h>0)!=y)[shell&v].sum()),
                        identity_common_union_errors=int(((base_pred>0)!=y)[shell&v].sum())))
                ensemble=probability_sum/counts
                assert torch.allclose(ensemble.sum(1),torch.ones_like(ensemble[:,0]),atol=2e-6)
                pp=[base_p[0].cpu().numpy(),ensemble[0].cpu().numpy()]
                mm=[]
                for q in pp:
                    pred=q.argmax(0).astype(np.uint8);m=spatial_metrics(q,pred,truth)
                    with torch.inference_mode():b=obj.bands(torch.from_numpy(np.log(q.clip(1e-30)))[None].to(device),label)
                    m.update(bands_bce=float(b.details['case_loss'][0]),bands_truth=float(b.truth[0]))
                    mm.append(m)
                ensemble_pred=pp[1].argmax(0)
                a=(base_pred>0)!=y;b=(ensemble_pred>0)!=y
                transitions=dict(shell_errors_corrected=int((a&~b&shell).sum()),
                    shell_errors_introduced=int((~a&b&shell).sum()))
                assert transitions['shell_errors_corrected']-transitions['shell_errors_introduced']==mm[0]['boundary_union_errors']-mm[1]['boundary_union_errors']
                name=str(batch['case_name'][0])
                rows.append(dict(split=split,case_name=name,identity=mm[0],tta13=mm[1],views=views,
                    gt_geometry=gt_geometry,
                    transitions=transitions,identity_drift={k:mm[0][k]-old[split,name][k] for k in ['macro_dice','all_fp','all_fn','boundary_union_errors']}))
            save(args.out/f'{arm}.json',dict(arm=arm,seed=args.seed,selected_epoch=done['selected_epoch'],cases=rows))
            print(json.dumps(dict(arm=arm,split=split,cases=len(rows))),flush=True)
        assert sha(folder/'checkpoint_best.pt')==done['checkpoint_sha256']
        bindings.append(dict(arm=arm,seed=args.seed,checkpoint_sha256=done['checkpoint_sha256'],selected_epoch=done['selected_epoch']))
        del model,obj;torch.cuda.empty_cache()
    save(args.out/'completion.json',dict(status='complete',training=False,job_id=os.getenv('SLURM_JOB_ID'),
        model_bindings=bindings,script_sha256=sha(__file__),source_manifest_sha256=sha(src/'PAYLOAD.json'),
        shifts=[list(s) for s in DEFAULT_TEACHER_SHIFTS],view_policy='identity plus fixed12integer shifts; equal valid-view probability average; no GT used for prediction'))


if __name__=='__main__':main()
