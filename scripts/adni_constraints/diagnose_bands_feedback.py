"""Measure frozen bands feedback on training cases without updating parameters."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from scripts.adni_baselines import core as adni
from scripts.adni_constraints import objectives
from scripts.capped_depth_delay.core import model_digest
from nnunet_band_objective import native_loss


def vector(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(parameters, gradients)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    original = adni.ROOT/'experiments/adni_constraints_noaug_5pct_fold0_seed0_20261013/protocols/nnunet'
    records = adni.ROOT/'reports/adni_constraints_noaug_5pct_fold0_seed0_20261013/comparison_20261013/results/nnunet'
    protocol = json.loads((original/'protocol.json').read_text())
    split = json.loads((original/'splits_10patients.json').read_text())[0]
    dataset = adni.ROOT/'datasets/Dataset103_ADNI/adni_hippocampus_full.pkl'
    assert adni.base.digest(dataset) == protocol['pkl_sha256']
    patients = adni.PreparedADNI(adni.base._load_pkl_dataframe(dataset), split['train'])
    architecture = json.loads((original/'nnunet_architecture.json').read_text())
    model = adni.build_model('nnunet', architecture)
    criterion, _ = adni.base.build_loss_and_optimizer('nnunet', model, protocol)
    calibration = json.loads((records/'dice_bands/fold0/seed0/run/calibration.json').read_text())
    weight = calibration['weights']['bands']
    result = dict(scope='All ten unaugmented training crops; no validation or optimizer updates',
                  diagnostic_runtime=dict(torch=torch.__version__,device='CPU',precision='FP32'),
                  limitation='Diagnostic FP32 gradients, not exact CUDA BF16 training gradients',
                  original_weight=weight, original_calibration_step=9,
                  original_calibration_statistics=calibration['statistics']['bands'], checkpoints={})
    for name, arm in [('baseline','dice'),('bands','dice_bands')]:
        path=args.checkpoints/(name+'.pt')
        done=json.loads((records/arm/'fold0/seed0/run/completion.json').read_text())
        assert adni.base.digest(path)==done['checkpoint_best_sha256']
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True)
        del checkpoint
        model.eval()
        before=model_digest(model)
        parameters=list(model.parameters())
        rows=[]
        for patient in patients:
            image,label=patient['image'][None],patient['label'][None]
            outputs=model(image)
            region=native_loss(outputs,label,criterion)
            band=objectives.bands()(outputs[0],label).loss
            gz_region=torch.autograd.grad(region,outputs[0],retain_graph=True)[0].detach()
            gz_band=torch.autograd.grad(band,outputs[0],retain_graph=True)[0].detach()
            g_region=vector(region,parameters,True)
            g_band=vector(band,parameters,False)
            row=dict(case_name=patient['case_name'],native_loss=float(region.detach()),band_loss=float(band.detach()),
                weighted_logit_norm_ratio=float(weight*gz_band.norm()/gz_region.norm()),
                weighted_parameter_norm_ratio=float(weight*g_band.norm()/g_region.norm()),
                parameter_cosine=float(torch.dot(g_band,g_region)/(g_band.norm()*g_region.norm())),
                logit_cosine=float((gz_band*gz_region).sum()/(gz_band.norm()*gz_region.norm())))
            assert all(np.isfinite(v) for k,v in row.items() if k!='case_name')
            rows.append(row)
            del outputs,region,band,gz_region,gz_band,g_region,g_band
            print(json.dumps(dict(checkpoint=name,completed_cases=len(rows))),flush=True)
        assert model_digest(model)==before
        aggregate={k:float(np.median([r[k] for r in rows])) for k in rows[0] if k!='case_name'}
        result['checkpoints'][name]=dict(model_sha256=before,checkpoint_sha256=adni.base.digest(path),
            selected_step=done['selected_step'],rows=rows,median=aggregate)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        adni.base.atomic_json(args.output,result)
        print(json.dumps(dict(checkpoint=name,median=aggregate)),flush=True)
        gc.collect()


if __name__=='__main__':
    main()
