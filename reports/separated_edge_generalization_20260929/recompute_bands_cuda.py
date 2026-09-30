"""Inference only: original selected weights, frozen source, and CUDA-AMP runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    base=args.experiment; source=base/'source'; out=args.output
    assert base.resolve()!=out.resolve() and base.resolve() not in out.resolve().parents
    out.mkdir(parents=True,exist_ok=True)
    for name,digest in read(source/'PAYLOAD.json')['files'].items(): assert sha(source/name)==digest,name
    sys.path[:0]=[str(source),str(source/'scripts')]
    import torch,monai,numpy as np
    from torch.utils.data import DataLoader
    from train_separated_edge import loaders,build_model,objectives
    from audit_edge_coherence import metrics
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    runtime=dict(torch=torch.__version__,monai=monai.__version__,device=torch.cuda.get_device_name(),precision='Original CUDA autocast inference; bands internal FP32',job_id=os.environ.get('SLURM_JOB_ID'))
    save(out/'inference_runtime.json',runtime)
    for arm in ['pooled','separated']:
        for seed in range(3):
            p=base/f'fold0/seed{seed}/{arm}'
            config=read(p/'config.json');complete=read(p/'completion.json')
            expected=config['runtime_provenance']
            # Fail rather than silently substitute a different model library/runtime.
            assert expected['torch']==torch.__version__,(expected['torch'],torch.__version__)
            assert expected['monai']==monai.__version__,(expected['monai'],monai.__version__)
            assert config['run']['training_augmentation']=='none'
            assert config['run']['amp'] is True
            for key in ['pkl','splits_json']: assert sha(config[key])==config[key+'_sha256']
            assert sha(p/'checkpoint_best.pt')==complete['checkpoint_sha256']
            assert sha(p/'selected_cases.json')==complete['selected_cases_sha256']
            train,val,_,_=loaders(config)
            checkpoint=torch.load(p/'checkpoint_best.pt',map_location='cpu',weights_only=True)
            assert checkpoint['epoch']==complete['selected_epoch'] and checkpoint['run']==complete['run']
            model=build_model(config,device);model.load_state_dict(checkpoint['model'],strict=True);model.eval();del checkpoint
            _,objective=objectives(config)
            bands=objective.bands
            assert bands.adherence_threshold==0.90 and bands.inner_focal_gamma==bands.outer_focal_gamma==bands.degree_alpha==0
            assert bands.foreground_class_ids==(1,2) and bands.complement_class_ids==(0,) and bands.steps==2
            original={(r['split'],r['case_name']):r['metrics'] for r in read(p/'selected_cases.json')}
            rows=[];started=time.perf_counter()
            for kind,loader in [('train',DataLoader(train.dataset,batch_size=1,shuffle=False)),('validation',val)]:
                for batch in loader:
                    with torch.inference_mode(),torch.autocast('cuda',enabled=config['run']['amp']):
                        logits=model(batch['image'].to(device))
                    with torch.inference_mode(): result=bands(logits,batch['label'].to(device))
                    probabilities=logits[0].float().softmax(0).cpu().numpy()
                    truth=batch['label'][0,0].numpy().astype(np.uint8)
                    fresh=metrics(probabilities[1:].sum(0),probabilities.argmax(0).astype(np.uint8),truth)
                    name=str(batch['case_name'][0]);old=original[(kind,name)]
                    b={k:float(result.details[k][0]) for k in ['inner_loss','outer_loss','case_loss']}
                    b.update(truth=float(result.truth[0]),inner_truth=float(result.value[0,0]),outer_truth=float(result.value[0,1]),confidence_adherent=bool(result.details['confidence_adherent'][0]),valid=bool(result.details['valid'][0]))
                    rows.append(dict(split=kind,case_name=name,bands=b,recomputed_edge_metrics=fresh,drift_from_saved={k:fresh[k]-old[k] for k in ['macro_dice','inner_both_correct','outer_both_correct','cross_correct_transition']}))
            assert {(r['split'],r['case_name']) for r in rows}==set(original)
            save(out/f'bands_{arm}_seed{seed}.json',dict(arm=arm,seed=seed,selected_epoch=complete['selected_epoch'],checkpoint_sha256=complete['checkpoint_sha256'],runtime=runtime,cases=rows))
            print(json.dumps(dict(arm=arm,seed=seed,cases=len(rows),seconds=time.perf_counter()-started,max_macro_dice_abs_drift=max(abs(r['drift_from_saved']['macro_dice']) for r in rows))),flush=True)
            del model,objective,bands
            torch.cuda.empty_cache()
    save(out/'completion.json',dict(status='complete',training=False,files={p.name:sha(p) for p in out.glob('bands_*.json')},runtime=runtime))

if __name__=='__main__': main()
