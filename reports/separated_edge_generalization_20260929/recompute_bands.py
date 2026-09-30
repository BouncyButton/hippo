"""Read-only selected-checkpoint CPU inference; saves new per-case bands audit."""
import os
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'reports/separated_edge_20260929/source'
RESULTS = ROOT/'reports/separated_edge_20260929/results'
sys.path[:0] = [str(SOURCE), str(SOURCE/'scripts')]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    for base, manifest in [(RESULTS, RESULTS/'RESULT_MANIFEST.json'), (SOURCE,SOURCE/'PAYLOAD.json')]:
        for name, expected in json.loads(manifest.read_text())['files'].items():
            assert sha(base/name)==expected, str(base/name)
    import torch
    from torch.utils.data import DataLoader
    from train_separated_edge import loaders, build_model
    from thesis.new_constraints.bands.outer_boundary import OuterBoundaryBandLoss
    from audit_edge_coherence import metrics
    torch.set_num_threads(4)
    device=torch.device('cpu')
    runtime=dict(torch=torch.__version__, device='cpu', precision='FP32', original_precision='CUDA AMP; hard edge/Dice primary metrics remain original saved audit values', threads=4)
    save(OUT/'inference_runtime.json',runtime)
    for arm in ['pooled','separated']:
        for seed in range(3):
            src=RESULTS/f'fold0/seed{seed}/{arm}'
            config=json.loads((src/'config.json').read_text())
            complete=json.loads((src/'completion.json').read_text())
            cp_path=OUT/f'checkpoints/fold0/seed{seed}/{arm}/checkpoint_best.pt'
            assert sha(cp_path)==complete['checkpoint_sha256']
            assert sha(src/'selected_cases.json')==complete['selected_cases_sha256']
            pkl=ROOT/'datasets/Dataset101_MSD/msd_hippocampus_full_cluster_20260906.pkl'
            split=OUT/'splits_5pct_seed0.json'
            assert sha(pkl)==config['pkl_sha256']
            assert sha(split)==config['splits_json_sha256']
            config['pkl'],config['splits_json']=str(pkl),str(split)
            train,val,_,_=loaders(config)
            cp=torch.load(cp_path,map_location='cpu',weights_only=True)
            assert cp['epoch']==complete['selected_epoch']
            assert cp['run']==complete['run']
            model=build_model(config,device)
            model.load_state_dict(cp['model'],strict=True)
            model.eval()
            del cp
            bands=OuterBoundaryBandLoss()
            rows=[]
            original={(r['split'],r['case_name']):r['metrics'] for r in json.loads((src/'selected_cases.json').read_text())}
            started=time.perf_counter()
            for kind,loader in [('train',DataLoader(train.dataset,batch_size=1,shuffle=False)),('validation',val)]:
                for batch in loader:
                    with torch.inference_mode():
                        logits=model(batch['image'])
                        result=bands(logits,batch['label'])
                    p=logits[0].float().softmax(0).numpy()
                    truth=batch['label'][0,0].numpy()
                    fresh=metrics(p[1:].sum(0),p.argmax(0),truth)
                    name=str(batch['case_name'][0])
                    old=original[(kind,name)]
                    bm={k:float(result.details[k][0]) for k in ['inner_loss','outer_loss','case_loss']}
                    bm.update(truth=float(result.truth[0]),inner_truth=float(result.value[0,0]),outer_truth=float(result.value[0,1]),confidence_adherent=bool(result.details['confidence_adherent'][0]),valid=bool(result.details['valid'][0]))
                    rows.append(dict(split=kind,case_name=name,bands=bm,recomputed_edge_metrics=fresh,drift_from_saved={k:fresh[k]-old[k] for k in ['macro_dice','inner_both_correct','outer_both_correct','cross_correct_transition']}))
                    print(json.dumps(dict(arm=arm,seed=seed,case=len(rows),elapsed=time.perf_counter()-started)),flush=True)
                    if args.limit and len(rows)>=args.limit: break
                if args.limit and len(rows)>=args.limit: break
            dest=OUT/(f'benchmark_{arm}_seed{seed}.json' if args.limit else f'bands_{arm}_seed{seed}.json')
            save(dest,dict(arm=arm,seed=seed,selected_epoch=complete['selected_epoch'],checkpoint_sha256=complete['checkpoint_sha256'],runtime=runtime,cases=rows))
            if args.limit: return

if __name__=='__main__': main()
