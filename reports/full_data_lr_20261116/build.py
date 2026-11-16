"""Freeze one all-fold, final-update LR contrast, without selecting runs on validation scores."""
import csv, hashlib, json, math, zipfile
from pathlib import Path

D = Path(__file__).resolve().parent
OLD = D.parent / 'full_data_baselines_20261115'
ROOT = OLD.parents[1]
def hash_json(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()
def lr(i,h,peak,low):
    return peak*(i+1)/20 if i<20 else low+.5*(peak-low)*(1+math.cos(math.pi*(i-20)/max(h-21,1)))
def build():
    plan={}
    for r in csv.DictReader((OLD/'paired_comparisons.csv').open()):
        k,ds,m=r['cell'],r['dataset'],r['model']
        row=dict(dataset=ds,model=m,fold=int(r['partition']),seed=int(r['seed']))
        plan[k]=row
        if r['development_archive']:
            with zipfile.ZipFile(ROOT/r['development_archive']) as a:
                receipt=json.loads(a.read('matched_receipt.json'))
                assert receipt['key']==k and receipt['development_peak_lr']==.001 and receipt['observed']==receipt['eligible']
            row.update(status='reuse',reference=r['development_archive'],peak=.001)
            continue
        if m!='medsam3':
            name=r['native_pass_archive'] or r['paper_matched_archive']
            with zipfile.ZipFile(ROOT/name) as a:
                trace=[json.loads(s) for s in a.read('control_exposure.jsonl').splitlines()]
            case=[x['case'] for x in trace];before=max(x['lr'] for x in trace)
            peak=.001 if m in ('basic_unet16','unetr_vit_b16') else before*(1.25 if ds=='HFH' else 2)
            row.update(status='new',reference=name,training=len(set(case)),updates=len(trace),peak=peak,
                old_peak=before,order_hash=hash_json(case),rates=[x['lr']*(peak/before) for x in trace],
                parent='native_pass' if r['native_pass_archive'] else 'matched')
            continue
        if ds=='MNI':
            p=json.loads((OLD/'mni_adapter/protocols'/(k+'.json')).read_bytes())
            order=json.loads((OLD/'archive'/k/'slab_schedule.json').read_bytes())
        else:
            file={'MSD':'MSD_standard_code.zip','HFH':'cluster_hfh_gain.zip','ADNI':'ADNI_medical_protocols.zip'}[ds]
            with zipfile.ZipFile(OLD/file) as a:
                p=json.loads(a.read('protocols/'+k+('_standard_fold' if ds=='MSD' else '')+'.json'))
            order=p.get('frozen_sample_schedule')
            if ds=='ADNI':
                with zipfile.ZipFile(ROOT/r['paper_matched_archive']) as a:order=json.loads(a.read('slab_schedule.json'))
        assert (p['learning_rate'],p['final_learning_rate'],p['warmup_updates'],p['accumulation'])==(5e-5,5e-6,20,2)
        assert len(set(p['train_cases']))=={'MSD':208,'ADNI':216,'HFH':40,'MNI':40}[ds]
        assert not set(p['train_cases']).intersection(p['evaluation_cases'])
        h=p.get('learning_rate_horizon_updates',p['updates'])
        old=p.get('frozen_learning_rates',[lr(i,h,5e-5,5e-6) for i in range(h)])
        if order:assert len(order)==p['updates']*2 and {c for c,_ in order}==set(p['train_cases'])
        row.update(status='new',peak=.0001,old_peak=.00005,training=len(p['train_cases']),validation=len(p['evaluation_cases']),
                   updates=p['updates'],train_hash=hash_json(p['train_cases']),val_hash=hash_json(p['evaluation_cases']),
                   order_hash=hash_json(order) if order else None,rates=[2*x for x in old[:p['updates']]])
    assert len(plan)==180 and sum(r['status']=='reuse' for r in plan.values())==54
    (D/'plan.json').write_text(json.dumps(plan,separators=(',',':')))
if __name__=='__main__':build()
