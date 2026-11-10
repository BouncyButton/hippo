"""Recover validation background Dice from saved confusion matrices, never pooled counts.

Read-only with respect to source archives and thesis; writes only this report folder.
All Dice values in generated CSVs are fractions in [0,1].
"""
from pathlib import Path
import collections, csv, hashlib, json, math, statistics, zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
RUNS=[]; CASES=[]; ISSUES=[]; MISSING=[]

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p): return list(csv.DictReader(open(p)))
def writecsv(name, rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def extract_archives():
    for campaign,pattern in [('MSD','remote/MSD_archives/*.zip'),('SAM3','remote/SAM3_colab_archives/*.zip')]:
        for p in ROOT.glob(pattern):
            with zipfile.ZipFile(p) as z:
                manifests=[x for x in z.namelist() if x.endswith('archive_manifest.json')]
                if not manifests: continue
                m=json.loads(z.read(manifests[0]));task=m.get('task',{})
                rid=task.get('id') if isinstance(task,dict) else task
                if not rid:
                    rid=next((part for part in Path(manifests[0]).parts if part.startswith('GeoFWI3D_SAM3_')),None)
                if not rid: raise ValueError(('missing task id',p))
                out=ROOT/'extracted'/campaign/rid;out.mkdir(parents=True,exist_ok=True)
                keep={'evaluation.json','validation_latest.json','archive_manifest.json','completion.json','config.json','protocol.json'}
                for member in z.namelist():
                    if Path(member).name in keep:
                        dest=out/Path(member).name
                        b=z.read(member)
                        if dest.exists() and dest.read_bytes()!=b: raise ValueError(('conflicting zip members',dest))
                        dest.write_bytes(b)
                (out/'recovery_receipt.json').write_text(json.dumps({'source':str(p.relative_to(REPO)),'sha256':sha(p)},indent=2)+'\n')

def manifest_check(p):
    mp=p.parent/'archive_manifest.json'
    if not mp.exists(): return 'manifest unavailable'
    m=read(mp);expected=m.get('metadata',{}).get(p.name) or m.get('files',{}).get(p.name)
    if isinstance(expected,dict): expected=expected.get('sha256')
    if not expected: return 'evaluation hash absent from manifest'
    assert sha(p)==expected, ('evaluation hash mismatch',p)
    return 'matched manifest sha256'

def add(campaign,rid,p,fold,seed,model,arm,expected=None,expected_hash=None,extra_control=False):
    try:
        d=read(p);records=d.get('cases',d.get('records',[]))
        records=[c for c in records if c.get('split','validation') in ['validation','val','heldout','test']]
        assert records, ('no validation cases',p)
        digest=sha(p)
        if expected_hash: assert digest==expected_hash, ('reference evaluation hash mismatch',p)
        integrity=manifest_check(p)
        vals=[];seen=set()
        for i,c in enumerate(records):
            cm=c.get('confusion',c.get('confusion_true_rows_predicted_columns'))
            assert cm and len(cm) in [2,3] and all(len(x)==len(cm) for x in cm)
            assert all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 and int(x)==x for row in cm for x in row)
            k=len(cm);dice=[]
            for j in range(k):
                den=sum(cm[j])+sum(row[j] for row in cm)
                dice.append(2*cm[j][j]/den if den else 1.0)
            bg=dice[0];fg=statistics.mean(dice[1:]);allc=statistics.mean(dice)
            saved=c.get('macro_hard_dice',c.get('ap_mean_dice'))
            if saved is not None: assert abs(fg-float(saved))<1e-10, ('per-case foreground mismatch',rid,i)
            if 'background' in c.get('per_class',{}): assert abs(bg-c['per_class']['background']['dice'])<1e-10
            name=c.get('case_name',c.get('case',str(i)))
            assert name not in seen, ('duplicate case',rid,name)
            seen.add(name)
            vals.append({'campaign':campaign,'run_id':rid,'fold':int(fold),'seed':int(seed),'model':model,'arm':arm,'case':name,'n_classes':k,'voxel_count':sum(map(sum,cm)),'foreground_dice':fg,'background_dice':bg,'dice_including_background':allc,'confusion_json':json.dumps(cm,separators=(',',':'))})
        fg=statistics.mean(x['foreground_dice'] for x in vals)
        if expected is not None: assert abs(fg-float(expected))<1e-8, ('reported score mismatch',rid,fg,expected)
        row={'campaign':campaign,'run_id':rid,'fold':int(fold),'seed':int(seed),'model':model,'arm':arm,'validation_cases':len(vals),'foreground_dice':fg,'background_dice':statistics.mean(x['background_dice'] for x in vals),'dice_including_background':statistics.mean(x['dice_including_background'] for x in vals),'extra_matched_control':extra_control,'evaluation_source':str(p.relative_to(REPO)),'evaluation_sha256':digest,'integrity':integrity,'foreground_reference_checked':expected is not None}
        RUNS.append(row);CASES.extend(vals)
        return True
    except Exception as e:
        ISSUES.append({'campaign':campaign,'run_id':rid,'source':str(p),'error':str(e)});return False

def main():
    extract_archives()
    local=read(ROOT/'local_conventional_matches.json')
    for dataset,folder in [('MSD','msd_3fold_3seed_final_20261020'),('ADNI','adni_colab_final_20261019')]:
        for r in csvread(REPO/'reports'/folder/'all_504_runs.csv'):
            rid=r['id'];p=ROOT/('extracted/MSD' if dataset=='MSD' else 'remote/ADNI_conventional')/rid/'validation_latest.json'
            if not p.exists() and rid in local:
                alternatives=[REPO/x for x in local[rid] if (REPO/x).exists()]
                alternatives.sort(key=lambda x: not (x.parent/'archive_manifest.json').exists())
                if alternatives: p=alternatives[0]
            if not p.exists():MISSING.append({'campaign':dataset+'_conventional','run_id':rid,'reason':'per-volume evaluation not located'});continue
            add(dataset+'_conventional',rid,p,r['fold'],r['seed'],r['model'],r['arm'],float(r['dice'])/100)
    for r in read(REPO/'reports/thesis_geofwi_update_20261022/results.json')['rows']:
        archived=r.get('archive');rid=Path(r['source']).name if not archived else Path(archived).name
        p=REPO/archived/'validation_latest.json' if archived else ROOT/'remote/GeoFWI3D_missing_baselines'/rid/'validation_latest.json'
        if not p.exists():MISSING.append({'campaign':'GeoFWI3D_conventional','run_id':rid,'reason':'per-volume evaluation not located'});continue
        add('GeoFWI3D_conventional',rid,p,r['fold'],r['seed'],r['model'],r['arm'],r.get('final_dice',r.get('dice')))
    for r in csvread(REPO/'reports/medsam3_msd_colab_final_20261019/all_108_runs.csv'):
        p=ROOT/'remote/MSD_MedSAM3_extension'/r['id']/'evaluation.json'
        if not p.exists():MISSING.append({'campaign':'MSD_MedSAM3_extension','run_id':r['id'],'reason':'per-volume evaluation not located'});continue
        add('MSD_MedSAM3_extension',r['id'],p,r['fold'],r['seed'],'MedSAM3',r['arm'],r['validation_dice'],r['evaluation_sha256'],r['arm']=='baseline')
    for r in read(REPO/'reports/medsam3_msd_colab_108_3fold_3seed_noaug_20261015/bundle/existing_results.json')['runs']:
        MISSING.append({'campaign':'MSD_MedSAM3_original','run_id':r['id'],'reason':'original foreground results preserved; per-volume confusion matrices not located in searched archives'})
    adniroot=REPO/'reports/adni_medsam3_2sample_3fold_3seed_14arms_20261018/runs'
    for p in sorted(adniroot.glob('*/evaluation.json')):
        m=read(p.parent/'archive_manifest.json');task=m.get('task',{})
        import re
        match=re.search(r'_(dice.*?)_f(\d+)s(\d+)$',p.parent.name)
        assert match, p
        arm,fold,seed=match.groups()
        completion=read(p.parent/'completion.json') if (p.parent/'completion.json').exists() else {}
        assert completion.get('status')=='complete' and completion.get('updates')==200
        add('ADNI_MedSAM3',p.parent.name,p,fold,seed,'MedSAM3',arm,completion['validation_dice'],completion.get('evaluation_sha256'))
    samroot=REPO/'reports/sam3_geofwi_f013_s012_126_20261019'
    state=read(samroot/'split_cluster_20261022/restart_colab_20261022/completion/campaign_state.json')
    for r in read(samroot/'bundle/plan.json')['tasks']:
        p=ROOT/'extracted/SAM3'/r['id']/'evaluation.json'
        if not p.exists():p=samroot/'split_cluster_20261022/cluster_results/runs'/r['id']/'evaluation.json'
        if not p.exists():MISSING.append({'campaign':'GeoFWI3D_SAM3','run_id':r['id'],'reason':'no completed evaluation archived locally at this snapshot'});continue
        completion=read(p.parent/'completion.json')
        assert completion.get('status')=='complete' and completion.get('updates')==25
        assert int(completion['fold'])==r['fold'] and int(completion['seed'])==r['seed'] and completion['arm']==r['arm']
        add('GeoFWI3D_SAM3',r['id'],p,r['fold'],r['seed'],'SAM3',r['arm'],completion['validation_dice'],completion.get('evaluation_sha256'))
    writecsv('background_dice_runs.csv',RUNS);writecsv('background_dice_cases.csv',CASES)
    writecsv('missing_runs.csv',MISSING)
    summaries=[]
    for key in sorted(set((r['campaign'],r['model'],r['arm']) for r in RUNS)):
        rows=[r for r in RUNS if (r['campaign'],r['model'],r['arm'])==key]
        s={'campaign':key[0],'model':key[1],'arm':key[2],'runs':len(rows),'folds':len(set(r['fold'] for r in rows)),'complete_3fold_3seed':len(rows)==9 and len(set((r['fold'],r['seed']) for r in rows))==9 and all(sum(r['fold']==f for r in rows)==3 for f in set(r['fold'] for r in rows))}
        for metric in ['foreground_dice','background_dice','dice_including_background']:
            means=[statistics.mean(r[metric] for r in rows if r['fold']==f) for f in sorted(set(r['fold'] for r in rows))]
            s[metric]=statistics.mean(means);s[metric+'_fold_sd']=statistics.stdev(means) if len(means)>1 else ''
        summaries.append(s)
    writecsv('background_dice_conditions.csv',summaries)
    result={'recovered_runs_including_extra_controls':len(RUNS),'extra_matched_controls':sum(r['extra_matched_control'] for r in RUNS),'recovered_validation_case_records':len(CASES),'coverage':dict(collections.Counter(r['campaign'] for r in RUNS)),'missing_by_campaign':dict(collections.Counter(r['campaign'] for r in MISSING)),'checks':{'saved_per_case_foreground_dice':'checked wherever present','saved_background_dice':'checked wherever present','foreground_run_reference_count':sum(r['foreground_reference_checked'] for r in RUNS),'evaluation_manifest_hashes':dict(collections.Counter(r['integrity'] for r in RUNS))},'issues':ISSUES,'aggregation':'Dice calculated separately for each class and volume, then averaged over classes and volumes. Condition summaries average seeds within fold and then folds equally; SD is across fold means. Incomplete conditions explicitly flagged. No pooled-voxel substitution.'}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
