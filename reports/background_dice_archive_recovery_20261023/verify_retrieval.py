"""Consolidate successful download receipts; retain earlier failed attempts as history."""
from pathlib import Path
import collections,hashlib,json
root=Path(__file__).resolve().parent
success={};attempts=collections.Counter()
for p in sorted(root.glob('*receipt.json')):
    try: records=json.loads(p.read_text())
    except (ValueError,OSError):continue
    if not isinstance(records,list):continue
    for r in records:
        if not isinstance(r,dict) or 'dest' not in r:continue
        attempts[r.get('status','unknown')]+=1
        if r.get('status')=='downloaded': success[r['dest']]=r
problems=[]
for name,r in success.items():
    p=Path(name)
    if not p.exists(): problems.append({'file':name,'error':'missing'});continue
    b=p.read_bytes()
    if hashlib.sha256(b).hexdigest()!=r['sha256'] or len(b)!=r['bytes']:
        problems.append({'file':name,'error':'download receipt mismatch'})
coverage={}
for campaign,file in [('MSD_MedSAM3_extension','evaluation.json'),('ADNI_conventional','validation_latest.json'),('GeoFWI3D_missing_baselines','validation_latest.json')]:
    coverage[campaign]=len(list((root/'remote'/campaign).glob('*/'+file)))
result={'verified_downloaded_files':len(success),'bytes':sum(r['bytes'] for r in success.values()),'evaluation_file_counts':coverage,'historical_download_attempts':dict(attempts),'problems':problems,'note':'Failed initial ADNI downloads used expired temporary export URLs. Successful refreshed receipts supersede those attempts; source data was not changed.'}
(root/'retrieval_verification.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'downloaded_file_inventory.json').write_text(json.dumps(list(success.values()),indent=2)+'\n')
print(json.dumps(result,indent=2))
