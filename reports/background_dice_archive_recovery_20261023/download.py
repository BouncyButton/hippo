"""Download only metadata exports selected through the authenticated Drive connector."""
import concurrent.futures, hashlib, json, pathlib, sys, urllib.request
manifest=pathlib.Path(sys.argv[1])
items=json.loads(manifest.read_text())
def fetch(x):
    p=pathlib.Path(x['dest']);p.parent.mkdir(parents=True,exist_ok=True)
    record={k:v for k,v in x.items() if k!='url'}
    try:
        b=urllib.request.urlopen(x['url'],timeout=60).read()
        if x.get('size') is not None: assert len(b)==int(x['size']), 'size mismatch'
        digest=hashlib.sha256(b).hexdigest()
        if x.get('sha256'): assert digest==x['sha256'], 'hash mismatch'
        p.write_bytes(b);record.update(status='downloaded',bytes=len(b),sha256=digest)
    except Exception as e: record.update(status='failed',error=type(e).__name__)
    return record
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(fetch,items))
out=pathlib.Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'downloaded':sum(x['status']=='downloaded' for x in results),'failed':sum(x['status']!='downloaded' for x in results),'receipt':str(out)}))
