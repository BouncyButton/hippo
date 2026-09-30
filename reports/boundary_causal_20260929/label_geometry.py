"""Ground-truth-only geometry; no local model inference is performed."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'source/scripts')]
import numpy as np
import torch
from scipy import ndimage as ndi
from train_edge_lowdata import data_arguments
from thesis.new_constraints import train_swinunetr_constraints as trainer
from audit_edge_coherence import geometry

torch.set_num_threads(4)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
config=json.loads((ROOT/'results/seed0/sum/config.json').read_text())
config['pkl']=str(REPO/'datasets/Dataset101_MSD/msd_hippocampus_full.pkl')
config['splits_json']=str(ROOT/'splits_50cases_seed0.json')
for k in ('pkl','splits_json'):assert sha(config[k])==config[k+'_sha256']
train,val,_,_,_=trainer.build_data(data_arguments(config),torch.Generator().manual_seed(0))
old={(r['split'],r['case_name']):r['metrics'] for r in json.loads((ROOT/'results/seed0/sum/selected_cases.json').read_text())}
rows=[]
for split,loader in [('train',torch.utils.data.DataLoader(train.dataset,batch_size=1,shuffle=False)),('validation',val)]:
 for batch in loader:
  t=batch['label'][0,0].numpy().astype(np.uint8);y,inner,outer=geometry(t)
  outer_faces=ap_faces=0
  outer_endpoints=np.zeros_like(y);ap_endpoints=np.zeros_like(y)
  for axis in range(3):
   a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None);a,b=tuple(a),tuple(b)
   om=y[a]!=y[b];am=(t[a]>0)&(t[b]>0)&(t[a]!=t[b])
   outer_faces+=int(om.sum());ap_faces+=int(am.sum())
   outer_endpoints[a]|=om;outer_endpoints[b]|=om;ap_endpoints[a]|=am;ap_endpoints[b]|=am
  name=str(batch['case_name'][0]);o=old[split,name]
  assert outer_faces==o['cross_face_count']
  assert int(y.sum())==o['gt_foreground_voxels']
  assert int(inner.sum())==o['inner_voxel_count'] and int(outer.sum())==o['outer_voxel_count']
  rows.append(dict(split=split,case_name=name,label_sha256=hashlib.sha256(t.tobytes()).hexdigest(),
   outer_faces=outer_faces,ap_faces=ap_faces,outer_endpoints=int(outer_endpoints.sum()),
   ap_endpoints=int(ap_endpoints.sum()),gt_foreground=int(y.sum()),
   inner_shell=int(inner.sum()),outer_shell=int(outer.sum())))
result=dict(scope='GT-only CPU geometry, no prediction inference. Original pickle/split hashes and every GPU-audit foreground/shell/outer-face count match. Exact label array hashes await original-runtime comparison.',
 cases=rows,summary={})
for split in ('train','validation'):
 r=[x for x in rows if x['split']==split]
 result['summary'][split]={k:sum(x[k] for x in r) for k in ['outer_faces','ap_faces','outer_endpoints','ap_endpoints','gt_foreground','inner_shell','outer_shell']}
 result['summary'][split]['case_count']=len(r)
 result['summary'][split]['mean_case_outer_to_ap_face_ratio']=float(np.mean([x['outer_faces']/x['ap_faces'] for x in r]))
(ROOT/'LABEL_GEOMETRY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['summary'],indent=2))
