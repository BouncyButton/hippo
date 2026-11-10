"""Small CPU loss/metric/loader checks; no optimizer updates or research training."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/geofwi-audit-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/geofwi-audit-cache')
import numpy as np
import torch
import monai
from monai.losses import DiceLoss
from monai.networks.nets import UNETR

HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/'geofwi3d_colab_seed_search_20261018/bundle'
SOURCE=BUNDLE/'source'
torch.set_num_threads(2)
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def extract(path,name,namespace):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

ns=dict(torch=torch,DiceLoss=DiceLoss)
extract(SOURCE/'thesis/new_constraints/constraint_result.py','differentiable_zero',ns)
loss=extract(SOURCE/'scripts/compute_constraint_objectives.py','training_loss',ns)
torch.manual_seed(0)
y=torch.zeros((1,1,8,8,8),dtype=torch.long);y[:,:,2:5,3:6,1:4]=1
z=torch.randn(1,2,8,8,8,requires_grad=True)
actual=loss(z,y,'dice',{},.2)
prob=z.softmax(1);target=torch.nn.functional.one_hot(y[:,0],2).permute(0,4,1,2,3)
manual=(1-(2*(prob*target).sum((2,3,4))+1e-5)/(prob.sum((2,3,4))+target.sum((2,3,4))+1e-5)).mean()
assert torch.allclose(actual,manual,atol=1e-7,rtol=0)
ga=torch.autograd.grad(actual,z,retain_graph=True)[0];gm=torch.autograd.grad(manual,z)[0]
assert torch.allclose(ga,gm,atol=1e-8,rtol=0) and torch.isfinite(ga).all() and ga.abs().sum()>0
oracle=torch.cat([(1-y).float(),y.float()],dim=1)*40-20
perfect=float(loss(oracle,y,'dice',{},.2));inverted=float(loss(oracle.flip(1),y,'dice',{},.2))
assert perfect<1e-6 and inverted>.99
metric=load_module(SOURCE/'scripts/cross_dataset_campaign/common.py','audit_common').confusion_metrics
assert metric([[400,30],[20,62]],['background','salt'])['macro_hard_dice']==124/174
data=load_module(SOURCE/'scripts/geofwi3d_data.py','audit_data_module')
manifest=json.loads((BUNDLE/'data/fold_0_manifest.json').read_text());splits=json.loads((BUNDLE/'splits.json').read_text())
for name in [splits[0]['train'][0],splits[0]['val'][0],splits[2]['val'][-1]]:
    x,label=data.load_case(BUNDLE/'data',manifest['cases'][name])
    assert x.dtype==np.float32 and label.dtype==np.uint8 and int(label.sum())==manifest['cases'][name]['salt_voxels']
lr=extract(SOURCE/'scripts/train_compute_matched_baselines.py','learning_rate',dict(math=math))
curve=[lr('unet',step,477,1e-4) for step in range(414)]
assert np.isfinite(curve).all() and min(curve)>0 and max(curve)==1e-4
records=[]
for path in (HERE/'evidence').glob('GeoFWI3D_unetr*/metrics.csv'):
    import csv
    for row in csv.DictReader(path.open()):
        assert abs(float(row['learning_rate'])-curve[int(row['step'])-1])<1e-15
        records.append(row)
tree=ast.parse((SOURCE/'baselines/simple_segmentation.py').read_text())
candidates=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CANDIDATES' for t in n.targets)))
kwargs=candidates['unetr_vit_b16']['kwargs'].copy();kwargs.update(out_channels=2,img_size=(96,96,96))
model=UNETR(**kwargs)
params=sum(p.numel() for p in model.parameters())
assert params==92802274
assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
assert all(m.p==0 for m in model.modules() if isinstance(m,torch.nn.Dropout))
assert all(not m.track_running_stats for m in model.modules() if isinstance(m,torch.nn.modules.instancenorm._InstanceNorm))
report=dict(torch=torch.__version__,monai=monai.__version__,device='cpu',
    manual_dice_value_matches=True,manual_dice_gradient_matches=True,perfect_mask_loss=perfect,
    inverted_mask_loss=inverted,confusion_metric_check=True,actual_loader_examples=3,
    lr_records_recomputed=len(records),lr_warmup_updates=24,lr_first=curve[0],lr_final=curve[-1],
    actual_model_class=type(model).__name__,parameters=params,model_kwargs=kwargs,
    no_batchnorm_no_dropout_no_instance_running_stats=True,
    limitation=f'CPU checks used torch {torch.__version__} / MONAI {monai.__version__}. They do not reproduce the training environment torch 2.12.1 CUDA/BF16 arithmetic.')
(HERE/'cpu_contract_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
