"""Verify the frozen Dice+CE loss and gradients without training a model."""
import ast,json,os
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/geofwi-audit-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/geofwi-audit-cache')
import torch,monai
from monai.losses import DiceLoss
import torch.nn.functional as F
root=Path(__file__).resolve().parent
source=root/'bundle/source'
torch.set_num_threads(2)
def extract(path,names,ns):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns
ns=dict(torch=torch,DiceLoss=DiceLoss)
extract(source/'thesis/new_constraints/constraint_result.py',{'differentiable_zero'},ns)
extract(source/'scripts/compute_constraint_objectives.py',{'training_loss'},ns)
old=ns['training_loss']
combined=extract(source/'scripts/capped_depth_delay/objectives.py',{'base_arm','full_loss','training_loss'},dict(old_loss=old))
ce=extract(source/'scripts/cross_dataset_campaign/msd_ce_constraints.py',{'training_loss'},dict(F=F,original=SimpleNamespace(**combined)))['training_loss']
checks=[]
for mode in ('mixed','background','foreground'):
    torch.manual_seed(101)
    y=torch.zeros((1,1,8,8,8),dtype=torch.long)
    if mode=='mixed': y[:,:,2:5,3:6,1:4]=1
    if mode=='foreground': y[:]=1
    z=torch.randn(1,2,8,8,8,requires_grad=True)
    actual=ce(z,y,'dice',{},.2)
    prob=z.softmax(1);target=F.one_hot(y[:,0],2).permute(0,4,1,2,3)
    dice=(1-(2*(prob*target).sum((2,3,4))+1e-5)/(prob.sum((2,3,4))+target.sum((2,3,4))+1e-5)).mean()
    cross_entropy=-z.log_softmax(1).gather(1,y).mean()
    expected=dice+cross_entropy
    print(mode, 'losses:', float(actual.detach()),float(expected.detach()),flush=True)
    assert torch.allclose(actual,expected,atol=4*torch.finfo(torch.float32).eps,rtol=0)
    ga=torch.autograd.grad(actual,z,retain_graph=True)[0]
    ge=torch.autograd.grad(expected,z)[0]
    assert torch.allclose(ga,ge,atol=1e-8,rtol=0) and torch.isfinite(ga).all()
    checks.append(dict(labels=mode,loss=float(actual.detach()),max_gradient_error=float((ga-ge).abs().max())))
report=dict(status='passed',device='cpu',torch=torch.__version__,monai=monai.__version__,checks=checks,
    optimizer_updates=0,limitation='CPU loss checks only; CUDA BF16 and resume preflight run in Colab before training')
(root/'cpu_loss_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
