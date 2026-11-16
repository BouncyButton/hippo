"""One unconstrained baseline, two full-volume held-out spatial measurements."""
import csv
import fcntl
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from copy import deepcopy
import inspect
import numpy as np
import nibabel as nib
import torch
from torch.nn import functional as F
from spatial_metrics import counts

HERE=Path(os.environ.get('HIPPO_SPATIAL_STAGE',Path(__file__).resolve().parent))
RESULTS=Path(os.environ.get('HIPPO_SPATIAL_RESULTS',HERE/'results'))
BASE=(Path(os.environ.get('HIPPO_CLUSTER_ROOT', Path.home())))
TIMES=(5,50)


def data(ds,case):
    with np.load(HERE/'data'/ds/(case+'.npz')) as a:
        return a['image'].astype('float32'),a['label'].astype('uint8'),(a['orientation'] if 'orientation' in a else None)


def example(ds,case):
    x,y,_=data(ds,case)
    nz=x!=0
    x[nz]=(x[nz]-x[nz].mean())/max(x[nz].std(),1e-8)
    pad=[((64-n)//2,(65-n)//2) for n in y.shape]
    crop=tuple(slice(a,a+n) for (a,_),n in zip(pad,y.shape))
    return torch.from_numpy(np.pad(x,pad))[None,None],torch.from_numpy(np.pad(y,pad))[None,None],crop


def write_checkpoint_metrics(out,t,job,predict,classes):
    rows=[]
    ds=job['dataset']
    affine=np.diag([1.,2. if ds=='HFH' else 1.,1.,1.])
    for c in job['val']:
        truth,pred=predict(c)
        assert truth.shape==pred.shape and np.isin(pred,np.arange(classes)).all()
        row=counts(truth,pred,affine,classes)
        rows.append(dict(model=job['model'],dataset=ds,fold=job['fold'],seed=job['seed'],step=t,
                         case=c,prediction_sha256=hashlib.sha256(pred.astype('uint8').tobytes()).hexdigest(),
                         **row))
    path=out/f'u{t}.csv'
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    print(out.name,t,len(rows),'mean foreground Dice',np.mean([r['dice'] for r in rows]),flush=True)


def scratch(key,job,p,out,device):
    from monai.networks.nets import UNETR,SwinUNETR
    from baselines.simple_segmentation import CANDIDATES
    from scripts.train_compute_matched_baselines import build_model,build_loss_and_optimizer
    ds,model,seed=job['dataset'],job['model'],job['seed']
    classes=p['num_classes']
    if model=='nnunet':
        architecture=json.loads((HERE/'configs'/(key+'_arch.json')).read_text())
        net=build_model('nnunet',architecture)
    elif model=='swinunetr':
        kw=dict(in_channels=1,out_channels=classes,use_checkpoint=False,drop_rate=0.)
        if 'img_size' in inspect.signature(SwinUNETR).parameters:
            kw['img_size']=(64,64,64)
        net=SwinUNETR(**kw)
    else:
        kw=deepcopy(CANDIDATES[model]['kwargs'])
        kw.update(out_channels=classes,img_size=(64,64,64))
        if ds=='MNI':
            kw['dropout_rate']=.1
        net=UNETR(**kw)
    net=net.to(device)
    criterion,opt=build_loss_and_optimizer(p['trainer_alias'],net,p)
    rng=np.random.default_rng(seed)
    training=[example(ds,c) for c in job['train']]
    if ds=='MNI' and model=='unetr_vit_b16':
        from scripts.unetr_subfields.geo_recipe import training_loss as native
    elif ds=='MNI' and model=='swinunetr':
        from scripts.unetr_subfields.log_squared_dice import training_loss as native
    else:
        native=None
    def prediction(c):
        x,y,crop=example(ds,c)
        with torch.inference_mode():
            logits=net(x.to(device))
            if isinstance(logits,(list,tuple)):
                logits=logits[0]
            assert logits.isfinite().all()
            return y[0,0].numpy()[crop],logits.argmax(1)[0].cpu().numpy()[crop]
    order=[]
    trace=[]
    for t in range(1,max(TIMES)+1):
        net.train()
        if not order:
            order=list(rng.permutation(len(training)))
        i=order.pop()
        x,y,_=training[i]
        x,y=x.to(device),y.long().to(device)
        if p['augmentation']=='xy_d4_v1':
            from scripts.unetr_subfields.geo_recipe import augment
            x,y=augment(x,y,seed,t-1)
        opt.zero_grad()
        output=net(x)
        if model=='nnunet':
            targets=[F.interpolate(y.float(),size=o.shape[2:],mode='nearest').long() for o in output]
            loss=criterion(output,targets)
        elif native:
            loss=native(output,y,'dice',{},0)
        else:
            loss=criterion(output,y)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        trace.append(dict(step=t,case=job['train'][i],loss=float(loss.detach()),lr=opt.param_groups[0]['lr']))
        if t in TIMES:
            net.eval()
            write_checkpoint_metrics(out,t,job,prediction,classes)
            (out/'training.json').write_text(json.dumps(trace))


def medical(job,p,out):
    from scripts.cross_dataset_campaign import foundation as e
    from scripts.train_medsam3_few_shot import is_adapter
    from evaluation.medsam3_zero_shot import normalize_image
    from scripts.colab_medsam3.fast_eval import FeatureReuse
    ds=job['dataset']
    p=dict(p,seed=job['seed'],lora_weights=str(BASE/'medsam3_runtime_20260928/weights/best_lora_weights.pt'),
           upstream=str(BASE/'medsam3_runtime_20260928/MedSAM3'),
           lora_sha256='499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91',
           dataset=ds,prompts=p.get('prompts',p.get('class_prompts')))
    assert len(p['classes'])==len(p['prompts'])+1
    classes=len(p['classes'])
    weights=BASE/'sam3_f013_s012_cluster_tail_20261022/bundle/weights/sam3.pt'
    net,processor=e.load_foundation(p,weights)
    params=[]
    for n,par in net.named_parameters():
        par.requires_grad_(is_adapter(n))
        if par.requires_grad:
            params.append(par)
    assert sum(q.numel() for q in params)==18497664
    loss_fn=e.native_loss(net)
    opt=torch.optim.AdamW(params,lr=p['learning_rate'],weight_decay=.01)
    def volume(c):
        x,y,o=data(ds,c)
        if ds=='HFH':
            x,y=x.transpose(0,2,1),y.transpose(0,2,1)
        if ds=='MSD':
            x,y=nib.orientations.apply_orientation(x,o),nib.orientations.apply_orientation(y,o)
        return normalize_image(x.copy()),y.copy(),o
    support={c:volume(c)[:2] for c in job['train']}
    def pixels(x,z):
        return np.round(x[:,:,z].T*255).astype('uint8')
    rng=random.Random(job['seed'])
    trace=[]
    for t in range(1,max(TIMES)+1):
        net.train()
        for par in params:
            par.requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        lr=e.lr_for_update(t-1,p.get('learning_rate_horizon_updates',p['updates']),p['warmup_updates'],
                           p['learning_rate'],p['final_learning_rate'])
        for group in opt.param_groups:
            group['lr']=lr
        draws=[]
        for micro in range(p['accumulation']):
            case=rng.choice(job['train']); x,y=support[case]; anchor=rng.randrange(x.shape[-1])
            draws.append((case,anchor))
            indices=(anchor,(anchor+1)%x.shape[-1])
            planes=[pixels(x,z) for z in indices]
            if ds=='HFH':
                from audit_hfh_augmentation import transform_pair, augmentation_seed
                planes,_=transform_pair(planes,augmentation_seed(p,t-1,micro,case,anchor),'gain')
            for px,z in zip(planes,indices):
                _,loss=e.forward_plane(net,processor,px,p,y[:,:,z].T.copy(),loss_fn)
                assert torch.isfinite(loss)
                (loss/(2*p['accumulation'])).backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        opt.step()
        trace.append(dict(step=t,sampled=draws,lr=lr,gradient_norm=float(norm)))
        if t not in TIMES:
            continue
        net.eval().requires_grad_(False)
        def reference(model,pr,px,labels=None,criterion=None):
            return e.forward_plane(model,pr,px,p,labels,criterion)
        cache=FeatureReuse(net,processor,reference)
        def predict(c):
            x,y,o=volume(c)
            pred=np.empty_like(y)
            with torch.inference_mode():
                for z in range(y.shape[2]):
                    px=pixels(x,z)
                    logp,_=cache(px)
                    if z==0:
                        slow,_=reference(net,processor,px)
                        if not torch.equal(slow.argmax(0),logp.argmax(0)):
                            raise ValueError('Feature cache changed an evaluated mask')
                    assert torch.isfinite(logp).all()
                    pred[:,:,z]=logp.argmax(0).cpu().numpy().T
            if ds=='HFH':
                return y.transpose(0,2,1),pred.transpose(0,2,1)
            if ds=='MSD':
                canon=nib.orientations.axcodes2ornt(('R','A','S'))
                back=nib.orientations.ornt_transform(canon,o)
                return nib.orientations.apply_orientation(y,back),nib.orientations.apply_orientation(pred,back)
            return y,pred
        write_checkpoint_metrics(out,t,job,predict,classes)
        cache.clear()
        (out/'training.json').write_text(json.dumps(trace))


def run_key(key, job, out):
    if (out/'complete.json').is_file():
        return
    p=json.loads((HERE/'configs'/(key+'.json')).read_text())
    torch.manual_seed(job['seed'])
    random.seed(job['seed'])
    np.random.seed(job['seed'])
    torch.set_num_threads(4)
    if job['model']=='medsam3':
        medical(job,p,out)
    else:
        scratch(key,job,p,out,os.environ.get('HIPPO_SPATIAL_DEVICE','cuda'))
    checks={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [out/'u5.csv',out/'u50.csv',out/'training.json']}
    tmp=out/'complete.tmp'
    tmp.write_text(json.dumps(dict(key=key,recipe_sha256=job['recipe_sha256'],checks=checks)))
    tmp.replace(out/'complete.json')


if __name__=='__main__':
    key=sys.argv[1]
    jobs=json.loads((HERE/'jobs.json').read_text())
    out=RESULTS/key
    out.mkdir(exist_ok=True,parents=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_key(key, jobs[key], out)
    ordered=sorted(jobs,key=lambda k:(jobs[k]['fold'],jobs[k]['dataset'],jobs[k]['model']))
    last_in_lane={ordered[::2][-1], ordered[1::2][-1]}
    if key in last_in_lane and 'SLURM_JOB_ID' in os.environ and '--single' not in sys.argv:
        # Give another remaining partition this same GPU after its lane finishes.
        # Exec releases all framework and CUDA state before switching environments.
        os.execv(sys.executable, [sys.executable, str(HERE/'finish_remaining.py')])
