"""Train the encoder to supply useful loss gradients, retaining its learned features."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from scripts.depth_supervision.utils import atomic_json,digest,save_checkpoint,seed_all,rng_state,restore_rng
from .prior import PriorTrainingModel,load_training,Corruptions,one_hot,balanced_ce
from .verify import counterfactual_probabilities


def run_refinement(root, p, device):
    root=Path(root);out=root/'encoder';settings=p['refinement']
    ident=dict(protocol=p['identity_sha'],initial_encoder_sha256=p['reused_encoder']['sha256'],settings=settings,train=p['splits']['5pct']['train'])
    complete=out/'refinement_completion.json'
    if complete.exists():
        saved=json.loads(complete.read_text());assert saved['identity']==ident and digest(out/'frozen.pt')==saved['encoder_sha256'];return
    initial=out/'initial_frozen.pt'
    if not initial.exists():shutil.copy2(out/'frozen.pt',initial)
    assert digest(initial)==ident['initial_encoder_sha256']
    seed_all(settings['seed'])
    patients,contexts=load_training(root,p,device)
    makers=[Corruptions(x,contexts.get(x)) for x in patients.items]
    old=torch.load(out/'latest.pt',map_location=device,weights_only=False)
    assert old['step']==p['encoder']['steps'] and old['identity']==p['reused_encoder']['identity']
    model=PriorTrainingModel().to(device);model.load_state_dict(old['model']);del old
    model.requires_grad_(False);model.encoder.requires_grad_(True)
    variances=torch.tensor(torch.load(initial,map_location=device,weights_only=False)['feature_variances'],device=device)
    opt=torch.optim.AdamW(model.encoder.parameters(),lr=settings['lr'],weight_decay=1e-5)
    rng=np.random.default_rng(settings['seed']);history=[];start=0
    checkpoint=out/'refinement_latest.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False);assert saved['identity']==ident
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);history=saved['history'];start=saved['step']
        rng.bit_generator.state=saved['numpy_rng'];restore_rng(saved['rng'])
    for step in range(start,settings['steps']):
        i=int(rng.integers(10));item=patients[i];context=contexts.get(item);kind=('missing','extra','swap')[step%3]
        wrong,selected=makers[i].make(kind,int(rng.integers(1,3)),float(rng.choice([1.5,3.,4.5])),rng)
        truth=item['label'].to(device).long();y=one_hot(truth)
        wrong=torch.from_numpy(wrong)[None].to(device);mask=torch.from_numpy(selected)[None].to(device)
        confidence=float(rng.uniform(.75,.98))
        base,prob=counterfactual_probabilities(truth,wrong,mask,confidence)
        if step%2:
            w=one_hot(wrong);prob=torch.where(mask[:,None],w*confidence+(1-w)*((1-confidence)/2),base)
        logits=prob.log().detach().requires_grad_(True)
        candidate=logits.softmax(1)
        # Both encoder paths participate in encoder fitting; no stop-gradient here.
        a=model.encoder(candidate,context);b=model.encoder(y,context)
        terms=[]
        for layer,(x,z) in enumerate(zip(a,b)):
            valid=F.interpolate(context[:,1:2],size=x.shape[2:],mode='nearest')
            terms.append(((x-z).square()*valid).sum()/(valid.sum()*x.shape[1]*variances[layer]))
        learned=torch.stack(terms).mean()
        g,=torch.autograd.grad(learned,logits,create_graph=True)
        # CE supplies an independently supervised correction direction, not an
        # extra segmentation objective. Fit directions rather than magnitudes.
        teacher=(candidate.detach()-y)
        a=g.movedim(1,-1)[mask];b=teacher.movedim(1,-1)[mask]
        a=a/(a.square().sum(-1,keepdim=True)+1e-20).sqrt()
        b=b/(b.square().sum(-1,keepdim=True)+1e-20).sqrt()
        alignment=(1-(a*b).sum(-1)).mean()
        heads=model(candidate.detach(),context)
        error=torch.zeros_like(truth);error[mask]=('missing','extra','swap').index(kind)+1
        retained=sum(balanced_ce(h,t,context[:,1:2]) for h,t in zip(heads,(wrong,truth,error)))
        objective=alignment+settings['representation_weight']*retained
        opt.zero_grad(set_to_none=True);objective.backward()
        assert torch.isfinite(objective) and all(torch.isfinite(q.grad).all() for q in model.encoder.parameters())
        opt.step()
        history.append(dict(step=step+1,alignment=float(alignment.detach()),representation=float(retained.detach()),case_name=item['case_name'],kind=kind))
        if (step+1)%50==0 or step+1==settings['steps']:
            save_checkpoint(checkpoint,dict(identity=ident,model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,
                history=history,numpy_rng=rng.bit_generator.state,rng=rng_state()))
            atomic_json(out/'refinement_history.json',history)
            print(json.dumps(dict(stage='gradient_refinement',step=step+1,alignment=np.mean([r['alignment'] for r in history[-50:]]))),flush=True)
    # Retain the initial scale constants used by the fitted gradient objective.
    save_checkpoint(out/'frozen.pt',dict(identity=ident,encoder=model.encoder.state_dict(),feature_variances=variances.tolist()))
    result=dict(identity=ident,encoder_sha256=digest(out/'frozen.pt'),steps=len(history),train_cases=ident['train'],validation_cases_used=[],
        correction_teacher='Cross-entropy logit-gradient direction on independently generated training corruptions; not used as a Swin auxiliary loss.')
    atomic_json(complete,result)
