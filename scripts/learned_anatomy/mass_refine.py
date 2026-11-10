"""Fit the learned metric against the probability change its gradient produces."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from scripts.depth_supervision.utils import atomic_json, digest, save_checkpoint, seed_all, rng_state, restore_rng
from .prior import PriorTrainingModel, load_training, Corruptions, one_hot, balanced_ce
from .verify import counterfactual_probabilities


def repair_objective(logits, gradient, target, mask, context, kind, swap_margin):
    """A real, unmasked logit step; differentiate its tissue/label changes."""
    probability = logits.softmax(1)
    step = .25 / gradient.detach().abs().max().clamp_min(1e-20)
    updated = (logits-step*gradient).softmax(1)
    change = updated-probability
    desired = target-probability.detach()
    a = change.movedim(1, -1)[mask]
    b = desired.movedim(1, -1)[mask]
    a = a/(a.square().sum(-1, keepdim=True)+1e-20).sqrt()
    b = b/(b.square().sum(-1, keepdim=True)+1e-20).sqrt()
    alignment = (1-(a*b).sum(-1)).mean()
    correct_gain = (change*target).sum(1)[mask].sum()
    # Use the whole real FOV so damage outside the selected patch also counts.
    mass_change = (change[:, 1:].sum(1)*context[:, 1]).sum()
    scale = correct_gain.detach().abs().clamp_min(1e-8)
    if kind == 'swap':
        mass_penalty = F.relu(mass_change.abs()/scale-swap_margin)
    else:
        signed = mass_change if kind == 'missing' else -mass_change
        mass_penalty = F.relu(.1-signed/scale)
    return alignment, mass_penalty, mass_change, correct_gain


def run_mass_refinement(root, p, device):
    root=Path(root); out=root/'encoder'; settings=p['mass_refinement']
    reuse=p['reused_refinement']
    ident=dict(protocol=p['identity_sha'],initial_encoder_sha256=reuse['encoder_sha256'],
               settings=settings,train=p['splits']['5pct']['train'])
    complete=out/'mass_refinement_completion.json'
    if complete.exists():
        saved=json.loads(complete.read_text())
        assert saved['identity']==ident and digest(out/'frozen.pt')==saved['encoder_sha256']
        return
    initial=out/'before_mass_refinement.pt'
    if not initial.exists():shutil.copy2(out/'frozen.pt',initial)
    assert digest(initial)==reuse['encoder_sha256']
    seed_all(settings['seed'])
    patients,contexts=load_training(root,p,device)
    makers=[Corruptions(x,contexts.get(x)) for x in patients.items]
    old=torch.load(out/'refinement_latest.pt',map_location=device,weights_only=False)
    assert old['step']==reuse['steps'] and old['identity']==reuse['identity']
    model=PriorTrainingModel().to(device);model.load_state_dict(old['model']);del old
    model.requires_grad_(False);model.encoder.requires_grad_(True)
    variances=torch.tensor(torch.load(initial,map_location=device,weights_only=False)['feature_variances'],device=device)
    opt=torch.optim.AdamW(model.encoder.parameters(),lr=settings['lr'],weight_decay=1e-5)
    rng=np.random.default_rng(settings['seed']);history=[];start=0
    checkpoint=out/'mass_refinement_latest.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        assert saved['identity']==ident
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];start=saved['step']
        rng.bit_generator.state=saved['numpy_rng'];restore_rng(saved['rng'])
    for step in range(start,settings['steps']):
        i=int(rng.integers(10));item=patients[i];context=contexts.get(item)
        kind=('missing','extra','swap')[step%3]
        wrong,selected=makers[i].make(kind,int(rng.integers(1,3)),float(rng.choice([1.5,3.,4.5])),rng)
        truth=item['label'].to(device).long();y=one_hot(truth)
        wrong=torch.from_numpy(wrong)[None].to(device);mask=torch.from_numpy(selected)[None].to(device)
        confidence=float(rng.uniform(.75,.98))
        base,prob=counterfactual_probabilities(truth,wrong,mask,confidence)
        if step%2:
            w=one_hot(wrong);prob=torch.where(mask[:,None],w*confidence+(1-w)*((1-confidence)/2),base)
        logits=prob.log().detach().requires_grad_(True)
        candidate=logits.softmax(1)
        a=model.encoder(candidate,context);b=model.encoder(y,context)
        terms=[]
        for layer,(x,z) in enumerate(zip(a,b)):
            valid=F.interpolate(context[:,1:2],size=x.shape[2:],mode='nearest')
            terms.append(((x-z).square()*valid).sum()/(valid.sum()*x.shape[1]*variances[layer]))
        learned=torch.stack(terms).mean()
        gradient,=torch.autograd.grad(learned,logits,create_graph=True)
        alignment,mass_penalty,mass,gain=repair_objective(logits,gradient,y,mask,context,kind,settings['swap_margin'])
        error=torch.zeros_like(truth);error[mask]=('missing','extra','swap').index(kind)+1
        heads=model(candidate.detach(),context)
        retained=sum(balanced_ce(h,t,context[:,1:2]) for h,t in zip(heads,(wrong,truth,error)))
        objective=alignment+settings['mass_weight']*mass_penalty+settings['representation_weight']*retained
        opt.zero_grad(set_to_none=True);objective.backward()
        assert torch.isfinite(objective) and all(torch.isfinite(q.grad).all() for q in model.encoder.parameters())
        opt.step()
        history.append(dict(step=step+1,alignment=float(alignment.detach()),mass_penalty=float(mass_penalty.detach()),
            foreground_change=float(mass.detach()),correct_gain=float(gain.detach()),representation=float(retained.detach()),
            case_name=item['case_name'],kind=kind,uncertain=bool(step%2)))
        if (step+1)%50==0 or step+1==settings['steps']:
            save_checkpoint(checkpoint,dict(identity=ident,model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,
                history=history,numpy_rng=rng.bit_generator.state,rng=rng_state()))
            atomic_json(out/'mass_refinement_history.json',history)
            print(json.dumps(dict(stage='probability_gradient_refinement',step=step+1,
                alignment=np.mean([r['alignment'] for r in history[-50:]]),mass_penalty=np.mean([r['mass_penalty'] for r in history[-50:]]))),flush=True)
    save_checkpoint(out/'frozen.pt',dict(identity=ident,encoder=model.encoder.state_dict(),feature_variances=variances.tolist()))
    atomic_json(complete,dict(identity=ident,encoder_sha256=digest(out/'frozen.pt'),steps=len(history),
        train_cases=ident['train'],validation_cases_used=[],correction_teacher='Actual probability repair and tissue-mass direction during encoder fitting only.'))
