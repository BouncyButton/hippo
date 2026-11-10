"""Missing SAM3 constraint matrix with adapted calibration and fixed run budgets."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

PREFIX = 5
UPDATES = 25
MONITOR_STEPS = (10, 25)


ARMS = ('dice_sdf', 'dice_bands_sdf', 'dice_bands_edge_sdf', 'dice_depth',
        'dice_bands_depth', 'dice_edge_depth', 'dice_sdf_depth', 'dice_bands_edge_depth',
        'dice_bands_sdf_depth', 'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth')
TERMS = ('bands', 'edge', 'sdf', 'depth')


def names_for(arm):
    if arm not in ARMS:
        raise ValueError('Only the11 missing established arms are allowed')
    return ['native']+[n for n in TERMS if n in arm.split('_')]


def choose_weights(rows, names, bands):
    """Sequential original median/p95 rule; retain bands only when this arm includes it."""
    coeff=np.zeros(len(names));coeff[0]=1.
    weights={n:0. for n in TERMS};stats={}
    for i,name in enumerate(names[1:],1):
        if name=='bands':
            weights[name]=bands;coeff[i]=bands
            stats[name]=dict(policy='saved fixed bands coefficient',weight=bands)
            continue
        ratios=[]
        for row in rows:
            gram=np.asarray(row['gram'],dtype=float)
            if gram.shape!=(len(names),len(names)) or not np.isfinite(gram).all():
                raise ValueError('Invalid calibration Gram matrix')
            denom=float(coeff@gram@coeff)
            if denom>0 and gram[i,i]>0:ratios.append(float(np.sqrt(gram[i,i]/denom)))
        if len(ratios)<5:raise ValueError('Too few nonzero gradients: '+name)
        weight=float(min(.1/np.median(ratios),.5/np.quantile(ratios,.95)))
        if not np.isfinite(weight) or weight<=0:raise ValueError('Invalid weight: '+name)
        weights[name]=weight;coeff[i]=weight
        stats[name]=dict(valid=len(ratios),target=.1,p95_cap=.5,
            weighted_median=float(weight*np.median(ratios)),weighted_p95=float(weight*np.quantile(ratios,.95)))
    return weights,stats


def ratio_metrics(gram,names,weights,ramp=1.):
    """Report each term against preceding terms AND all auxiliaries against native."""
    gram=np.asarray(gram,dtype=float)
    if not np.isfinite(gram).all() or gram[0,0]<=0:raise ValueError('Invalid monitoring Gram')
    coeff=np.zeros(len(names));coeff[0]=1.;result={}
    for i,name in enumerate(names[1:],1):
        base=float(coeff@gram@coeff);edge=max(float(gram[i,i]),0.)
        if base<=0:raise ValueError('Nonpositive sequential reference norm')
        result[name]=dict(relative_norm=float(ramp*weights[name]*np.sqrt(edge/base)),
            cosine=float(coeff@gram[:,i]/np.sqrt(base*edge)) if edge>0 else None)
        coeff[i]=ramp*weights[name]
    auxiliary=coeff.copy();auxiliary[0]=0.
    result['combined_auxiliary_over_native']=float(np.sqrt(max(float(auxiliary@gram@auxiliary),0.)/gram[0,0]))
    return result


def probe(original, diag, model, criterion, parameters, slabs, item, weights, step):
    """Selected loss gradients from one forward; restore RNG, buffers and mode."""
    with diag.preserve_probe_state(model):
        original.reset_seed(item['seed'])
        losses = original.forward_slab(model, criterion, slabs, item['case'], item['anchor'])
        names = names_for(slabs.protocol['arm'])
        gradients = [torch.autograd.grad(losses[n], parameters, retain_graph=i < len(names)-1,
                                        allow_unused=True) for i, n in enumerate(names)]
        gram = diag.gram_matrix(gradients)
        return dict(**item, update=step, gram=gram,
                    losses={n:float(losses[n].detach()) for n in names},
                    full_strength=ratio_metrics(gram, names, weights),
                    scheduled_strength=ratio_metrics(gram, names, weights, min((step+1)/20., 1.)))


def train(original, diag, rt, model, criterion, parameters, slabs, p, shared,
          bands, output, recovery, writer, durable):
    names=names_for(p['arm'])
    schedule = slabs.schedule(p)
    rt.write(output/'slab_schedule.json', schedule)
    original.reset_seed(p['seed']); model.train()
    optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], weight_decay=.01)
    frozen = original.parameter_digest(model, False)
    initial = original.parameter_digest(model, True)
    latest = output/'checkpoint_latest.pt'
    step, history = 0, []
    calibration = dict(status='awaiting_update5', names=names,
                       weights={n:(bands if n=='bands' and n in names else 0.) for n in TERMS},
                       prefix_updates=PREFIX, target=.1, p95_cap=.5, monitoring={})
    elapsed = 0.
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if (state['protocol_sha256'] != rt.digest(output/'protocol.json') or
                state['frozen_before'] != frozen or state['adapter_before'] != initial or
                state['step'] != rt.read(output/'checkpoint_ledger.json')['step'] or
                len(state['history']) != state['step']):
            raise ValueError('Recovery state differs from this run')
        model.load_state_dict(state['adapter'], strict=False)
        optimizer.load_state_dict(state['optimizer'])
        step, history, calibration = state['step'], state['history'], state['calibration']
        elapsed = state['elapsed_seconds']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    evaluation_resume = step == UPDATES and str(UPDATES) in calibration['monitoring']
    began = time.monotonic()

    def commit(force=False):
        rt.write(output/'calibration.json', calibration)
        original.save_checkpoint(latest, dict(step=step, history=history, calibration=calibration,
            adapter={n:v.detach().cpu() for n,v in model.state_dict().items() if original.is_adapter(n)},
            optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            protocol_sha256=rt.digest(output/'protocol.json'), frozen_before=frozen, adapter_before=initial,
            elapsed_seconds=elapsed+time.monotonic()-began))
        rt.write(output/'checkpoint_ledger.json', dict(step=step, protocol_sha256=rt.digest(output/'protocol.json')))
        rt.write(output/'training_history.json', history)
        writer.capture(force=force)

    if not latest.exists():
        commit(force=True)
    probes = [dict(case=r['case'], anchor=r['anchor'], seed=p['calibration_seed']+i)
              for i,r in enumerate(shared['rows'])]
    if len(probes) != 20:
        raise ValueError('Expected original20 calibration slabs')
    selected = [next(item for i,item in enumerate(probes)
                     if item['case']==case and sum(x['case']==case for x in probes[:i])==1)
                for case in p['train_cases']]

    def collect(items, label):
        path = durable/'probes'/output.name/f'{label}.json'
        identity = dict(adapter=original.parameter_digest(model, True), step=step,
                        protocol_sha256=rt.digest(output/'protocol.json'), probes=items)
        saved = rt.read(path) if path.exists() else dict(identity=identity, rows=[])
        if saved['identity'] != identity:
            raise ValueError('Probe recovery belongs to another model state')
        for item in items[len(saved['rows']):]:
            if original.stopping():
                commit(force=True); raise SystemExit(75)
            saved['rows'].append(probe(original,diag,model,criterion,parameters,slabs,item,calibration['weights'],step))
            rt.write(path,saved)
            print(f"PROBE {label}: {len(saved['rows'])}/{len(items)}", flush=True)
        if original.parameter_digest(model,True) != identity['adapter']:
            raise ValueError('Probe modified model parameters')
        return saved['rows']

    while step <= UPDATES:
        if original.stopping():
            commit(force=True); raise SystemExit(75)
        if step == PREFIX and calibration['status'] != 'complete':
            rows = collect(probes, 'calibration_update5')
            weights, stats = choose_weights(rows, names, bands)
            calibration.update(status='complete', weights=weights,
                               rows=rows, statistics=stats,
                               adapter_at_calibration=original.parameter_digest(model,True),
                               scope='20 original support slabs after5 prefix updates; bands fixed when present, new terms sequentially calibrated')
            # Recompute ratios with calibrated weights instead of the prefix's zero weights.
            for row in rows:
                row['full_strength'] = ratio_metrics(row['gram'],names,calibration['weights'])
                row['scheduled_strength'] = ratio_metrics(row['gram'],names,calibration['weights'],.3)
            commit(force=True)
            print('CALIBRATED CONSTRAINTS',json.dumps(dict(weights=calibration['weights'],statistics=stats)),flush=True)
        if step in MONITOR_STEPS and str(step) not in calibration['monitoring']:
            calibration['monitoring'][str(step)] = collect(selected,f'monitor_update{step}')
            commit(force=True)
            ratios=[r['full_strength']['combined_auxiliary_over_native'] for r in calibration['monitoring'][str(step)]]
            print('COMBINED AUXILIARY MONITOR',json.dumps(dict(update=step,median=float(np.median(ratios)),maximum=max(ratios))),flush=True)
        if step == UPDATES:
            break
        charged = dict(step=step+1,protocol_sha256=rt.digest(output/'protocol.json'))
        rt.write(recovery/'charged_ledger.json', charged)
        if rt.read(recovery/'charged_ledger.json') != charged:
            raise IOError('Charge readback failed')
        lr=original.lr_for_update(step,200,p['warmup_updates'],p['learning_rate'],p['final_learning_rate'])
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        ramp=min((step+1)/p['constraint_warmup_updates'],1.)
        weights=calibration['weights']; values=[]
        for micro in range(p['accumulation']):
            case,anchor=schedule[step*p['accumulation']+micro]
            losses=original.forward_slab(model,criterion,slabs,case,anchor)
            total=losses['native']+ramp*sum(weights[n]*losses[n] for n in TERMS)
            if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
            (total/p['accumulation']).backward()
            values.append({**{n:float(v.detach()) for n,v in losses.items()},'total':float(total.detach())})
            del losses,total
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step();step+=1
        history.append(dict(update=step,lr=lr,ramp=ramp,gradient_norm=float(norm),
                            weights=dict(weights),
                            **{n:float(np.mean([v[n] for v in values])) for n in values[0]}))
        commit();print(json.dumps(history[-1]),flush=True)
    if not evaluation_resume:
        commit(force=True)
    else:
        # Preserve the evaluated checkpoint hash for partial-volume/audit resume.
        rt.write(output/'calibration.json',calibration)
        rt.write(output/'training_history.json',history)
    frozen_after=original.parameter_digest(model,False); final=original.parameter_digest(model,True)
    if frozen_after != frozen or final == initial:raise ValueError('Frozen base changed or adapter did not learn')
    rt.write(output/'training_summary.json',dict(updates=UPDATES,frozen_before=frozen,frozen_after=frozen_after,
        adapter_before=initial,adapter_after=final,weights=calibration['weights'],
        schedule_sha256=rt.digest(output/'slab_schedule.json'),augmentation='none',
        learning_rate_horizon_updates=200,elapsed_seconds=elapsed+time.monotonic()-began))


def tasks_for(template):
    return [dict(template,id=f'GeoFWI3D_SAM3_5vol_{arm}_u25_f0s0_warm5_missing_v1',
                 arm=arm,scientific_arm=arm,bands_target=.1 if 'bands' in arm else None,
                 edge_target=.1 if 'edge' in arm else None,
                 auxiliary_targets={n:.1 for n in TERMS if n in arm.split('_')}) for arm in ARMS]


def context(rt,root,durable,args,diag):
    plan=rt.verify_bundle(root);binding=rt.binding(root,durable)
    if binding!=rt.read(args.source_results/'environment_binding.json'):
        raise ValueError('Original GPU/software differs')
    shared=rt.fold_calibration(root,args.source_results,0)
    old=next(t for t in plan['tasks'] if t['scientific_arm']=='dice_bands_edge')
    band_task=next(t for t in plan['tasks'] if t['scientific_arm']=='dice_bands')
    for item in plan['tasks']:
        if not rt.completed(root,args.source_results,item):raise ValueError('Original constraints are incomplete')
    warm_task={**old,'id':old['id']+'_warm5_edge_v1'}
    if not rt.completed(root,args.warm_results,warm_task):raise ValueError('Adapted edge reference is incomplete')
    if rt.read(args.warm_results/'environment_binding.json')!=binding:raise ValueError('Warm reference environment differs')
    baseline,records=rt.previous_baseline(root,durable,args.previous_results)
    bands=rt.read(args.source_results/'runs'/band_task['id']/'calibration.json')['weights']['bands']
    tasks=tasks_for(old)
    prototype=rt.protocol(root,old)
    if (prototype['dataset'],prototype['fold'],prototype['seed'],prototype['updates'],
            prototype['constraint_warmup_updates'])!=('GeoFWI3D',0,0,25,20):
        raise ValueError('Only original fold0/seed0 and25-update scope is supported')
    identity=dict(tasks=tasks,binding=binding,prototype=prototype,bands_weight=bands,
        prefix_updates=PREFIX,monitor_steps=list(MONITOR_STEPS),target=.1,p95_cap=.5,
        extension_sha256=rt.digest(Path(__file__)),probe_source_sha256=rt.digest(Path(diag.__file__)),
        calibration_sha256=rt.digest(args.source_results/'calibration/fold0/calibration.json'),
        reference_folders=[str(args.previous_results),str(args.source_results),str(args.warm_results)],
        policy='Fixed bands when present; otherwise native prefix. Sequential adapted calibration for remaining terms. No online cap or validation-driven tuning.')
    if (durable/'identity.json').exists() and rt.read(durable/'identity.json')!=identity:
        raise ValueError('Campaign identity changed; preserve this result folder')
    rt.write(durable/'identity.json',identity)
    for file in (Path(__file__),Path(diag.__file__)):rt.copy_verified(file,durable/'source'/file.name)
    references=[('Saved native baseline',baseline,records)]
    for item in plan['tasks']:
        folder=args.source_results/'runs'/item['id']
        references.append(('Saved '+item['scientific_arm'],rt.read(folder/'completion.json'),rt.read(folder/'evaluation.json')['records']))
    folder=args.warm_results/'runs'/warm_task['id']
    references.append(('Saved adapted-calibration bands+edge',rt.read(folder/'completion.json'),rt.read(folder/'evaluation.json')['records']))
    return dict(tasks=tasks,prototype=prototype,shared=shared,bands=bands,identity=identity,references=references)


def report(rt,root,durable,ctx):
    entries=list(ctx['references'])
    for task in ctx['tasks']:
        if rt.completed(root,durable,task):
            folder=durable/'runs'/task['id']
            entries.append((task['scientific_arm'],rt.read(folder/'completion.json'),rt.read(folder/'evaluation.json')['records']))
    bands=next(e[2] for e in entries if e[0]=='Saved dice_bands')
    reference={r['case_name']:r for r in bands if r['split']=='validation'}
    rows=[]
    for label,done,records in entries:
        val=[r for r in records if r['split']=='validation']
        if len(val)!=96 or {r['case_name']:r['truth_sha256'] for r in val}!={k:r['truth_sha256'] for k,r in reference.items()}:
            raise ValueError('Validation cohorts or truth hashes differ')
        delta=[r['macro_hard_dice']-reference[r['case_name']]['macro_hard_dice'] for r in val]
        rows.append(dict(run=label,reused=label.startswith('Saved '),validation_dice=done['validation_dice'],
            training_dice=done['training_dice'],foreground_fn=done['foreground_fn'],foreground_fp=done['foreground_fp'],
            delta_vs_bands_pp=100*float(np.mean(delta)),wins_vs_bands=sum(x>0 for x in delta),losses_vs_bands=sum(x<0 for x in delta)))
    result=dict(rows=rows,new_completed=len(entries)-4,new_total=len(ARMS),reused_runs=4,validation_volumes=96,
        interpretation='One seed/fold exploration; report every arm including negative results. Existing controls have earlier calibration policies; this is not a fully matched factorial comparison.')
    rt.write(durable/'comparison.json',result)
    return result


def run_one(rt,root,durable,args,ctx,task,diag):
    import engine
    from parallel import restore_supplement,save_supplement
    p={**ctx['prototype'],'arm':task['scientific_arm'],
       'requested_arms':list(ARMS),'constraint_calibration_after_updates':PREFIX,
       'gradient_ratio_target':.1,'gradient_ratio_p95_cap':.5,
       'gradient_ratio_targets':{n:.1 for n in TERMS if n in task['arm'].split('_')},
       'gradient_ratio_p95_caps':{n:.5 for n in TERMS if n in task['arm'].split('_')},
       'fixed_bands_weight':ctx['bands'] if 'bands' in task['arm'] else 0.,
       'prefix_objective':'native+bands' if 'bands' in task['arm'] else 'native',
       'monitor_updates':list(MONITOR_STEPS),
       'extension_sha256':ctx['identity']['extension_sha256'],
       'probe_source_sha256':ctx['identity']['probe_source_sha256']}
    if rt.completed(root,durable,task):return True
    rt.quota(durable)
    output=root/'work'/task['id'];recovery=durable/'recovery'/task['id']
    # Only restore when no local checkpoint AND no already-complete local archive awaiting cleanup.
    if not (output/'checkpoint_latest.pt').exists() and not (output/'completion.json').exists():
        rt.restore(output,recovery)
    output.mkdir(parents=True,exist_ok=True);restore_supplement(rt,root,durable,task)
    if (output/'protocol.json').exists() and rt.read(output/'protocol.json')!=p:
        raise ValueError('Resume protocol changed')
    rt.write(output/'protocol.json',p);rt.check_ledger(output,recovery)
    if not (output/'completion.json').exists():
        original,model,processor,parameters,criterion,volumes,slabs=engine.load(root,p)
        shared=ctx['shared']
        if shared['initialization']!=dict(frozen=original.parameter_digest(model,False),adapter=original.parameter_digest(model,True)) or slabs.inventory!=shared['inventory']:
            raise ValueError('Original initialization/support differs')
        # Catch silent zero SDF/depth fields before training any update.
        for term in ('sdf','depth'):
            if term in p['arm']:
                fields=getattr(slabs,term)
                if any(not torch.isfinite(v).all() or not torch.any(v!=0) for v in fields.values()):
                    raise ValueError('Missing/nonfinite geometry fields: '+term)
        rt.write(output/'training_inventory.json',slabs.inventory)
        rt.write(output/'runtime_recipe.json',dict(binding=ctx['identity']['binding'],updates=UPDATES,
            prefix_updates=PREFIX,prefix_objective=p['prefix_objective'],calibration_slabs=20,
            monitor_updates=list(MONITOR_STEPS),learning_rate_horizon_updates=200,augmentation='none',
            extension_sha256=p['extension_sha256'],probe_source_sha256=p['probe_source_sha256']))
        writer=rt.RecoveryWriter(output,recovery)
        try:
            with (output/'train.log').open('a') as log:
                console=sys.stdout
                class Tee:
                    def write(self,value):console.write(value);log.write(value);log.flush()
                    def flush(self):console.flush();log.flush()
                with redirect_stdout(Tee()):
                    train(original,diag,rt,model,criterion,parameters,slabs,p,shared,ctx['bands'],output,recovery,writer,durable)
                    os.environ['HIPPO_AUDITED_FAST_EVAL']='1'
                    result=engine.evaluate(original,model,processor,volumes,p,output,root,durable)
                    import matplotlib;matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    history=rt.read(output/'training_history.json')
                    fig,ax=plt.subplots()
                    for name in names_for(p['arm']):
                        ax.plot([r['update'] for r in history],[r[name] for r in history],label=name)
                    ax.axvline(5,color='gray',linestyle='--');ax.set_xlabel('Update');ax.legend()
                    fig.tight_layout();fig.savefig(output/'curves.png',dpi=140);plt.close(fig)
                    val=[r for r in result['records'] if r['split']=='validation']
                    rt.write(output/'completion.json',dict(status='complete',updates=UPDATES,
                        validation_dice=result['mean_dice']['validation'],training_dice=result['mean_dice']['training_support'],
                        foreground_fn=sum(r['foreground_fn'] for r in val),foreground_fp=sum(r['foreground_fp'] for r in val),
                        checkpoint_sha256=rt.digest(output/'checkpoint_latest.pt'),primary_checkpoint='final_update',
                        evaluation_sha256=rt.digest(output/'evaluation.json'),protocol_sha256=rt.digest(output/'protocol.json')))
        except SystemExit as error:
            if error.code!=75:raise
            print('Stopped cleanly; rerun the notebook to resume this arm.',flush=True);return False
        finally:
            writer.close();save_supplement(rt,root,durable,task)
    rt.archive_run(root,output,durable,recovery,task)
    print('ARCHIVED',task['arm'],'and removed its completed checkpoints.',flush=True)
    return True


def supervise(rt,root,durable,args,ctx,stopping):
    tasks=sorted(enumerate(ctx['tasks']),key=lambda pair:(
        not (root/'work'/pair[1]['id']/'checkpoint_latest.pt').exists()
        and not (durable/'recovery'/pair[1]['id']/'latest.json').exists(),pair[0]))
    for _,task in tasks:
        if stopping():break
        if rt.completed(root,durable,task):continue
        print('RUN',task['arm'],'—25 updates, one final evaluation',flush=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--bundle',str(root),
            '--source-results',str(args.source_results),'--previous-results',str(args.previous_results),
            '--warm-results',str(args.warm_results),'--durable',str(durable),'--arm',task['arm']]
        child=subprocess.Popen(command)
        requested=False
        while child.poll() is None:
            if stopping() and not requested:
                child.send_signal(signal.SIGINT);requested=True
            time.sleep(1)
        if child.returncode==75:break
        if child.returncode!=0:raise RuntimeError('Arm failed; recovery retained: '+task['arm'])
        result=report(rt,root,durable,ctx)
        print(f"Completed {result['new_completed']}/{result['new_total']} missing arms",flush=True)
    return report(rt,root,durable,ctx)


def main():
    parser=argparse.ArgumentParser()
    for name in ('bundle','source-results','previous-results','warm-results','durable'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,help=argparse.SUPPRESS)
    args=parser.parse_args();root=args.bundle.resolve();durable=args.durable.resolve()
    sys.path.insert(0,str(root))
    import runtime as rt
    rt.activate(root)
    import sam3_edge_diagnose_v1 as diag
    from parallel import SweepLock
    from scripts.cross_dataset_campaign.common import install_signals,stopping
    install_signals();durable.mkdir(parents=True,exist_ok=True)
    if args.arm:
        # Parent owns the sweep lock; one child releases GPU memory fully between arms.
        ctx=context(rt,root,durable,args,diag)
        task=next(t for t in ctx['tasks'] if t['arm']==args.arm)
        if not run_one(rt,root,durable,args,ctx,task,diag):raise SystemExit(75)
        return
    with SweepLock(root):
        ctx=context(rt,root,durable,args,diag)
        report(rt,root,durable,ctx)
        result=supervise(rt,root,durable,args,ctx,stopping)
        print(f"Saved {result['new_completed']}/{result['new_total']} new results; four references reused.",flush=True)


if __name__=='__main__':main()

