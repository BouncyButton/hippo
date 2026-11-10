"""One 25-update SAM3 retry: five bands updates, then adapted edge calibration."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

PREFIX = 5
UPDATES = 25
MONITOR_STEPS = (10, 25)


def choose_edge_weight(rows, bands):
    """Original median/p95 calibration, with the established bands weight fixed."""
    ratios = []
    for row in rows:
        gram = np.asarray(row['gram'], dtype=float)
        coeff = np.array([1., bands])
        norm = float(coeff @ gram[:2, :2] @ coeff)
        if not np.isfinite(gram).all():
            raise ValueError('Nonfinite calibration Gram matrix')
        if norm > 0 and gram[2, 2] > 0:
            ratios.append(float(np.sqrt(gram[2, 2] / norm)))
    if len(ratios) < 5:
        raise ValueError('Too few nonzero edge gradients')
    weight = float(min(.1 / np.median(ratios), .5 / np.quantile(ratios, .95)))
    return weight, dict(valid=len(ratios), target=.1, p95_cap=.5,
                       weighted_median=float(weight*np.median(ratios)),
                       weighted_p95=float(weight*np.quantile(ratios, .95)))


def probe(original, diag, model, criterion, parameters, slabs, item, weights, step):
    """Three loss gradients from one forward; restore RNG, buffers and model mode."""
    with diag.preserve_probe_state(model):
        original.reset_seed(item['seed'])
        losses = original.forward_slab(model, criterion, slabs, item['case'], item['anchor'])
        names = ('native', 'bands', 'edge')
        gradients = [torch.autograd.grad(losses[n], parameters, retain_graph=i < 2,
                                        allow_unused=True) for i, n in enumerate(names)]
        gram = diag.gram_matrix(gradients)
        return dict(**item, update=step, gram=gram,
                    losses={n:float(losses[n].detach()) for n in names},
                    full_strength=diag.ratio_metrics(gram, weights),
                    scheduled_strength=diag.ratio_metrics(gram, weights, min((step+1)/20., 1.)))


def train(original, diag, rt, model, criterion, parameters, slabs, p, shared,
          bands, output, recovery, writer, durable):
    schedule = slabs.schedule(p)
    rt.write(output/'slab_schedule.json', schedule)
    original.reset_seed(p['seed']); model.train()
    optimizer = torch.optim.AdamW(parameters, lr=p['learning_rate'], weight_decay=.01)
    frozen = original.parameter_digest(model, False)
    initial = original.parameter_digest(model, True)
    latest = output/'checkpoint_latest.pt'
    step, history = 0, []
    calibration = dict(status='awaiting_update5', weights=dict(bands=bands, edge=0.),
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
        path = durable/'probes'/f'{label}.json'
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
            edge, stats = choose_edge_weight(rows, bands)
            calibration.update(status='complete', weights=dict(bands=bands,edge=edge),
                               rows=rows, statistics=dict(edge=stats),
                               adapter_at_calibration=original.parameter_digest(model,True),
                               scope='20 original training-support slabs after five bands-only updates; bands fixed')
            # Recompute ratios with the newly selected coefficient, not the prior zero edge weight.
            for row in rows:
                row['full_strength'] = diag.ratio_metrics(row['gram'],calibration['weights'])
                row['scheduled_strength'] = diag.ratio_metrics(row['gram'],calibration['weights'],.3)
            commit(force=True)
            print('CALIBRATED EDGE',json.dumps(dict(weights=calibration['weights'],statistics=stats)),flush=True)
        if step in MONITOR_STEPS and str(step) not in calibration['monitoring']:
            calibration['monitoring'][str(step)] = collect(selected,f'monitor_update{step}')
            commit(force=True)
            ratios=[r['full_strength']['edge_over_native_bands'] for r in calibration['monitoring'][str(step)]]
            print('EDGE MONITOR',json.dumps(dict(update=step,median=float(np.median(ratios)),maximum=max(ratios))),flush=True)
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
            total=losses['native']+ramp*(weights['bands']*losses['bands']+weights['edge']*losses['edge'])
            if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
            (total/p['accumulation']).backward()
            values.append({**{n:float(v.detach()) for n,v in losses.items()},'total':float(total.detach())})
            del losses,total
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step();step+=1
        history.append(dict(update=step,lr=lr,ramp=ramp,gradient_norm=float(norm),
                            bands_weight=weights['bands'],edge_weight=weights['edge'],
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


def comparison(rt, root, durable, prior, previous, task):
    baseline, records=rt.previous_baseline(root,durable,previous)
    entries=[('Saved native baseline',baseline,records)]
    for old in rt.read(root/'plan.json')['tasks']:
        if not rt.completed(root,prior,old):raise ValueError('Saved reference run is incomplete')
        folder=prior/'runs'/old['id']
        entries.append(('Saved '+old['scientific_arm'],rt.read(folder/'completion.json'),rt.read(folder/'evaluation.json')['records']))
    folder=durable/'runs'/task['id']
    entries.append(('Adapted-calibration bands+edge',rt.read(folder/'completion.json'),rt.read(folder/'evaluation.json')['records']))
    truth={r['case_name']:r['truth_sha256'] for r in records if r['split']=='validation'}
    bands_entry=next(e for e in entries if e[0]=='Saved dice_bands')
    reference={r['case_name']:r['macro_hard_dice'] for r in bands_entry[2] if r['split']=='validation'}
    rows=[]
    for label,done,rs in entries:
        val=[r for r in rs if r['split']=='validation']
        if len(val)!=96 or {r['case_name']:r['truth_sha256'] for r in val}!=truth:raise ValueError('Comparison cohorts differ')
        delta=[r['macro_hard_dice']-reference[r['case_name']] for r in val]
        rows.append(dict(run=label,validation_dice=done['validation_dice'],training_dice=done['training_dice'],
                         foreground_fn=done['foreground_fn'],foreground_fp=done['foreground_fp'],
                         delta_vs_bands_pp=100*float(np.mean(delta)),wins_vs_bands=sum(x>0 for x in delta),
                         losses_vs_bands=sum(x<0 for x in delta)))
    rt.write(durable/'comparison.json',dict(rows=rows,new_runs=1,reused_runs=3,validation_volumes=96))


def main():
    parser=argparse.ArgumentParser()
    for arg in ('bundle','source-results','previous-results','durable'):parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args();root=args.bundle.resolve();durable=args.durable.resolve()
    sys.path.insert(0,str(root))
    import runtime as rt
    rt.activate(root)
    import engine
    import sam3_edge_diagnose_v1 as diag
    from parallel import SweepLock, save_supplement, restore_supplement
    from scripts.cross_dataset_campaign.common import install_signals
    install_signals();durable.mkdir(parents=True,exist_ok=True)
    with SweepLock(root):
        plan=rt.verify_bundle(root);binding=rt.binding(root,durable)
        if binding!=rt.read(args.source_results/'environment_binding.json'):raise ValueError('Source environment differs')
        shared=rt.fold_calibration(root,args.source_results,0)
        old=next(t for t in plan['tasks'] if t['scientific_arm']=='dice_bands_edge')
        band_task=next(t for t in plan['tasks'] if t['scientific_arm']=='dice_bands')
        for item in (old,band_task):
            if not rt.completed(root,args.source_results,item):raise ValueError('Original result archive is incomplete')
        # Validate saved baseline now, before spending compute.
        rt.previous_baseline(root,durable,args.previous_results)
        bands=rt.read(args.source_results/'runs'/band_task['id']/'calibration.json')['weights']['bands']
        task={**old,'id':old['id']+'_warm5_edge_v1'}
        p=rt.protocol(root,old)
        p.update(edge_calibration_after_updates=PREFIX,edge_target=.1,edge_p95_cap=.5,
                 fixed_bands_weight=bands,monitor_updates=list(MONITOR_STEPS),
                 extension_sha256=rt.digest(Path(__file__)),probe_source_sha256=rt.digest(Path(diag.__file__)))
        if p['updates']!=UPDATES or p['constraint_warmup_updates']!=20:raise ValueError('Unexpected original schedule')
        identity=dict(task=task,protocol=p,source_calibration_sha256=rt.digest(args.source_results/'calibration/fold0/calibration.json'))
        if (durable/'identity.json').exists() and rt.read(durable/'identity.json')!=identity:raise ValueError('Retry identity changed')
        rt.write(durable/'identity.json',identity)
        for file in (Path(__file__),Path(diag.__file__)):rt.copy_verified(file,durable/'source'/file.name)
        if rt.completed(root,durable,task):
            comparison(rt,root,durable,args.source_results,args.previous_results,task)
            print('Already complete: reused saved retry; no training or evaluation repeated.',flush=True);return
        rt.quota(durable)
        output=root/'work'/task['id'];recovery=durable/'recovery'/task['id']
        if not (output/'checkpoint_latest.pt').exists():rt.restore(output,recovery)
        output.mkdir(parents=True,exist_ok=True)
        restore_supplement(rt,root,durable,task)
        if (output/'protocol.json').exists() and rt.read(output/'protocol.json')!=p:raise ValueError('Resume protocol changed')
        rt.write(output/'protocol.json',p);rt.check_ledger(output,recovery)
        if not (output/'completion.json').exists():
            original,model,processor,parameters,criterion,volumes,slabs=engine.load(root,p)
            if shared['initialization']!=dict(frozen=original.parameter_digest(model,False),adapter=original.parameter_digest(model,True)) or slabs.inventory!=shared['inventory']:
                raise ValueError('Original initialization/support differs')
            rt.write(output/'training_inventory.json',slabs.inventory)
            rt.write(output/'runtime_recipe.json',dict(binding=binding,updates=UPDATES,prefix_updates=PREFIX,
                calibration_slabs=20,monitor_updates=list(MONITOR_STEPS),learning_rate_horizon_updates=200,
                augmentation='none',extension_sha256=p['extension_sha256'],probe_source_sha256=p['probe_source_sha256']))
            writer=rt.RecoveryWriter(output,recovery)
            try:
                with (output/'train.log').open('a') as log:
                    class Tee:
                        def write(self,value):sys.__stdout__.write(value);log.write(value);log.flush()
                        def flush(self):sys.__stdout__.flush();log.flush()
                    with redirect_stdout(Tee()):
                        train(original,diag,rt,model,criterion,parameters,slabs,p,shared,bands,output,recovery,writer,durable)
                        os.environ['HIPPO_AUDITED_FAST_EVAL']='1'
                        result=engine.evaluate(original,model,processor,volumes,p,output,root,durable)
                        import matplotlib;matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        history=rt.read(output/'training_history.json')
                        fig,ax=plt.subplots()
                        for n in ('native','bands','edge'):ax.plot([r['update'] for r in history],[r[n] for r in history],label=n)
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
                print('Stopped cleanly. Rerun the same cell to resume.',flush=True);return
            finally:
                writer.close()
                save_supplement(rt,root,durable,task)
        rt.archive_run(root,output,durable,recovery,task)
        comparison(rt,root,durable,args.source_results,args.previous_results,task)
        print('Completed one retry; final metrics archived and checkpoints removed.',flush=True)


if __name__=='__main__':main()
