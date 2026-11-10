"""Frozen SAM3 optimizer loop; only durable-ledger and checkpoint cadence additions.
Derived from the archived foundation.py train function. No objective, schedule,
precision, optimizer or sampling changes.
"""
from scripts.cross_dataset_campaign.foundation import *

def train(model, criterion, parameters, slabs, protocol, calibration, arm, output, writer, recovery):
    schedule = slabs.schedule(protocol)
    write(output/'slab_schedule.json',schedule)
    reset_seed(protocol['seed'])
    model.train()
    optimizer = torch.optim.AdamW(parameters,lr=protocol['learning_rate'],weight_decay=.01)
    frozen_before, adapter_before = parameter_digest(model,False),parameter_digest(model,True)
    start, history = 0, []
    latest = output/'checkpoint_latest.pt'
    elapsed = 0.
    if latest.exists():
        state = torch.load(latest,map_location='cpu',weights_only=False)
        if state['protocol_sha256'] != digest(output/'protocol.json'):
            raise ValueError('Resume protocol mismatch')
        model.load_state_dict(state['adapter'],strict=False)
        optimizer.load_state_dict(state['optimizer'])
        start,history,elapsed = state['step'],state['history'],state['elapsed_seconds']
        if state['frozen_before'] != frozen_before or state['adapter_before'] != adapter_before:
            raise ValueError('Initial model changed')
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
    began = time.monotonic()
    def commit(step):
        save_checkpoint(latest,dict(step=step,history=history,
            adapter={n:p.detach().cpu() for n,p in model.state_dict().items() if is_adapter(n)},
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),protocol_sha256=digest(output/'protocol.json'),
            frozen_before=frozen_before,adapter_before=adapter_before,elapsed_seconds=elapsed+time.monotonic()-began))
        write(output/'checkpoint_ledger.json',dict(step=step,protocol_sha256=digest(output/'protocol.json')))
        write(output/'training_history.json',history)
        writer.capture()
    if not latest.exists():
        commit(0)
        writer.capture(force=True)
    for update in range(start,protocol['updates']):
        if stopping():
            commit(update)
            raise SystemExit(75)
        ledger=dict(step=update+1,protocol_sha256=digest(output/'protocol.json'))
        write(recovery/'charged_ledger.json',ledger)
        if read(recovery/'charged_ledger.json') != ledger:
            raise IOError('Drive charge readback failed')
        horizon=protocol.get('learning_rate_horizon_updates',protocol['updates'])
        if horizon < protocol['updates']:
            raise ValueError('Learning-rate horizon cannot precede the stopping update')
        lr=lr_for_update(update,horizon,protocol['warmup_updates'],protocol['learning_rate'],protocol['final_learning_rate'])
        for group in optimizer.param_groups:
            group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        ramp=min((update+1)/protocol['constraint_warmup_updates'],1.)
        rows=[]
        for micro in range(protocol['accumulation']):
            case,anchor=schedule[update*protocol['accumulation']+micro]
            losses=forward_slab(model,criterion,slabs,case,anchor)
            total=losses['native']+ramp*sum(calibration['weights'][name]*losses[name] for name in calibration['weights'])
            if not torch.isfinite(total):
                raise ValueError('Nonfinite fine-tuning loss')
            (total/protocol['accumulation']).backward()
            rows.append({**{k:float(v.detach()) for k,v in losses.items()},'total':float(total.detach())})
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(update=update+1,lr=lr,ramp=ramp,gradient_norm=float(norm),
                            **{k:float(np.mean([r[k] for r in rows])) for k in rows[0]}))
        if True:  # Local checkpoint every update; asynchronous Drive snapshots.
            commit(update+1)
            print(json.dumps(history[-1]),flush=True)
    commit(protocol['updates'])
    model.eval().requires_grad_(False)
    frozen_after,adapter_after=parameter_digest(model,False),parameter_digest(model,True)
    if frozen_before!=frozen_after or adapter_before==adapter_after:
        raise ValueError('Frozen base changed or adapter did not learn')
    write(output/'training_summary.json',dict(updates=protocol['updates'],frozen_before=frozen_before,
        frozen_after=frozen_after,adapter_before=adapter_before,adapter_after=adapter_after,
        schedule_sha256=digest(output/'slab_schedule.json'),elapsed_seconds=elapsed+time.monotonic()-began,
        augmentation='none',weights=calibration['weights'],
        learning_rate_horizon_updates=protocol.get('learning_rate_horizon_updates',protocol['updates'])))
