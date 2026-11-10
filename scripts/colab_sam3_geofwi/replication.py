"""126 fresh SAM3 runs: folds 0/1/3, seeds 0/1/2, the established 14 arms."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ARMS = ('dice', 'dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
        'dice_bands_edge_sdf', 'dice_depth', 'dice_bands_depth', 'dice_edge_depth',
        'dice_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_sdf_depth',
        'dice_edge_sdf_depth', 'dice_bands_edge_sdf_depth')
FOLDS, SEEDS = (0, 1, 3), (0, 1, 2)


def tasks():
    return [dict(id=f'GeoFWI3D_SAM3_f{f}s{s}_{arm}_u25_v1', fold=f, seed=s,
                 arm=arm, scientific_arm=arm, updates=25)
            for f in FOLDS for s in SEEDS for arm in ARMS]


def protocol(root, task):
    p = json.loads((root/f'protocols/f{task["fold"]}s{task["seed"]}.json').read_text())
    return dict(p, data=str(root/'data'), upstream=str(root/'upstream'), arm=task['arm'])


def verify(root):
    import runtime as rt
    manifest = rt.read(root/'bundle_manifest.json')
    for name, sha in manifest.items():
        if rt.digest(root/rt.relative(name)) != sha: raise ValueError('Source differs: '+name)
    for name, sha in rt.read(root/'data_manifest.json').items():
        if rt.digest(root/'data'/rt.relative(name)) != sha: raise ValueError('Data differs: '+name)
    expected = protocol(root, tasks()[0])['base_sha256']
    if rt.digest(root/'weights/sam3.pt') != expected: raise ValueError('SAM3 base differs')


def restore_run(transport, task):
    import runtime as rt
    root = transport.root; name = task['id']; output = root/'work'/name
    ledger = output/'checkpoint_ledger.json'
    charge = transport.state['charges'].get(name)
    local_coherent = (output/'checkpoint_latest.pt').exists() and ledger.exists() and (
        charge is None or rt.read(ledger) == charge)
    if not local_coherent:
        transport.restore('recovery/'+name)
        transport.restore('progress/'+name)
    if charge is not None and (not ledger.exists() or rt.read(ledger) != charge):
        raise RuntimeError('VM loss left charged updates newer than recovery; preserve files for audit, do not replay.')
    return output


def validate_complete(rt, output, p):
    evaluation = rt.read(output/'evaluation.json')
    expected = {(split, c) for split, cases in [('training_support', p['train_cases']),
                ('validation', p['evaluation_cases'])] for c in cases}
    records = evaluation['records']
    history = rt.read(output/'training_history.json')
    summary = rt.read(output/'training_summary.json')
    if (len(records) != len(expected) or {(r['split'], r['case_name']) for r in records} != expected
            or len(history) != 25 or [h['update'] for h in history] != list(range(1,26))
            or summary['frozen_before'] != summary['frozen_after']
            or any(not all(k in r for k in ('confusion','foreground_fn','foreground_fp','truth_sha256')) for r in records)):
        raise ValueError('Incomplete run cannot be retired')
    return evaluation


def run_one(root, task, transport):
    import runtime as rt
    import engine
    import missing_constraints as recipe
    import sam3_edge_diagnose_v1 as diag
    from replication_store import Recovery
    recipe.ARMS = ARMS
    p = protocol(root, task); name = task['id']; durable = root/'durable'
    output = restore_run(transport, task); output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists() and rt.read(output/'protocol.json') != p:
        raise ValueError('Resume protocol changed')
    rt.write(output/'protocol.json', p)
    writer = Recovery(transport, name, output)
    if not (output/'completion.json').exists():
        original, model, processor, parameters, criterion, volumes, slabs = engine.load(root, p)
        for term in ('sdf','depth'):
            if term in p['arm']:
                import torch
                if any(not torch.isfinite(v).all() or not torch.any(v != 0) for v in getattr(slabs,term).values()):
                    raise ValueError('Invalid support geometry: '+term)
        cell = f'f{p["fold"]}s{p["seed"]}'
        folder = durable/'calibration'/cell
        if not (folder/'calibration.json').exists(): transport.restore('calibration/'+cell)
        initialization = dict(frozen=original.parameter_digest(model,False), adapter=original.parameter_digest(model,True))
        if (folder/'calibration.json').exists():
            shared = rt.read(folder/'calibration.json')
            if shared['initialization'] != initialization or shared['inventory'] != slabs.inventory:
                raise ValueError('Cell calibration initialization/support differs')
        else:
            # Once per fold/seed: fresh band coefficient, no historical coefficient imported.
            folder.mkdir(parents=True,exist_ok=True)
            rt.write(folder/'protocol.json',p)
            from scripts.cross_dataset_campaign.foundation_msd import native_training_check
            with diag.preserve_probe_state(model):
                native_training_check(model,criterion,parameters,slabs,p,folder)
                shared = original.calibrate(model,criterion,parameters,slabs,p,'dice_bands',folder)
            shared.update(initialization=initialization, inventory=slabs.inventory)
            if len(shared['rows']) != 20: raise ValueError('Expected 20 calibration slabs')
            rt.write(folder/'calibration.json',shared)
            rt.write(folder/'protocol.json',p)
            transport.publish('calibration/'+cell, list(folder.glob('*')))
        rt.write(output/'training_inventory.json',slabs.inventory)
        rt.write(output/'runtime_recipe.json',dict(binding=transport.state['binding'],
            source_sha256=transport.source, cell_calibration_sha256=rt.digest(folder/'calibration.json'),
            updates=25, prefix_updates=5, monitoring=[10,25], augmentation='none',
            policy='Fresh bands calibration per cell; edge/SDF/depth calibrated sequentially after five prefix updates; fixed coefficients; global 20-update ramp.'))
        raw_write = rt.write
        def durable_write(path, value):
            raw_write(path,value)
            if Path(path).name == 'charged_ledger.json': transport.charge(name,value)
            elif Path(path) == output/'evaluation.json': writer.progress()
        rt.write = durable_write; engine.write = durable_write
        try:
            with (output/'train.log').open('a') as log:
                console = sys.stdout
                class Tee:
                    def write(self, value): console.write(value); log.write(value); log.flush()
                    def flush(self): console.flush(); log.flush()
                with redirect_stdout(Tee()):
                    recipe.train(original,diag,rt,model,criterion,parameters,slabs,p,shared,
                        shared['weights']['bands'],output,durable/'recovery'/name,writer,durable)
                    writer.capture(force=True)
                    os.environ['HIPPO_AUDITED_FAST_EVAL'] = '1'
                    result = engine.evaluate(original,model,processor,volumes,p,output,root,durable)
                    import matplotlib; matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    history = rt.read(output/'training_history.json')
                    fig, ax = plt.subplots()
                    for term in recipe.names_for(p['arm']): ax.plot([r['update'] for r in history],[r[term] for r in history],label=term)
                    ax.legend(); ax.set_xlabel('Update'); fig.tight_layout()
                    fig.savefig(output/'curves.png',dpi=140); plt.close(fig)
                    val = [r for r in result['records'] if r['split']=='validation']
                    rt.write(output/'completion.json',dict(status='complete',updates=25,fold=p['fold'],seed=p['seed'],arm=p['arm'],
                        validation_dice=result['mean_dice']['validation'],training_dice=result['mean_dice']['training_support'],
                        foreground_fn=sum(r['foreground_fn'] for r in val),foreground_fp=sum(r['foreground_fp'] for r in val),
                        checkpoint_sha256=rt.digest(output/'checkpoint_latest.pt'),protocol_sha256=rt.digest(output/'protocol.json')))
        except SystemExit as exc:
            if exc.code == 75:
                writer.capture(force=True); writer.progress(force=True)
                print('Clean stop: server-verified recovery saved.',flush=True)
            raise
        finally:
            rt.write = raw_write; engine.write = raw_write
    validate_complete(rt,output,p)
    done = rt.read(output/'completion.json')
    if rt.digest(output/'checkpoint_latest.pt') != done['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint changed')
    rt.write(output/'archive_manifest.json',dict(task=task,bundle_sha256=transport.source,
        files={str(f.relative_to(output)):rt.digest(f) for f in output.rglob('*') if f.is_file() and not f.name.endswith('.pt') and f.name!='archive_manifest.json'}))
    transport.publish('completed/'+name, writer.paths(checkpoint=False))
    transport.state['completed'][name] = done
    transport.save()
    transport.retire(name)
    print('COMPLETED AND ARCHIVED',name,done['validation_dice'],flush=True)


def report(root, transport):
    import runtime as rt
    rows = list(transport.state['completed'].values())
    result = dict(completed=len(rows),total=126,rows=rows,folds=list(FOLDS),seeds=list(SEEDS),
        policy='All 126 fresh runs; report every fold/seed, no selection of winning seeds. Native SAM3 loss in every arm.')
    rt.write(root/'durable/comparison.json', result)
    transport.publish('comparison', [root/'durable/comparison.json'])
    print(f'Completed {len(rows)}/126 runs.',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--task'); parser.add_argument('--hours',type=float,default=11)
    args = parser.parse_args(); root = args.bundle.resolve()
    sys.path.insert(0,str(root))
    import runtime as rt
    rt.activate(root)
    from scripts.cross_dataset_campaign.common import install_signals, stopping
    from parallel import SweepLock
    from replication_store import Transport
    install_signals()
    def context():
        transport = Transport(root)
        if 'binding' in transport.state:
            rt.write(root/'durable/environment_binding.json',transport.state['binding'])
        binding = rt.binding(root,root/'durable')
        if 'binding' not in transport.state:
            transport.state['binding']=binding; transport.save()
        return transport
    if args.task:
        transport = context()
        task = next(t for t in tasks() if t['id']==args.task)
        if task['id'] not in transport.state['completed']: run_one(root,task,transport)
        return
    with SweepLock(root):
        verify(root); transport = context(); transport.probe()
        # Complete any interrupted cleanup only after checking the completed archive on Google's server.
        for name in list(transport.state['completed']):
            transport.restore('completed/'+name)
            validate_complete(rt,root/'work'/name,rt.read(root/'work'/name/'protocol.json'))
            if name in transport.state['charges'] or 'recovery/'+name in transport.state['objects']:
                transport.retire(name)
        os.environ['HIPPO_DEADLINE'] = str(time.time()+args.hours*3600)
        pending = sorted(tasks(), key=lambda t:(t['id'] not in transport.state['charges'], tasks().index(t)))
        for task in pending:
            if stopping(): break
            if task['id'] in transport.state['completed']: continue
            print('RUN',task['id'],flush=True)
            child = subprocess.Popen([sys.executable,__file__,'--bundle',str(root),'--task',task['id']])
            signalled = False
            while child.poll() is None:
                if stopping() and not signalled:
                    child.send_signal(signal.SIGINT); signalled=True
                time.sleep(1)
            transport = context()  # Reload the child's acknowledged remote state.
            report(root,transport)
            if child.returncode == 75: break
            if child.returncode: raise RuntimeError('Run failed; local/recovery files retained: '+task['id'])
        report(root,transport)


if __name__ == '__main__': main()
