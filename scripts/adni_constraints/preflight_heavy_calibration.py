"""Check sequential calibration, actual native gradients and archive safeguards."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch

from heavy_calibration_pilot import select_at_target, NAME, PARENT, TARGET
from verified_archive import read, write, digest, diagnostics, prepare, retire
from scripts.adni_constraints import objectives as obj
from scripts.adni_constraints.train import validation_steps
from scripts.adni_baselines.core import base
from nnunet_band_objective import native_loss


def independent_weights(rows, arm, target):
    weights = {}
    for term in ('bands', 'edge', 'sdf', 'depth'):
        if term not in arm: continue
        ratios = []
        for row in rows:
            if term == 'bands':
                ratios.append(row['bands_over_dice']); continue
            coefficients = {'dice': 1., **weights}
            if term == 'depth':
                mse = sum(a*b*row['_depth_pair_means'][x+':'+y]
                          for x,a in coefficients.items() for y,b in coefficients.items())
            else:
                names = list(coefficients)
                mse = sum(coefficients[x]**2 * row[x+'_rms']**2 for x in names)
                for i,x in enumerate(names):
                    for y in names[i+1:]:
                        mse += 2*coefficients[x]*coefficients[y]*row[x+'_'+y+'_cross_mean']
            # Edge without preceding bands is saved as its directly measured ratio.
            ratios.append(row['edge_over_base'] if term == 'edge' and not weights
                          else row[term+'_rms']/math.sqrt(mse))
        values = sorted(ratios)
        median = (values[4]+values[5])/2
        p95 = values[8]*.45 + values[9]*.55
        weights[term] = min(target/median, .5/p95)
    return weights


def main():
    root = Path.cwd()
    stage = root/'experiments'/NAME
    plan = read(stage/'plan.json')
    prior = read(root/'experiments'/PARENT/'protocols/nnunet/protocol.json')
    originals = root/'reports'/PARENT/'comparison_20261013/results/nnunet'
    torch.set_num_threads(2)
    torch.manual_seed(17)
    labels = torch.zeros(1,1,64,64,64,dtype=torch.long)
    labels[:,:,25:38,19:45,26:39] = 1
    outputs = [torch.randn(1,2,n,n,n,requires_grad=True) for n in (64,32,16,8)]
    criterion, _ = base.build_loss_and_optimizer('nnunet', torch.nn.Linear(1,1), prior)
    native_gradients = torch.autograd.grad(native_loss(outputs,labels,criterion), outputs,
                                         retain_graph=True, allow_unused=True)
    sdf = obj.signed_distance_maps(labels)
    depth = obj.geometry_weights(labels, np.eye(4), obj.GEOMETRY)
    terms = dict(bands=obj.bands()(outputs[0],labels).loss, edge=obj.BinaryEdge()(outputs[0],labels),
                 sdf=obj.sdf_loss(outputs[0],labels,sdf), depth=obj.depth_loss(outputs[0],labels,depth))
    term_gradients = {k:torch.autograd.grad(v,outputs[0],retain_graph=True)[0] for k,v in terms.items()}
    checked = []
    for spec in plan['runs']:
        arm = spec['objective']
        cal = read(originals/arm/'fold0/seed0/run/calibration.json')
        light,_ = select_at_target(deepcopy(cal['cases']),arm,.1)
        heavy,statistics = select_at_target(deepcopy(cal['cases']),arm)
        oracle = independent_weights(cal['cases'],arm,TARGET)
        for name, weight in heavy.items():
            assert math.isclose(weight,oracle[name],rel_tol=1e-10,abs_tol=1e-12)
            assert math.isclose(light[name],cal['weights'][name],rel_tol=1e-10,abs_tol=1e-12)
            stats = statistics[name]
            assert stats['weighted_p95_ratio'] <= .5+1e-12
            assert math.isclose(stats['weighted_median_ratio'],.4,abs_tol=1e-12) or math.isclose(stats['weighted_p95_ratio'],.5,abs_tol=1e-12)
        protocol = read(stage/'conditions'/arm/'protocols/nnunet/protocol.json')
        original_schedule = prior['models']['nnunet']['objectives'][arm]
        assert protocol['models']['nnunet']['objectives'][arm] == original_schedule
        allowed = {'name','requested_arm_order','models','calibration_target_ratio','calibration_p95_cap',
                   'pilot_controller_sha256','calibration_policy','pilot_change'}
        assert {k for k in set(prior)|set(protocol) if prior.get(k)!=protocol.get(k)} <= allowed
        assert {k:v for k,v in protocol['models']['nnunet'].items() if k!='objectives'} == {k:v for k,v in prior['models']['nnunet'].items() if k!='objectives'}
        assert len(validation_steps(protocol,arm)) == 20 and original_schedule['prefix_updates'] == 9
        assert protocol['constraint_warmup_updates'] == 9 and protocol['augmentation'] == 'none'
        assert protocol['lr_horizon_updates'] == 174
        for early_scale, sdf_scale in ((1.,1/9),(1.,1.)) if 'sdf' in arm else ((1/9,0.),(1.,0.)):
            loss = obj.training_loss(outputs,labels,arm,heavy,early_scale,sdf,criterion,depth,sdf_scale)
            grads = torch.autograd.grad(loss,outputs,retain_graph=True,allow_unused=True)
            expected = native_gradients[0] + sum(heavy[k]*early_scale*(sdf_scale if k=='sdf' else 1.)*term_gradients[k] for k in heavy)
            assert torch.allclose(grads[0],expected,rtol=2e-5,atol=1e-10),arm
            for actual, native in zip(grads[1:],native_gradients[1:]):
                assert actual is None and native is None or torch.equal(actual,native),arm
        checked.append(dict(objective=arm,weights=heavy,statistics=statistics,updates=spec['updates']))
    # Exercise the exact runtime injection, including calibration metadata.
    import heavy_calibration_pilot as pilot
    from scripts.adni_constraints import train as entry
    from scripts.capped_depth_delay import core
    old_main, old_configure, old_calibrate = entry.main, entry.configure, core.calibrate
    arm = plan['runs'][-1]['objective']
    protocol = read(stage/'conditions'/arm/'protocols/nnunet/protocol.json')
    assert digest(Path(pilot.__file__)) == protocol['pilot_controller_sha256']
    cal = read(originals/arm/'fold0/seed0/run/calibration.json')
    core.calibrate = lambda: dict(weights=core.select_weights(deepcopy(cal['cases']),arm)[0])
    def fake_main():
        entry.configure(protocol,arm)
        result = core.calibrate()
        assert result['target_ratio'] == .4 and result['weights'] == plan['runs'][-1]['expected_weights']
    entry.main = fake_main
    try: pilot.train([])
    finally: entry.main,entry.configure,core.calibrate = old_main,old_configure,old_calibrate
    # FN/FP reversibility for every original NN arm, and fail-closed cleanup.
    with tempfile.TemporaryDirectory(prefix='adni-heavy-preflight-') as temp:
        directory = Path(temp)
        for source in sorted(originals.glob('*/fold0/seed0/run')):
            output = directory/source.parts[-4]
            shutil.copytree(source,output)
            diagnostics(output)
        output = directory/'cleanup_case'
        shutil.copytree(originals/'dice_bands/fold0/seed0/run',output)
        done = read(output/'completion.json')
        for name in ('checkpoint_best.pt','checkpoint_latest.pt'):
            (output/name).write_bytes(b'Only synthetic preflight checkpoint bytes')
            done[Path(name).stem+'_sha256'] = digest(output/name)
        write(output/'completion.json',done)
        prepare(output)
        try: retire(output)
        except FileNotFoundError: pass
        else: raise AssertionError('Cleanup must require a durable acknowledgment')
        assert (output/'checkpoint_best.pt').exists()
        write(output/'archive_acknowledgment.json',dict(status='verified',durable_archive='/private/tmp/test-only-durable',
            archive_manifest_sha256=digest(output/'archive_manifest.json'),source_archive_verified=True))
        saved = (output/'config.json').read_bytes()
        (output/'config.json').write_text('{}')
        try: retire(output)
        except AssertionError: pass
        else: raise AssertionError('Cleanup must reject modified metadata')
        assert (output/'checkpoint_latest.pt').exists()
        (output/'config.json').write_bytes(saved)
        receipt = retire(output)
        assert receipt['status']=='complete' and not list(output.glob('*.pt'))
        assert retire(output)['status']=='complete' and (output/'metrics.csv').exists()
    result = dict(status='passed',manifest_sha256=digest(stage/'manifest.json'),checks=checked,
        assertions=['Independent joint-gradient calibration oracle for all 12 arms',
                    'Original 10% calibration reproduces every archived coefficient',
                    'Actual runtime wrapper selects 40% weights and correct metadata',
                    'Native lower-resolution head gradients exactly unchanged',
                    'Full-resolution auxiliary gradients match weights and ramp',
                    'Unchanged timing, split, optimizer, augmentation, FLOPs schedules and 20 validation points',
                    'Exact FN/FP reconstruction verified for all 14 old nnUNet conditions',
                    'Cleanup requires acknowledgment and rejects metadata tampering; repeat cleanup is safe'])
    write(stage/'preflight.json',result)
    write(root/'reports'/NAME/'preflight.json',result)
    print('Passed calibration, gradients, protocol, diagnostics and cleanup preflight for all 12 conditions.')


if __name__ == '__main__': main()
