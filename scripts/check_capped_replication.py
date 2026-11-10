"""Exercise phase gradients and durable metadata failure/resume behavior."""
import importlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / 'experiments/capped_replication_31976_20261007'
SOURCE = STAGE / 'source'
sys.path[:0] = [str(SOURCE), str(SOURCE / 'scripts')]

import torch
from scripts.capped_replication_worker import check_frozen, digest, report, store_metadata, verify_stored, verify_checkpoints
from scripts.capped_depth_delay.objectives import full_loss
from thesis.new_constraints.signed_distance import signed_distance_maps


def main():
    torch.set_num_threads(2)
    plan = check_frozen(STAGE)
    from scripts.capped_depth_delay.schedule import training_flops
    assert len(plan['runs']) == 448 and len(plan['reused_runs']) == 56
    # The delay is fixed from the pilot, never chosen independently per new seed.
    for model in plan['models']:
        protocol = json.loads((SOURCE/'experiments/capped_replication_31976_20261007'/model/'protocol.json').read_text())
        for arm, schedule in protocol['models'][protocol['trainer_alias']]['objectives'].items():
            if protocol['sdf_policy'][arm] == 'delayed_5_of_9':
                late = schedule['sdf_prefix_updates']
                target = plan['budget_flops'] * 5 // 9
                choices = [late-1, late, late+1]
                assert late == min(choices, key=lambda n:(abs(training_flops(n,schedule)-target),n))
    coordinates = torch.meshgrid(*(torch.arange(64) for _ in range(3)), indexing='ij')
    sphere = sum((v-32)**2 for v in coordinates) < 14**2
    labels = torch.zeros(1,1,64,64,64,dtype=torch.long)
    labels[0,0][sphere & (coordinates[0] < 32)] = 1
    labels[0,0][sphere & (coordinates[0] >= 32)] = 2
    sdf = signed_distance_maps(labels)
    torch.manual_seed(7)
    outputs = torch.randn(1,3,64,64,64,requires_grad=True)
    weights = dict(bands=.003,edge=.01,sdf=.08)
    adapter = importlib.import_module('scripts.capped_replication_train')
    results = {}
    with tempfile.TemporaryDirectory(prefix='hippo-replication-check-') as temporary:
        tmp = Path(temporary)
        original_main = adapter.trainer.main
        try:
            for arm in ('dice_sdf','dice_bands_sdf','dice_bands_edge_sdf'):
                adapter.trainer.main = lambda: None
                sys.argv = ['capped_replication_train','--protocol-dir',str(SOURCE/'experiments/capped_replication_31976_20261007/unetr_vit_b16'),
                            '--output',str(tmp/arm),'--objective',arm,'--model','unet']
                adapter.main()
                for scale in (1/50,.5,1.):
                    actual = adapter.trainer.training_loss(outputs,labels,arm,weights,scale,sdf,None,None,.5)
                    expected = full_loss(outputs,labels,arm,weights,scale,None,sdf,None)
                    ga = torch.autograd.grad(actual,outputs,retain_graph=True)[0]
                    gb = torch.autograd.grad(expected,outputs,retain_graph=True)[0]
                    assert torch.equal(actual,expected) and torch.equal(ga,gb)
                expected = full_loss(outputs,labels,'dice',{},.2)
                # Before calibration, the actual active arm is Dice only.
                actual = adapter.trainer.training_loss(outputs,labels,'dice',{},.2,None,None,None,0.)
                assert torch.equal(actual,expected)
                results[arm] = 'Original early-ramp loss and gradients exactly reproduced; Dice prefix accepted'
        finally:
            adapter.trainer.main = original_main
        # Validate durable storage, deliberate corruption rejection, and report
        # regeneration from retained rows without requiring deleted checkpoints.
        output = tmp/'output'; output.mkdir()
        row = dict(plan['reused_runs'][0])
        row.update(fold=0, seed=1)
        configuration = dict(identity='test fixture')
        (output/'config.json').write_text(json.dumps(configuration))
        (output/'completion.json').write_text(json.dumps(dict(status='complete', selected_step=row['updates'])))
        (output/'metrics.csv').write_text('step,dice\n90,0.67\n')
        payload = dict(configuration=configuration, step=row['updates'], score=row['best_dice'],
                       charged_attempts=row['updates'], model={'weight':torch.ones(16)})
        for name in ('checkpoint_best.pt','checkpoint_latest.pt'):
            torch.save(payload,output/name)
        proofs = verify_checkpoints(output,row)
        destination = tmp/'stored/run'; destination.parent.mkdir()
        stored = store_metadata(output,destination,row,proofs)
        assert stored == row and not list(destination.glob('*.pt'))
        assert (output/'checkpoint_best.pt').exists()
        assert verify_stored(destination) == row
        payload['model']['weight'][0] = float('nan')
        torch.save(payload,output/'checkpoint_best.pt')
        try:
            verify_checkpoints(output,row)
        except ValueError:
            results['nonfinite_checkpoint_rejected_before_discard'] = True
        else:
            raise AssertionError('Nonfinite checkpoint accepted')
        (destination/'metrics.csv').write_text('corrupted\n')
        try:
            verify_stored(destination)
        except ValueError:
            results['storage_corruption_rejected'] = True
        else:
            raise AssertionError('Corrupted stored metrics accepted')
        result_root = tmp/'report'; result_root.mkdir()
        report(result_root,plan,[stored],'paused_walltime')
        assert json.loads((result_root/'summary.json').read_text())['completed_new_runs']==1
        assert (result_root/'results.md').exists() and (result_root/'results.csv').exists()
        results['metadata_only_storage_and_report'] = True
    receipt = dict(status='passed', strict_budget_and_504_condition_coverage=True,
                   sdf_delay_nearest_boundary_verified=True, checks=results,
                   no_training_or_submission=True)
    path = REPO/'reports/capped_replication_31976_20261007/local_checks.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
