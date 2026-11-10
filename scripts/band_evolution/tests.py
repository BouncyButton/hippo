"""Scientific controls, geometry, gradients, restart and checkpoint-retention gates."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.edge_ablation.tests import synthetic_item, TinyPatients, tiny_model
from scripts.band_chamfer.geometry import Targets as OldTargets
from scripts.band_chamfer.objectives import components as old_components
from scripts.depth_supervision.utils import atomic_json, digest, seed_all
from scripts.depth_supervision.losses import combined_loss
from scripts.native_branch_constraints.objectives import gradient_record
from .geometry import Targets, pair_geometry, move
from .matrix import experiment_matrix, pair
from .objectives import components, independent_loss, point_error, select_weights, local_mass
from . import train


def target_for(item,entry):
    p=dict(native_geometry={item['case_name']:dict(native_shape=list(item['label'].shape[1:]))})
    return Targets(Path('.'),p,entry).get(item,'cpu')


class ScientificTests(unittest.TestCase):
    def test_nested_pair_samples_transfer_device(self):
        # Meta exercises a real device change on CPU-only development machines.
        item=synthetic_item()
        groups=pair_geometry(item['label'][0].numpy(),np.ones(item['label'].shape[1:],bool),np.ones(3),pair())
        source=dict(pairs=groups,tuple_value=(torch.ones(2),),metadata='unchanged')
        moved=move(source,torch.device('meta'))
        for group in moved['pairs']:
            for side in ('inside','outside'):
                for tensor in group[side].values():self.assertEqual(tensor.device.type,'meta')
        self.assertEqual(moved['tuple_value'][0].device.type,'meta')
        self.assertEqual(moved['metadata'],'unchanged')
        self.assertEqual(source['pairs'][0]['inside']['weight'].device.type,'cpu')

    def test_exact_old_controls_and_generic_independent_equivalence(self):
        item=synthetic_item();labels=item['label'][None]
        p=dict(native_geometry={'synthetic':dict(native_shape=list(labels.shape[2:]))})
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        for entry in experiment_matrix()[:3]:
            a=components(z,labels,entry,target_for(item,entry))
            b=old_components(z,labels,entry,OldTargets(Path('.'),p,entry).get(item,'cpu'))
            for key in a:
                torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
                torch.testing.assert_close(torch.autograd.grad(a[key],z,retain_graph=True)[0],
                    torch.autograd.grad(b[key],z,retain_graph=True)[0],atol=0,rtol=0)
        generic=dict(entry,pair=pair())
        c=components(z,labels,generic,target_for(item,generic))
        torch.testing.assert_close(a['pair'],c['pair'],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(torch.autograd.grad(a['pair'],z,retain_graph=True)[0],
                                  torch.autograd.grad(c['pair'],z)[0],atol=1e-8,rtol=1e-5)

    def test_geometry_distances_support_and_thin_region(self):
        y=np.zeros((17,17,17),np.int64);y[4:13,4:13,4:13]=1
        valid=np.ones_like(y,bool)
        for support in ('all','common7','separate'):
            spec=pair((1,3,5,7));spec['support']=support
            groups=pair_geometry(y,valid,np.ones(3),spec)
            for group,d in zip(groups,spec['distances']):
                a,b=[np.stack(np.unravel_index(group[s]['index'],y.shape),1) for s in ('inside','outside')]
                self.assertTrue((np.abs(a-b).sum(1)==d).all())
                self.assertTrue((y[tuple(a.T)]>0).all());self.assertTrue((y[tuple(b.T)]==0).all())
                for side in group:self.assertAlmostEqual(float(group[side]['weight'].sum()),1.,places=6)
        thin=np.zeros_like(y);thin[8,4:13,4:13]=1
        joint=pair_geometry(thin,valid,np.ones(3),pair((7,)))[0]
        spec=pair((7,));spec['support']='separate'
        separate=pair_geometry(thin,valid,np.ones(3),spec)[0]
        self.assertGreater(len(separate['outside']['index']),len(joint['outside']['index']))
        for tag in ('common3','common5'):
            spec=pair((1,3));spec['support']=tag
            groups=pair_geometry(y,valid,np.ones(3),spec)
            self.assertEqual(len(groups[0]['inside']['index']),len(groups[1]['inside']['index']))

    def test_side_scale_weights_and_gradients(self):
        z=torch.tensor([-.7,.2,.5,-.1],dtype=torch.double,requires_grad=True)
        groups=[{side:dict(index=torch.tensor([index]),weight=torch.ones(1,dtype=torch.double))
                 for side,index in zip(('inside','outside'),indices)} for indices in ((0,1),(2,3))]
        spec=pair(inside_weight=.6,inside_weights=[.75,.25],outside_weights=[.25,.75])
        q=z.sigmoid()
        expected=.6*(.75*(1-q[0])**2+.25*(1-q[2])**2)+.4*(.25*q[1]**2+.75*q[3]**2)
        loss=independent_loss(z,groups,spec)
        torch.testing.assert_close(loss,expected)
        self.assertTrue(torch.autograd.gradcheck(lambda x:independent_loss(x,groups,spec),(z,)))
        gradient=torch.autograd.grad(loss,z)[0]
        self.assertTrue((gradient[[0,2]]<0).all());self.assertTrue((gradient[[1,3]]>0).all())

    def test_empty_full_and_padding(self):
        for value in (0,1):
            y=np.full((9,9,9),value);valid=np.ones_like(y,bool)
            for reduction in ('pairs','unique_voxel','anchor','class_balanced'):
                spec=pair(reduction=reduction)
                groups=pair_geometry(y,valid,np.ones(3),spec)
                z=torch.randn(*y.shape,requires_grad=True)
                loss=independent_loss(z,groups,spec)
                self.assertEqual(float(loss),0.)
                self.assertTrue((torch.autograd.grad(loss,z)[0]==0).all())
        valid=np.zeros((9,9,9),bool);valid[2:7,2:7,2:7]=True
        y=valid.astype(np.int64)
        groups=pair_geometry(y,valid,np.ones(3),pair())
        self.assertTrue(all(not len(g[s]['index']) for g in groups for s in g))

    def test_all_matrix_terms_and_calibrated_budgets(self):
        item=synthetic_item();labels=item['label'][None]
        z=torch.randn(1,3,*labels.shape[2:],requires_grad=True)
        entries=experiment_matrix()
        self.assertEqual(len(entries),len({e['id'] for e in entries}))
        for entry in entries:
            values=components(z,labels,entry,target_for(item,entry))
            record=gradient_record(values,z)
            result=select_weights([record],entry,dict(prefix_epochs=5,target_ratio=.1,cap_ratio=.5))
            coeff=result['coefficients']
            for k,v in values.items():
                self.assertTrue(torch.isfinite(v),entry['id'])
                self.assertTrue(torch.isfinite(torch.autograd.grad(v,z,retain_graph=True)[0]).all())
            extras=[k for k in values if k not in ('dice','bands')]
            if extras:
                a=torch.autograd.grad(sum(coeff[k]*values[k] for k in extras),z,retain_graph=True)[0]
                b=torch.autograd.grad(values['dice']+coeff['bands']*values['bands'],z,retain_graph=True)[0]
                self.assertAlmostEqual(float(a.norm()/b.norm()),.1*entry['extra_budget'],places=5,msg=entry['id'])

    def test_local_mass_perfect_and_padding_gradient(self):
        y=torch.zeros(9,9,9,dtype=torch.bool);y[3:6,3:6,3:6]=True
        valid=torch.zeros_like(y);valid[1:8,1:8,1:8]=True
        target=dict(foreground=y,valid=valid,distance=torch.zeros(9,9,9))
        good=y.float().requires_grad_()
        self.assertEqual(float(local_mass(good,dict(kernel=3),target)),0.)
        bad=torch.full((9,9,9),.5,requires_grad=True)
        loss=local_mass(bad,dict(kernel=3),target)
        self.assertGreater(float(loss),0.)
        self.assertTrue((torch.autograd.grad(loss,bad)[0][~valid]==0).all())

    def test_restart_and_retention(self):
        p=dict(name='test',identity_sha='test',splits={'5pct':dict(train=['a','b'],val=['v'])},pkl='unused',
            native_geometry={n:dict(native_shape=[16]*3) for n in ('a','b','v')},geometry={},
            learning_rate=.001,weight_decay=1e-5,step_size=3,lr_gamma=.5,
            calibration=dict(prefix_epochs=2,ramp_epochs=2,target_ratio=.1,cap_ratio=.5,max_cases=10),
            max_epochs=4,early_stopping=dict(patience=0,min_epochs=1,min_delta=0.))
        entry=next(e for e in experiment_matrix() if e['id']=='combo_sdf8_in50');p['runs']=[entry]
        with tempfile.TemporaryDirectory() as directory,patch.object(train,'_load_pkl_dataframe',return_value=None),\
             patch.object(train,'Patients',TinyPatients),patch.object(train,'model_factory',side_effect=tiny_model):
            root=Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                expected=train.run(p,root/'whole',entry,device_override='cpu')
                train.run(p,root/'test',entry,stop_after=3,device_override='cpu')
                actual=train.run(p,root/'test',entry,device_override='cpu')
            self.assertEqual(expected['validation'],actual['validation'])
            a=torch.load(root/'whole/runs'/entry['id']/'best.pt',weights_only=False)
            b=torch.load(root/'test/runs'/entry['id']/'best.pt',weights_only=False)
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
            from .retention import retain_checkpoints
            retain_checkpoints(p,root/'test',dict(completed=1,rows=[dict(objective='fake'+str(i),macro_dice=1.) for i in range(5)]))
            folder=root/'test/runs'/entry['id']
            self.assertFalse((folder/'best.pt').exists())
            self.assertTrue((folder/'validation_predictions.npz').exists())
            self.assertIn('best.pt',train.completed_record(folder,train.identity(p,entry))['retired_artifacts'])


def model_smoke(device_name, root, pkl_override=None):
    seed_all(0)
    device = torch.device(device_name)
    p = json.loads((root/'protocol.json').read_text())
    dataset = pkl_override or p['pkl']
    if digest(dataset) != p['pkl_sha256']:
        # Local pickle serialization differs. Permit only the independently
        # cluster-verified ten image/label array fingerprints, never guess parity.
        from scripts.band_chamfer.audit_ap import audit
        actual, _ = audit(p, dataset)
        expected = json.loads((root/'ap_training_geometry_cluster.json').read_text())
        if actual != expected:
            raise ValueError('Training arrays differ from the frozen cluster dataset')
    patients = train.Patients(train._load_pkl_dataframe(dataset), p['splits']['5pct']['train'], p['native_geometry'])
    model = train.model_factory(device)
    prefix = root/'shared_warmup/latest.pt'
    if prefix.exists():
        state = torch.load(prefix, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); del state
    assert sum(q.numel() for q in model.parameters()) == p['model_parameters']
    item = patients[0]
    model.eval()
    with train.autocast(device):
        logits = model(item['image'][None].to(device))
    detached = logits.detach().float().requires_grad_(True)
    labels = item['label'][None].to(device)
    checks, geometry = {}, {}
    for entry in p['runs']:
        cache = Targets(root, p, entry)
        for other in patients.items:
            target = cache.get(other, 'cpu')
            if not len(target['gt']):
                raise ValueError('No GT surface in real training case')
            if entry['pair'] and not (len(target['legacy_pair']['i']) if 'legacy_pair' in target else any(len(g['inside']['index']) or len(g['outside']['index']) for g in target['pairs'])):
                raise ValueError('No pair support in real training case')
        values = components(detached, labels, entry, cache.get(item, device))
        record = gradient_record(values, detached)
        calibrated = select_weights([record], entry, p['calibration'])
        checks[entry['id']] = dict(losses=record['losses'], coefficients=calibrated['coefficients'])
        geometry[entry['id']] = cache.statistics()
        print(json.dumps(dict(stage='smoke_objective', id=entry['id'], losses=record['losses'])), flush=True)
    entry = next(e for e in p['runs'] if e['id'] == 'combo_chamfer_in50')
    values = components(logits, labels, entry, Targets(root, p, entry).get(item, device))
    surface_gradient = torch.autograd.grad(values['surface'], logits, retain_graph=True)[0]
    assert float(surface_gradient.abs().sum()) > 0
    loss = combined_loss(values, checks[entry['id']]['coefficients'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    loss.backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
    watched = {n: (q, q.detach().clone()) for n, q in model.named_parameters()
               if n in ('swinViT.patch_embed.proj.weight', 'out.conv.conv.weight')}
    assert len(watched) == 2
    optimizer.step()
    assert all(not torch.equal(q, before) for q, before in watched.values())
    return dict(passed=True, objectives_checked=len(checks), real_training_cases_checked=len(patients),
                device=device_name, parameters=p['model_parameters'], parameters_updated=True,
                ordinary_swinunetr=True, chamfer_logit_gradient_nonzero=True,
                objective_checks=checks, geometry=geometry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('unit', 'model'), required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pkl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == 'unit':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        value = dict(passed=True, tests=result.testsRun)
    else:
        value = model_smoke(args.device, args.root, args.pkl)
    from .prepare import ROOT, sha
    paths = [f for f in (ROOT/'scripts/band_evolution').glob('*') if f.suffix in ('.py', '.sh')]
    value['tested_sources'] = {str(f.relative_to(ROOT)): sha(f) for f in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k not in ('tested_sources', 'geometry', 'objective_checks')}))


if __name__ == '__main__':
    main()
