"""Scientific estimator, scheduling and interrupted-recovery regression checks."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import runtime
import engine
import diagnose_preflight
from audit_numerics import audit_empty_point_grid


class DiagnosticTests(unittest.TestCase):
    def test_empty_grid_matches_native_output_and_zero_derivatives(self):
        native = torch.nn.functional.grid_sample
        enabled = torch.are_deterministic_algorithms_enabled()
        warning = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            for shape in ((2, 0, 1, 2), (2, 3, 0, 2)):
                for mode in ('bilinear', 'nearest', 'bicubic'):
                    for padding in ('zeros', 'border', 'reflection'):
                        value = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
                        grid = torch.empty(shape, dtype=torch.float64, requires_grad=True)
                        expected = native(value, grid, mode, padding, align_corners=False)
                        expected_grads = torch.autograd.grad(expected.sum(), (value, grid))
                        with audit_empty_point_grid() as audit:
                            actual = torch.nn.functional.grid_sample(value, grid, mode, padding, align_corners=False)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            for _ in range(2):
                                grads = torch.autograd.grad(actual.sum(), (value, grid), retain_graph=True)
                                for a, b in zip(grads, expected_grads):
                                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                            self.assertEqual(audit['empty_calls'], 1)
                            self.assertEqual(audit['nonempty_calls'], 0)
                        self.assertIs(torch.nn.functional.grid_sample, native)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warning)

    def test_grid_audit_preserves_nonempty_gradients_and_restores_after_failure(self):
        native = torch.nn.functional.grid_sample
        enabled = torch.are_deterministic_algorithms_enabled()
        warning = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            value = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
            grid = torch.zeros(1, 2, 2, 2, dtype=torch.float64, requires_grad=True)
            expected = native(value, grid, align_corners=False)
            expected_grads = torch.autograd.grad(expected.sum(), (value, grid))
            with self.assertRaisesRegex(RuntimeError, 'native error preserved'):
                with audit_empty_point_grid() as audit:
                    actual = torch.nn.functional.grid_sample(value, grid, align_corners=False)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    grads = torch.autograd.grad(actual.sum(), (value, grid))
                    for a, b in zip(grads, expected_grads):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(audit['empty_calls'], 0)
                    self.assertEqual(audit['nonempty_calls'], 1)
                    raise RuntimeError('native error preserved')
            self.assertIs(torch.nn.functional.grid_sample, native)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warning)

    def test_equivalence_keeps_strict_gradients_and_restores_diagnostic_updates(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.adapter = torch.nn.Parameter(torch.tensor([.4, .6]))
        model = Tiny(); initial = model.adapter.detach().clone()
        processor = SimpleNamespace(transform=SimpleNamespace(enabled=True))
        slabs = SimpleNamespace(calibration_schedule=lambda _: [('case',0),('case',1)], samples=[('pixels',None)])
        def values(*_):
            x = model.adapter
            return dict(native=x.square().sum(), bands=x.pow(3).sum(), edge=x.pow(4).sum(),
                        sdf=x.sum(), depth=x.exp().sum())
        original = SimpleNamespace(reset_seed=torch.manual_seed,
            forward_slab=lambda *_: tuple(values()[name] for name in ('native','bands','edge')),
            forward_plane=lambda *_: (model.adapter.square(), None))
        enabled = torch.are_deterministic_algorithms_enabled()
        warning = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            with patch.object(engine, 'forward', side_effect=values):
                result = engine.equivalence(original, model, processor, None, [model.adapter], slabs, {})
            self.assertTrue(result['original_repeatability'])
            self.assertTrue(result['optimizer_checkpoint_resume_exact'])
            torch.testing.assert_close(model.adapter.detach(), initial, rtol=0, atol=0)
            def broken(*args):
                result = values()
                v = result['native']
                result['native'] = v.detach() + 1.01*(v-v.detach())
                return result
            with patch.object(engine, 'forward', side_effect=broken):
                with self.assertRaisesRegex(AssertionError, 'extension_cached: native gradient for adapter'):
                    engine.equivalence(original, model, processor, None, [model.adapter], slabs, {})
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warning)

    def test_strict_comparison_rejects_nondeterministic_or_warning_only_mode(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warning = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            for mode, warn_only in ((False, False), (True, True)):
                torch.use_deterministic_algorithms(mode, warn_only=warn_only)
                with self.assertRaisesRegex(RuntimeError, 'isolated deterministic audit'):
                    engine.equivalence(None, None, None, None, None, None, None)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warning)

    def test_audit_child_settings_do_not_leak_and_matching_receipt_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'bundle'; durable = Path(directory)/'results'
            value = dict(bundle_sha256='current', environment='same GPU')
            checks = dict(original_repeatability=True, original_loss_gradient_equivalence=True,
                          optimizer_checkpoint_resume_exact=True, label_free_inference_exact=True)
            path = durable/'calibration/fold0/equivalence.json'
            runtime.write(path, dict(status='passed', binding={'bundle_sha256':'old'}, checks=checks))
            previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic)
            def child(command, env, check):
                self.assertEqual(command[2], 'equivalence')
                self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                self.assertTrue(Path(env['TORCHINDUCTOR_CACHE_DIR']).is_dir())
                self.assertNotEqual(env['TORCHINDUCTOR_CACHE_DIR'], 'original cache')
                self.assertTrue(check)
                runtime.write(path, dict(status='passed', binding=value, checks=checks))
            with patch.dict(os.environ, CUBLAS_WORKSPACE_CONFIG='original setting', TORCHINDUCTOR_CACHE_DIR='original cache'), \
                    patch.object(engine.subprocess, 'run', side_effect=child) as run:
                self.assertEqual(engine.ensure_equivalence_audit(root, durable, 0, value), checks)
                self.assertEqual(engine.ensure_equivalence_audit(root, durable, 0, value), checks)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], 'original setting')
                self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'], 'original cache')
            self.assertEqual(previous, (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic))

    def test_audit_exit_zero_cannot_approve_failed_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'bundle'; durable = Path(directory)/'results'
            value = dict(bundle_sha256='current')
            path = durable/'calibration/fold0/equivalence.json'
            def child(*args, **kwargs):
                runtime.write(path, dict(status='passed', binding=value, checks={}))
            with patch.object(engine.subprocess, 'run', side_effect=child):
                with self.assertRaisesRegex(ValueError, 'invalid deterministic equivalence'):
                    engine.ensure_equivalence_audit(root, durable, 0, value)

    def test_exact_reference_and_named_mismatch(self):
        reference = [torch.tensor([[1., 2.]]), None]
        names = ['backbone.qkv.lora_A', 'unused.lora_B']
        result = diagnose_preflight.gradient_difference(reference, reference, names)
        self.assertTrue(result['passed'])
        self.assertEqual(result['relative_l2'], 0.)
        actual = [torch.tensor([[1., 2.1]]), None]
        result = diagnose_preflight.gradient_difference(reference, actual, names)
        self.assertFalse(result['passed'])
        self.assertEqual(result['mismatched_elements'], 1)
        self.assertEqual(result['first_failures'][0]['parameter'], names[0])

    def test_connectivity_and_nonfinite_are_failures(self):
        result = diagnose_preflight.gradient_difference([None], [torch.ones(1)], ['adapter'])
        self.assertFalse(result['passed'])
        result = diagnose_preflight.gradient_difference([torch.ones(1)], [torch.tensor([float('nan')])], ['adapter'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['first_failures'][0]['actual_finite'])
        json.dumps(result, allow_nan=False)


class EstimatorTests(unittest.TestCase):
    def test_batched_gram_preserves_original_fp64_summation(self):
        gradients=[[torch.randn(7),None,torch.randn(11)],[torch.randn(7),torch.randn(4),torch.randn(11)]]
        expected=[[sum(float((a.double()*b.double()).sum()) for a,b in zip(left,right)
                       if a is not None and b is not None) for right in gradients] for left in gradients]
        self.assertEqual(engine.gram_matrix(gradients),expected)

    def test_sdf_and_depth_uniform_slice_expectation(self):
        torch.manual_seed(8)
        logits=torch.randn(3,5,6,7,requires_grad=True)
        logp=logits.log_softmax(0)
        truth=torch.randint(0,3,(5,6,7))
        sdf=torch.randn(2,5,6,7)
        depth=torch.rand(5,6,7);depth/=depth.sum()
        sampled=[engine.auxiliary(logp[:,z],truth[z],sdf[:,z],depth[z],5) for z in range(5)]
        full_sdf=((logp[1:].exp()-torch.stack([truth==c for c in (1,2)]).float())*sdf).mean()
        full_depth=-(logp.gather(0,truth[None])[0]*depth).sum()
        for i,full in enumerate((full_sdf,full_depth)):
            estimate=torch.stack([s[i] for s in sampled]).mean()
            torch.testing.assert_close(estimate,full)
            a=torch.autograd.grad(estimate,logits,retain_graph=True)[0]
            b=torch.autograd.grad(full,logits,retain_graph=True)[0]
            torch.testing.assert_close(a,b)

    def test_all_coefficients_are_fresh_and_sequential(self):
        rows=[dict(gram=np.eye(5).tolist()) for _ in range(20)]
        old=dict(bands=dict(weight=12.9067),edge=dict(weight=142.75))
        result=engine.weights_for('bands_edge_sdf_depth',rows,old)
        self.assertEqual(result['weights']['bands'],.1)
        self.assertAlmostEqual(result['weights']['edge'],.1*np.sqrt(1.01))
        self.assertAlmostEqual(result['diagnostics']['sdf']['median'],.1)
        self.assertAlmostEqual(result['diagnostics']['depth']['median'],.1)
        self.assertTrue(all(not d['imported_original_coefficient'] for d in result['diagnostics'].values()))
        self.assertFalse(engine.weights_for('edge_depth',rows,old)['diagnostics']['edge']['imported_original_coefficient'])

    def test_bands_p95_failure_is_recalibrated_not_waived(self):
        ratios = np.array([.004]*18 + [.6736139612372511/12.906762993482202]*2)
        rows = []
        for ratio in ratios:
            gram = np.eye(5); gram[1, 1] = ratio**2
            rows.append(dict(gram=gram.tolist()))
        result = engine.weights_for('bands_edge', rows, dict(bands=dict(weight=12.906762993482202)))
        self.assertAlmostEqual(result['weights']['bands'], 9.580237150797679)
        self.assertAlmostEqual(result['diagnostics']['bands']['p95'], .5)
        # Edge denominator must contain the NEW bands coefficient.
        edge_ratios = 1/np.sqrt(1+(result['weights']['bands']*ratios)**2)
        self.assertAlmostEqual(result['weights']['edge'], min(.1/np.median(edge_ratios), .5/np.quantile(edge_ratios,.95)))

    def test_calibration_rejects_degenerate_gradients(self):
        rows=[dict(gram=np.zeros((5,5)).tolist()) for _ in range(20)]
        with self.assertRaises(ValueError):engine.weights_for('sdf',rows,{})

    def test_transform_cache_returns_independent_values(self):
        calls=[]
        cache=engine.TransformCache(lambda x:(calls.append(1) or x.float()*2),maximum=1000)
        data=torch.arange(8).reshape(2,4)
        a=cache(data);a.zero_();b=cache(data)
        torch.testing.assert_close(b,data.float()*2)
        self.assertEqual(len(calls),1)
        cache.enabled=False;cache(data);self.assertEqual(len(calls),2)

    def test_original_forward_graph_preserved(self):
        from scripts import train_medsam3_multiclass as original
        from scripts.medsam3_supervised_constraints import VolumeSupport
        labels=torch.zeros(4,5,6,dtype=torch.long);labels[:,1:3,1:3]=1;labels[:,2:4,3:5]=2
        logits=torch.randn(3,4,5,6,requires_grad=True)
        samples=[(z,labels[z].numpy()) for z in range(4)]
        slabs=SimpleNamespace(indices={'case':list(range(4))},samples=samples,processor=None,
            supports={'case':[VolumeSupport.build(labels==c) for c in range(3)]},
            sdf={'case':torch.zeros(2,4,5,6)},depth={'case':torch.ones(4,5,6)/120})
        saved=original.forward_plane
        def fake(model,processor,pixels,labels=None,criterion=None):
            x=model[:,pixels];return x.log_softmax(0),x.square().mean()
        original.forward_plane=fake
        try:
            for anchor in (0,3):
                a=original.forward_slab(logits,None,slabs,'case',anchor)
                b=engine.forward(original,logits,None,slabs,'case',anchor)
                for x,key in zip(a,('native','bands','edge')):
                    torch.testing.assert_close(x,b[key],rtol=0,atol=0)
                    torch.testing.assert_close(torch.autograd.grad(x,logits,retain_graph=True)[0],
                        torch.autograd.grad(b[key],logits,retain_graph=True)[0],rtol=0,atol=0)
        finally:original.forward_plane=saved


class RecoveryTests(unittest.TestCase):
    def test_training_resume_matches_uninterrupted_updates(self):
        from scripts.train_medsam3_few_shot import parameter_digest
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.adapter=torch.nn.Module()
                self.adapter.lora_A=torch.nn.Parameter(torch.tensor([.5,.7]))
                self.adapter.lora_B=torch.nn.Parameter(torch.tensor([.2,.9]))
                self.frozen=torch.nn.Parameter(torch.ones(2),requires_grad=False)
        def fake_load(root,p):
            model=Tiny();parameters=list(model.adapter.parameters())
            original=SimpleNamespace(reset_seed=lambda seed:torch.manual_seed(seed))
            processor=SimpleNamespace(transform=SimpleNamespace(maximum=1000))
            slabs=SimpleNamespace(schedule=lambda p:[('case',0)]*(p['updates']*p['accumulation']))
            old=dict(context=dict(frozen_before=parameter_digest(model,False),adapter_before=parameter_digest(model,True)))
            return original,model,processor,None,parameters,slabs,old
        def fake_forward(original,model,*_):
            base=sum(v.square().sum() for v in model.adapter.parameters())
            return {name:base*(i+1)/5 for i,name in enumerate(('native','bands','edge','sdf','depth'))}
        def fake_evaluate(original,model,processor,root,p,output):
            rows=[dict(split=s,ap_mean_dice=.8,per_class=dict(background=dict(fp=2,fn=3)),
                       anterior_to_posterior=1,posterior_to_anterior=1) for s in ('training_support','validation')]
            runtime.write(output/'evaluation.json',dict(records=rows));return rows
        cal=dict(arms=dict(sdf=dict(weights=dict(sdf=.1))),rows=[],geometry=[],original_calibration_sha256='old')
        task=dict(id='run',fold=0,seed=17,arm='sdf')
        p=dict(seed=17,fold_index=0,updates=200,accumulation=2,learning_rate=5e-5,final_learning_rate=5e-6,
               warmup_updates=20,constraint_warmup_updates=20,classes=['background','anterior','posterior'],class_prompts={})
        with tempfile.TemporaryDirectory() as temp:
            roots=[Path(temp)/name for name in ('continuous','resumed')]
            for root in roots:
                runtime.write(root/'protocols/fold0_seed17.json',p)
                runtime.write(root/'drive/calibration/fold0/calibration.json',cal)
                source=root/'source/scripts/train_medsam3_few_shot.py';source.parent.mkdir(parents=True);source.write_text('source')
                (root/'work/run').mkdir(parents=True)
            patches=[patch.object(engine,'load',fake_load),patch.object(engine,'forward',fake_forward),
                patch.object(engine,'evaluate',fake_evaluate),patch.object(engine,'fold_calibration',lambda *_:cal),
                patch.object(engine,'rng_state',lambda:dict(torch=torch.get_rng_state())),
                patch.object(engine,'set_rng',lambda s:torch.set_rng_state(s['torch'])),
                patch.object(torch.cuda,'get_device_name',lambda:'test CPU')]
            from contextlib import ExitStack,redirect_stdout
            import io
            with ExitStack() as stack,redirect_stdout(io.StringIO()):
                for mocked in patches:stack.enter_context(mocked)
                with patch.object(engine,'stopped',lambda _:False):engine.train(roots[0],roots[0]/'drive',task,None)
                calls=[]
                with patch.object(engine,'stopped',lambda _:(calls.append(1) or len(calls)>35)):
                    with self.assertRaises(SystemExit) as error:engine.train(roots[1],roots[1]/'drive',task,None)
                    self.assertEqual(error.exception.code,75)
                # Restore from the Drive snapshot into a fresh local work directory.
                import shutil
                shutil.rmtree(roots[1]/'work/run')
                runtime.restore(roots[1]/'work/run',roots[1]/'drive/recovery/run')
                with patch.object(engine,'stopped',lambda _:False):engine.train(roots[1],roots[1]/'drive',task,None)
            states=[torch.load(r/'work/run/checkpoint_latest.pt',weights_only=False) for r in roots]
            self.assertEqual(states[0]['history'],states[1]['history'])
            for key in states[0]['adapter']:torch.testing.assert_close(states[0]['adapter'][key],states[1]['adapter'][key],rtol=0,atol=0)
            self.assertEqual(states[1]['step'],200)

    def test_snapshot_restore_and_lost_update_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';out.mkdir();durable=root/'drive'
            (out/'checkpoint_latest.pt').write_bytes(b'coherent-checkpoint')
            for name in ('protocol.json','calibration.json'):runtime.write(out/name,dict(ok=True))
            ledger=dict(step=7,protocol_sha256='frozen')
            runtime.write(out/'checkpoint_ledger.json',ledger);runtime.write(durable/'charged_ledger.json',ledger)
            writer=runtime.RecoveryWriter(out,durable,interval=0);writer.close(force=True)
            restored=root/'restored';runtime.restore(restored,durable);runtime.check_ledger(restored,durable)
            self.assertEqual((restored/'checkpoint_latest.pt').read_bytes(),b'coherent-checkpoint')
            runtime.write(durable/'charged_ledger.json',dict(step=8,protocol_sha256='frozen'))
            with self.assertRaisesRegex(RuntimeError,'lost charged updates'):runtime.check_ledger(restored,durable)

    def test_corrupt_recovery_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runtime.write(root/'latest.json',dict(file='recovery.tar',sha256='wrong'))
            (root/'recovery.tar').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):runtime.restore(root/'work',root)

    def test_path_traversal_rejected(self):
        for name in ('../escape','/absolute','a/../../escape'):
            with self.assertRaises(ValueError):runtime.relative(name)

    def test_completed_archive_cleanup_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';out.mkdir();durable=root/'drive';recovery=durable/'recovery'/'run'
            runtime.write(root/'bundle_manifest.json',{})
            task=dict(id='run',fold=0,seed=17,arm='sdf')
            checkpoint=out/'checkpoint_latest.pt';checkpoint.write_bytes(b'completed checkpoint')
            for name in ('protocol.json','calibration.json','training_history.json','training_summary.json','slab_schedule.json','runtime_recipe.json','evaluation.json'):
                runtime.write(out/name,{})
            (out/'curves.png').write_bytes(b'plot');(out/'train.log').write_text('finished')
            runtime.write(out/'completion.json',dict(updates=200,status='complete',checkpoint_sha256=runtime.digest(checkpoint)))
            runtime.archive_run(root,out,durable,recovery,task)
            self.assertFalse(checkpoint.exists());self.assertTrue(runtime.completed(root,durable,task))
            # Simulate interruption after deletions but before the final cleanup receipt.
            (durable/'runs/run/cleanup.json').unlink()
            runtime.archive_run(root,out,durable,recovery,task)
            self.assertTrue(runtime.completed(root,durable,task))

    def test_shared_calibration_hash_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);durable=root/'drive';folder=durable/'calibration/fold0'
            runtime.write(root/'bundle_manifest.json',{})
            bound=dict(bundle_sha256=runtime.digest(root/'bundle_manifest.json'),environment={})
            runtime.write(durable/'environment_binding.json',bound);runtime.write(folder/'calibration.json',dict(weights={}))
            runtime.write(folder/'receipt.json',dict(status='passed',binding=bound,calibration_sha256=runtime.digest(folder/'calibration.json')))
            runtime.fold_calibration(root,durable,0)
            runtime.write(folder/'calibration.json',dict(weights={'sdf':999}))
            with self.assertRaises(ValueError):runtime.fold_calibration(root,durable,0)


class MatchedStudyTests(unittest.TestCase):
    def fixture(self, directory):
        import shutil
        import reuse_preflight
        from scripts.colab_medsam3.prepare import EXPORT
        root = Path(directory)/'bundle'; previous = Path(directory)/'previous'
        durable = Path(directory)/'results'; previous_bundle = Path(directory)/'old_bundle'
        for name in ('engine.py', 'runtime.py', 'bundle_manifest.json',
                     'compatibility/engine.py', 'compatibility/runtime.py', 'compatibility/bundle_manifest.json',
                     *(f'calibration/fold{f}.json' for f in runtime.FOLDS)):
            target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EXPORT/'bundle'/name, target)
        old = dict(bundle_sha256=reuse_preflight.LEGACY_MANIFEST, environment=dict(gpu='same A100'))
        value = dict(bundle_sha256=runtime.digest(root/'bundle_manifest.json'), environment=old['environment'])
        runtime.write(previous/'environment_binding.json', old)
        all_rows = {}
        for fold in (0, 1):
            folder = previous/'calibration'/f'fold{fold}'
            runtime.write(folder/'equivalence.json', dict(status='passed', binding=old,
                checks={k: True for k in reuse_preflight.CHECKS}))
            rows = [dict(case=r['case'], anchor=r['anchor'], gram=np.eye(5).tolist(),
                         losses={n: .5 for n in ('native', 'bands', 'edge', 'sdf', 'depth')})
                    for r in runtime.read(root/'calibration'/f'fold{fold}.json')['slabs']]
            all_rows[fold] = rows
            if fold == 0:
                runtime.write(folder/'calibration.json', dict(rows=rows, arms={'bands': {'weight': 999.}}))
                runtime.write(folder/'receipt.json', dict(status='passed', binding=old,
                    calibration_sha256=runtime.digest(folder/'calibration.json')))
            else:
                torch.save(dict(binding=old, rows=rows, rng={}), folder/'partial.pt')
        return root, durable, previous, previous_bundle, value, all_rows

    def test_reuse_preserves_complete_samples_evidence_and_manual_pause(self):
        import reuse_preflight
        with tempfile.TemporaryDirectory() as directory:
            root, durable, previous, old_bundle, value, rows = self.fixture(directory)
            before = {str(p.relative_to(previous)): runtime.digest(p) for p in previous.rglob('*') if p.is_file()}
            (previous/'STOP').write_text('manual pause')
            with patch.object(reuse_preflight, 'binding', return_value=value):
                result = reuse_preflight.reuse_previous(root, durable, previous, old_bundle)
                self.assertEqual(result['folds']['0']['measurements_reused'], 20)
                self.assertEqual(result['folds']['1']['measurements_reused'], 20)
                self.assertFalse(result['campaign_approved'])
                self.assertFalse(result['coefficients_imported'])
                self.assertEqual((durable/'STOP').read_text(), 'manual pause')
                # Deliberately resuming the new study is respected on future calls.
                (durable/'STOP').unlink()
                reuse_preflight.reuse_previous(root, durable, previous, old_bundle)
                self.assertFalse((durable/'STOP').exists())
            for name, sha in before.items(): self.assertEqual(runtime.digest(previous/name), sha)
            for fold in (0, 1):
                folder = durable/'calibration'/f'fold{fold}'
                self.assertEqual(runtime.read(folder/'measurements.json')['rows'], rows[fold])
                self.assertFalse((folder/'calibration.json').exists())
                self.assertFalse((folder/'receipt.json').exists())
                reuse_preflight.verify_reused_evidence(folder, value)
            self.assertFalse((durable/'preflight.json').exists())
            runtime.write(durable/'calibration/fold1/measurements.json', dict(tampered=True))
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                reuse_preflight.verify_reused_evidence(durable/'calibration/fold1', value)

    def test_reuse_rejects_source_audit_schedule_and_recovery_changes(self):
        import reuse_preflight
        cases = ('source', 'audit', 'schedule', 'recovery', 'receipt_hash')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root, durable, previous, old_bundle, value, rows = self.fixture(directory)
                if case == 'source':
                    path = root/'engine.py'
                    path.write_text(path.read_text().replace('def forward(original,', 'def changed_forward(original,'))
                elif case == 'audit':
                    path = previous/'calibration/fold0/equivalence.json'; record = runtime.read(path)
                    record['checks']['original_repeatability'] = False; runtime.write(path, record)
                elif case == 'schedule':
                    path = previous/'calibration/fold1/partial.pt'
                    state = torch.load(path, weights_only=False); state['rows'][0]['anchor'] = 999
                    torch.save(state, path)
                elif case == 'recovery':
                    runtime.write(previous/'recovery/run/charged_ledger.json', dict(step=1))
                else:
                    path = previous/'calibration/fold0/calibration.json'
                    runtime.write(path, dict(rows=[]))
                with patch.object(reuse_preflight, 'binding', return_value=value):
                    with self.assertRaises(ValueError): reuse_preflight.reuse_previous(root, durable, previous, old_bundle)
                self.assertFalse((durable/'preflight.json').exists())

    def test_changed_environment_keeps_old_evidence_and_requires_fresh_audit(self):
        import reuse_preflight
        with tempfile.TemporaryDirectory() as directory:
            root, durable, previous, old_bundle, value, rows = self.fixture(directory)
            value['environment'] = dict(gpu='different H100')
            with patch.object(reuse_preflight, 'binding', return_value=value):
                reuse_preflight.reuse_previous(root, durable, previous, old_bundle)
                # Rerunning notebook setup after starting the NEW study must be safe.
                runtime.write(durable/'recovery/run/charged_ledger.json', dict(step=1))
                reuse_preflight.reuse_previous(root, durable, previous, old_bundle)
            self.assertFalse((durable/'calibration').exists())
            self.assertTrue((previous/'calibration/fold1/partial.pt').exists())

    def test_preflight_recalculates_complete_reused_measurements_without_forward_updates(self):
        import reuse_preflight
        with tempfile.TemporaryDirectory() as directory:
            root, durable, previous, old_bundle, value, rows = self.fixture(directory)
            with patch.object(reuse_preflight, 'binding', return_value=value):
                reuse_preflight.reuse_previous(root, durable, previous, old_bundle)
            runtime.write(root/'protocols/fold1_seed17.json', dict(calibration_seed=20260929))
            original = SimpleNamespace(reset_seed=lambda seed: None)
            slabs = SimpleNamespace(geometry_audit=[], calibration_schedule=lambda _: [(r['case'],r['anchor']) for r in rows[1]])
            with patch.object(engine, 'binding', return_value=value), \
                 patch.object(engine, 'load', return_value=(original,None,None,None,[],slabs,{})) as load, \
                 patch.object(engine, 'forward', side_effect=AssertionError('No measurement recomputation expected')):
                engine.preflight(root, durable, 1)
                engine.preflight(root, durable, 1)
                self.assertEqual(load.call_count, 1)
            cal = runtime.read(durable/'calibration/fold1/calibration.json')
            self.assertEqual(cal['rows'], rows[1])
            self.assertEqual(cal['arms']['bands']['weights'], dict(bands=.1))
            self.assertEqual(len(cal['arms']), 14)
            self.assertEqual(runtime.read(durable/'calibration/fold1/receipt.json')['status'], 'passed')
            self.assertFalse((durable/'preflight.json').exists())

    def test_historical_results_never_fill_matched_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); durable = root/'results'
            tasks = [dict(id=a, fold=0, seed=17, arm=a) for a in ('baseline','bands')]
            runtime.write(root/'plan.json', dict(tasks=tasks))
            runtime.write(root/'existing_results.json', dict(runs=[dict(fold=0, seed=17, arm='baseline', validation_dice=.99)]))
            runtime.write(durable/'runs/bands/completion.json', dict(validation_dice=.8))
            with patch.object(runtime, 'completed', side_effect=lambda r,d,t:t['arm']=='bands'):
                report = runtime.report(root, durable)
            self.assertEqual(report['new_completed'], 1)
            self.assertIsNone(report['runs'][0]['gain_pp'])
            runtime.write(durable/'runs/baseline/completion.json', dict(validation_dice=.7))
            with patch.object(runtime, 'completed', return_value=True): report = runtime.report(root, durable)
            self.assertEqual(report['new_completed'], 2)
            self.assertAlmostEqual(report['runs'][1]['gain_pp'], 10.)
            self.assertEqual(len(report['historical_reference']), 1)
            self.assertFalse(report['historical_results_used_as_matched_controls'])


class PackageTests(unittest.TestCase):
    def test_preflight_upgrade_preserves_evidence_and_stop(self):
        from scripts.colab_medsam3.prepare import SOURCE_EXPORT as EXPORT, prepare_preflight_upgrade
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'bundle'; durable = Path(directory)/'results'
            root.mkdir(); durable.mkdir()
            old_manifest = EXPORT/'preflight_original_bundle/bundle/bundle_manifest.json'
            (root/'bundle_manifest.json').write_bytes(old_manifest.read_bytes())
            (root/'engine.py').write_text('old engine'); (root/'runtime.py').write_text('old runtime')
            (durable/'STOP').write_text('manual pause')
            runtime.write(durable/'environment_binding.json', dict(old=True))
            runtime.write(durable/'calibration/fold0/receipt.json', dict(status='failed'))
            prepare_preflight_upgrade(root, durable, 'corrected manifest')
            backups = list((durable/'preflight_migrations').iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0]/'engine.py').read_text(), 'old engine')
            self.assertEqual(runtime.read(backups[0]/'calibration/fold0/receipt.json')['status'], 'failed')
            self.assertFalse((durable/'calibration').exists())
            self.assertEqual((durable/'STOP').read_text(), 'manual pause')

    def test_preflight_upgrade_refuses_any_scientific_or_recovery_work(self):
        from scripts.colab_medsam3.prepare import SOURCE_EXPORT as EXPORT, prepare_preflight_upgrade
        for location in ('bundle/work/run/checkpoint_latest.pt', 'results/recovery/run/charged_ledger.json',
                         'results/runs/run/archive_manifest.json'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                base = Path(directory); root = base/'bundle'; durable = base/'results'
                root.mkdir(); durable.mkdir()
                old_manifest = EXPORT/'preflight_original_bundle/bundle/bundle_manifest.json'
                (root/'bundle_manifest.json').write_bytes(old_manifest.read_bytes())
                artifact = base/location; artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text('must remain')
                with self.assertRaisesRegex(RuntimeError, 'Scientific work or recovery exists'):
                    prepare_preflight_upgrade(root, durable, 'corrected manifest')
                self.assertEqual(artifact.read_text(), 'must remain')
                self.assertFalse((durable/'preflight_migrations').exists())

    def test_scope_and_notebook_syntax(self):
        from scripts.colab_medsam3.prepare import EXPORT,notebook
        plan=runtime.read(EXPORT/'bundle/plan.json')
        self.assertEqual(len(plan['tasks']),108)
        self.assertEqual(len({r['id'] for r in plan['tasks']}),108)
        self.assertEqual(set(r['arm'] for r in plan['tasks'][:9]), {'baseline'})
        self.assertFalse(set(r['arm'] for r in plan['tasks'][9:]) & set(runtime.EXISTING))
        self.assertFalse({'bands', 'bands_edge'} & set(r['arm'] for r in plan['tasks']))
        self.assertEqual(plan['historical_results_role'], 'reference_only')
        self.assertEqual(plan['calibration_policy'], 'fresh_runtime_sequential')
        self.assertIsNone(plan['optimization_flop_cap'])
        for index,cell in enumerate(notebook({},'hash')['cells']):
            if cell['cell_type']=='code':compile(cell['source'],f'cell-{index}','exec')

    def test_all_native_support_geometry(self):
        import nibabel as nib
        from scripts.depth_supervision.geometry import build_geometry,signed_distance
        from scripts.colab_medsam3.prepare import EXPORT,ROOT
        for fold in runtime.FOLDS:
            p=runtime.read(EXPORT/'bundle/protocols'/f'fold{fold}_seed17.json')
            for case in p['train_cases']:
                image=nib.as_closest_canonical(nib.load(ROOT/'datasets/Dataset101_MSD/labelsTr'/f'{case}.nii.gz'))
                labels=np.asarray(image.dataobj).transpose(2,1,0)
                affine=image.affine.copy();affine[:3,:3]=image.affine[:3,:3][:,::-1]
                geometry=build_geometry(labels,affine,fields=False)
                self.assertTrue(np.isfinite(geometry['weights']).all())
                self.assertAlmostEqual(float(geometry['weights'].sum()),1.,places=6)
                for c in (1,2):
                    sdf=signed_distance(labels==c,(1.,1.,1.),8.)
                    self.assertTrue((sdf[labels==c]<0).all());self.assertTrue((sdf[labels!=c]>0).all())


if __name__=='__main__':unittest.main()
