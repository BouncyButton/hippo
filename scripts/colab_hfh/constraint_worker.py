"""Bind the frozen HFH worker to one declared multi-constraint objective."""
from functools import partial
import inspect
from pathlib import Path


def install(trainer, protocol, arm):
    from scripts.colab_hfh.constraint_calibration import calibration_case, select_joint_weights
    from scripts.colab_hfh.constraint_geometry import install as install_geometry
    install_geometry(trainer, protocol, arm)
    trainer.calibration_case = calibration_case
    trainer.select_weights = partial(select_joint_weights,
        targets=protocol['constraint_component_targets'], component_cap=protocol['calibration_p95_cap'],
        joint_target=protocol['constraint_joint_target'], joint_cap=protocol['constraint_joint_cap'])
    original = trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original(*args, **kwargs)
        report.update(method=protocol['constraint_calibration'],
            scope='Exactly the ten frozen training crops; actual native loss versus auxiliary gradients at detached FP32 full-resolution logits; joint and ramp-subset caps; no parameter updates')
        return report
    trainer.calibrate = calibrate
    # A prefix-zero calibration must have a ledger before RecoveryWriter captures
    # its initial checkpoint. The same audited fix is used by the BAND worker.
    source = inspect.getsource(trainer.run)
    anchor = '    step, best_score, best_step, epoch, cursor = 0, -1., 0, 0, 0'
    if source.count(anchor) != 1:
        raise ValueError('Training recovery anchor changed')
    source = source.replace(anchor, '    if not ledger_path.exists():\n        base.atomic_json(ledger_path, ledger)\n'+anchor)
    audit_anchor='    base.atomic_json(args.output / "training_final.json", training_evaluation)'
    if source.count(audit_anchor)!=1:
        raise ValueError('Final gradient diagnostic anchor changed')
    source=source.replace(audit_anchor,audit_anchor+'''
    from scripts.colab_hfh.constraint_diagnostics import audit as audit_fixed_gradients
    audit_fixed_gradients(calibrate,model,train,args.objective,device,amp,native_criterion,
                          protocol,args.output,schedule)
    from scripts.colab_hfh.parameter_gradient_diagnostics import audit_final as audit_parameter_gradients
    audit_parameter_gradients(model,train,args.objective,device,amp,native_criterion,
                             protocol,args.output,schedule)
''')
    exec(compile(source, trainer.__file__, 'exec'), trainer.__dict__)


def worker_source():
    source = Path(__file__).with_name('baseline.py').read_text()
    replacements = [
        ("p['trainer_alias'], 'dice', 4, True", "p['trainer_alias'], p['active_objective'], 4, True"),
        ("exp.configure(p, 'dice')", "exp.configure(p, p['active_objective'])"),
        ("    binary.bind()", "    if p.get('band_inner_weight', .5) != .5 or p.get('band_adaptive_sides', False):\n        from scripts.colab_hfh.bands_variants import install as install_band_variant\n        install_band_variant(p)\n    binary.bind()"),
        ("    exp.base.PreparedPatients = PreparedPatients", "    from scripts.colab_hfh.constraint_worker import install\n    install(exp.trainer, p, p['active_objective'])\n    exp.base.PreparedPatients = PreparedPatients"),
        ("policy=p['sdf_policy']['dice']", "policy=p['sdf_policy'][p['active_objective']]"),
        ("    builder = exp.base.build_model", "    if p.get('online_sdf_gradient_ratio') is not None:\n        from scripts.colab_hfh.constraint_online_sdf import OnlineSDF\n        exp.trainer.training_loss = OnlineSDF(exp.trainer.training_loss, p, args.output)\n    builder = exp.base.build_model"),
        ("        return model\n    exp.base.build_model = build",
         "        reference=read(args.protocol_dir/'baseline_references.json')[f'{p[\"model_identity\"]}_f{args.fold}_s{args.seed}']\n        if exp.trainer.model_digest(model) != reference['initialization']['model_state_sha256']:\n            raise ValueError('Initialization differs from matched baseline')\n        return model\n    exp.base.build_model = build"),
        ("loss=p['training_loss'], augmentation='none'",
         "loss=p['training_loss'], objective=p['active_objective'], constraint_definition=p['constraint_definition'], augmentation='none'"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Frozen HFH worker integration changed')
        source = source.replace(before, after)
    return source


if __name__ == '__main__':
    exec(compile(worker_source(), __file__, 'exec'), {'__name__': '__main__', '__file__': __file__})
