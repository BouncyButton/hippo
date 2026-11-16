"""Frozen seed-zero long-schedule ablation of canonical non-BAND constraints."""
import inspect
from scripts.colab_hfh import constraint_worker
from scripts.colab_hfh.nnunet_seed0_long_worker import learning_rate, SCHEDULE


def modified_run(source):
    anchor = '            report.update(config_sha256=base.digest(config_path), anchor_successful_updates=step,'
    if source.count(anchor) != 1:
        raise ValueError('Calibration integration changed')
    source = source.replace(anchor, '''            expected = protocol['matched_readiness_anchor']
            if step != expected['after_update'] or report['anchor_model_sha256'] != expected['model_state_sha256']:
                raise ValueError('Native prefix differs from the saved matched control')
'''+anchor)
    anchor = '        latest_validation = result'
    if source.count(anchor) != 1:
        raise ValueError('Validation integration changed')
    source = source.replace(anchor, '''        base.atomic_json(args.output / f"validation_update{step}.json", result)
        base.atomic_json(args.output / f"training_update{step}.json", base.evaluate(model, train, device, amp))
'''+anchor)
    return source


def install(trainer, protocol, arm):
    if protocol['long_schedule'] != SCHEDULE or protocol['augmentation'] != 'none':
        raise ValueError('Matched long recipe changed')
    # Pass the additional instrumentation through the existing canonical installer.
    original = inspect.getsource
    run = trainer.run
    def getsource(obj):
        value = original(obj)
        return modified_run(value) if obj is run else value
    inspect.getsource = getsource
    try:
        constraint_worker.install(trainer, protocol, arm)
    finally:
        inspect.getsource = original
    trainer.base.learning_rate = lambda name, step, maximum, initial: learning_rate(step)
    trainer.base.validation_steps = lambda maximum, count: sorted(
        {n for n in range(10, maximum+1, 10)} | {60, min(179, maximum), maximum})


def source():
    value = constraint_worker.worker_source()
    old = 'from scripts.colab_hfh.constraint_worker import install'
    if value.count(old) != 1:
        raise ValueError('Worker integration changed')
    return value.replace(old, 'from scripts.colab_hfh.nnunet_long_constraints_worker import install')
