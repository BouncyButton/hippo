"""Explicit delayed-constraint ablation without changing native optimization."""
import inspect


def constraint_ramp(update, protocol):
    delay = protocol['constraint_recipe'].get('delay', 0)
    warmup = protocol['constraint_warmup_updates']
    if not isinstance(delay, int) or not 0 <= delay < protocol['updates'] or warmup <= 0:
        raise ValueError('Invalid constraint activation schedule')
    return min(max((update + 1 - delay) / warmup, 0.), 1.)


def delayed_source(source, full_native_graph=False):
    replacements = (
        ("ramp=min((update+1)/protocol['constraint_warmup_updates'],1.)",
         'ramp=hfh_constraint_ramp(update, protocol)'),
        ("total=losses['native']+ramp*sum(calibration['weights'][name]*losses[name] for name in calibration['weights'])",
         "total=losses['native'] if ramp == 0 else losses['native']+ramp*sum(calibration['weights'][name]*losses[name] for name in calibration['weights'])"),
    )
    for old, new in replacements[:1] if full_native_graph else replacements:
        if source.count(old) != 1:
            raise ValueError('Audited native training loop changed: '+old)
        source = source.replace(old, new)
    return source


def install(original, protocol):
    if not protocol['constraint_recipe'].get('delay', 0):
        return
    constraint_ramp(0, protocol)
    source = delayed_source(inspect.getsource(original.train),
                            protocol['constraint_recipe'].get('full_native_graph', False))
    # Keep the live module globals: the optimized runner installs recovery hooks
    # after this adapter. Copying globals would silently bypass those hooks.
    original.__dict__['hfh_constraint_ramp'] = constraint_ramp
    exec(compile(source, __file__, 'exec'), original.__dict__)
